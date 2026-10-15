#include "batch_mma.cuh"
namespace batch_mma {
static int splits(const Plan& p){return 1+(p.k/128-1)/p.groups_per_split;}
cudaError_t validate(const Plan& p){
  if(p.n<=0||p.n%32||p.n>2147483520||p.k<=0||p.k%128||p.r<1||p.r>32||p.groups_per_split<=0||
     (p.n_tile!=32&&p.n_tile!=64&&p.n_tile!=128))return cudaErrorInvalidValue;
  if(splits(p)>65535)return cudaErrorInvalidValue;
  return cudaSuccess;
}
cudaError_t select(int n,int k,int r,Plan* out){
  if(!out)return cudaErrorInvalidValue;
  Plan p{n,k,r,16,32};auto e=validate(p);if(e!=cudaSuccess)return e;
  if(r!=5&&r!=8&&r!=16&&r!=32)return cudaErrorNotSupported;
  if(n==5120&&k==17408)p.n_tile=r==32?128:r==16?32:64;
  else if(n==248320&&k==5120){if(r==32){p.n_tile=64;p.groups_per_split=k/128;}}
  else if(!((n==34816||n==65536)&&k==5120))return cudaErrorNotSupported;
  *out=p;return cudaSuccess;
}
size_t partial_bytes(const Plan& p){return splits(p)>1?size_t(p.r)*p.n*splits(p)*sizeof(float):0;}
__device__ __forceinline__ unsigned expand(unsigned w){return (w*0x00204081u)&0x01010101u;}
__device__ __forceinline__ void mma(int* c,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ int swizzle(int row,int col){return row*32+(col^((row&7)<<2));}
template<int NT,int PACKS> __global__ void projection(const uint32_t* __restrict__ signs,const __half* __restrict__ ws,
  bitserial::Carrier in,float* y,int ldo,float* partial,int n,int groups,int r,int chunk,int ns,float alpha,float beta){
  __shared__ unsigned q[PACKS*16*32];
  __shared__ float sx[PACKS*16];__shared__ int dc[PACKS*16];__shared__ unsigned status[PACKS*16];
  int lane=threadIdx.x%32,warp=threadIdx.x/32,gid=lane/4,tid=lane%4;
  int token=(warp/(NT/8))*16+gid,feature=blockIdx.x*NT+(warp%(NT/8))*8;
  int bf=feature+gid,cf=feature+tid*2,split=blockIdx.y;
  float sum[4]={};int first=split*chunk,last=min(first+chunk,groups);
  for(int g=first;g<last;g++){
    for(int i=threadIdx.x;i<PACKS*16*32;i+=blockDim.x){
      int row=i/32,col=i%32;
      q[swizzle(row,col)]=row<r?reinterpret_cast<const unsigned*>(in.q)[(size_t(row)*groups+g)*32+col]:0;
    }
    if(threadIdx.x<PACKS*16){int row=threadIdx.x;size_t ix=size_t(row)*groups+g;
      sx[row]=row<r?in.scales[ix]:0;dc[row]=row<r?in.dc[ix]:0;status[row]=row<r?unsigned(in.status[ix]):0;}
    __syncthreads();
    int c[4]={};
    #pragma unroll
    for(int s=0;s<4;s++){
      unsigned w=bf<n?signs[(size_t(g)*4+s)*n+bf]:0;
      mma(c,q[swizzle(token,s*8+tid)],q[swizzle(token+8,s*8+tid)],
            q[swizzle(token,s*8+tid+4)],q[swizzle(token+8,s*8+tid+4)],
            expand((w>>(tid*4))&15),expand((w>>(tid*4+16))&15));
    }
    #pragma unroll
    for(int j=0;j<4;j++){
      int row=token+(j/2)*8,f=cf+j%2;
      if(row<r&&f<n){float scale=__half2float(ws[size_t(g)*n+f])*sx[row];
        sum[j]=status[row]?__int_as_float(0x7fc00000):fmaf(float(2*c[j]-dc[row]),scale,sum[j]);}
    }
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<4;j++){
    int row=token+(j/2)*8,f=cf+j%2;
    if(row<r&&f<n){if(ns>1)partial[(size_t(row)*ns+split)*n+f]=sum[j];
      else {size_t ix=size_t(row)*ldo+f;y[ix]=beta==0?alpha*sum[j]:fmaf(alpha,sum[j],beta*y[ix]);}}
  }
}
__global__ void continuation(const float* partial,float* y,int n,int ldo,int ns,float alpha,float beta){
  int f=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;if(f>=n)return;
  float sum=0;for(int s=0;s<ns;s++)sum+=partial[(size_t(row)*ns+s)*n+f];
  size_t ix=size_t(row)*ldo+f;y[ix]=beta==0?alpha*sum:fmaf(alpha,sum,beta*y[ix]);
}
template<int NT> static void launch(const Plan& p,const uint32_t* signs,const __half* ws,bitserial::Carrier in,
  float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t st){
  dim3 grid((p.n+NT-1)/NT,splits(p));
  if(p.r<=16)projection<NT,1><<<grid,(NT/8)*32,0,st>>>(signs,ws,in,y,ldo,partial,p.n,p.k/128,p.r,p.groups_per_split,splits(p),alpha,beta);
  else projection<NT,2><<<grid,(NT/8)*64,0,st>>>(signs,ws,in,y,ldo,partial,p.n,p.k/128,p.r,p.groups_per_split,splits(p),alpha,beta);
}
cudaError_t project_carrier(const Plan& p,const uint32_t* signs,const __half* ws,bitserial::Carrier in,
  float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t st){
  auto e=validate(p);if(e!=cudaSuccess)return e;
  if(!signs||!ws||!in.q||!in.scales||!in.dc||!in.status||!y||ldo<p.n||(splits(p)>1&&!partial))return cudaErrorInvalidValue;
  switch(p.n_tile){case 32:launch<32>(p,signs,ws,in,y,ldo,partial,alpha,beta,st);break;
    case 64:launch<64>(p,signs,ws,in,y,ldo,partial,alpha,beta,st);break;
    case 128:launch<128>(p,signs,ws,in,y,ldo,partial,alpha,beta,st);break;}
  e=cudaGetLastError();if(e!=cudaSuccess)return e;
  if(splits(p)>1){continuation<<<dim3((p.n+255)/256,p.r),256,0,st>>>(partial,y,p.n,ldo,splits(p),alpha,beta);e=cudaGetLastError();}
  return e;
}
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* ws,const float* x,bitserial::Carrier scratch,
  float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t st){
  auto e=validate(p);if(e!=cudaSuccess)return e;
  e=bitserial::produce(bitserial::Route::Dp4a01,x,p.k,p.r,scratch,st);if(e!=cudaSuccess)return e;
  return project_carrier(p,signs,ws,scratch,y,ldo,partial,alpha,beta,st);
}
}
