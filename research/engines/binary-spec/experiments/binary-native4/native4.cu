#include "native4.cuh"
namespace native4 {
static int splits(const Plan& p){return 1+(p.k/128-1)/p.groups_per_split;}
cudaError_t validate(const Plan& p){
  if(p.n<=0||p.n%32||p.n>2147483520||p.k<=0||p.k%128||p.r<1||p.r>32||p.groups_per_split<=0||
     (p.n_tile!=32&&p.n_tile!=64&&p.n_tile!=128))return cudaErrorInvalidValue;
  if(splits(p)>65535)return cudaErrorInvalidValue;return cudaSuccess;
}
size_t weight_bytes(int n,int k){return size_t(n)*k/2;}
size_t dc_bytes(int n,int k){return size_t(n)*(k/128);}
size_t plane_bytes(int k,int r){return size_t(k)*r/2;}
size_t partial_bytes(const Plan& p){return splits(p)>1?size_t(p.r)*p.n*splits(p)*4:0;}
__global__ void prepare_kernel(const uint32_t* signs,Weights out,int n,int groups){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*groups)return;
  int f=i%n,g=i/n;unsigned w[4];int dc=0;
  #pragma unroll
  for(int j=0;j<4;j++){w[j]=signs[(size_t(g)*4+j)*n+f];dc+=2*__popc(w[j])-32;}
  out.dc[i]=int8_t(dc/2);
  #pragma unroll
  for(int j=0;j<16;j++){
    unsigned src=w[j/4]>>(8*(j%4)),word=0;
    #pragma unroll
    for(int b=0;b<8;b++)word|=((src>>b)&1?1u:15u)<<(4*b);
    out.nibbles[(size_t(g)*16+j)*n+f]=word;
  }
}
cudaError_t prepare(const uint32_t* signs,Weights out,int n,int k,cudaStream_t st){
  if(!signs||!out.nibbles||!out.dc||!out.scales||n<=0||n%32||k<=0||k%128)return cudaErrorInvalidValue;
  prepare_kernel<<<(size_t(n)*(k/128)+255)/256,256,0,st>>>(signs,out,n,k/128);return cudaGetLastError();
}
__global__ void repack_kernel(int k,Carrier out){
  int g=blockIdx.x,t=blockIdx.y,lane=threadIdx.x;size_t base=(size_t(t)*(k/128)+g)*128;
  unsigned u=unsigned(int(out.a8.q[base+lane])+128),low=(u&15)<<(4*(lane%8)),high=(u>>4)<<(4*(lane%8));
  #pragma unroll
  for(int d=4;d;d/=2){low|=__shfl_down_sync(0xffffffff,low,d);high|=__shfl_down_sync(0xffffffff,high,d);}
  if(lane%8==0){size_t dst=(base/128)*16+lane/8;out.low[dst]=low;out.high[dst]=high;}
}
cudaError_t repack(int k,int r,Carrier out,cudaStream_t st){
  if(k<=0||k%128||r<=0||r>65535||!out.low||!out.high||!out.a8.q)return cudaErrorInvalidValue;
  repack_kernel<<<dim3(k/128,r),128,0,st>>>(k,out);return cudaGetLastError();
}
cudaError_t produce(const float* x,int k,int r,Carrier out,cudaStream_t st){
  if(!out.low||!out.high)return cudaErrorInvalidValue;
  auto e=bitserial::produce(bitserial::Route::Dp4a01,x,k,r,out.a8,st);if(e!=cudaSuccess)return e;return repack(k,r,out,st);
}
__device__ __forceinline__ void mma(int* c,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ int swizzle(int row,int col){return row*32+(col^((row&7)<<2));}
template<int NT,int PACKS> __global__ __launch_bounds__(NT/8*32*PACKS,1) void projection(Weights w,Carrier in,float* y,int ldo,float* partial,
  int n,int groups,int r,int chunk,int ns,float alpha,float beta){
  __shared__ unsigned low[PACKS*16*32],high[PACKS*16*32];
  __shared__ float sx[PACKS*16];__shared__ unsigned status[PACKS*16];
  int lane=threadIdx.x%32,warp=threadIdx.x/32,gid=lane/4,tid=lane%4;
  int token=warp/(NT/8)*16+gid,feature=blockIdx.x*NT+warp%(NT/8)*8;
  int bf=feature+gid,cf=feature+tid*2,split=blockIdx.y;float sum[4]={};
  int first=split*chunk,last=min(first+chunk,groups);
  for(int g=first;g<last;g++){
    for(int i=threadIdx.x;i<PACKS*16*16;i+=blockDim.x){
      int row=i/16,col=i%16;size_t ix=(size_t(row)*groups+g)*16+col;
      low[swizzle(row,col)]=row<r?in.low[ix]:0;high[swizzle(row,col)]=row<r?in.high[ix]:0;
    }
    if(threadIdx.x<PACKS*16){int row=threadIdx.x;size_t ix=size_t(row)*groups+g;
      sx[row]=row<r?in.a8.scales[ix]:0;status[row]=row<r?unsigned(in.a8.status[ix]):0;}
    __syncthreads();int cl[4]={},ch[4]={};
    #pragma unroll
    for(int s=0;s<2;s++){
      unsigned b0=bf<n?w.nibbles[(size_t(g)*16+s*8+tid)*n+bf]:0;
      unsigned b1=bf<n?w.nibbles[(size_t(g)*16+s*8+tid+4)*n+bf]:0;
      mma(cl,low[swizzle(token,s*8+tid)],low[swizzle(token+8,s*8+tid)],
             low[swizzle(token,s*8+tid+4)],low[swizzle(token+8,s*8+tid+4)],b0,b1);
      mma(ch,high[swizzle(token,s*8+tid)],high[swizzle(token+8,s*8+tid)],
             high[swizzle(token,s*8+tid+4)],high[swizzle(token+8,s*8+tid+4)],b0,b1);
    }
    #pragma unroll
    for(int j=0;j<4;j++){
      int row=token+j/2*8,f=cf+j%2;
      if(row<r&&f<n){size_t ix=size_t(g)*n+f;int dot=cl[j]+16*ch[j]-256*int(w.dc[ix]);
        sum[j]=status[row]?__int_as_float(0x7fc00000):fmaf(float(dot),__half2float(w.scales[ix])*sx[row],sum[j]);}
    }
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<4;j++){
    int row=token+j/2*8,f=cf+j%2;
    if(row<r&&f<n){if(ns>1)partial[(size_t(row)*ns+split)*n+f]=sum[j];
      else{size_t ix=size_t(row)*ldo+f;y[ix]=beta==0?alpha*sum[j]:fmaf(alpha,sum[j],beta*y[ix]);}}
  }
}
__global__ void continuation(const float* partial,float* y,int n,int ldo,int ns,float alpha,float beta){
  int f=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;if(f>=n)return;
  float sum=0;for(int s=0;s<ns;s++)sum+=partial[(size_t(row)*ns+s)*n+f];
  size_t ix=size_t(row)*ldo+f;y[ix]=beta==0?alpha*sum:fmaf(alpha,sum,beta*y[ix]);
}
template<int NT> static void launch(const Plan& p,Weights w,Carrier in,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t st){
  dim3 grid((p.n+NT-1)/NT,splits(p));
  if(p.r<=16)projection<NT,1><<<grid,NT/8*32,0,st>>>(w,in,y,ldo,partial,p.n,p.k/128,p.r,p.groups_per_split,splits(p),alpha,beta);
  else projection<NT,2><<<grid,NT/8*64,0,st>>>(w,in,y,ldo,partial,p.n,p.k/128,p.r,p.groups_per_split,splits(p),alpha,beta);
}
cudaError_t project_carrier(const Plan& p,Weights w,Carrier in,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t st){
  auto e=validate(p);if(e!=cudaSuccess)return e;
  if(!w.nibbles||!w.dc||!w.scales||!in.low||!in.high||!in.a8.scales||!in.a8.status||!y||ldo<p.n||(splits(p)>1&&!partial))return cudaErrorInvalidValue;
  switch(p.n_tile){case 32:launch<32>(p,w,in,y,ldo,partial,alpha,beta,st);break;
    case 64:launch<64>(p,w,in,y,ldo,partial,alpha,beta,st);break;case 128:launch<128>(p,w,in,y,ldo,partial,alpha,beta,st);break;}
  e=cudaGetLastError();if(e!=cudaSuccess)return e;
  if(splits(p)>1){continuation<<<dim3((p.n+255)/256,p.r),256,0,st>>>(partial,y,p.n,ldo,splits(p),alpha,beta);e=cudaGetLastError();}return e;
}
cudaError_t project(const Plan& p,Weights w,const float* x,Carrier scratch,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t st){
  auto e=validate(p);if(e!=cudaSuccess)return e;e=produce(x,p.k,p.r,scratch,st);if(e!=cudaSuccess)return e;
  return project_carrier(p,w,scratch,y,ldo,partial,alpha,beta,st);
}
}
