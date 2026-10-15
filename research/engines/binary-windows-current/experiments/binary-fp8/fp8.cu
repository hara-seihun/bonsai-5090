#include "fp8.cuh"
namespace binary_fp8 {
static int splits(const Plan& p){return 1+(p.k/128-1)/p.groups_per_split;}
cudaError_t validate(const Plan& p){
  if(p.n<=0||p.n%32||p.n>2147483520||p.k<=0||p.k%128||p.r<1||p.r>32||p.groups_per_split<=0||
     (p.n_tile!=32&&p.n_tile!=64&&p.n_tile!=128))return cudaErrorInvalidValue;
  if(splits(p)>65535)return cudaErrorInvalidValue;return cudaSuccess;
}
size_t partial_bytes(const Plan& p){return splits(p)>1?size_t(p.r)*p.n*splits(p)*4:0;}
__global__ void producer(const float* x,int k,Carrier out){
  int g=blockIdx.x,row=blockIdx.y,t=threadIdx.x,lane=t%32,warp=t/32;size_t base=(size_t(row)*(k/128)+g)*128;
  float v=x[base+t];bool bad=!isfinite(v);float a=bad?0:fabsf(v);
  __shared__ float maxima[4],inverse;__shared__ unsigned status;
  for(int d=16;d;d/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));if(lane==0)maxima[warp]=a;
  int nonfinite=__syncthreads_or(bad);
  if(t==0){float maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
    status=nonfinite?1:maximum!=0&&(maximum<1e-30f||maximum>1e30f)?2:0;
    inverse=status||maximum==0?0:448.f/maximum;out.scales[base/128]=status?0:maximum/448.f;
    out.status[base/128]=static_cast<bitserial::InputStatus>(status);
    if(status&&out.error)atomicMax(reinterpret_cast<unsigned*>(out.error),status);
  }
  __syncthreads();out.q[base+t]=status?0:__nv_cvt_float_to_fp8(v*inverse,__NV_SATFINITE,__NV_E4M3);
}
cudaError_t produce(const float* x,int k,int r,Carrier out,cudaStream_t st){
  if(!x||!out.q||!out.scales||!out.status||k<=0||k%128||r<1||r>65535)return cudaErrorInvalidValue;
  producer<<<dim3(k/128,r),128,0,st>>>(x,k,out);return cudaGetLastError();
}
__device__ __forceinline__ unsigned expand(unsigned nibble){return 0xb8b8b8b8u^(((nibble*0x00204081u)&0x01010101u)<<7);}
__device__ __forceinline__ void mma(float* c,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ int swizzle(int row,int col){return row*32+(col^((row&7)<<2));}
template<int NT,int PACKS> __global__ void projection(const uint32_t* __restrict__ signs,const __half* __restrict__ ws,
  Carrier in,float* y,int ldo,float* partial,int n,int groups,int r,int chunk,int ns,float alpha,float beta){
  __shared__ unsigned q[PACKS*8*32];__shared__ float sx[PACKS*8];__shared__ unsigned status[PACKS*8];
  int lane=threadIdx.x%32,warp=threadIdx.x/32,gid=lane/4,tid=lane%4;
  int token=warp/(NT/16)*8,feature=blockIdx.x*NT+warp%(NT/16)*16+gid;
  int split=blockIdx.y;float sum[4]={};int first=split*chunk,last=min(first+chunk,groups);
  for(int g=first;g<last;g++){
    for(int i=threadIdx.x;i<PACKS*8*32;i+=blockDim.x){int row=i/32,col=i%32;
      q[swizzle(row,col)]=row<r?reinterpret_cast<const unsigned*>(in.q)[(size_t(row)*groups+g)*32+col]:0;}
    if(threadIdx.x<PACKS*8){int row=threadIdx.x;size_t ix=size_t(row)*groups+g;
      sx[row]=row<r?in.scales[ix]:0;status[row]=row<r?unsigned(in.status[ix]):0;}
    __syncthreads();float c[4]={};
    #pragma unroll
    for(int s=0;s<4;s++){
      unsigned w0=feature<n?signs[(size_t(g)*4+s)*n+feature]:0;
      unsigned w8=feature+8<n?signs[(size_t(g)*4+s)*n+feature+8]:0;
      mma(c,expand((w0>>(tid*4))&15),expand((w8>>(tid*4))&15),
            expand((w0>>(tid*4+16))&15),expand((w8>>(tid*4+16))&15),
            q[swizzle(token+gid,s*8+tid)],q[swizzle(token+gid,s*8+tid+4)]);}
    float scale0=feature<n?__half2float(ws[size_t(g)*n+feature]):0;
    float scale8=feature+8<n?__half2float(ws[size_t(g)*n+feature+8]):0;
    #pragma unroll
    for(int j=0;j<4;j++){int row=token+tid*2+j%2,f=feature+j/2*8;
      if(row<r&&f<n)sum[j]=status[row]?__int_as_float(0x7fc00000):fmaf(c[j],(j<2?scale0:scale8)*sx[row],sum[j]);}
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<4;j++){int row=token+tid*2+j%2,f=feature+j/2*8;
    if(row<r&&f<n){if(ns>1)partial[(size_t(row)*ns+split)*n+f]=sum[j];
      else{size_t ix=size_t(row)*ldo+f;y[ix]=beta==0?alpha*sum[j]:fmaf(alpha,sum[j],beta*y[ix]);}}}
}
__global__ void continuation(const float* partial,float* y,int n,int ldo,int ns,float alpha,float beta){
  int f=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;if(f>=n)return;
  float sum=0;for(int s=0;s<ns;s++)sum+=partial[(size_t(row)*ns+s)*n+f];
  size_t ix=size_t(row)*ldo+f;y[ix]=beta==0?alpha*sum:fmaf(alpha,sum,beta*y[ix]);
}
template<int NT> static void launch(const Plan& p,const uint32_t* signs,const __half* ws,Carrier in,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t st){
  dim3 grid((p.n+NT-1)/NT,splits(p));
  if(p.r<=8)projection<NT,1><<<grid,NT/16*32,0,st>>>(signs,ws,in,y,ldo,partial,p.n,p.k/128,p.r,p.groups_per_split,splits(p),alpha,beta);
  else if(p.r<=16)projection<NT,2><<<grid,NT/16*64,0,st>>>(signs,ws,in,y,ldo,partial,p.n,p.k/128,p.r,p.groups_per_split,splits(p),alpha,beta);
  else projection<NT,4><<<grid,NT/16*128,0,st>>>(signs,ws,in,y,ldo,partial,p.n,p.k/128,p.r,p.groups_per_split,splits(p),alpha,beta);
}
cudaError_t project_carrier(const Plan& p,const uint32_t* signs,const __half* ws,Carrier in,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t st){
  auto e=validate(p);if(e!=cudaSuccess)return e;
  if(!signs||!ws||!in.q||!in.scales||!in.status||!y||ldo<p.n||(splits(p)>1&&!partial))return cudaErrorInvalidValue;
  switch(p.n_tile){case 32:launch<32>(p,signs,ws,in,y,ldo,partial,alpha,beta,st);break;
    case 64:launch<64>(p,signs,ws,in,y,ldo,partial,alpha,beta,st);break;case 128:launch<128>(p,signs,ws,in,y,ldo,partial,alpha,beta,st);break;}
  e=cudaGetLastError();if(e!=cudaSuccess)return e;
  if(splits(p)>1){continuation<<<dim3((p.n+255)/256,p.r),256,0,st>>>(partial,y,p.n,ldo,splits(p),alpha,beta);e=cudaGetLastError();}return e;
}
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* ws,const float* x,Carrier scratch,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t st){
  auto e=validate(p);if(e!=cudaSuccess)return e;e=produce(x,p.k,p.r,scratch,st);if(e!=cudaSuccess)return e;
  return project_carrier(p,signs,ws,scratch,y,ldo,partial,alpha,beta,st);
}
}
