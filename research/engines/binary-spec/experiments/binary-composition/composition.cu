#include "composition.cuh"
#include <climits>
namespace composition {
using bitserial::Carrier;
using bitserial::InputStatus;
cudaError_t validate(Plan p) {
  if(p.n<=0||p.n>INT_MAX/2||p.n%128||p.k<=0||p.k%128||p.r<1||p.r>16||
     (p.tile!=1&&p.tile!=4&&p.tile!=8)||p.r%p.tile||p.chunk<1||p.chunk>p.k/128)return cudaErrorInvalidValue;
  return cudaSuccess;
}
size_t partial_bytes(Plan p){int ns=(p.k/128+p.chunk-1)/p.chunk;return ns==1?0:size_t(p.r)*ns*2*p.n*4;}
__device__ float gated(float g,float u){return (g/(1.f+__expf(-g)))*u;}
__device__ void quant(float v,int feature,int token,int n,Carrier out) {
  __shared__ float maxima[4],inverse;
  __shared__ int sums[4];__shared__ unsigned status;
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  float a=isfinite(v)?fabsf(v):0;
  for(int d=16;d;d/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
  if(lane==0)maxima[warp]=a;
  int bad=__syncthreads_or(!isfinite(v));
  size_t base=size_t(token)*(n/128)+feature/128;
  if(threadIdx.x==0) {
    float m=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
    status=bad?1:m!=0&&(m<1e-30f||m>1e30f)?2:0;
    inverse=status||m==0?0:127.f/m;
    out.scales[base]=status?0:m/127.f;out.status[base]=InputStatus(status);
    if(status&&out.error)atomicMax(reinterpret_cast<unsigned*>(out.error),status);
  }
  __syncthreads();
  int q=status?0:__float2int_rn(v*inverse);out.q[size_t(token)*n+feature]=int8_t(q);
  for(int d=16;d;d/=2)q+=__shfl_xor_sync(0xffffffff,q,d);
  if(lane==0)sums[warp]=q;__syncthreads();
  if(threadIdx.x==0)out.dc[base]=sums[0]+sums[1]+sums[2]+sums[3];
  __syncthreads();
}
template<int T> __global__ void paired(const uint32_t* signs,const __half* scales,Carrier in,Carrier out,
                                      float* partial,int n,int groups,int chunk,int ns) {
  __shared__ uint32_t q[T*32];__shared__ float sx[T];__shared__ int dc[T];__shared__ unsigned status[T];
  int feature=blockIdx.x*128+threadIdx.x,split=blockIdx.y,token=blockIdx.z*T;
  float gate[T]={},up[T]={};
  int last=min((split+1)*chunk,groups);
  for(int g=split*chunk;g<last;g++) {
    for(int j=threadIdx.x;j<T*32;j+=128)q[j]=reinterpret_cast<const uint32_t*>(in.q)[(size_t(token+j/32)*groups+g)*32+j%32];
    if(threadIdx.x<T){size_t i=size_t(token+threadIdx.x)*groups+g;sx[threadIdx.x]=in.scales[i];dc[threadIdx.x]=in.dc[i];status[threadIdx.x]=unsigned(in.status[i]);}
    __syncthreads();
    int dg[T][4]={},du[T][4]={};
    #pragma unroll
    for(int w=0;w<4;w++) {
      size_t offset=(size_t(g)*4+w)*(2*n)+feature;
      unsigned wg=signs[offset],wu=signs[offset+n];
      #pragma unroll
      for(int j=0;j<8;j++) {
        unsigned eg=(((wg>>(4*j))&15)*0x00204081u)&0x01010101u;
        unsigned eu=(((wu>>(4*j))&15)*0x00204081u)&0x01010101u;
        #pragma unroll
        for(int t=0;t<T;t++){int x=int(q[t*32+w*8+j]);dg[t][w]=__dp4a(int(eg),x,dg[t][w]);du[t][w]=__dp4a(int(eu),x,du[t][w]);}
      }
    }
    float sg=__half2float(scales[size_t(g)*(2*n)+feature]);
    float su=__half2float(scales[size_t(g)*(2*n)+feature+n]);
    #pragma unroll
    for(int t=0;t<T;t++) {
      int ig=2*((dg[t][0]+dg[t][1])+(dg[t][2]+dg[t][3]))-dc[t];
      int iu=2*((du[t][0]+du[t][1])+(du[t][2]+du[t][3]))-dc[t];
      gate[t]=status[t]?__int_as_float(0x7fc00000):fmaf(float(ig),sg*sx[t],gate[t]);
      up[t]=status[t]?__int_as_float(0x7fc00000):fmaf(float(iu),su*sx[t],up[t]);
    }
    __syncthreads();
  }
  #pragma unroll
  for(int t=0;t<T;t++) {
    if(ns>1){size_t i=(size_t(token+t)*ns+split)*(2*n)+feature;partial[i]=gate[t];partial[i+n]=up[t];}
    else quant(gated(gate[t],up[t]),feature,token+t,n,out);
  }
}
__global__ void finalize(const float* partial,int n,int ns,Carrier out) {
  int feature=blockIdx.x*128+threadIdx.x,token=blockIdx.y;
  float g=0,u=0;
  for(int s=0;s<ns;s++){size_t i=(size_t(token)*ns+s)*(2*n)+feature;g+=partial[i];u+=partial[i+n];}
  quant(gated(g,u),feature,token,n,out);
}
__global__ void activation(const float* gu,float* h,int n) {
  int feature=blockIdx.x*128+threadIdx.x,token=blockIdx.y;
  size_t i=size_t(token)*(2*n)+feature;
  h[size_t(token)*n+feature]=gated(gu[i],gu[i+n]);
}
cudaError_t unfused_tail(const float* gu,int n,int r,Carrier out,float* h,cudaStream_t st) {
  if(!gu||!h||n<=0||n>INT_MAX/2||n%128||r<=0||r>16)return cudaErrorInvalidValue;
  activation<<<dim3(n/128,r),128,0,st>>>(gu,h,n);
  cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return e;
  return bitserial::produce(bitserial::Route::Dp4a01,h,n,r,out,st);
}
cudaError_t fused(Plan p,const uint32_t* signs,const __half* scales,Carrier in,Carrier out,float* partial,cudaStream_t st) {
  cudaError_t e=validate(p);if(e!=cudaSuccess)return e;
  if(!signs||!scales||!in.q||!in.scales||!in.dc||!in.status||!out.q||!out.scales||!out.dc||!out.status||
     (partial_bytes(p)&&!partial))return cudaErrorInvalidValue;
  int ns=(p.k/128+p.chunk-1)/p.chunk;dim3 grid(p.n/128,ns,p.r/p.tile);
  switch(p.tile){
    case 1:paired<1><<<grid,128,0,st>>>(signs,scales,in,out,partial,p.n,p.k/128,p.chunk,ns);break;
    case 4:paired<4><<<grid,128,0,st>>>(signs,scales,in,out,partial,p.n,p.k/128,p.chunk,ns);break;
    case 8:paired<8><<<grid,128,0,st>>>(signs,scales,in,out,partial,p.n,p.k/128,p.chunk,ns);break;
  }
  e=cudaGetLastError();if(e!=cudaSuccess)return e;
  if(ns>1){finalize<<<dim3(p.n/128,p.r),128,0,st>>>(partial,p.n,ns,out);e=cudaGetLastError();}
  return e;
}
}
