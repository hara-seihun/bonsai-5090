#pragma once
#include "model.hpp"
#include "q2k.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
namespace halo6000 {
__device__ __forceinline__ float warp_sum(float x) {
 #pragma unroll
 for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);
 return x;
}
__device__ __forceinline__ float warp_max(float x) {
 #pragma unroll
 for(int d=16;d;d>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));
 return x;
}
__device__ __forceinline__ int pane_operand(uint32_t x) {return __byte_perm(0x00000100u,0x0000ff00u,x);}
__device__ __forceinline__ int load_i8x4(const int8_t*p){return *reinterpret_cast<const int*>(p);}
__device__ __forceinline__ float half_value(const uint16_t*p){return __half2float(*reinterpret_cast<const half*>(p));}
__device__ __forceinline__ float pane_group(const MatrixView&w,ActivationView a,int row,int group,int expert=0,int ar=0) {
 const size_t b=(size_t(expert)*w.n+row)*(w.k/128)+group;
 const auto*base=static_cast<const char*>(w.data)+(b>>4)*544;
 const auto*words=reinterpret_cast<const uint32_t*>(base+(b&15)*32);
 const int8_t*q=a.qs+ar*a.q_stride+group*128;
 const float*s=a.scale+ar*a.scale_stride+group*4;
 float v=0;
 #pragma unroll
 for(int c=0;c<4;c++){
  int isum=0;
  #pragma unroll
  for(int h=0;h<2;h++){
   const uint32_t p=words[2*c+h],lo=p&0x55555555u,hi=(p>>1)&0x55555555u;
   isum=__dp4a(pane_operand(lo),load_i8x4(q+c*32+h*16),isum);
   isum=__dp4a(pane_operand(lo>>16),load_i8x4(q+c*32+h*16+4),isum);
   isum=__dp4a(pane_operand(hi),load_i8x4(q+c*32+h*16+8),isum);
   isum=__dp4a(pane_operand(hi>>16),load_i8x4(q+c*32+h*16+12),isum);
  }
  v+=s[c]*float(isum);
 }
 return half_value(reinterpret_cast<const uint16_t*>(base+512+(b&15)*2))*v;
}
__device__ __forceinline__ float q8_group(const MatrixView&w,ActivationView a,int row,int group,int expert=0,int ar=0) {
 const size_t b=(size_t(expert)*w.n+row)*(w.k/128)+group;
 const auto*base=static_cast<const char*>(w.data)+b*136;
 const auto*q=reinterpret_cast<const int8_t*>(base);
 const auto*scales=reinterpret_cast<const uint16_t*>(base+128);
 const auto*x=a.qs+ar*a.q_stride+group*128;
 const auto*ds=a.scale+ar*a.scale_stride+group*4;
 float v=0;
 #pragma unroll
 for(int c=0;c<4;c++){
  int sum=0;
  #pragma unroll
  for(int t=0;t<8;t++)sum=__dp4a(load_i8x4(q+c*32+t*4),load_i8x4(x+c*32+t*4),sum);
  v+=half_value(scales+c)*ds[c]*float(sum);
 }
 return v;
}
// A warp owns one output row; all lanes must enter even for zero-sized tails.
__device__ __forceinline__ float quantized_row(const MatrixView&w,ActivationView a,int row,int expert=0,int ar=0){
 if(w.format==WeightFormat::Q2K)return q2k_row<32>(w,a,row,expert,ar);
 float v=0;const int lane=threadIdx.x&31;
 for(int b=lane;b<w.k/128;b+=32){
  v+=w.format==WeightFormat::Pane2?pane_group(w,a,row,b,expert,ar):q8_group(w,a,row,b,expert,ar);
 }
 return warp_sum(v);
}
__device__ __forceinline__ float float_row(const MatrixView&w,const float*x,int row,int expert=0){
 const auto*p=reinterpret_cast<const float*>(static_cast<const char*>(w.data)+expert*w.expert_stride+row*w.row_stride);
 float v=0;
 for(int i=threadIdx.x&31;i<w.k;i+=32)v=fmaf(p[i],x[i],v);
 return warp_sum(v);
}
__device__ __forceinline__ void matvec_phase(const MatrixView&w,ActivationView a,const float*x,float*out,int expert=0,int ar=0){
 const int warp=threadIdx.x/32,warps=blockDim.x/32;
 for(int row=blockIdx.x*warps+warp;row<w.n;row+=gridDim.x*warps){
  const float v=w.format==WeightFormat::F32?float_row(w,x,row,expert):quantized_row(w,a,row,expert,ar);
  if((threadIdx.x&31)==0)out[row]=v;
 }
}
__device__ __forceinline__ void matvec_one_warp_per_cta(const MatrixView&w,ActivationView a,const float*x,float*out){
 if(threadIdx.x<32)for(int row=blockIdx.x;row<w.n;row+=gridDim.x){const float v=w.format==WeightFormat::F32?float_row(w,x,row):quantized_row(w,a,row);if(threadIdx.x==0)out[row]=v;}
}
__device__ __forceinline__ void quantize_phase(const float*x,ActivationBuffer a){
 const int lane=threadIdx.x&31,warp=threadIdx.x/32,warps=blockDim.x/32;
 for(int b=blockIdx.x*warps+warp;b<a.k/32;b+=gridDim.x*warps){
  const float v=x[b*32+lane],amax=warp_max(fabsf(v)),d=amax/127.f;
  a.qs[b*32+lane]=amax==0?0:int8_t(roundf(v/d));
  if(lane==0)a.scale[b]=__half2float(__float2half(d));
 }
}
__device__ __forceinline__ void quantize_cta(const float*x,ActivationBuffer a){
 const int lane=threadIdx.x&31,warp=threadIdx.x/32;
 for(int b=warp;b<a.k/32;b+=blockDim.x/32){
  const float v=x[b*32+lane],amax=warp_max(fabsf(v)),d=amax/127.f;
  a.qs[b*32+lane]=amax==0?0:int8_t(roundf(v/d));
  if(lane==0)a.scale[b]=__half2float(__float2half(d));
 }
}
// Reference 1024 logical threads mapped to 256 physical threads. Every caller
// must provide at least 32 floats scratch and enter with blockDim.x==256.
template<int Width>
__device__ __forceinline__ void norm_phase(const float*x,const float*gain,float*out,float*scratch,float eps=NormEpsilon){
 const int lane=threadIdx.x&31,warp=threadIdx.x/32;
 #pragma unroll
 for(int z=0;z<4;z++){
  float sum=0;
  for(int i=threadIdx.x+z*256;i<Width;i+=1024)sum=fmaf(x[i],x[i],sum);
  sum=warp_sum(sum);
  if(lane==0)scratch[z*8+warp]=sum;
 }
 __syncthreads();
 float sum=warp_sum(scratch[lane]);
 const float inverse=rsqrtf(sum/Width+eps);
 for(int i=threadIdx.x;i<Width;i+=256)out[i]=inverse*x[i]*gain[i];
 __syncthreads();
}
__device__ __forceinline__ float sigmoid(float x){return 1.f/(1.f+expf(-x));}
__device__ __forceinline__ float silu(float x){return x*sigmoid(x);}
} // namespace halo6000
