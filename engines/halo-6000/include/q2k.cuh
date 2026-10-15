#pragma once
#include "model.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace halo6000 {
struct Q2KBlock {
 uint8_t scales[16];
 uint8_t qs[64];
 uint16_t d,dmin;
};
static_assert(sizeof(Q2KBlock)==84 && offsetof(Q2KBlock,d)==80,"native GGML Q2_K ABI");

// Four contiguous activation bytes in each of four scale32 groups. This is the
// native MMVQ affine rule: mins use integer activation sums, NOT Q8_1.ds.y.
__device__ __forceinline__ float q2k_slice(const Q2KBlock& w,const int8_t*qs,const float*ds,int slice){
 const uint32_t v=*reinterpret_cast<const uint32_t*>(w.qs+slice*4);
 const int part=slice/8,offset=slice%8,scale=part*8+offset/4;
 float sum_d=0,sum_m=0;
 #pragma unroll
 for(int h=0;h<4;h++){
  const int a=*reinterpret_cast<const int*>(qs+(part*4+h)*32+offset*4);
  const int code=(v>>(2*h))&0x03030303;
  const int sc=w.scales[scale+2*h];
  sum_d+=ds[part*4+h]*float(__dp4a(code,a,0)*(sc&15));
  const int minimum=sc>>4;
  sum_m+=ds[part*4+h]*float(__dp4a(minimum*0x01010101,a,0));
 }
 const float d=__half2float(*reinterpret_cast<const half*>(&w.d));
 const float minimum=__half2float(*reinterpret_cast<const half*>(&w.dmin));
 return d*sum_d-minimum*sum_m;
}

__device__ __forceinline__ const Q2KBlock* q2k_weights(const MatrixView&w,int row,int expert=0){
 const size_t row_bytes=w.row_stride?w.row_stride:size_t(w.k/256)*84;
 const size_t expert_bytes=w.expert_stride?w.expert_stride:row_bytes*w.n;
 return reinterpret_cast<const Q2KBlock*>(static_cast<const uint8_t*>(w.data)+expert*expert_bytes+row*row_bytes);
}

// Every lane in its subgroup enters, including lanes owning no K slice.
// group32 effective scales and original signed INT8 codes are the only inputs.
template<int Lanes=32>
__device__ __forceinline__ float q2k_row(const MatrixView&w,ActivationView a,int row,int expert=0,int ar=0){
 static_assert(Lanes==1||Lanes==4||Lanes==8||Lanes==16||Lanes==32);
 const int lane=threadIdx.x%Lanes;
 const Q2KBlock* weights=q2k_weights(w,row,expert);
 const int8_t*qs=a.qs+ar*a.q_stride;
 const float*ds=a.scale+ar*a.scale_stride;
 float result=0;
 for(int i=lane;i<w.k/16;i+=Lanes){
  const int b=i/16;
  result+=q2k_slice(weights[b],qs+b*256,ds+b*8,i%16);
 }
 #pragma unroll
 for(int shift=Lanes/2;shift;shift/=2)result+=__shfl_down_sync(0xffffffff,result,shift,Lanes);
 return result;
}

// Callable from a fused phase. No allocations, preparation or synchronizations.
template<int Lanes=32>
__device__ __forceinline__ void q2k_matvec_phase(const MatrixView&w,ActivationView a,float*out,int expert=0,int ar=0){
 const int group=threadIdx.x/Lanes,groups=blockDim.x/Lanes;
 for(int row=blockIdx.x*groups+group;row<w.n;row+=gridDim.x*groups){
  const float result=q2k_row<Lanes>(w,a,row,expert,ar);
  if(threadIdx.x%Lanes==0)out[row]=result;
 }
}
}
