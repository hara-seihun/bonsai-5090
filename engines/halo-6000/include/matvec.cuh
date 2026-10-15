#pragma once
#include "device.cuh"
namespace halo6000 {
// Virtualize the original32-lane tree, retaining each logical lane's serial
// accumulation and descending-XOR additions. Fewer physical lanes own a row.
template<int Lanes>
__device__ __forceinline__ float row_exact(const MatrixView&w,ActivationView a,const float*x,int row,int expert=0,int ar=0){
 static_assert(Lanes==4||Lanes==8||Lanes==16||Lanes==32);
 if(w.format!=WeightFormat::F32&&w.format!=WeightFormat::Pane2&&w.format!=WeightFormat::Q8Grouped128){asm volatile("trap;");return 0;}
 constexpr int Slots=32/Lanes;const int lane=threadIdx.x&(Lanes-1);float part[Slots];
 #pragma unroll
 for(int s=0;s<Slots;s++){
  float v=0;const int logical=lane+s*Lanes;
  if(w.format==WeightFormat::F32){const auto*p=reinterpret_cast<const float*>(static_cast<const char*>(w.data)+expert*w.expert_stride+row*w.row_stride);
   for(int i=logical;i<w.k;i+=32)v=fmaf(p[i],x[ar*w.k+i],v);
  }else{
   for(int b=logical;b<w.k/128;b+=32)v+=w.format==WeightFormat::Pane2?pane_group(w,a,row,b,expert,ar):q8_group(w,a,row,b,expert,ar);
  }
  part[s]=v;
 }
 #pragma unroll
 for(int d=16;d>=Lanes;d>>=1){
  #pragma unroll
  for(int s=0;s<d/Lanes;s++)part[s]+=part[s+d/Lanes];
 }
 float v=part[0];const unsigned mask=__activemask();
 #pragma unroll
 for(int d=Lanes/2;d;d>>=1)v+=__shfl_xor_sync(mask,v,d,Lanes);
 return v;
}
template<int Lanes>
__device__ __forceinline__ void matvec_exact_phase(const MatrixView&w,ActivationView a,const float*x,float*out){
 const int groups=blockDim.x/Lanes,lane=threadIdx.x&(Lanes-1);
 for(int row=blockIdx.x*groups+threadIdx.x/Lanes;row<w.n;row+=gridDim.x*groups){float v=row_exact<Lanes>(w,a,x,row);if(!lane)out[row]=v;}
}
} // namespace halo6000
