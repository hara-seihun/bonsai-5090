#pragma once
#include "moe.cuh"
namespace halo6000 {
inline constexpr size_t RouterBf16Rows=Experts+1;
inline constexpr size_t RouterBf16LayerElements=RouterBf16Rows*Hidden;
inline constexpr size_t RouterBf16LayerBytes=RouterBf16LayerElements*2;
inline constexpr size_t RouterBf16ImageBytes=Layers*RouterBf16LayerBytes;
// Separate mixed-precision image: RNE BF16 coefficients, not an F32 cache view.
__device__ __forceinline__ float router_bf16_row(const uint16_t*row,const float*input){
 float value=0;
 for(int i=threadIdx.x&31;i<Hidden;i+=32)value=fmaf(__uint_as_float(uint32_t(row[i])<<16),input[i],value);
 return warp_sum(value);
}
template<int RowsPerCTA=2>
__device__ __forceinline__ void moe_router_bf16_layer(const uint16_t*base,const MoeView&v,MoeScratch s){
 static_assert(RowsPerCTA>=1&&RowsPerCTA*32<=256);
 const int group=threadIdx.x/32,lane=threadIdx.x&31;
 if(group<RowsPerCTA)for(int row=blockIdx.x*RowsPerCTA+group;row<Experts;row+=gridDim.x*RowsPerCTA){
  float value=router_bf16_row(base+size_t(row)*Hidden,v.input);if(!lane)s.logits[row]=value;
 }
 if(blockIdx.x==0&&threadIdx.x<32){float gain=router_bf16_row(base+size_t(Experts)*Hidden,v.input);if(!lane)*s.shared_gain=sigmoid(gain);}
}
template<int RowsPerCTA=2>
__device__ __forceinline__ void moe_router_bf16(const uint16_t*image,int layer,const MoeView&v,MoeScratch s){
 moe_router_bf16_layer<RowsPerCTA>(image+size_t(layer)*RouterBf16LayerElements,v,s);
}
} // namespace halo6000
