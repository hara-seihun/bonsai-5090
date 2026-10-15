#pragma once
#include "device.cuh"
#include "q4_0-format.hpp"
namespace halo6000 {
enum class Q4Storage:uint32_t{Raw18=0,Grouped128=1};
enum class Q4Arithmetic:uint32_t{ForkAffine16=0,SignedGroup32=1};
__device__ __forceinline__ uint32_t q4_load_b2(const uint8_t*p){const auto*h=reinterpret_cast<const uint16_t*>(p);return uint32_t(h[0])|(uint32_t(h[1])<<16);}
__device__ __forceinline__ int q4_signed_nibbles(uint32_t nibble_bytes){return int((nibble_bytes+0x78787878u)^0x80808080u);}
__device__ __forceinline__ int q4_low_operand(uint32_t packed){return q4_signed_nibbles(packed&0x0f0f0f0fu);}
__device__ __forceinline__ int q4_high_operand(uint32_t packed){return q4_signed_nibbles((packed>>4)&0x0f0f0f0fu);}
template<Q4Storage Storage>
__device__ __forceinline__ uint32_t q4_packed_word(const MatrixView&w,int row,int group,int word){
 const size_t b=size_t(row)*(w.k/32)+group;const auto*base=static_cast<const uint8_t*>(w.data);
 if constexpr(Storage==Q4Storage::Grouped128)return *reinterpret_cast<const uint32_t*>(base+(b/4)*72+(b%4)*16+word*4);
 else return q4_load_b2(base+b*18+2+word*4);
}
template<Q4Storage Storage>
__device__ __forceinline__ float q4_scale(const MatrixView&w,int row,int group){
 const size_t b=size_t(row)*(w.k/32)+group;const auto*base=static_cast<const uint8_t*>(w.data);
 if constexpr(Storage==Q4Storage::Grouped128)return half_value(reinterpret_cast<const uint16_t*>(base+(b/4)*72+64+(b%4)*2));
 else return half_value(reinterpret_cast<const uint16_t*>(base+b*18));
}
template<Q4Storage Storage,Q4Arithmetic Arithmetic>
__device__ __forceinline__ float q4_partial(const MatrixView&w,ActivationView a,const float*original_sums,int row,int unit){
 if constexpr(Arithmetic==Q4Arithmetic::ForkAffine16){
  const int group=unit/2,part=unit%2;int sum=0;
  #pragma unroll
  for(int i=0;i<2;i++){
   uint32_t packed=q4_packed_word<Storage>(w,row,group,part*2+i);
   sum=__dp4a(int(packed&0x0f0f0f0fu),load_i8x4(a.qs+group*32+part*8+i*4),sum);
   sum=__dp4a(int((packed>>4)&0x0f0f0f0fu),load_i8x4(a.qs+group*32+16+part*8+i*4),sum);
  }
  return q4_scale<Storage>(w,row,group)*(float(sum)*a.scale[group]-4.f*original_sums[group]);
 }else{
  const int group=unit;int sum=0;
  #pragma unroll
  for(int i=0;i<4;i++){
   uint32_t packed=q4_packed_word<Storage>(w,row,group,i);
   sum=__dp4a(q4_low_operand(packed),load_i8x4(a.qs+group*32+i*4),sum);
   sum=__dp4a(q4_high_operand(packed),load_i8x4(a.qs+group*32+16+i*4),sum);
  }
  return (q4_scale<Storage>(w,row,group)*a.scale[group])*float(sum);
 }
}
template<int Lanes,Q4Storage Storage=Q4Storage::Grouped128,Q4Arithmetic Arithmetic=Q4Arithmetic::ForkAffine16>
__device__ __forceinline__ float q4_row(const MatrixView&w,ActivationView a,const float*original_sums,int row){
 static_assert(Lanes==4||Lanes==8||Lanes==16||Lanes==32);
 constexpr int Slots=32/Lanes;const int lane=threadIdx.x&(Lanes-1);float part[Slots];
 #pragma unroll
 for(int s=0;s<Slots;s++){
  float value=0;const int count=w.k/(Arithmetic==Q4Arithmetic::ForkAffine16?16:32);
  for(int unit=lane+s*Lanes;unit<count;unit+=32)value+=q4_partial<Storage,Arithmetic>(w,a,original_sums,row,unit);
  part[s]=value;
 }
 #pragma unroll
 for(int d=16;d>=Lanes;d>>=1){
  #pragma unroll
  for(int s=0;s<d/Lanes;s++)part[s]+=part[s+d/Lanes];
 }
 float value=part[0];const unsigned mask=__activemask();
 #pragma unroll
 for(int d=Lanes/2;d;d>>=1)value+=__shfl_xor_sync(mask,value,d,Lanes);
 return value;
}
template<int Lanes=32,Q4Storage Storage=Q4Storage::Grouped128,Q4Arithmetic Arithmetic=Q4Arithmetic::ForkAffine16>
__device__ __forceinline__ void q4_phase(const MatrixView&w,ActivationView a,float*out,const float*original_sums=nullptr){
 const int rows=blockDim.x/Lanes,lane=threadIdx.x&(Lanes-1);
 for(int row=blockIdx.x*rows+threadIdx.x/Lanes;row<w.n;row+=gridDim.x*rows){float v=q4_row<Lanes,Storage,Arithmetic>(w,a,original_sums,row);if(!lane)out[row]=v;}
}
// S is half(sum original F32), not sum(qs)*effectiveDS. This is the fork's map.
__device__ __forceinline__ void q4_quantize_phase(const float*x,ActivationBuffer a,float*original_sums){
 const int lane=threadIdx.x&31,warps=blockDim.x/32,warp=threadIdx.x/32;
 for(int group=blockIdx.x*warps+warp;group<a.k/32;group+=gridDim.x*warps){
  float v=x[group*32+lane],amax=warp_max(fabsf(v)),sum=warp_sum(v),d=amax/127.f;
  a.qs[group*32+lane]=amax==0?0:int8_t(roundf(v/d));
  if(!lane){a.scale[group]=__half2float(__float2half(d));if(original_sums)original_sums[group]=__half2float(__float2half(sum));}
 }
}
static __global__ void q4_repack_kernel(const uint8_t*source,uint8_t*dest,size_t groups128){
 for(size_t g=size_t(blockIdx.x)*blockDim.x+threadIdx.x;g<groups128;g+=size_t(gridDim.x)*blockDim.x){
  #pragma unroll
  for(int c=0;c<4;c++){
   #pragma unroll
   for(int word=0;word<4;word++)*reinterpret_cast<uint32_t*>(dest+g*72+c*16+word*4)=q4_load_b2(source+(g*4+c)*18+2+word*4);
   *reinterpret_cast<uint16_t*>(dest+g*72+64+c*2)=*reinterpret_cast<const uint16_t*>(source+(g*4+c)*18);
  }
 }
}
// Compact M16/K32 image: two raw-nibble words per lane, exact half scales per32.
static __global__ void q4_prepare_mma(MatrixView w,uint32_t*out,uint16_t*scales){
 const uint64_t id=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,tile=id/32;const int lane=id%32,kt=w.k/32;
 if(tile>=uint64_t(w.n/16)*kt)return;const int m=tile/kt,g=tile%kt;
 #pragma unroll
 for(int h=0;h<2;h++){int row=m*16+lane/4+h*8;out[(tile*32+lane)*2+h]=q4_packed_word<Q4Storage::Grouped128>(w,row,g,lane%4);}
 if(lane%4==0){
  #pragma unroll
  for(int h=0;h<2;h++){int row=m*16+lane/4+h*8;const size_t b=size_t(row)*kt+g;const auto*p=static_cast<const uint8_t*>(w.data);scales[b]=*reinterpret_cast<const uint16_t*>(p+(b/4)*72+64+(b%4)*2);}
 }
}
__device__ __forceinline__ void q4_mma_operand(const uint32_t*p,int lane,uint32_t a[4]){
 uint32_t lo=p[lane*2],hi=p[lane*2+1];a[0]=uint32_t(q4_low_operand(lo));a[1]=uint32_t(q4_low_operand(hi));a[2]=uint32_t(q4_high_operand(lo));a[3]=uint32_t(q4_high_operand(hi));
}
} // namespace halo6000
