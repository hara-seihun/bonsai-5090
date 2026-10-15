#pragma once
#include "device.cuh"
namespace halo6000 {
// Prepared bytes are only a permutation of the immutable grouped128 image.
// Lanes selects tile rows=32/Lanes; a view must be consumed with the same Lanes.
struct HeadPreparedView {
 const int* words=nullptr;
 const uint16_t* scales=nullptr;
 int n=0,k=2048,lanes=0;
};
template<int Lanes>inline constexpr size_t head_prepared_rows(int n) {
 static_assert(Lanes==4||Lanes==8||Lanes==16||Lanes==32);
 constexpr int Rows=32/Lanes;
 return size_t((n+Rows-1)/Rows)*Rows;
}
template<int Lanes>inline constexpr size_t head_prepared_word_bytes(int n){return head_prepared_rows<Lanes>(n)*2048;}
template<int Lanes>inline constexpr size_t head_prepared_scale_bytes(int n){return head_prepared_rows<Lanes>(n)*64*2;}

template<int Lanes,bool Prepared>
__device__ __forceinline__ float head_group(const MatrixView&w,HeadPreparedView p,ActivationView a,int row,int g) {
 constexpr int Rows=32/Lanes;
 float v=0;
 #pragma unroll
 for(int c=0;c<4;c++) {
  int sum=0;
  #pragma unroll
  for(int t=0;t<8;t++) {
   int q;
   if constexpr(Prepared) q=p.words[((size_t(row/Rows)*32+c*8+t)*16+g)*Rows+row%Rows];
   else q=load_i8x4(static_cast<const int8_t*>(w.data)+(size_t(row)*16+g)*136+c*32+t*4);
   sum=__dp4a(q,load_i8x4(a.qs+g*128+c*32+t*4),sum);
  }
  const uint16_t* s;
  if constexpr(Prepared) s=p.scales+((size_t(row/Rows)*4+c)*16+g)*Rows+row%Rows;
  else s=reinterpret_cast<const uint16_t*>(static_cast<const char*>(w.data)+(size_t(row)*16+g)*136+128)+c;
  v+=half_value(s)*a.scale[g*4+c]*float(sum);
 }
 return v;
}
template<int Lanes,bool Prepared,bool Contiguous=false>
__device__ __forceinline__ void head_phase_impl(const MatrixView&w,HeadPreparedView p,ActivationView a,float*out) {
 static_assert(Lanes==4||Lanes==8||Lanes==16||Lanes==32);
 constexpr int Rows=32/Lanes,Registers=(16+Lanes-1)/Lanes;
 const int lane=threadIdx.x&31,warp=threadIdx.x/32;
 const int localrow=lane%Rows,g=lane/Rows;
 const int warps=blockDim.x/32,n=Prepared?p.n:w.n;
 const int tiles=(n+Rows-1)/Rows;
 const int begin=Contiguous?int(size_t(blockIdx.x)*tiles/gridDim.x):blockIdx.x*warps;
 const int end=Contiguous?int(size_t(blockIdx.x+1)*tiles/gridDim.x):tiles;
 const int step=Contiguous?warps:gridDim.x*warps;
 for(int tile=begin+warp;tile<end;tile+=step) {
  const int row=tile*Rows+localrow;
  float v[Registers];
  #pragma unroll
  for(int r=0;r<Registers;r++) v[r]=(row<n&&g+r*Lanes<16)?0.f+head_group<Lanes,Prepared>(w,p,a,row,g+r*Lanes):0.f;
  // Fold the original sixteen-group XOR tree, not a new serial accumulation.
  #pragma unroll
  for(int d=Registers/2;d;d/=2) {
   #pragma unroll
   for(int r=0;r<d;r++)v[r]+=v[r+d];
  }
  float sum=v[0];
  #pragma unroll
  for(int d=Lanes/2;d;d/=2)sum+=__shfl_xor_sync(0xffffffff,sum,d*Rows);
  if(g==0&&row<n)out[row]=sum;
 }
}
template<int Lanes=16,bool Contiguous=false>
__device__ __forceinline__ void head_q8_phase(const MatrixView&w,ActivationView a,float*out) {
 head_phase_impl<Lanes,false,Contiguous>(w,{},a,out);
}
template<int Lanes=4,bool Contiguous=false>
__device__ __forceinline__ void head_prepared_phase(HeadPreparedView p,ActivationView a,float*out) {
 head_phase_impl<Lanes,true,Contiguous>({},p,a,out);
}
template<int Lanes,bool Prepared>
__global__ void head_kernel(MatrixView w,HeadPreparedView p,ActivationView a,float*out) {
 head_phase_impl<Lanes,Prepared>(w,p,a,out);
}
template<int Lanes=16>
inline cudaError_t launch_head_q8(MatrixView w,ActivationView a,float*out,cudaStream_t stream=nullptr,int blocks=0) {
 if(w.format!=WeightFormat::Q8Grouped128||w.k!=2048||a.k!=2048||a.group!=32)return cudaErrorInvalidValue;
 if(!blocks)blocks=(w.n+(256/Lanes)-1)/(256/Lanes);
 head_kernel<Lanes,false><<<blocks,256,0,stream>>>(w,{},a,out);
 return cudaGetLastError();
}
template<int Lanes=4>
inline cudaError_t launch_head_prepared(HeadPreparedView p,ActivationView a,float*out,cudaStream_t stream=nullptr,int blocks=0) {
 if(p.lanes!=Lanes||p.k!=2048||a.k!=2048||a.group!=32)return cudaErrorInvalidValue;
 if(!blocks)blocks=(p.n+(256/Lanes)-1)/(256/Lanes);
 head_kernel<Lanes,true><<<blocks,256,0,stream>>>({},p,a,out);
 return cudaGetLastError();
}
template<int Lanes>
__global__ void head_prepare_kernel(const uint8_t*src,int*words,uint16_t*scales,int n) {
 constexpr int Rows=32/Lanes;
 const size_t tiles=(n+Rows-1)/Rows,total=tiles*Rows*512;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x) {
  const int r=i%Rows,g=(i/Rows)%16,t=(i/Rows/16)%32;
  const int row=int(i/Rows/16/32)*Rows+r;
  words[i]=row<n?load_i8x4(reinterpret_cast<const int8_t*>(src)+(size_t(row)*16+g)*136+t*4):0;
 }
 const size_t count=tiles*Rows*64;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
  const int r=i%Rows,g=(i/Rows)%16,c=(i/Rows/16)%4;
  const int row=int(i/Rows/16/4)*Rows+r;
  scales[i]=row<n?*reinterpret_cast<const uint16_t*>(src+(size_t(row)*16+g)*136+128+c*2):0;
 }
}
template<int Lanes>
inline cudaError_t prepare_head_q8(MatrixView w,HeadPreparedView p,cudaStream_t stream=nullptr) {
 if(w.format!=WeightFormat::Q8Grouped128||w.k!=2048||p.n!=w.n||p.k!=2048||p.lanes!=Lanes)return cudaErrorInvalidValue;
 head_prepare_kernel<Lanes><<<188,256,0,stream>>>(static_cast<const uint8_t*>(w.data),const_cast<int*>(p.words),const_cast<uint16_t*>(p.scales),w.n);
 return cudaGetLastError();
}
} // namespace halo6000
