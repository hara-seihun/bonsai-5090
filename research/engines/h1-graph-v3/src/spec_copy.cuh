#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
namespace h5 {
// A static map of disjoint live byte intervals. All addresses and extents are
// byte coordinates; no interpretation or rounding of model state is performed.
struct SpecCopyRegion { const uint8_t* src; uint8_t* dst; size_t pitch, bytes; int rows, first_tile, tiles_per_row; };
struct SpecCopyPlan { SpecCopyRegion r[12]; int count=0, tiles=0; };
inline void spec_copy_add(SpecCopyPlan& p,const void* src,void* dst,size_t bytes,int rows=1,size_t pitch=0) {
  if(!bytes||!rows)return;
  if(p.count>=12) throw std::runtime_error("spec copy region overflow");
  int n=int((bytes+4095)/4096);
  p.r[p.count++]={static_cast<const uint8_t*>(src),static_cast<uint8_t*>(dst),pitch?pitch:bytes,bytes,rows,p.tiles,n};
  p.tiles+=n*rows;
}
__global__ void spec_copy_kernel(SpecCopyPlan p) {
  int b=blockIdx.x, k=0;
  while(k+1<p.count&&b>=p.r[k+1].first_tile)++k;
  const auto r=p.r[k];int t=b-r.first_tile,row=t/r.tiles_per_row;
  size_t off=size_t(t%r.tiles_per_row)*4096+threadIdx.x*16;
  const uint8_t* s=r.src+row*r.pitch;uint8_t* d=r.dst+row*r.pitch;
  if(off+16<=r.bytes&&!(reinterpret_cast<uintptr_t>(s+off)&15)&&!(reinterpret_cast<uintptr_t>(d+off)&15))
    *reinterpret_cast<uint4*>(d+off)=*reinterpret_cast<const uint4*>(s+off);
  else for(int j=0;j<16&&off+j<r.bytes;++j)d[off+j]=s[off+j];
}
inline void spec_copy_launch(const SpecCopyPlan& p,cudaStream_t st){if(p.tiles)spec_copy_kernel<<<p.tiles,256,0,st>>>(p);}
}
