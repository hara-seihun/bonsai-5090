#pragma once
#include "f1.cuh"
#include <cuda_fp4.h>
#include <cuda_fp8.h>
namespace f1 {
__device__ __forceinline__ void q8(const float* v,float bm,float fixed,unsigned& packed,uint8_t& sb) {
  float target=bm/(6.f*fixed);
  sb=__nv_cvt_float_to_fp8(target,__NV_SATFINITE,__NV_E4M3);
  float sv=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3)));
  if(sv<target && sb<126) { ++sb; sv=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3))); }
  float rs=sv>0?1.f/(fixed*sv):0;
  packed=0;
  #pragma unroll
  for(int i=0;i<4;i++) packed|=(unsigned)__nv_cvt_float2_to_fp4x2(make_float2(v[2*i]*rs,v[2*i+1]*rs),__NV_E2M1,cudaRoundNearest)<<(8*i);
}
// Called by all nine warps after the streaming producer drains and all h stores finish.
template<int TOKW> __device__ void emit(float* tile,int R,int F,const float* signs,float fixed,uint8_t* Q,uint8_t* QS,float* QG) {
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  int cls=blockIdx.y%4,chunk=blockIdx.y/4;
  for(int tr=warp;tr<TOKW;tr+=9) {
    int row=blockIdx.x*TOKW+tr;
    if(row>=R) continue;
    float v[8];
    #pragma unroll
    for(int i=0;i<8;i++) {
      int j=lane*8+i, k=chunk*1024+(j>>5)*128+cls*32+(j&31);
      v[i]=tile[tr*256+j]*signs[k];
    }
    #pragma unroll
    for(int b=1;b<8;b*=2)
      #pragma unroll
      for(int i=0;i<8;i++) if(!(i&b)) {float a=v[i],c=v[i+b];v[i]=a+c;v[i+b]=a-c;}
    #pragma unroll
    for(int b=1;b<32;b*=2)
      #pragma unroll
      for(int i=0;i<8;i++){float a=__shfl_xor_sync(0xffffffff,v[i],b);v[i]=(lane&b)?a-v[i]:v[i]+a;}
    float bm=0;
    #pragma unroll
    for(int i=0;i<8;i++){v[i]*=.0625f;bm=fmaxf(bm,fabsf(v[i]));}
    bm=fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,1));
    unsigned wd;uint8_t sb;q8(v,bm,fixed,wd,sb);
    int j=lane*8,k=chunk*1024+(j>>5)*128+cls*32+(j&31),rr=row&63;
    size_t base=((size_t)(row>>6)*(F/128)+(k>>7))*64+rr;
    int off=(((k&127)>>5)^((rr>>1)&3))*16+((k&31)>>1);
    *(unsigned*)(Q+base*64+off)=wd;
    if(!(lane&1)) QS[base*8+((k&127)>>4)]=sb;
    if(lane==0 && blockIdx.y==0) QG[row]=fixed;
  }
}
}
