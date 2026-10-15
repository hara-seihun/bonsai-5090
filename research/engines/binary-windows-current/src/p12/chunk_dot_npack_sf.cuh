#pragma once
#include <cuda_fp16.h>
#include "sf_dot.cuh"

namespace fp4_sf_round2 {
template<bool SCALED=true> __device__ __forceinline__ void chunk_dot_npack_sf(
    const uint8_t* ch,const uint2* act,const float* asc,int lane,float& rg,float& r8) {
  const uint32_t* sg=reinterpret_cast<const uint32_t*>(ch);
  const uint32_t* sc=reinterpret_cast<const uint32_t*>(ch+2048);
  const int g=lane>>2,t=lane&3;
  const uint32_t sf_b=SCALED ? 0x40404040u+0x10101010u*uint32_t(g&3) : 0x38383838u;
  float accg=0.f,acc8=0.f;
#pragma unroll 2
  for(int block=0;block<8;block++) {
    float d[4]={};
#pragma unroll
    for(int half=0;half<2;half++) {
      int s=block*2+half;uint32_t w=sg[s*32+lane];
      uint32_t a[4];
      a[0]=(w&0x88888888u)|0x22222222u;
      a[1]=((w<<1)&0x88888888u)|0x22222222u;
      a[2]=((w<<2)&0x88888888u)|0x22222222u;
      a[3]=((w<<3)&0x88888888u)|0x22222222u;
      uint2 p=make_uint2(0,0);
      if(g<3)p=act[s*12+4*g+t];
      mma_scaled(d,a,p.x,p.y,sf_b);
    }
    float cg=__shfl_xor_sync(0xffffffffu,d[0],1);
    float c8=__shfl_xor_sync(0xffffffffu,d[2],1);
    float dg,d8;
    if constexpr(SCALED) {
      dg=__fadd_rn(__fadd_rn(d[0],d[1]),cg);
      d8=__fadd_rn(__fadd_rn(d[2],d[3]),c8);
    } else {
      dg=__fmaf_rn(32.f,cg,__fmaf_rn(8.f,d[1],__fmul_rn(2.f,d[0])));
      d8=__fmaf_rn(32.f,c8,__fmaf_rn(8.f,d[3],__fmul_rn(2.f,d[2])));
    }
    uint32_t su=sc[block*8+g];
    float2 ws=__half22float2(*reinterpret_cast<const __half2*>(&su));
    float as=asc[block];
    accg=__fmaf_rn(dg,__fmul_rn(ws.x,as),accg);
    acc8=__fmaf_rn(d8,__fmul_rn(ws.y,as),acc8);
  }
  rg=accg;r8=acc8;
}
}
