#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace fp4_sf_round2 {
__device__ __forceinline__ void mma_scaled(float (&d)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1,uint32_t sf_b) {
  constexpr uint32_t sf_a=0x38383838u;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1),"r"(sf_a),"r"(sf_b));
}

__device__ __forceinline__ void exact_k128(float (&d)[4],const uint32_t (&a)[2][4],const uint32_t (&b)[3][4]) {
  d[0]=d[1]=d[2]=d[3]=0.f;
  mma_scaled(d,a[0],b[0][0],b[0][1],0x40404040u);
  mma_scaled(d,a[1],b[0][2],b[0][3],0x40404040u);
  mma_scaled(d,a[0],b[1][0],b[1][1],0x50505050u);
  mma_scaled(d,a[1],b[1][2],b[1][3],0x50505050u);
  mma_scaled(d,a[0],b[2][0],b[2][1],0x60606060u);
  mma_scaled(d,a[1],b[2][2],b[2][3],0x60606060u);
}

__device__ __forceinline__ float continue_original(float dot_float,float weight_scale,float activation_scale,float previous,bool bad) {
  int dot=__float2int_rn(dot_float);
  float scale=__fmul_rn(weight_scale,activation_scale);
  return bad ? __int_as_float(0x7fc00000) : __fmaf_rn(float(dot),scale,previous);
}
}
