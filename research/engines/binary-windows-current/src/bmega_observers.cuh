#pragma once
#include "fp4_ops.cuh"
#include "p12/observer_chain.cuh"
#include "p12/chunk_dot_npack_sf.cuh"
namespace bmega {
using h5::mma_fp4;
__device__ __forceinline__ void chunk_dot_original(const uint8_t* ch, const uint2* act, const float* asc, int lane, float& rg, float& r8) {
  const uint32_t* sg = (const uint32_t*)ch;
  const uint32_t* sc = (const uint32_t*)(ch + 2048);
  const int g = lane >> 2, t = lane & 3;
  const uint2* ap = act + t;
  float accg = 0.f, acc8 = 0.f;
#pragma unroll 2
  for (int b = 0; b < 8; b++) {
    float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f}, d2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int s = b * 2 + hh;
      const uint32_t w = sg[s * 32 + lane];
      unsigned A[4];
      A[0] = (w & 0x88888888u) | 0x22222222u;
      A[1] = ((w << 1) & 0x88888888u) | 0x22222222u;
      A[2] = ((w << 2) & 0x88888888u) | 0x22222222u;
      A[3] = ((w << 3) & 0x88888888u) | 0x22222222u;
      const uint2 p0 = ap[s * 12 + 0], p1 = ap[s * 12 + 4], p2 = ap[s * 12 + 8];
      mma_fp4(d0, A, p0.x, p0.y, 0x38383838u, d0);
      mma_fp4(d1, A, p1.x, p1.y, 0x38383838u, d1);
      mma_fp4(d2, A, p2.x, p2.y, 0x38383838u, d2);
    }
    const float dg = fmaf(32.f, d2[0], fmaf(8.f, d1[0], 2.f * d0[0]));
    const float d8 = fmaf(32.f, d2[2], fmaf(8.f, d1[2], 2.f * d0[2]));
    const uint32_t su = sc[b * 8 + g];
    const float2 ws = __half22float2(*(const __half2*)&su);
    const float as = asc[b];
    accg = fmaf(dg, __fmul_rn(ws.x, as), accg);
    acc8 = fmaf(d8, __fmul_rn(ws.y, as), acc8);
  }
  rg = accg; r8 = acc8;
}


__device__ __forceinline__ void chunk_dot(const uint8_t* ch,const uint2* act,const float* asc,int lane,float& rg,float& r8,int observer) {
  if (observer == 1) fp4_sf_round2::chunk_dot_npack_sf<false>(ch,act,asc,lane,rg,r8);
  else if (observer == 2) fp4_sf_round2::chunk_dot_npack_sf<true>(ch,act,asc,lane,rg,r8);
  else if (observer == 3) mma_observer_chains::chunk_b1(ch,act,asc,lane,rg,r8);
  else chunk_dot_original(ch,act,asc,lane,rg,r8);
}
}
