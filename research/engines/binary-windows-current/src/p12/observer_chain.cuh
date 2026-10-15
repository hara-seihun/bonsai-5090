#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace mma_observer_chains {

__device__ __forceinline__ void fp4(float d[4], const unsigned a[4], uint2 b) {
  const unsigned sf = 0x38383838u;
  const uint16_t z = 0;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},%14,{%15,%15},%14,{%15,%15};"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y),
      "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "r"(sf), "h"(z));
}

__device__ __forceinline__ unsigned coefficient(int t) {
  return t == 0 ? 0x40000000u : t == 1 ? 0x41000000u : t == 2 ? 0x42000000u : 0u;
}

__device__ __forceinline__ void observe_b1(float d[4], unsigned cb) {
  const unsigned a0 = __float_as_uint(d[0]), a1 = __float_as_uint(d[2]);
  const float zero = 0.f;
  asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3},{%4,%5},{%6},{%7,%7,%7,%7};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(cb), "f"(zero));
}

__device__ __forceinline__ void observe_b2(float d[4], unsigned cb0, unsigned cb1) {
  const unsigned a0 = __float_as_uint(d[0]), a1 = __float_as_uint(d[2]);
  const unsigned a2 = __float_as_uint(d[1]), a3 = __float_as_uint(d[3]);
  const float zero = 0.f;
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(cb0), "r"(cb1), "f"(zero));
}

// Batch-lane five-symbol encoder: physical columns d0,d1,d2,d3,d4,0,0,0.
// Prepare these once per warp, then pass to observe_b2 after the two FP4 steps.
__device__ __forceinline__ void five_pair_coefficients(int lane, unsigned& cb0, unsigned& cb1) {
  const int t=lane&3;
  const bool token1=((lane>>2)&1)!=0;
  const float b0=token1 ? (t==1?30.f:t==2?2.f:0.f) : (t==0?2.f:t==2?22.f:0.f);
  const float b1=token1 ? (t==1?38.f:0.f) : (t==0?22.f:0.f);
  cb0=__float_as_uint(b0); cb1=__float_as_uint(b1);
}

__device__ __forceinline__ void signs(unsigned a[4], unsigned w) {
  a[0] = (w & 0x88888888u) | 0x22222222u;
  a[1] = ((w << 1) & 0x88888888u) | 0x22222222u;
  a[2] = ((w << 2) & 0x88888888u) | 0x22222222u;
  a[3] = ((w << 3) & 0x88888888u) | 0x22222222u;
}

// Original opus-bmega chunk: eight original K128 groups, sixteen feature rows.
// All lanes call; lane%4==0 owns the two feature outputs. No producer changes.
__device__ __forceinline__ void chunk_b1(
    const uint8_t* ch, const uint2* act, const float* asc,
    int lane, float& rg, float& r8) {
  const auto* sg = reinterpret_cast<const uint32_t*>(ch);
  const auto* sc = reinterpret_cast<const uint32_t*>(ch + 2048);
  const int g = lane >> 2, t = lane & 3;
  const unsigned cb = coefficient(t);
  float accg = 0.f, acc8 = 0.f;
#pragma unroll 2
  for (int b = 0; b < 8; ++b) {
    float d[4] = {0.f,0.f,0.f,0.f};
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int s = 2*b+hh;
      unsigned a[4];
      signs(a,sg[s*32+lane]);
      uint2 p = make_uint2(0u,0u);
      if (g < 6 && (g & 1) == 0) p = act[s*12+4*(g/2)+t];
      fp4(d,a,p);
    }
    observe_b1(d,cb);
    const uint32_t su = sc[b*8+g];
    const float2 ws = __half22float2(*reinterpret_cast<const __half2*>(&su));
    const float as = asc[b];
    accg = fmaf(d[0],__fmul_rn(ws.x,as),accg);
    acc8 = fmaf(d[2],__fmul_rn(ws.y,as),acc8);
  }
  rg = accg;
  r8 = acc8;
}

// Each token pointer retains the original act[K64][plane][t] layout and own scale.
// Even N columns carry token0, odd N columns token1; no intermediate memory.
__device__ __forceinline__ void chunk_b2(
    const uint8_t* ch, const uint2* act0, const uint2* act1,
    const float* asc0, const float* asc1, int lane,
    float& rg0, float& r80, float& rg1, float& r81) {
  const auto* sg = reinterpret_cast<const uint32_t*>(ch);
  const auto* sc = reinterpret_cast<const uint32_t*>(ch + 2048);
  const int g = lane >> 2, t = lane & 3;
  const unsigned cb = coefficient(t);
  const unsigned cb0 = (g & 1) == 0 ? cb : 0u;
  const unsigned cb1 = (g & 1) == 1 ? cb : 0u;
  float ag0=0.f,a80=0.f,ag1=0.f,a81=0.f;
#pragma unroll 2
  for (int b = 0; b < 8; ++b) {
    float d[4] = {0.f,0.f,0.f,0.f};
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int s = 2*b+hh;
      unsigned a[4];
      signs(a,sg[s*32+lane]);
      uint2 p = make_uint2(0u,0u);
      if (g < 6) {
        const uint2* act = (g & 1) == 0 ? act0 : act1;
        p = act[s*12+4*(g/2)+t];
      }
      fp4(d,a,p);
    }
    observe_b2(d,cb0,cb1);
    const uint32_t su = sc[b*8+g];
    const float2 ws = __half22float2(*reinterpret_cast<const __half2*>(&su));
    const float as0=asc0[b],as1=asc1[b];
    ag0=fmaf(d[0],__fmul_rn(ws.x,as0),ag0);
    a80=fmaf(d[2],__fmul_rn(ws.y,as0),a80);
    ag1=fmaf(d[1],__fmul_rn(ws.x,as1),ag1);
    a81=fmaf(d[3],__fmul_rn(ws.y,as1),a81);
  }
  rg0=ag0; r80=a80; rg1=ag1; r81=a81;
}
}
