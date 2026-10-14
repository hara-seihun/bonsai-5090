#pragma once
#include "kernels.cuh"
#include "attention_fp4_kernels.cuh"
namespace p12kv4h5 {
using namespace h5;
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o)); return v; }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float block_sum_256(float v, float* red) {
  v = warp_sum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) t += red[i];
  return t;
}

__device__ __forceinline__ void attn_pre_body(const AttnArgs& a, int bx, int row) {
  __shared__ float s[HD];
  __shared__ float red[8];
  const int d = threadIdx.x;
  const int pos = a.row_pos[row], slot = a.row_slot[row];
  const bool is_q = bx < NH;
  const int hq = bx, hk = bx - NH;
  const float* pr = a.proj + (size_t)row * ATT_IN;
  float v = is_q ? pr[hq * 2 * HD + d] : pr[Q_OUT + hk * HD + d];
  const float ss = block_sum_256(v * v, red);
  v = v * rsqrtf(ss / (float)HD + NORM_EPS) * (is_q ? a.q_norm[d] : a.k_norm[d]);
  s[d] = v;
  __syncthreads();
  float outv = v;
  if (d < NROT) {
    const int i = d < NROT / 2 ? d : d - NROT / 2;
    int rp = pos;
    if (a.rope_pos) {
      // Qwen35 IMRoPE interleaves t/h/w frequencies, not contiguous M-RoPE sections.
      const int axis = i % 3 == 1 && i < 3 * a.rope_sections[1] ? 1 :
                       i % 3 == 2 && i < 3 * a.rope_sections[2] ? 2 :
                       i % 3 == 0 && i < 3 * a.rope_sections[0] ? 0 : 3;
      rp = a.rope_pos[4 * row + axis];
    }
    const float theta = (float)rp * powf(ROPE_BASE, -(float)i / (float)(NROT / 2));
    float c, sn; sincosf(theta, &sn, &c);
    outv = d < NROT / 2 ? s[d] * c - s[d + NROT / 2] * sn : s[d - NROT / 2] * sn + s[d] * c;
  }
  if (is_q) { a.qrot[(size_t)row * NH * HD + hq * HD + d] = outv * 0.0625f; return; }
  const size_t pidx = (((size_t)slot * NATTN + a.al) * NKV + hk) * a.ctx + pos;
  const float vv = pr[Q_OUT + KV_OUT + hk * HD + d];
  p12kv4::store(outv,d,pidx,(uint8_t*)a.kc,(uint8_t*)a.ksc);
  p12kv4::store(vv,d,pidx,(uint8_t*)a.vc,(uint8_t*)a.vsc);
}

__global__ void prepare(AttnArgs a){h5::pdl_enter_e(a.early);attn_pre_body(a,blockIdx.x,blockIdx.y);}
}
