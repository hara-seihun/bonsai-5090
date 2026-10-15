// DFlash2 drafter kernels (semantics: z-lab/dflash model.py, DFlash2DraftModel).
#define H5_PDLCLOCK_MODULE draft
#include "kernels.cuh"
#include <cuda_bf16.h>

namespace h5 {

__device__ __forceinline__ float d_warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); return v; }

// Grouped dynamic causal conv over each runtime-length block (kernel 2, group 16):
//   out[t][e] = sum_off (base[which][off][e] + dyn[t][which*640 + off*320 + e/16]) * src[t-off][e]; src[-1] = 0
// add: h[t][e] += out. Grid (rows), 256 threads x float4 x 5.
__global__ void k_dconv(const float* __restrict__ src, const float* __restrict__ dyn, int which, const float* __restrict__ base,
                        float* __restrict__ out, int add, int block_rows) {
  pdl_wait(); pdl_launch();
  const int row = blockIdx.x, first = (row % block_rows) == 0;
  const float* bw = base + (size_t)which * 2 * D;
  for (int e = threadIdx.x * 4; e < D; e += 1024) {
    const int g = e >> 4;
    const float4 x0 = *(const float4*)(src + (size_t)row * D + e);
    const float4 x1 = first ? make_float4(0.f, 0.f, 0.f, 0.f) : *(const float4*)(src + (size_t)(row - 1) * D + e);
    const float d0 = dyn[(size_t)row * DF_KPROJ + which * 640 + g], d1 = dyn[(size_t)row * DF_KPROJ + which * 640 + 320 + g];
    const float4 b0 = *(const float4*)(bw + e), b1 = *(const float4*)(bw + D + e);
    float4 v;
    v.x = (b0.x + d0) * x0.x + (b1.x + d1) * x1.x; v.y = (b0.y + d0) * x0.y + (b1.y + d1) * x1.y;
    v.z = (b0.z + d0) * x0.z + (b1.z + d1) * x1.z; v.w = (b0.w + d0) * x0.w + (b1.w + d1) * x1.w;
    float4* o = (float4*)(out + (size_t)row * D + e);
    if (add) { const float4 t = *o; v.x += t.x; v.y += t.y; v.z += t.z; v.w += t.w; }
    *o = v;
  }
}
void launch_dconv(const float* src, const float* dyn, int which, const float* base, float* out, int add, int rows, int block_rows, cudaStream_t st) {
  pdl(k_dconv, dim3(rows), dim3(256), 0, st, src, dyn, which, base, out, add, block_rows);
}

// q/k RMSNorm (128) + full-dim RoPE (theta 1e7, pairs (i, i+64)); K/V into the ring cache.
// Grid (rows, nq + DF_NKV), 128 threads. qkv row layout: [q (nq*128) | k (1024) | v (1024)] with stride ld.
__global__ void __launch_bounds__(128) k_dattn_pre(DAttnPreArgs a) {
  pdl_wait(); pdl_launch();
  __shared__ float s[DF_HD];
  __shared__ float red[4];
  const int row = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
  const int nq = a.do_q ? DF_NH : 0;
  const bool is_q = hh < nq;
  const int hk = hh - nq;
  const float* pr = a.qkv + (size_t)row * a.ld;
  float v = is_q ? pr[hh * DF_HD + d] : pr[a.koff + hk * DF_HD + d];
  float ss = d_warp_sum(v * v);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  ss = red[0] + red[1] + red[2] + red[3];
  v = v * rsqrtf(ss / (float)DF_HD + NORM_EPS) * (is_q ? a.q_norm[d] : a.k_norm[d]);
  s[d] = v;
  __syncthreads();
  const int pos = a.pos[row];
  const int i = d < DF_HD / 2 ? d : d - DF_HD / 2;
  const float theta = (float)pos * powf(1e7f, -(float)i / (float)(DF_HD / 2));
  float sn, cs; sincosf(theta, &sn, &cs);
  const float lo = s[i], hi = s[i + DF_HD / 2];
  const float outv = d < DF_HD / 2 ? lo * cs - hi * sn : lo * sn + hi * cs;
  if (is_q) { a.qrot[(size_t)row * DF_Q + hh * DF_HD + d] = outv * 0.08838834764831845f; return; }
  const size_t idx = ((((size_t)a.slot[row] * DF_LAYERS + a.layer) * DF_NKV + hk) * DF_RING + pos % DF_RING) * DF_HD + d;
  a.kc[idx] = __float2half(outv);
  a.vc[idx] = __float2half(pr[a.koff + DF_KV + hk * DF_HD + d]);
}
void launch_dattn_pre(const DAttnPreArgs& a, int rows, cudaStream_t st) {
  pdl(k_dattn_pre, dim3(rows, (a.do_q ? DF_NH : 0) + DF_NKV), dim3(128), 0, st, a);
}

// Top-16 of each logits row: chunk maxima -> 16th largest as threshold -> candidates -> sort.
// One CTA (1024 threads) per row.
__device__ __forceinline__ void bitonic_desc(float* v, int* ix, int n) {   // n power of two, in shared memory
  for (int k = 2; k <= n; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int l = i ^ j;
        if (l > i) {
          const bool desc = (i & k) == 0;
          const bool sw = desc ? (v[i] < v[l] || (v[i] == v[l] && ix[i] > ix[l])) : (v[i] > v[l] || (v[i] == v[l] && ix[i] < ix[l]));
          if (sw) { float t = v[i]; v[i] = v[l]; v[l] = t; int u = ix[i]; ix[i] = ix[l]; ix[l] = u; }
        }
      }
      __syncthreads();
    }
}
constexpr int TK_CAND = 2048;
static_assert(VOCAB % 4 == 0, "topk float4 loads");
__global__ void __launch_bounds__(1024) k_topk(const float* __restrict__ logits, float* __restrict__ topv, int* __restrict__ topi) {
  pdl_wait(); pdl_launch();
  __shared__ float v[TK_CAND]; __shared__ int ix[TK_CAND]; __shared__ int cnt; __shared__ float thr;
  const float* l = logits + (size_t)blockIdx.x * VOCAB;
  const int t = threadIdx.x;
  // coalesced: thread t owns indices t, t + 1024, ... (float4 loads, VOCAB is a multiple of 4096)
  const float4* l4 = (const float4*)l;
  float m = -INFINITY;
  for (int i = t; i < VOCAB / 4; i += 1024) { const float4 q = l4[i]; m = fmaxf(fmaxf(m, fmaxf(q.x, q.y)), fmaxf(q.z, q.w)); }
  v[t] = m; ix[t] = t;
  if (t == 0) cnt = 0;
  __syncthreads();
  bitonic_desc(v, ix, 1024);
  if (t == 0) thr = v[DF_TOPK - 1];
  __syncthreads();
  const float th = thr;
  for (int i = t; i < TK_CAND; i += 1024) { v[i] = -INFINITY; ix[i] = 0x7fffffff; }
  __syncthreads();
  for (int i = t; i < VOCAB / 4; i += 1024) {
    const float4 q = l4[i]; const float xs[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int e = 0; e < 4; e++)
      if (xs[e] >= th) { const int k = atomicAdd(&cnt, 1); if (k < TK_CAND) { v[k] = xs[e]; ix[k] = 4 * i + e; } }
  }
  __syncthreads();
  bitonic_desc(v, ix, TK_CAND);
  if (t < DF_TOPK) { topv[(size_t)blockIdx.x * DF_TOPK + t] = v[t]; topi[(size_t)blockIdx.x * DF_TOPK + t] = ix[t]; }
}
void launch_topk(const float* logits, float* topv, int* topi, int rows, cudaStream_t st) { pdl(k_topk, dim3(rows), dim3(1024), 0, st, logits, topv, topi); }

// Candidate selector: one CTA (256 threads = rank) per runtime-length block; anchor query is excluded.
__global__ void __launch_bounds__(256) k_selector(const __nv_bfloat16* __restrict__ pred_cb, const __nv_bfloat16* __restrict__ succ_cb,
                                                  const float* __restrict__ hproj, const float* __restrict__ topv, const int* __restrict__ topi,
                                                  const int* __restrict__ anchor, int* __restrict__ drafts, float* __restrict__ conf, int proposals) {
  pdl_wait(); pdl_launch();
  __shared__ float part[8][DF_TOPK];
  __shared__ int prev;
  const int b = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
  if (t == 0) prev = anchor[b];
  __syncthreads();
  for (int p = 0; p < proposals; p++) {
    const int r = b * proposals + p;
    const float pr = __bfloat162float(pred_cb[(size_t)prev * DF_RANK + t]) * hproj[(size_t)r * DF_RANK + t];
#pragma unroll
    for (int k = 0; k < DF_TOPK; k++) {
      float sc = pr * __bfloat162float(succ_cb[(size_t)topi[r * DF_TOPK + k] * DF_RANK + t]);
      sc = d_warp_sum(sc);
      if (lane == 0) part[w][k] = sc;
    }
    __syncthreads();
    if (t == 0) {
      float best = -INFINITY, second = -INFINITY; int bk = 0;
      for (int k = 0; k < DF_TOPK; k++) {
        float sc = topv[r * DF_TOPK + k];
        for (int q = 0; q < 8; q++) sc += part[q][k];
        if (sc > best) { second = best; best = sc; bk = k; } else if (sc > second) second = sc;
      }
      prev = topi[r * DF_TOPK + bk];
      drafts[r] = prev;
      conf[r] = best - second;
    }
    __syncthreads();
  }
}
void launch_selector(const __nv_bfloat16* pred_cb, const __nv_bfloat16* succ_cb, const float* hproj, const float* topv, const int* topi,
                     const int* anchor, int* drafts, float* conf, int blocks, int proposals, cudaStream_t st) {
  pdl(k_selector, dim3(blocks), dim3(256), 0, st, pred_cb, succ_cb, hproj, topv, topi, anchor, drafts, conf, proposals);
}

__global__ void __launch_bounds__(256) k_selector_two(const __nv_bfloat16* __restrict__ pred_cb,
    const __nv_bfloat16* __restrict__ succ_cb, const float* __restrict__ hproj,
    const float* __restrict__ topv, const int* __restrict__ topi, const int* __restrict__ anchor,
    int* __restrict__ drafts, float* __restrict__ conf, int* __restrict__ alt,
    float* __restrict__ alt_conf, int* __restrict__ branches, int fork_span) {
  pdl_wait(); pdl_launch();
  __shared__ float part[8][DF_TOPK], margin[7];
  __shared__ int prev, second_tok[7], fork;
  const int b = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
  if (t == 0) prev = anchor[b];
  __syncthreads();
  for (int p = 0; p < 7; p++) {
    const int r = b * 7 + p;
    const float pr = __bfloat162float(pred_cb[(size_t)prev * DF_RANK + t]) * hproj[(size_t)r * DF_RANK + t];
#pragma unroll
    for (int k = 0; k < DF_TOPK; k++) {
      float sc = d_warp_sum(pr * __bfloat162float(succ_cb[(size_t)topi[r * DF_TOPK + k] * DF_RANK + t]));
      if (lane == 0) part[w][k] = sc;
    }
    __syncthreads();
    if (t == 0) {
      float best = -INFINITY, second = -INFINITY; int bk = 0, sk = 0;
      for (int k = 0; k < DF_TOPK; k++) {
        float sc = topv[r * DF_TOPK + k];
        for (int q = 0; q < 8; q++) sc += part[q][k];
        if (sc > best) { second = best; sk = bk; best = sc; bk = k; }
        else if (sc > second) { second = sc; sk = k; }
      }
      prev = topi[r * DF_TOPK + bk]; drafts[r] = prev;
      margin[p] = conf[r] = best - second;
      second_tok[p] = topi[r * DF_TOPK + sk];
    }
    __syncthreads();
  }
  if (t == 0) {
    fork = 0;
    for (int p = 1; p < fork_span; p++) if (margin[p] < margin[fork]) fork = p;
    branches[b] = fork; prev = anchor[b];
  }
  __syncthreads();
  for (int p = 0; p < 7; p++) {
    const int r = b * 7 + p;
    if (p <= fork) {
      if (t == 0) {
        prev = p == fork ? second_tok[p] : drafts[r];
        alt[r] = prev; alt_conf[r] = p == fork ? -margin[p] : margin[p];
      }
      __syncthreads();
      continue;
    }
    // Only the predecessor changes; both paths share the same masked DFlash trunk/top16 evidence.
    const float pr = __bfloat162float(pred_cb[(size_t)prev * DF_RANK + t]) * hproj[(size_t)r * DF_RANK + t];
#pragma unroll
    for (int k = 0; k < DF_TOPK; k++) {
      float sc = d_warp_sum(pr * __bfloat162float(succ_cb[(size_t)topi[r * DF_TOPK + k] * DF_RANK + t]));
      if (lane == 0) part[w][k] = sc;
    }
    __syncthreads();
    if (t == 0) {
      float best = -INFINITY, second = -INFINITY; int bk = 0;
      for (int k = 0; k < DF_TOPK; k++) {
        float sc = topv[r * DF_TOPK + k];
        for (int q = 0; q < 8; q++) sc += part[q][k];
        if (sc > best) { second = best; best = sc; bk = k; } else if (sc > second) second = sc;
      }
      prev = topi[r * DF_TOPK + bk]; alt[r] = prev; alt_conf[r] = best - second;
    }
    __syncthreads();
  }
}
void launch_selector_two(const __nv_bfloat16* pred_cb, const __nv_bfloat16* succ_cb, const float* hproj,
    const float* topv, const int* topi, const int* anchor, int* drafts, float* conf,
    int* alt, float* alt_conf, int* branches, int fork_span, int blocks, cudaStream_t st) {
  pdl(k_selector_two, dim3(blocks), dim3(256), 0, st, pred_cb, succ_cb, hproj, topv, topi,
      anchor, drafts, conf, alt, alt_conf, branches, fork_span);
}

}  // namespace h5
