// DFlash2 drafter kernels (semantics: z-lab/dflash model.py, DFlash2DraftModel).
#define H5_PDLCLOCK_MODULE draft
#define H5_LIT_MODULE draft
#define H5_LIT_MODULE_ID 3u
#include "kernels.cuh"
#include "draft_lz.cuh"
#include <cuda_bf16.h>
#include <stdexcept>
#include "sampling_gpu.cuh"
#include "sampling_noise.h"

namespace h5 {

// Diagnostic stream boundary: conventional launch deliberately excludes PDL overlap.
// This records an ordinary dependency in graphs as well as eager execution.
__global__ void k_draft_dependency_fence() {}
cudaError_t launch_draft_dependency_fence(cudaStream_t st) {
  k_draft_dependency_fence<<<1, 1, 0, st>>>();
  return cudaPeekAtLastError();
}

__device__ __forceinline__ float d_warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); return v; }

// Grouped dynamic causal conv over each runtime-length block (kernel 2, group 16):
//   out[t][e] = sum_off (base[which][off][e] + dyn[t][which*640 + off*320 + e/16]) * src[t-off][e]; src[-1] = 0
// add: h[t][e] += out. Grid (rows), 256 threads x float4 x 5.
__global__ void k_dconv(const float* __restrict__ src, const float* __restrict__ dyn, int which, const float* __restrict__ base,
                        float* __restrict__ out, int add, int block_rows) { H5_LIT(0);
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
    v.x = dconv_elem(b0.x, d0, x0.x, b1.x, d1, x1.x); v.y = dconv_elem(b0.y, d0, x0.y, b1.y, d1, x1.y);
    v.z = dconv_elem(b0.z, d0, x0.z, b1.z, d1, x1.z); v.w = dconv_elem(b0.w, d0, x0.w, b1.w, d1, x1.w);
    float4* o = (float4*)(out + (size_t)row * D + e);
    if (add) { const float4 t = *o; v.x = __fadd_rn(v.x, t.x); v.y = __fadd_rn(v.y, t.y); v.z = __fadd_rn(v.z, t.z); v.w = __fadd_rn(v.w, t.w); }
    *o = v;
  }
}
void launch_dconv(const float* src, const float* dyn, int which, const float* base, float* out, int add, int rows, int block_rows, cudaStream_t st) {
  pdl(k_dconv, dim3(rows), dim3(256), 0, st, src, dyn, which, base, out, add, block_rows);
}

// q/k RMSNorm (128) + full-dim RoPE (theta 1e7, pairs (i, i+64)); K/V into the ring cache.
// Grid (rows, nq + DF_NKV), 128 threads. qkv row layout: [q (nq*128) | k (1024) | v (1024)] with stride ld.
__global__ void __launch_bounds__(128) k_dattn_pre(DAttnPreArgs a) { H5_LIT(0);
  pdl_wait(); pdl_launch();
  __shared__ float s[DF_HD];
  __shared__ float red[4];
  const int row = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
  const int nq = a.do_q ? DF_NH : 0;
  const bool is_q = hh < nq;
  const int hk = hh - nq, z = blockIdx.z, koff = a.koff + z * 2 * DF_KV, layer = a.layer + z;
  const float* k_norm = a.layers > 1 ? a.kn[z] : a.k_norm;
  const float* pr = a.qkv + (size_t)row * a.ld;
  float v = is_q ? pr[hh * DF_HD + d] : pr[koff + hk * DF_HD + d];
  float ss = d_warp_sum(v * v);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  ss = red[0] + red[1] + red[2] + red[3];
  v = v * rsqrtf(ss / (float)DF_HD + NORM_EPS) * (is_q ? a.q_norm[d] : k_norm[d]);
  s[d] = v;
  __syncthreads();
  const int pos = a.pos[row];
  const int i = d < DF_HD / 2 ? d : d - DF_HD / 2;
  const float theta = (float)pos * powf(1e7f, -(float)i / (float)(DF_HD / 2));
  float sn, cs; sincosf(theta, &sn, &cs);
  const float lo = s[i], hi = s[i + DF_HD / 2];
  const float outv = d < DF_HD / 2 ? lo * cs - hi * sn : lo * sn + hi * cs;
  if (is_q) { a.qrot[(size_t)row * DF_Q + hh * DF_HD + d] = outv * 0.08838834764831845f; return; }
  const size_t idx = ((((size_t)a.slot[row] * DF_LAYERS + layer) * DF_NKV + hk) * DF_RING + pos % DF_RING) * DF_HD + d;
  a.kc[idx] = __float2half(outv);
  a.vc[idx] = __float2half(pr[koff + DF_KV + hk * DF_HD + d]);
}
void launch_dattn_pre(const DAttnPreArgs& a, int rows, cudaStream_t st) {
  if (a.layers < 1 || a.layer + a.layers > DF_LAYERS || (a.layers > 1 && a.do_q)) throw std::runtime_error("drafter attention prep: layers must stay in range and fused layers carry k/v only");
  for (int z = 1; a.layers > 1 && z <= a.layers; z++) if (!a.kn[z - 1]) throw std::runtime_error("drafter attention prep: fused layer k norm unset");
  pdl(k_dattn_pre, dim3(rows, (a.do_q ? DF_NH : 0) + DF_NKV, a.layers), dim3(128), 0, st, a);
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
__global__ void __launch_bounds__(1024) k_topk(const float* __restrict__ logits, float* __restrict__ topv, int* __restrict__ topi) { H5_LIT(0);
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
// Split top-k (H5_TOPK_SPLIT=1, read per launch): TK_SPLIT CTAs per row each take the exact top-16 of
// one slice of the n logits (thread-maximum threshold, so every slice top-16 element is a candidate; at
// most 16 threads x 60 elements = 960 <= TK_SLICE_CAND for n <= VOCAB), then one CTA per row merges the
// 16 x 16 candidates. Ordering is value descending, index ascending at ties, as in k_topk: same output.
// map (optional): logits column -> token id, for a pruned candidate vocabulary (H5_DRAFT_VOCAB).
constexpr int TK_SPLIT = TOPK_SPLITS, TK_SLICE_CAND = 1024;
static_assert(TOPK_PARTS_PER_ROW == TK_SPLIT * DF_TOPK);
static_assert(VOCAB % (4 * TK_SPLIT) == 0, "split top-k slices");
static_assert(DF_TOPK * TK_SPLIT == 256, "merge sorts 256 candidates");
__global__ void __launch_bounds__(256) k_topk_slice(const float* __restrict__ logits, int n, float* __restrict__ cv, int* __restrict__ ci) { H5_LIT(0);
  pdl_wait(); pdl_launch();
  __shared__ float v[TK_SLICE_CAND]; __shared__ int ix[TK_SLICE_CAND]; __shared__ int cnt; __shared__ float thr;
  const int row = blockIdx.x, sl = blockIdx.y, t = threadIdx.x, slice4 = n / 4 / TK_SPLIT;
  const float4* l4 = (const float4*)(logits + (size_t)row * n) + (size_t)sl * slice4;
  float m = -INFINITY;
  for (int i = t; i < slice4; i += 256) { const float4 q = l4[i]; m = fmaxf(fmaxf(m, fmaxf(q.x, q.y)), fmaxf(q.z, q.w)); }
  v[t] = m; ix[t] = t;
  if (t == 0) cnt = 0;
  __syncthreads();
  bitonic_desc(v, ix, 256);
  if (t == 0) thr = v[DF_TOPK - 1];
  __syncthreads();
  const float th = thr;
  for (int i = t; i < TK_SLICE_CAND; i += 256) { v[i] = -INFINITY; ix[i] = 0x7fffffff; }
  __syncthreads();
  const int base = sl * slice4 * 4;
  for (int i = t; i < slice4; i += 256) {
    const float4 q = l4[i]; const float xs[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int e = 0; e < 4; e++)
      if (xs[e] >= th) { const int k = atomicAdd(&cnt, 1); if (k < TK_SLICE_CAND) { v[k] = xs[e]; ix[k] = base + 4 * i + e; } }
  }
  __syncthreads();
  int nn = 32; while (nn < cnt && nn < TK_SLICE_CAND) nn <<= 1;   // sort only the occupied power-of-two prefix
  bitonic_desc(v, ix, nn);
  if (t < DF_TOPK) { cv[((size_t)row * TK_SPLIT + sl) * DF_TOPK + t] = v[t]; ci[((size_t)row * TK_SPLIT + sl) * DF_TOPK + t] = ix[t]; }
}
__global__ void __launch_bounds__(256) k_topk_merge(const float* __restrict__ cv, const int* __restrict__ ci, const int* __restrict__ map,
                                                    float* __restrict__ topv, int* __restrict__ topi) { H5_LIT(0);
  pdl_wait(); pdl_launch();
  __shared__ float v[256]; __shared__ int ix[256];
  const int row = blockIdx.x, t = threadIdx.x;
  v[t] = cv[(size_t)row * 256 + t]; ix[t] = ci[(size_t)row * 256 + t];
  __syncthreads();
  bitonic_desc(v, ix, 256);
  if (t < DF_TOPK) { topv[(size_t)row * DF_TOPK + t] = v[t]; topi[(size_t)row * DF_TOPK + t] = map ? map[ix[t]] : ix[t]; }
}
void launch_topk(const float* logits, int n, const int* map, float* topv, int* topi, int rows, TopkScratch scratch, cudaStream_t st) {
  const char* e = getenv("H5_TOPK_SPLIT");
  if (e && strcmp(e, "0") && strcmp(e, "1")) { fprintf(stderr, "H5_TOPK_SPLIT must be 0 or 1\n"); abort(); }
  if (n <= 0 || n > VOCAB || n % (4 * TK_SPLIT)) { fprintf(stderr, "top-k: logits row length must be a positive multiple of %d up to the vocabulary\n", 4 * TK_SPLIT); abort(); }
  if (!e || e[0] == '0') {
    if (n != VOCAB || map) { fprintf(stderr, "a pruned candidate vocabulary requires H5_TOPK_SPLIT=1\n"); abort(); }
    pdl(k_topk, dim3(rows), dim3(1024), 0, st, logits, topv, topi); return;
  }
  if (!logits || !topv || !topi || !scratch.values || !scratch.indices || rows < 1 || rows > scratch.row_capacity)
    throw std::invalid_argument("split top-k requires caller-owned scratch for every row");
  pdl(k_topk_slice, dim3(rows, TK_SPLIT), dim3(256), 0, st, logits, n, scratch.values, scratch.indices);
  pdl(k_topk_merge, dim3(rows), dim3(256), 0, st, (const float*)scratch.values, (const int*)scratch.indices, map, topv, topi);
}


// Candidate selector: one CTA (256 threads = rank) per runtime-length block; anchor query is excluded.
__global__ void __launch_bounds__(256) k_selector(const __nv_bfloat16* __restrict__ pred_cb, const __nv_bfloat16* __restrict__ succ_cb,
                                                  const float* __restrict__ hproj, const float* __restrict__ topv, const int* __restrict__ topi,
                                                  const int* __restrict__ anchor, int* __restrict__ drafts, float* __restrict__ conf, int proposals,
                                                  DraftLz lz) { H5_LIT(0);
  pdl_wait(); pdl_launch();
  __shared__ float part[8][DF_TOPK];
  __shared__ int prev;
  __shared__ int lz_tok[DRAFT_LZ_MAX_WINDOW + DRAFT_MAX_PROPOSALS];
  __shared__ short lz_su[DRAFT_LZ_MAX_WINDOW + DRAFT_MAX_PROPOSALS];
  __shared__ float lz_adj[DF_TOPK];
  __shared__ int lz_red[1];
  const int b = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
  const float lz_alpha = lz.ctx ? lz.alpha[b] : 0.f;
  const int lz_len = lz_alpha > 0.f ? lz.len[b] : 0;
  for (int i = t; i < lz_len; i += 256) lz_tok[i] = lz.ctx[(size_t)b * lz.w + i];
  // Variant bit 0x100: every candidate's predecessor and successor codebook rows (the next step's predecessor
  // is one of this step's candidates) are L2-prefetched up front, so the serial per-proposal chain hits L2.
  if (g_variant & 0x100u) for (int i = t; i < proposals * DF_TOPK; i += 256) {
    const int c = topi[(b * proposals + i / DF_TOPK) * DF_TOPK + i % DF_TOPK];
    asm volatile("prefetch.global.L2 [%0];" :: "l"(pred_cb + (size_t)c * DF_RANK) : "memory");
    asm volatile("prefetch.global.L2 [%0];" :: "l"(pred_cb + (size_t)c * DF_RANK + 128) : "memory");
    asm volatile("prefetch.global.L2 [%0];" :: "l"(succ_cb + (size_t)c * DF_RANK) : "memory");
    asm volatile("prefetch.global.L2 [%0];" :: "l"(succ_cb + (size_t)c * DF_RANK + 128) : "memory");
  }
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
    if (lz_alpha > 0.f) draft_lz_adjust(lz, lz_alpha, lz_len + p, lz_tok, topi + r * DF_TOPK, lz_adj, lz_su, lz_red);
    if (t == 0) {
      float best = -INFINITY, second = -INFINITY; int bk = 0;
      for (int k = 0; k < DF_TOPK; k++) {
        float sc = topv[r * DF_TOPK + k];
        for (int q = 0; q < 8; q++) sc += part[q][k];
        if (lz_alpha > 0.f) sc += lz_adj[k];
        if (sc > best) { second = best; best = sc; bk = k; } else if (sc > second) second = sc;
      }
      prev = topi[r * DF_TOPK + bk];
      drafts[r] = prev;
      conf[r] = best - second;
      if (lz_alpha > 0.f) lz_tok[lz_len + p] = prev;
    }
    __syncthreads();
  }
}
__global__ void __launch_bounds__(256) k_selector_sampled(
    const __nv_bfloat16* pred_cb, const __nv_bfloat16* succ_cb,
    const float* hproj, const float* topv, const int* topi,
    const int* anchor, int* drafts, float* conf, int proposals,
    H5SamplingConfig c, uint64_t seed, uint64_t first,
    const unsigned char* seen, int* error, double* transition_scores, DraftLz lz) {
  __shared__ float part[8][DF_TOPK];
  __shared__ int prev;
  __shared__ int lz_tok[DRAFT_LZ_MAX_WINDOW + DRAFT_MAX_PROPOSALS];
  __shared__ short lz_su[DRAFT_LZ_MAX_WINDOW + DRAFT_MAX_PROPOSALS];
  __shared__ float lz_adj[DF_TOPK];
  __shared__ int lz_red[1];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const float lz_alpha = lz.ctx ? lz.alpha[0] : 0.f;
  const int lz_len = lz_alpha > 0.f ? lz.len[0] : 0;
  for (int i = t; i < lz_len; i += 256) lz_tok[i] = lz.ctx[i];
  if (t == 0) prev = anchor[0];
  __syncthreads();
  for (int p = 0; p < proposals; ++p) {
    const float pr = __bfloat162float(pred_cb[size_t(prev) * DF_RANK + t]) * hproj[size_t(p) * DF_RANK + t];
    for (int k = 0; k < DF_TOPK; ++k) {
      const int candidate = topi[p * DF_TOPK + k];
      if (candidate < 0 || candidate >= VOCAB) atomicExch(error, 1);
      const int safe = candidate >= 0 && candidate < VOCAB ? candidate : 0;
      const float score = d_warp_sum(pr * __bfloat162float(succ_cb[size_t(safe) * DF_RANK + t]));
      if (lane == 0) part[warp][k] = score;
    }
    __syncthreads();
    if (lz_alpha > 0.f) draft_lz_adjust(lz, lz_alpha, lz_len + p, lz_tok, topi + p * DF_TOPK, lz_adj, lz_su, lz_red);
    if (t == 0) {
      double score[DF_TOPK]; int id[DF_TOPK];
      float best = -INFINITY, second = -INFINITY;
      for (int k = 0; k < DF_TOPK; ++k) {
        float raw = topv[p * DF_TOPK + k];
        for (int w = 0; w < 8; ++w) raw += part[w][k];
        if (lz_alpha > 0.f) raw += lz_adj[k];
        if (raw > best) { second = best; best = raw; } else if (raw > second) second = raw;
        if (!isfinite(raw)) { atomicExch(error, 1); raw = -INFINITY; }
        const int candidate = topi[p * DF_TOPK + k];
        const int token = candidate >= 0 && candidate < VOCAB ? candidate : 0;
        bool present = seen[token] != 0;
        for (int j = 0; !present && j < p; ++j) present = drafts[j] == token;
        double s = raw;
        if (present) { s = s < 0 ? s * c.repetition_penalty : s / c.repetition_penalty; s -= c.presence_penalty; }
        if (transition_scores) transition_scores[p * DF_TOPK + k] = s;
        int j = k;
        while (j > 0 && (score[j - 1] < s || (score[j - 1] == s && id[j - 1] > token))) {
          score[j] = score[j - 1]; id[j] = id[j - 1]; --j;
        }
        score[j] = s; id[j] = token;
      }
      int selected = id[0];
      if (c.temperature > 0) {
        const int limit = c.top_k ? min(c.top_k, DF_TOPK) : DF_TOPK;
        double weight[DF_TOPK], total = 0; int count = 0;
        for (int k = 0; k < limit; ++k) {
          weight[k] = exp((score[k] - score[0]) / c.temperature);
          if (weight[k] < c.min_p) break;
          total += weight[k]; ++count;
        }
        double sum = 0; int retained = 0;
        do { sum += weight[retained++]; } while (retained < count && sum < total * c.top_p);
        double maximum = -INFINITY;
        for (int k = 0; k < retained; ++k) {
          const double draw = (score[k] - score[0]) / c.temperature + sampling::noise_gumbel(seed, first + p, uint64_t(id[k]));
          if (draw > maximum || (draw == maximum && id[k] < selected)) { maximum = draw; selected = id[k]; }
        }
      }
      drafts[p] = prev = selected;
      conf[p] = best - second;
      if (lz_alpha > 0.f) lz_tok[lz_len + p] = selected;
    }
    __syncthreads();
  }
}
cudaError_t launch_selector_sampled(const __nv_bfloat16* pred, const __nv_bfloat16* succ,
    const float* hp, const float* topv, const int* topi, const int* anchor,
    int* drafts, float* conf, int proposals, const H5SamplingConfig& c,
    uint64_t seed, uint64_t first, const unsigned char* seen, int* error, double* transition_scores, const DraftLz& lz, cudaStream_t st) {
  k_selector_sampled<<<1, 256, 0, st>>>(pred, succ, hp, topv, topi, anchor, drafts, conf, proposals, c, seed, first, seen, error, transition_scores, lz);
  return cudaPeekAtLastError();
}

// Beam selection over the same candidate lattice (PrismML llama.cpp #284): keep the W best prefixes by
// summed log-softmax transition score (each beam normalises its own 16 transition scores), emit only the
// best full path. Verification is unchanged: one flat block of `proposals` tokens.
// Measured 2026-10-08 (relaxed15-vocab64, stock drafter, mixed speedbench): W=2 3.558 vs 3.565 tokens/round,
// W=4 3.487, drafter +0.024/+0.048 ms: negative; opt-in only.
// conf[r] = best - second transition score of the emitted path at position r (same meaning as k_selector).
template <int W>
__global__ void __launch_bounds__(256) k_selector_beam(const __nv_bfloat16* __restrict__ pred_cb, const __nv_bfloat16* __restrict__ succ_cb,
                                                       const float* __restrict__ hproj, const float* __restrict__ topv, const int* __restrict__ topi,
                                                       const int* __restrict__ anchor, int* __restrict__ drafts, float* __restrict__ conf, int proposals) { H5_LIT(0);
  pdl_wait(); pdl_launch();
  constexpr int PM = 64;
  __shared__ float part[8][W][DF_TOPK];
  __shared__ int prev[W], nbeam;
  __shared__ float score[W];
  __shared__ unsigned char path[2][W][PM];
  __shared__ float pconf[2][W][PM];
  const int b = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
  if (t == 0) { prev[0] = anchor[b]; score[0] = 0.f; nbeam = 1; }
  __syncthreads();
  const int P = proposals < PM ? proposals : PM;
  for (int p = 0; p < P; p++) {
    const int r = b * proposals + p, cur = p & 1, nxt = cur ^ 1;
    const float hp = hproj[(size_t)r * DF_RANK + t];
    float sv[DF_TOPK];
#pragma unroll
    for (int k = 0; k < DF_TOPK; k++) sv[k] = __bfloat162float(succ_cb[(size_t)topi[r * DF_TOPK + k] * DF_RANK + t]);
    for (int q = 0; q < nbeam; q++) {
      const float pr = __bfloat162float(pred_cb[(size_t)prev[q] * DF_RANK + t]) * hp;
#pragma unroll
      for (int k = 0; k < DF_TOPK; k++) {
        float sc = d_warp_sum(pr * sv[k]);
        if (lane == 0) part[w][q][k] = sc;
      }
    }
    __syncthreads();
    if (t == 0) {
      float cs[W]; int cb[W], ck[W]; float cc[W]; int nc = 0;
      for (int q = 0; q < nbeam; q++) {
        float s[DF_TOPK], mx = -INFINITY, best = -INFINITY, second = -INFINITY;
        for (int k = 0; k < DF_TOPK; k++) {
          float v = topv[r * DF_TOPK + k];
          for (int u = 0; u < 8; u++) v += part[u][q][k];
          s[k] = v; mx = fmaxf(mx, v);
          if (v > best) { second = best; best = v; } else if (v > second) second = v;
        }
        float z = 0.f; for (int k = 0; k < DF_TOPK; k++) z += __expf(s[k] - mx);
        const float lse = mx + __logf(z);
        for (int k = 0; k < DF_TOPK; k++) {
          const float c = score[q] + s[k] - lse;
          if (!(c == c)) continue;                                   // NaN sorts last
          const float cf = (s[k] == best) ? best - second : s[k] - best;
          int pos = nc < W ? nc++ : W;                               // insertion into the W best
          if (pos == W) { if (c <= cs[W - 1]) continue; pos = W - 1; }
          while (pos > 0 && cs[pos - 1] < c) { cs[pos] = cs[pos - 1]; cb[pos] = cb[pos - 1]; ck[pos] = ck[pos - 1]; cc[pos] = cc[pos - 1]; pos--; }
          cs[pos] = c; cb[pos] = q; ck[pos] = k; cc[pos] = cf;
        }
      }
      for (int j = 0; j < nc; j++) {
        for (int i = 0; i < p; i++) { path[nxt][j][i] = path[cur][cb[j]][i]; pconf[nxt][j][i] = pconf[cur][cb[j]][i]; }
        path[nxt][j][p] = (unsigned char)ck[j]; pconf[nxt][j][p] = cc[j];
      }
      for (int j = 0; j < nc; j++) { score[j] = cs[j]; prev[j] = topi[r * DF_TOPK + ck[j]]; }
      nbeam = nc;
    }
    __syncthreads();
  }
  const int fin = P & 1;
  for (int p = t; p < P; p += blockDim.x) {
    const int r = b * proposals + p;
    drafts[r] = topi[r * DF_TOPK + path[fin][0][p]];
    conf[r] = pconf[fin][0][p];
  }
}
void launch_selector(const __nv_bfloat16* pred_cb, const __nv_bfloat16* succ_cb, const float* hproj, const float* topv, const int* topi,
                     const int* anchor, int* drafts, float* conf, int blocks, int proposals, const DraftLz& lz, cudaStream_t st) {
  // H5_DRAFT_BEAM=W (2..4): beam selection (#284); unset or 1 = greedy walk.
  static const int beam = [] { const char* e = getenv("H5_DRAFT_BEAM"); const int v = e ? atoi(e) : 1;
    if (v < 1 || v > 4) { fprintf(stderr, "H5_DRAFT_BEAM must be 1..4\n"); abort(); } return v; }();
  if (beam > 1 && proposals > 64) { fprintf(stderr, "H5_DRAFT_BEAM supports at most 64 proposals\n"); abort(); }
  if (beam > 1 && lz.ctx) { fprintf(stderr, "H5_DRAFT_LZ excludes H5_DRAFT_BEAM\n"); abort(); }
  if (beam == 2) pdl(k_selector_beam<2>, dim3(blocks), dim3(256), 0, st, pred_cb, succ_cb, hproj, topv, topi, anchor, drafts, conf, proposals);
  else if (beam == 3) pdl(k_selector_beam<3>, dim3(blocks), dim3(256), 0, st, pred_cb, succ_cb, hproj, topv, topi, anchor, drafts, conf, proposals);
  else if (beam == 4) pdl(k_selector_beam<4>, dim3(blocks), dim3(256), 0, st, pred_cb, succ_cb, hproj, topv, topi, anchor, drafts, conf, proposals);
  else pdl(k_selector, dim3(blocks), dim3(256), 0, st, pred_cb, succ_cb, hproj, topv, topi, anchor, drafts, conf, proposals, lz);
}

__global__ void __launch_bounds__(256) k_selector_two(const __nv_bfloat16* __restrict__ pred_cb,
    const __nv_bfloat16* __restrict__ succ_cb, const float* __restrict__ hproj,
    const float* __restrict__ topv, const int* __restrict__ topi, const int* __restrict__ anchor,
    int* __restrict__ drafts, float* __restrict__ conf, int* __restrict__ alt,
    float* __restrict__ alt_conf, int* __restrict__ branches, int fork_span) { H5_LIT(0);
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

H5_LIT_EXPORT

extern "C" void h5_set_variant_draft(unsigned v) { if (cudaMemcpyToSymbol(h5::g_variant, &v, sizeof(v)) != cudaSuccess) abort(); }
