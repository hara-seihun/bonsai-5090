// Producer-tail row preparation (H5_TAIL_PREP). Included inside namespace h5 after kernels.cuh.
//
// The CTA that lands the last contribution to a 1024-feature chunk of a producer's output prepares
// that chunk for the next GEMM (sign, 1024-point Hadamard, NVFP4), so no separate prep launch and no
// extra PDL transition stand between producer and consumer. Every CTA counts its arrival per chunk;
// the arrival that completes a chunk's expected count finalises it. One warp prepares one or two rows
// of a chunk, in k_prep_chunk's lane layout (lane owns features 128 i + 4 lane + j).
//
// Latency is the whole cost, so the finaliser is a short chain of L2 round trips: the per-lane
// multipliers (norm weight times sign, precomputed at load) and the GDN gate z are loaded by every
// CTA before it arrives; arrival is one barrier and one release/acquire atomic by thread 0; all rows'
// inputs are loaded together.
//
// Row RMSNorm (TAIL_NORM) needs the whole row. The Hadamard is linear, so the chunk finalisers
// transform first, then exchange (sum of squares of the input, max |H(x w s)|) per row through a
// monotonic rendezvous ticket counter; only scaling and quantisation wait for it. The quantisation
// scale g = max|row| / (6 * 448) matches k_prep2w (dynamic per row); the RMS and max arrive in a
// different association, so results equal the separate prep up to rounding.
//
// The finaliser writes the next GEMM's activations while other CTAs of the producer may still read
// the current ones: the engine ping-pongs the activation buffers across every tail.
#pragma once
enum TailKind : int { TAIL_NONE = 0, TAIL_CHUNK = 1, TAIL_NORM = 2, TAIL_SIDE = 3 };   // TAIL_SIDE: no prep; only clearing and/or the flag start
constexpr int TAIL_MAX_CHUNKS = 32, TAIL_MAX_ROWS = 8, TAIL_DBG_STAMPS = 8;
struct TailArgs {
  const float* x = nullptr; int ldx = 0, n = 0, R = 0; unsigned flags = 0;
  const float* mul = nullptr;      // [n] per-feature multiplier before the Hadamard: sign, or norm weight * sign (TAIL_NORM)
  const float* gnorm_w = nullptr; const float* gz = nullptr; int gz_ld = 0;   // P_GDN_NORM
  uint8_t* q4 = nullptr; uint8_t* s4 = nullptr; float* g4 = nullptr; float fixed_g = 0.f;
  unsigned* cnt = nullptr;   // [TAIL_MAX_CHUNKS] arrival counters, zero between producers
  unsigned* rdv = nullptr;   // monotonic rendezvous ticket counter (TAIL_NORM)
  float* part = nullptr;     // [TAIL_MAX_CHUNKS][TAIL_MAX_ROWS][2] per chunk and row: sum of squares, max |H x|
  // The next split-K GEMM's outputs: every CTA zeroes its share after its predecessor wait.
  float* clear_a = nullptr; int clear_a_n4 = 0; float* clear_b = nullptr; int clear_b_n4 = 0;
  unsigned long long* dbg = nullptr;   // optional [TAIL_MAX_CHUNKS][TAIL_DBG_STAMPS] finaliser globaltimer stamps
  // H5_GEMM_FLAG: start the main loop when *wait_sig >= wait_expect (every CTA of the preceding prep has
  // published its activations, release) instead of at the prep grid's completion; griddepcontrol.wait then
  // runs at exit so this grid still completes after its predecessor.
  const unsigned* wait_sig = nullptr; unsigned wait_expect = 0;
};
#ifdef H5_GEMM_FLAG_POLL_NS
constexpr unsigned gemm_flag_poll_ns = H5_GEMM_FLAG_POLL_NS;
static_assert(gemm_flag_poll_ns == 64 || gemm_flag_poll_ns == 512, "H5_GEMM_FLAG_POLL_NS must be 64 or 512");
#else
constexpr unsigned gemm_flag_poll_ns = 64;
#endif
// Acquire the producer-done counter, then order the async proxy (TMA activation loads) after it.
__device__ __forceinline__ void tail_flag_wait(const unsigned* sig, unsigned expect) {
  unsigned v;
  for (;;) { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(sig) : "memory"); if (v >= expect) break; __nanosleep(gemm_flag_poll_ns); }
  if (!(g_variant & 8u)) asm volatile("fence.proxy.async.global;" ::: "memory");   // variant bit 0x8: measure without it
}

__device__ __forceinline__ float tp_wsum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); return v; }
__device__ __forceinline__ float tp_wmax(float v) { for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o)); return v; }
__device__ __forceinline__ float tp_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ void tail_bar(int nthr) { asm volatile("bar.sync 1, %0;" :: "r"(nthr) : "memory"); }
__device__ __forceinline__ unsigned tail_ld_acquire(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned tail_atom_add(unsigned* p) {
  unsigned v; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned long long tail_clock() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void tail_stamp(const TailArgs& t, int c, int k, int tid) {
  if (t.dbg && tid == 0) t.dbg[c * TAIL_DBG_STAMPS + k] = tail_clock();
}

__device__ __forceinline__ void tail_clear_share(const TailArgs& t, int cta, int ncta, int tid, int nthr) {
  auto one = [&](float* p, int n4) {
    if (!p) return;
    const int per = (n4 + ncta - 1) / ncta, b0 = cta * per, b1 = min(n4, b0 + per);
    for (int i = b0 + tid; i < b1; i += nthr) reinterpret_cast<float4*>(p)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  };
  one(t.clear_a, t.clear_a_n4); one(t.clear_b, t.clear_b_n4);
}

// Per-lane operands every CTA loads before arriving (independent of the producer's output).
template <int RPW> struct TailPre { float4 m[8]; float4 gw; float4 z[RPW][8]; };
template <int RPW, int NWARPS>
__device__ __forceinline__ void tail_preload(const TailArgs& t, int c, TailPre<RPW>& p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, off = c * 1024 + lane * 4;
#pragma unroll
  for (int i = 0; i < 8; i++) p.m[i] = __ldg((const float4*)(t.mul + off + i * 128));
  if (t.flags & P_GDN_NORM) {
    p.gw = __ldg((const float4*)(t.gnorm_w + lane * 4));
#pragma unroll
    for (int k = 0; k < RPW; k++) {
      const int row = warp + k * NWARPS;
      if (row >= t.R) continue;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int q = c * 8 + i, head = (t.flags & P_PERMUTE_GDN) ? q / 3 + 16 * (q % 3) : q;
        p.z[k][i] = __ldcg((const float4*)(t.gz + (size_t)row * t.gz_ld + head * 128 + lane * 4));
      }
    }
  }
}

// All nthr threads (barrier 1) call this after their output stores. Returns true in every thread
// when this CTA completed chunk c (expected arrivals `expect`). Thread 0's release/acquire atomic
// after the barrier publishes the CTA's stores and, for the finaliser, acquires everyone else's.
__device__ __forceinline__ bool tail_arrive(const TailArgs& t, int c, unsigned expect, int tid, int nthr, volatile unsigned* sm_flag) {
  tail_bar(nthr);
  if (tid == 0) {
    const bool last = tail_atom_add(t.cnt + c) == expect - 1;
    if (last) t.cnt[c] = 0;   // the next producer arrives only after this grid completes
    *sm_flag = last;
  }
  tail_bar(nthr);
  return *sm_flag != 0;
}

// Hadamard in k_prep_chunk's butterfly order (registers carry feature bits 0..1 and 7..9, lanes 2..6).
__device__ __forceinline__ void tail_hadamard(float (&v)[32]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int len = 1; len < 4; len <<= 1)
#pragma unroll
    for (int r = 0; r < 32; r++) if (!(r & len)) { const float p = v[r], q = v[r + len]; v[r] = p + q; v[r + len] = p - q; }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1)
#pragma unroll
    for (int r = 0; r < 32; r++) { const float q = __shfl_xor_sync(0xffffffff, v[r], o); v[r] = (lane & o) ? q - v[r] : v[r] + q; }
#pragma unroll
  for (int len = 4; len < 32; len <<= 1)
#pragma unroll
    for (int r = 0; r < 32; r++) if (!(r & len)) { const float p = v[r], q = v[r + len]; v[r] = p + q; v[r + len] = p - q; }
#pragma unroll
  for (int r = 0; r < 32; r++) v[r] *= .03125f;
}
// NVFP4 with row scale g, values scaled by mul first; layout of k_prep_chunk.
__device__ __forceinline__ void tail_quant(const TailArgs& t, int row, int c, const float (&v)[32], float mul, float g) {
  const int lane = threadIdx.x & 31;
  const float ig = 1.f / g;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float a0 = v[4 * i] * mul, a1 = v[4 * i + 1] * mul, a2 = v[4 * i + 2] * mul, a3 = v[4 * i + 3] * mul;
    float bm = fmaxf(fmaxf(fabsf(a0), fabsf(a1)), fmaxf(fabsf(a2), fabsf(a3)));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, 1));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, 2));
    const float target = bm / 6.f * ig;
    uint8_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
    float sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
    if (sv < target && sb < 0x7e) { sb++; sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3))); }
    const float rs = sv > 0.f ? ig / sv : 0.f;
    const uint16_t word = (uint16_t)__nv_cvt_float2_to_fp4x2(make_float2(a0 * rs, a1 * rs), __NV_E2M1, cudaRoundNearest) |
      ((uint16_t)__nv_cvt_float2_to_fp4x2(make_float2(a2 * rs, a3 * rs), __NV_E2M1, cudaRoundNearest) << 8);
    const int rr = row & 63, cc = lane >> 3, kb = c * 8 + i;
    const size_t dst = ((size_t)(row >> 6) * (t.n / 128) + kb) * 64 + rr;
    *(uint16_t*)(t.q4 + dst * 64 + ((cc ^ ((rr >> 1) & 3)) << 4) + (lane & 7) * 2) = word;
    if (!(lane & 3)) t.s4[dst * 8 + (lane >> 2)] = sb;
  }
}

// Finalise chunk c: warp w prepares rows w, w + NWARPS (< t.R <= TAIL_MAX_ROWS).
// TAIL_NORM: every chunk of the row has exactly one finaliser, and all of them run this exchange.
template <int KIND, int NWARPS, int RPW>
__device__ __forceinline__ void tail_finalise(const TailArgs& t, int c, int tid, int nthr, const TailPre<RPW>& p) {
  static_assert(RPW * NWARPS >= TAIL_MAX_ROWS, "rows per warp must cover the tail rows");
  const int lane = tid & 31, warp = tid >> 5;
  tail_stamp(t, c, 1, tid);
  float v[RPW][32];
  // every row's input in flight together: one L2 round trip
#pragma unroll
  for (int k = 0; k < RPW; k++) {
    const int row = warp + k * NWARPS;
    if (row >= t.R) continue;
    const float* x = t.x + (size_t)row * t.ldx;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int gi = c * 1024 + lane * 4 + i * 128, q = gi >> 7;
      const int in = (t.flags & P_PERMUTE_GDN) ? (gi & 127) + 128 * (q / 3 + 16 * (q % 3)) : gi;
      const float4 a = __ldcg((const float4*)(x + in));
      v[k][4 * i] = a.x; v[k][4 * i + 1] = a.y; v[k][4 * i + 2] = a.z; v[k][4 * i + 3] = a.w;
    }
  }
  float ssr[RPW], amr[RPW];
#pragma unroll
  for (int k = 0; k < RPW; k++) {
    const int row = warp + k * NWARPS;
    if (row >= t.R) continue;
    if (t.flags & P_GDN_NORM) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) ss = fmaf(v[k][4 * i + j], v[k][4 * i + j], ss);
        ss = tp_wsum(ss);
        const float inv = rsqrtf(ss / 128.f + NORM_EPS);
        const float4 z = p.z[k][i], gw = p.gw;
        v[k][4 * i] = v[k][4 * i] * inv * gw.x * tp_silu(z.x); v[k][4 * i + 1] = v[k][4 * i + 1] * inv * gw.y * tp_silu(z.y);
        v[k][4 * i + 2] = v[k][4 * i + 2] * inv * gw.z * tp_silu(z.z); v[k][4 * i + 3] = v[k][4 * i + 3] * inv * gw.w * tp_silu(z.w);
      }
    }
    if constexpr (KIND == TAIL_NORM) {
      float ss = 0.f;
#pragma unroll
      for (int r = 0; r < 32; r++) ss = fmaf(v[k][r], v[k][r], ss);
      ssr[k] = tp_wsum(ss);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      v[k][4 * i] *= p.m[i].x; v[k][4 * i + 1] *= p.m[i].y; v[k][4 * i + 2] *= p.m[i].z; v[k][4 * i + 3] *= p.m[i].w;
    }
    tail_hadamard(v[k]);
    if constexpr (KIND == TAIL_CHUNK) {
      tail_quant(t, row, c, v[k], 1.f, t.fixed_g);
      if (c == 0 && lane == 0) t.g4[row] = t.fixed_g;
    } else {
      float am = 0.f;
#pragma unroll
      for (int r = 0; r < 32; r++) am = fmaxf(am, fabsf(v[k][r]));
      amr[k] = tp_wmax(am);
      if (lane == 0) { float* q = t.part + ((size_t)c * TAIL_MAX_ROWS + row) * 2; __stcg(q, ssr[k]); __stcg(q + 1, amr[k]); }
    }
  }
  tail_stamp(t, c, 2, tid);
  if constexpr (KIND == TAIL_NORM) {
    __shared__ unsigned rdv_word;
    tail_bar(nthr);
    if (tid == 0) {
      const unsigned nch = (unsigned)(t.n >> 10);
      const unsigned ticket = tail_atom_add(t.rdv), target = (ticket / nch + 1) * nch;
      while ((int)(tail_ld_acquire(t.rdv) - target) < 0) {}
      rdv_word = target;
    }
    tail_bar(nthr);
    tail_stamp(t, c, 3, tid);
    const int nch = t.n >> 10;
#pragma unroll
    for (int k = 0; k < RPW; k++) {
      const int row = warp + k * NWARPS;
      if (row >= t.R) continue;
      float ss = 0.f, am = 0.f;
      for (int cc = 0; cc < nch; cc++) {
        const float* q = t.part + ((size_t)cc * TAIL_MAX_ROWS + row) * 2;
        ss += __ldcg(q); am = fmaxf(am, __ldcg(q + 1));
      }
      const float inv = rsqrtf(ss / (float)t.n + NORM_EPS);
      const float g = fmaxf(am * inv / (6.f * 448.f), 1e-30f);
      tail_quant(t, row, c, v[k], inv, g);
      if (c == 0 && lane == 0) t.g4[row] = g;
    }
  }
  tail_stamp(t, c, 4, tid);
}
