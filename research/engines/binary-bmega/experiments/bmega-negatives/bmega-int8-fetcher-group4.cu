// Binary Bonsai 27B persistent single-stream decode step. See bmega.cuh for the contract.
#include "bmega.cuh"
#include "model.h"
#include "fp4_ops.cuh"
#include <cuda_fp16.h>
#include <thread>
#include <atomic>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace bmega {
using h5::D; using h5::FF; using h5::NLAYER; using h5::VOCAB; using h5::HV; using h5::HK; using h5::KDIM; using h5::VDIM;
using h5::CONV_CH; using h5::NH; using h5::NKV; using h5::HD; using h5::NROT; using h5::Q_OUT; using h5::KV_OUT;
using h5::ATTN_OUT; using h5::NATTN; using h5::NGDN; using h5::NORM_EPS; using h5::ROPE_BASE; using h5::GDN_IN; using h5::ATT_IN;
using h5::mbar_init; using h5::mbar_expect; using h5::mbar_arrive; using h5::mbar_wait; using h5::ring_release; using h5::smem_u32; using h5::mma_fp4;

constexpr int IN_ROWS_GDN = GDN_IN + 2 * HV;     // qkv | z | alpha | beta = 16480
constexpr int WORK_BYTES = 40960;
constexpr int SMEM = NSLOT * CHUNK + 2 * NSLOT * 8 + 256 * 4 + 64 * 4 + 32 * 4 + 32 * 4 + WORK_BYTES;
static_assert(SMEM <= 101376, "bmega shared memory");
constexpr int PART_STRIDE = HD + 2;                // attention partial: o[256], m, l
constexpr int GROUP = 4;                           // chunks per TMA copy / ring barrier
constexpr int NGRP = NSLOT / GROUP;                // ring barrier groups
static_assert(NSLOT % GROUP == 0, "ring groups");
constexpr int ASTEP = 64;                          // activation bytes per k64 step: A8 bytes for lanes t=0..3 (16 B each)

enum Kind { K_IN_GDN, K_IN_ATT, K_OUT, K_GU, K_DOWN, K_HEAD };

// ------------------------------------------------------------------------------------------------
// Device helpers
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float bm_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float bm_sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ void csync() { asm volatile("bar.sync 1, %0;" :: "n"(NCW * 32) : "memory"); }
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }

__device__ __forceinline__ void stamp(const Args& a, int code) {
  if (a.prof && blockIdx.x == 0 && threadIdx.x == 0) {
    const unsigned long long i = a.prof[0]++;
    if (i < 4095) a.prof[1 + i] = (gtime() << 8) | (unsigned)code;
  }
}

// Consumer-only grid barrier on a monotonic 64-bit counter (never reset; every launch adds the same
// multiple of NCTA). The producer warp never joins: it only waits for ring slots.
__device__ __forceinline__ void gsync(const Args& a, int bi = 0) {
  csync();
  if (threadIdx.x == 0) {
    if (a.arrive) a.arrive[(size_t)blockIdx.x * 1024 + 2 * bi] = gtime();
    unsigned long long old;
    asm volatile("atom.add.release.gpu.u64 %0, [%1], 1;" : "=l"(old) : "l"(a.bar) : "memory");
    const unsigned long long target = (old / NCTA + 1) * NCTA;
    unsigned long long cur;
    do { asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(cur) : "l"(a.bar) : "memory"); } while (cur < target);
    if (a.arrive) a.arrive[(size_t)blockIdx.x * 1024 + 2 * bi + 1] = gtime();
  }
  csync();
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// Sum over the 256 consumer threads; red holds 8 floats. Ends synchronized.
__device__ __forceinline__ float cta_sum(float v, float* red) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  csync();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NCW; i++) s += red[i];
  csync();
  return s;
}

// Exact signed digits of an original A8 byte: q = a + 4b + 16c with a,b,c in 2xE2M1. Nibbles hold the
// half-digit E2M1 codes; the table returns them in bytes 0,1,2 for byte-permute packing.
__host__ __device__ inline uint32_t exact_codes(int q) {
  unsigned a = unsigned(q) & 3, b = (unsigned(q) >> 2) & 3; int c = q >> 4;
  unsigned cm = c < 0 ? -c : c, bcode = b;
  if (cm == 5 || cm == 7) { int delta = (b & 1) ? 1 : -1; c += delta; bcode = (0x95b4u >> (4 * b)) & 15; }
  cm = c < 0 ? -c : c;
  const unsigned ccode = (cm <= 4 ? cm : cm / 2 + 2) | (c < 0 ? 8u : 0u);
  return a | (bcode << 8) | (ccode << 16);
}

// One octet (8 consecutive K values) of the activation: quantize with the block's 127/max and write the
// three FP4 digit-plane words into the B-fragment layout act[k64 step][plane][t][half].
// One octet (8 consecutive K values) of the activation: original A8 (round to nearest even with the
// block's 127/max) written as INT8 B-fragment bytes: per k64 step, lane t holds k = 4t.., 16+4t.., 32+4t.., 48+4t..
__device__ __forceinline__ void encode_octet(const float* v, float inv, const uint32_t* enc, uint32_t* act, int o) {
  (void)enc;
  uint32_t w[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    uint32_t word = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int q = __float2int_rn(v[4 * h + j] * inv);
      q = max(-127, min(127, q));
      word |= (uint32_t)(q & 0xff) << (8 * j);
    }
    w[h] = word;
  }
  const int s = o >> 3, oo = o & 7;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int u = 2 * oo + h;                       // 4-byte group within the k64 step
    const int m = u >> 3, half = (u >> 2) & 1, t = u & 3;
    act[(s * 4 + t) * 4 + m * 2 + half] = w[h];
  }
}

// ------------------------------------------------------------------------------------------------
// Encoded activation exchange, dataflow. Each GEMV input is encoded exactly once, distributed: the last
// producer of each K128 block (modular arrival counter) quantizes it, writes its FP4 digit words and
// final block scale, then publishes a generation-tagged ready flag. Consumers wait only for the blocks
// of the K piece they are about to use; no grid barrier exists anywhere in the step.
// ------------------------------------------------------------------------------------------------
constexpr int NBLK = FF / 128;                     // 136 blocks in the widest input
constexpr int AB_FRAG = (FF / 64) * ASTEP;         // 17408 B of A8 bytes
constexpr int AB_SCALE = AB_FRAG, AB_KEY = AB_SCALE + NBLK * 4, AB_CNT = AB_KEY + NBLK * 8, AB_SS = AB_CNT + NBLK * 4,
              AB_HSS = AB_SS + NCTA * 4, AB_RDY = ((AB_HSS + GDN_ITEMS * 4) + 15) / 16 * 16;
constexpr int AB_BYTES = ((AB_RDY + NBLK * 8) + 255) / 256 * 256;
__device__ __forceinline__ uint8_t* abuf(const Args& a, int g) { return a.actbuf + (size_t)(g % 3) * AB_BYTES; }
__device__ __forceinline__ uint32_t* ab_frag(uint8_t* b) { return (uint32_t*)b; }
__device__ __forceinline__ float* ab_scale(uint8_t* b) { return (float*)(b + AB_SCALE); }
__device__ __forceinline__ unsigned long long* ab_key(uint8_t* b) { return (unsigned long long*)(b + AB_KEY); }
__device__ __forceinline__ unsigned* ab_cnt(uint8_t* b) { return (unsigned*)(b + AB_CNT); }
__device__ __forceinline__ float* ab_ss(uint8_t* b) { return (float*)(b + AB_SS); }
__device__ __forceinline__ float* ab_hss(uint8_t* b) { return (float*)(b + AB_HSS); }
__device__ __forceinline__ unsigned long long* ab_rdy(uint8_t* b) { return (unsigned long long*)(b + AB_RDY); }
__device__ __forceinline__ unsigned long long gen_of(const Args& a, int g) { return (unsigned long long)a.step * 512ull + (unsigned long long)g + 1ull; }

__device__ __forceinline__ unsigned long long ld_acq64(const unsigned long long* p) {
  unsigned long long v; asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_rel64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ int g_nowait = 0;
__device__ unsigned long long* g_wacc = nullptr;   // per-CTA wait accumulators (diagnostic)
__device__ int g_wkind = 0;
__device__ __forceinline__ void wait_ge(const unsigned long long* p, unsigned long long v) {
  if (g_nowait) return;
  while (ld_acq64(p) < v) __nanosleep(20);
}
__device__ __forceinline__ unsigned atom_add_acqrel(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
// Generation-tagged block maximum: a newer generation always wins over stale values, no reset needed.
__device__ __forceinline__ void tag_max(unsigned long long* key, unsigned long long gen, float m) {
  atomicMax(key, (gen << 32) | (unsigned long long)__float_as_uint(m));
}
__device__ __forceinline__ float tag_value(unsigned long long k) { return __uint_as_float((unsigned)(k & 0xffffffffull)); }

// Encode one 128-value block with lanes 0..15 of a warp (one octet each): v = src * w (w optional).
__device__ __forceinline__ void encode_block(const float* src, const float* w, float bmax, uint32_t* frag, int blk,
                                             const uint32_t* enc, int lane) {
  if (lane >= 16) return;
  const int o = blk * 16 + lane;
  const float4 p0 = __ldcg((const float4*)(src + lane * 8)), p1 = __ldcg((const float4*)(src + lane * 8 + 4));
  float v[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
  if (w) {
    const float4 w0 = __ldg((const float4*)(w + lane * 8)), w1 = __ldg((const float4*)(w + lane * 8 + 4));
    v[0] *= w0.x; v[1] *= w0.y; v[2] *= w0.z; v[3] *= w0.w; v[4] *= w1.x; v[5] *= w1.y; v[6] *= w1.z; v[7] *= w1.w;
  }
  encode_octet(v, bmax > 0.f ? 127.f / bmax : 0.f, enc, frag, o);
}
// After a warp's encode_block: publish the block scale and its ready flag (lane 0).
__device__ __forceinline__ void publish_block(uint8_t* nb, int blk, float scale, unsigned long long gen, int lane) {
  __syncwarp();
  if (lane == 0) {
    ab_scale(nb)[blk] = scale;
    __threadfence();
    st_rel64(ab_rdy(nb) + blk, gen);
  }
  __syncwarp();
}

// Residual producer epilogue (embedding, OUT, DOWN): this CTA owns rows n0..n0+31 (a quarter block).
__device__ __forceinline__ void residual_publish(const Args& a, int n0, const float* xv, const float* wnext, uint8_t* nb,
                                                 unsigned long long gen, const uint32_t* enc, float* red) {
  const int tid = threadIdx.x, lane = tid & 31;
  if (tid >= 32) return;
  const float xn = xv[lane];
  a.x[n0 + lane] = xn;
  float m = fabsf(xn * __ldg(wnext + n0 + lane)), ss = xn * xn;
  m = warp_max(m); ss = warp_sum(ss);
  const int blk = n0 >> 7;
  unsigned last = 0;
  if (lane == 0) {
    ab_ss(nb)[blockIdx.x] = ss;
    tag_max(ab_key(nb) + blk, gen, m);
    last = ((atom_add_acqrel(ab_cnt(nb) + blk, 32u) + 32u) & 127u) == 0u;
  }
  last = __shfl_sync(0xffffffffu, last, 0);
  if (last) {
    const float bmax = tag_value(__ldcg(ab_key(nb) + blk));
    encode_block(a.x + blk * 128, wnext + blk * 128, bmax, ab_frag(nb), blk, enc, lane);
    publish_block(nb, blk, bmax / 127.f, gen, lane);
  }
}

// Inverse RMS of a residual input: every producer CTA stored its 32-row sum of squares.
__device__ __forceinline__ float residual_inv(uint8_t* nb, float* red) {
  const int tid = threadIdx.x;
  float v = 0.f;
  if (tid < NCTA) v = __ldcg(ab_ss(nb) + tid);
  v = warp_sum(v);
  if ((tid & 31) == 0) red[32 + (tid >> 5)] = v;
  csync();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NCW; i++) s += red[32 + i];
  csync();
  return rsqrtf(s / (float)D + NORM_EPS);
}

// Input fetch warp: for each GEMV phase, as soon as the consumers announce it (fphase), copy every K piece
// (8 blocks = 1 KB of A8 bytes + 8 scales) into shared memory the moment its blocks are ready, four pieces
// in flight at a time, and tag each piece with the phase. Consumers only test the shared tag.
__device__ __forceinline__ void fetcher(const Args& a, volatile int* fphase, volatile int* pflag, float* ssacc, uint8_t* work) {
  const int lane = threadIdx.x & 31;
  auto phase = [&](int ph, int K, bool rms) {
    while (*fphase != ph) __nanosleep(16);
    __threadfence_block();
    uint8_t* nb = abuf(a, ph);
    const unsigned long long gen = gen_of(a, ph);
    const int P = K / 1024;
    float* asc = (float*)(work + (size_t)(K / 64) * ASTEP);
    for (int p0 = 0; p0 < P; p0 += 4) {
      const int np = min(4, P - p0);
      if (lane < 8 * np) wait_ge(ab_rdy(nb) + p0 * 8 + lane, gen);
      __syncwarp();
      uint4 v[8];
      float sc = 0.f, ssv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int q = 0; q < 4; q++)
        if (q < np) {
          const uint4* src = (const uint4*)ab_frag(nb) + (size_t)(p0 + q) * 64;
          v[2 * q] = __ldcg(src + lane); v[2 * q + 1] = __ldcg(src + 32 + lane);
          if (rms) ssv[q] = __ldcg(ab_ss(nb) + (p0 + q) * 32 + lane);
        }
      if (lane < 8 * np) sc = __ldcg(ab_scale(nb) + p0 * 8 + lane);
#pragma unroll
      for (int q = 0; q < 4; q++)
        if (q < np) {
          uint4* dst = (uint4*)work + (size_t)(p0 + q) * 64;
          dst[lane] = v[2 * q]; dst[32 + lane] = v[2 * q + 1];
          if (rms) { const float t = warp_sum(ssv[q]); if (lane == 0) ssacc[p0 + q] = t; }
        }
      if (lane < 8 * np) asc[p0 * 8 + lane] = sc;
      __syncwarp();
      __threadfence_block();
      if (lane < np) pflag[p0 + lane] = ph + 1;
      __syncwarp();
    }
  };
  for (int l = 0; l < NLAYER; l++) {
    phase(4 * l, D, true);
    phase(4 * l + 1, VDIM, false);
    phase(4 * l + 2, D, true);
    phase(4 * l + 3, FF, false);
  }
  phase(NPH - 1, D, true);
}

__device__ __forceinline__ void imma(int* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// One 16-row x 1024-K chunk against the shared A8 activation: binary signs expand in registers to INT8
// +-1 A fragments (bit 8j+r of the lane word is byte j of register r), exact integer K128 dots on INT8 MMA,
// then original FP16 weight scale times activation scale, FP32 accumulation per row.
__device__ __forceinline__ void chunk_dot(const uint8_t* ch, const uint4* act, const float* asc, int lane, float& rg, float& r8) {
  const uint32_t* sg = (const uint32_t*)ch;
  const uint32_t* sc = (const uint32_t*)(ch + 2048);
  const int g = lane >> 2, t = lane & 3;
  const uint4* ap = act + t;
  float accg = 0.f, acc8 = 0.f;
#pragma unroll 2
  for (int b = 0; b < 8; b++) {
    int d[4] = {0, 0, 0, 0};
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int s = b * 2 + hh;
      const uint32_t w = sg[s * 32 + lane];
      unsigned A[8];
#pragma unroll
      for (int r = 0; r < 8; r++) A[r] = 0x01010101u + ((w >> r) & 0x01010101u) * 0xFEu;
      const uint4 bv = ap[s * 4];
      imma(d, A, bv.x, bv.y);
      imma(d, A + 4, bv.z, bv.w);
    }
    const uint32_t su = sc[b * 8 + g];
    const float2 ws = __half22float2(*(const __half2*)&su);
    const float as = asc[b];
    accg = fmaf((float)d[0], __fmul_rn(ws.x, as), accg);
    acc8 = fmaf((float)d[2], __fmul_rn(ws.y, as), acc8);
  }
  rg = accg; r8 = acc8;
}

// A weight phase: consume this CTA's chunks of projection ph (piece-major order), then its epilogue.
// xs holds this CTA's 32 residual rows (OUT/DOWN own rows 32c..32c+31); wnext is the next RMSNorm weight.
template <int KIND>
__device__ __forceinline__ void gemv(const Args& a, const LayerP& L, int ph, int K, uint8_t* ring, uint64_t* full, uint64_t* empty,
                                     uint8_t* work, float* xs, const float* wnext, const uint32_t* enc, float* red, int* pflag) {
  const int4 e = a.tab[ph * NCTA + blockIdx.x];
  const int t0 = e.x, nt = e.y, base = e.z, P = K / 1024, nch = nt * P;
  const uint4* act = (const uint4*)work;
  const float* asc = (const float*)(work + (size_t)(K / 64) * ASTEP);
  float* part = (float*)(work + (size_t)(K / 64) * ASTEP + (size_t)(K / 128) * 4);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr bool RMS_IN = KIND == K_IN_GDN || KIND == K_IN_ATT || KIND == K_GU || KIND == K_HEAD;
  float* ssacc = red + 48;
  if (g_wacc && tid == 0) g_wacc[(size_t)blockIdx.x * 1024 + 3 * ph] = gtime();
  if (tid == 0) { __threadfence_block(); *(volatile int*)(pflag + 31) = ph; }   // the previous phase's buffers are free
  // conv operands of this CTA's in-projection rows do not depend on the projection: load them first
  float4 cst = make_float4(0.f, 0.f, 0.f, 0.f), cwt = make_float4(0.f, 0.f, 0.f, 0.f);
  if constexpr (KIND == K_IN_GDN) {
    const int n = t0 * 16 + tid;
    if (tid < nt * 16 && n < CONV_CH) {
      const float* st = a.conv_state + ((size_t)a.slot * NGDN + L.gl) * 3 * CONV_CH;
      cst = make_float4(st[n], st[CONV_CH + n], st[2 * CONV_CH + n], 0.f);
      cwt = __ldg((const float4*)(L.conv_w + (size_t)n * 4));
    }
  }
  csync();
  for (int j = warp; j < nch; j += NCW) {
    const int piece = j / nt, tl = j - piece * nt;
    while (((volatile int*)pflag)[piece] != ph + 1) __nanosleep(16);
    __threadfence_block();
    const int i = base + j, slot = i % NSLOT, grp = (i / GROUP) % NGRP;
    {
      unsigned long long tw = g_wacc ? gtime() : 0ull;
      mbar_wait(full + grp, (unsigned)(i / NSLOT) & 1u);
      if (g_wacc && lane == 0) atomicAdd(g_wacc + (size_t)blockIdx.x * 1024 + 920 + KIND, gtime() - tw);
    }
    float rg, r8;
    chunk_dot(ring + (size_t)slot * CHUNK, act + piece * 16 * 4, asc + piece * 8, lane, rg, r8);
    ring_release(empty + grp, lane == 0);
    if ((lane & 3) == 0) { part[(tl * P + piece) * 16 + (lane >> 2)] = rg; part[(tl * P + piece) * 16 + 8 + (lane >> 2)] = r8; }
  }
  csync();
  if (g_wacc && tid == 0) g_wacc[(size_t)blockIdx.x * 1024 + 3 * ph + 1] = gtime();
  stamp(a, 32 + KIND);
  if (g_nowait >= 2) return;
  float inv = 1.f;
  if constexpr (RMS_IN) { float ssum = 0.f; for (int pc = 0; pc < P; pc++) ssum += ssacc[pc]; inv = rsqrtf(ssum / (float)D + NORM_EPS); }
  if constexpr (KIND == K_GU) {
    uint8_t* nb = abuf(a, ph + 1);
    const unsigned long long gout = gen_of(a, ph + 1);
    const int h0 = t0 * 8, hn = nt * 8;
    float v = 0.f;
    if (tid < hn) {
      const int tl = tid >> 3, rr = tid & 7;
      float gsum = 0.f, usum = 0.f;
      for (int pc = 0; pc < P; pc++) { gsum += part[(tl * P + pc) * 16 + rr]; usum += part[(tl * P + pc) * 16 + 8 + rr]; }
      v = bm_silu(gsum * inv) * (usum * inv);
      a.h[h0 + tid] = v;
    }
    const int b0 = h0 >> 7, split = min(hn, (b0 + 1) * 128 - h0);
    float m0 = tid < split ? fabsf(v) : 0.f, m1 = (tid >= split && tid < hn) ? fabsf(v) : 0.f;
    m0 = warp_max(m0); m1 = warp_max(m1);
    if (lane == 0) { red[warp] = m0; red[8 + warp] = m1; }
    csync();
    if (tid == 0) {
      float mm0 = 0.f, mm1 = 0.f;
      for (int w = 0; w < NCW; w++) { mm0 = fmaxf(mm0, red[w]); mm1 = fmaxf(mm1, red[8 + w]); }
      unsigned flags = 0;
      tag_max(ab_key(nb) + b0, gout, mm0);
      if (split < hn) tag_max(ab_key(nb) + b0 + 1, gout, mm1);
      if (((atom_add_acqrel(ab_cnt(nb) + b0, (unsigned)split) + (unsigned)split) & 127u) == 0u) flags |= 1u;
      if (split < hn && ((atom_add_acqrel(ab_cnt(nb) + b0 + 1, (unsigned)(hn - split)) + (unsigned)(hn - split)) & 127u) == 0u) flags |= 2u;
      ((unsigned*)red)[16] = flags;
    }
    csync();
    const unsigned flags = ((unsigned*)red)[16];
    if (warp < 2 && (flags & (1u << warp))) {
      const int b = b0 + warp;
      const float bmax = tag_value(__ldcg(ab_key(nb) + b));
      encode_block(a.h + b * 128, nullptr, bmax, ab_frag(nb), b, enc, lane);
      publish_block(nb, b, bmax / 127.f, gout, lane);
    }
  } else if constexpr (KIND == K_HEAD) {
    unsigned long long best = 0ull;
    for (int r = tid; r < nt * 16; r += NCW * 32) {
      const int tl = r >> 4, rr = r & 15;
      float v = 0.f;
      for (int pc = 0; pc < P; pc++) v += part[(tl * P + pc) * 16 + rr];
      v *= inv;
      const unsigned n = (unsigned)((t0 + tl) * 16 + rr);
      if (a.logits) a.logits[n] = v;
      unsigned u = __float_as_uint(v); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
      const unsigned long long key = ((unsigned long long)u << 32) | (0xFFFFFFFFu - n);
      best = key > best ? key : best;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) { const unsigned long long ob = __shfl_xor_sync(0xffffffffu, best, o); best = ob > best ? ob : best; }
    if (lane == 0) atomicMax(a.amax + (a.step & 1u), best);
  } else if constexpr (KIND == K_OUT || KIND == K_DOWN) {
    if (tid < 32) {
      float v = 0.f;
      for (int pc = 0; pc < P; pc++) v += part[((tid >> 4) * P + pc) * 16 + (tid & 15)];
      xs[tid] += v;
    }
    csync();
    residual_publish(a, t0 * 16, xs, wnext, abuf(a, ph + 1), gen_of(a, ph + 1), enc, red);
  } else {
    for (int r = tid; r < nt * 16; r += NCW * 32) {
      const int tl = r >> 4, rr = r & 15;
      float v = 0.f;
      for (int pc = 0; pc < P; pc++) v += part[(tl * P + pc) * 16 + rr];
      v *= inv;
      const int n = (t0 + tl) * 16 + rr;
      if constexpr (KIND == K_IN_GDN) {
        if (n < CONV_CH) {
          float* st = a.conv_state + ((size_t)a.slot * NGDN + L.gl) * 3 * CONV_CH;
          a.conv_out[n] = bm_silu(cwt.x * cst.x + cwt.y * cst.y + cwt.z * cst.z + cwt.w * v);
          st[n] = cst.y; st[CONV_CH + n] = cst.z; st[2 * CONV_CH + n] = v;
        } else if (n < GDN_IN) a.zbuf[n - CONV_CH] = v;
        else a.ab[n - GDN_IN] = v;
      } else {
        a.att[n] = v;
      }
    }
    csync();
    if (tid == 0) { __threadfence(); st_rel64(a.indone + blockIdx.x, gen_of(a, ph)); }
  }
  csync();
  if (g_wacc && tid == 0) g_wacc[(size_t)blockIdx.x * 1024 + 3 * ph + 2] = gtime();
  stamp(a, 48 + KIND);
}

// Mix-phase input readiness: wait for every CTA whose input-projection rows intersect [r0,r1) ranges.
__device__ __forceinline__ void wait_rows(const Args& a, int ph, const int2* ranges, int nr) {
  const int c = threadIdx.x;
  const unsigned long long tw = gtime();
  if (c < NCTA) {
    const int4 e = a.tab[ph * NCTA + c];
    const int lo = e.x * 16, hi = (e.x + e.y) * 16;
    bool need = false;
    for (int i = 0; i < nr; i++) need |= ranges[i].x < hi && ranges[i].y > lo;
    if (need) wait_ge(a.indone + c, gen_of(a, ph));
  }
  csync();
  if (g_wacc && threadIdx.x == 0) atomicAdd(g_wacc + (size_t)blockIdx.x * 1024 + 910, gtime() - tw);
}

// GDN recurrence for one (v-head, column half) item; q/k L2 norms and gates computed locally.
__device__ __forceinline__ void gdn_item(const Args& a, const LayerP& L, int ph, uint8_t* work, uint8_t* nb, unsigned long long gout,
                                         const uint32_t* enc, float* red) {
  const int item = blockIdx.x;
  const int h = item >> 1, half = item & 1, kh = h % HK, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  {
    const int2 rg[6] = {make_int2(kh * 128, kh * 128 + 128), make_int2(KDIM + kh * 128, KDIM + kh * 128 + 128),
                        make_int2(2 * KDIM + h * 128 + half * 64, 2 * KDIM + h * 128 + half * 64 + 64),
                        make_int2(CONV_CH + h * 128 + half * 64, CONV_CH + h * 128 + half * 64 + 64),
                        make_int2(GDN_IN + h, GDN_IN + h + 1), make_int2(GDN_IN + HV + h, GDN_IN + HV + h + 1)};
    wait_rows(a, ph, rg, 6);
  }
  float* sq = (float*)work; float* sk = sq + 128; float* sv = sk + 128; float* rq = sv + 64; float* gates = rq + 8;
  {
    const int d = tid & 127;
    const float v = __ldcg(a.conv_out + (tid < 128 ? 0 : KDIM) + kh * 128 + d);
    const float ss = warp_sum(v * v);
    if (lane == 0) rq[warp] = ss;
    if (tid < 64) sv[tid] = __ldcg(a.conv_out + 2 * KDIM + h * 128 + half * 64 + tid);
    if (tid == 0) {
      const float xg = __ldcg(a.ab + h) + L.ssm_dt[h];
      const float sp = xg > 20.f ? xg : log1pf(__expf(xg));
      gates[0] = __expf(L.ssm_a[h] * sp);
      gates[1] = bm_sigm(__ldcg(a.ab + HV + h));
    }
    csync();
    const int b = tid < 128 ? 0 : 4;
    const float s = rq[b] + rq[b + 1] + rq[b + 2] + rq[b + 3];
    (tid < 128 ? sq : sk)[d] = v * (1.f / fmaxf(sqrtf(s), NORM_EPS));
    csync();
  }
  const int jl = tid >> 2, cg = tid & 3, j = half * 64 + jl;
  __half* M = a.gdn_state + ((((size_t)a.slot * NGDN + L.gl) * HV + h) * 128 + j) * 128 + 4 * cg;
  float m[32], kr[32];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint2 raw = *(const uint2*)(M + 16 * i);
    const float2 f0 = __half22float2(*(const __half2*)&raw.x), f1 = __half22float2(*(const __half2*)&raw.y);
    m[4 * i] = f0.x; m[4 * i + 1] = f0.y; m[4 * i + 2] = f1.x; m[4 * i + 3] = f1.y;
    const float4 k4 = *(const float4*)(sk + 16 * i + 4 * cg);
    kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w;
  }
  float dot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 32; ++i) dot[i & 3] = fmaf(m[i], kr[i], dot[i & 3]);
  float s = (dot[0] + dot[1]) + (dot[2] + dot[3]);
  s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
  const float g = gates[0], delta = gates[1] * (sv[jl] - g * s);
#pragma unroll
  for (int i = 0; i < 32; ++i) m[i] = fmaf(g, m[i], delta * kr[i]);
  float out[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float4 q = *(const float4*)(sq + 16 * i + 4 * cg);
    out[0] = fmaf(m[4 * i], q.x, out[0]); out[1] = fmaf(m[4 * i + 1], q.y, out[1]);
    out[2] = fmaf(m[4 * i + 2], q.z, out[2]); out[3] = fmaf(m[4 * i + 3], q.w, out[3]);
  }
  float o = (out[0] + out[1]) + (out[2] + out[3]);
  o += __shfl_xor_sync(0xffffffffu, o, 1); o += __shfl_xor_sync(0xffffffffu, o, 2);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const __half2 lo = __floats2half2_rn(m[4 * i], m[4 * i + 1]), hi = __floats2half2_rn(m[4 * i + 2], m[4 * i + 3]);
    uint2 v; v.x = *(const uint32_t*)&lo; v.y = *(const uint32_t*)&hi;
    *(uint2*)(M + 16 * i) = v;
  }
  // gated RMSNorm input of ssm_out: u = y * w * silu(z); block h of the next GEMV input
  const float y = o * 0.08838834764831845f;
  const float u = y * __ldg(L.ssm_norm + j) * bm_silu(__ldcg(a.zbuf + h * 128 + j));
  if (cg == 0) a.ubuf[h * 128 + j] = u;
  float mu = cg == 0 ? fabsf(u) : 0.f, sy = cg == 0 ? y * y : 0.f;
  mu = warp_max(mu); sy = warp_sum(sy);
  if (lane == 0) { red[warp] = mu; red[8 + warp] = sy; }
  csync();
  if (tid == 0) {
    float mm = 0.f, ss = 0.f;
    for (int w = 0; w < NCW; w++) { mm = fmaxf(mm, red[w]); ss += red[8 + w]; }
    ab_hss(nb)[item] = ss;
    tag_max(ab_key(nb) + h, gout, mm);
    ((unsigned*)red)[16] = ((atom_add_acqrel(ab_cnt(nb) + h, 64u) + 64u) & 127u) == 0u;
  }
  csync();
  if (((unsigned*)red)[16] && warp == 0) {
    const float bmax = tag_value(__ldcg(ab_key(nb) + h));
    const float hs = __ldcg(ab_hss(nb) + 2 * h) + __ldcg(ab_hss(nb) + 2 * h + 1);
    encode_block(a.ubuf + h * 128, nullptr, bmax, ab_frag(nb), h, enc, lane);
    publish_block(nb, h, rsqrtf(hs / 128.f + NORM_EPS) * (bmax / 127.f), gout, lane);
  }
}

// RMSNorm (per 256-dim head) and NeoX RoPE on the first 64 dims, as attn_pre_body.
__device__ __forceinline__ float rope_dim(float v, float partner_lo, float partner_hi, int d, int pos) {
  if (d >= NROT) return v;
  const int i = d < NROT / 2 ? d : d - NROT / 2;
  const float theta = (float)pos * powf(ROPE_BASE, -(float)i / (float)(NROT / 2));
  float c, sn; sincosf(theta, &sn, &c);
  return d < NROT / 2 ? v * c - partner_hi * sn : partner_lo * sn + v * c;
}

// Attention for one (kv head, key split) item over keys 0..pos; last arriver of a kv head merges its
// six query heads, applies the output gate and writes y.
__device__ __forceinline__ void attn_item(const Args& a, const LayerP& L, int ph, uint8_t* work, int S, uint8_t* nb,
                                          unsigned long long gout, const uint32_t* enc) {
  const int item = blockIdx.x, hk = item & 3, sp = item >> 2;
  {
    const int2 rg[3] = {make_int2(hk * 6 * 2 * HD, (hk * 6 + 6) * 2 * HD), make_int2(Q_OUT + hk * HD, Q_OUT + hk * HD + HD),
                        make_int2(Q_OUT + KV_OUT + hk * HD, Q_OUT + KV_OUT + hk * HD + HD)};
    wait_rows(a, ph, rg, 3);
  }
  const int P = a.pos + 1, Lp = (P + S - 1) / S, p0 = min(P, sp * Lp), p1 = min(P, p0 + Lp), n = p1 - p0;
  const int d = threadIdx.x, lane = d & 31, warp = d >> 5;
  float* sq = (float*)work;              // [6][256]
  float* skn = sq + 6 * HD;              // [256]
  float* svn = skn + HD;                 // [256]
  float* red = svn + HD;                 // [8][8]
  float* ssc = red + 64;                 // [6][Lp]
  int* flag = (int*)(ssc + 6 * Lp);
  // query heads hq = hk*6 + i: RMSNorm with q_norm, RoPE, 1/16
  float qv[6];
  float ss[6];
#pragma unroll
  for (int i = 0; i < 6; i++) { qv[i] = __ldcg(a.att + (size_t)(hk * 6 + i) * 2 * HD + d); ss[i] = warp_sum(qv[i] * qv[i]); }
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 6; i++) red[warp * 8 + i] = ss[i];
  }
  const bool last = n > 0 && p1 == P;
  float kv = 0.f, vv = 0.f, kss = 0.f;
  if (last) {
    kv = __ldcg(a.att + Q_OUT + hk * HD + d);
    vv = __ldcg(a.att + Q_OUT + KV_OUT + hk * HD + d);
    kss = warp_sum(kv * kv);
    if (lane == 0) red[warp * 8 + 6] = kss;
  }
  csync();
#pragma unroll
  for (int i = 0; i < 6; i++) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < NCW; w++) t += red[w * 8 + i];
    qv[i] = qv[i] * rsqrtf(t / (float)HD + NORM_EPS) * L.q_norm[d];
    sq[i * HD + d] = qv[i];
  }
  if (last) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < NCW; w++) t += red[w * 8 + 6];
    kv = kv * rsqrtf(t / (float)HD + NORM_EPS) * L.k_norm[d];
    skn[d] = kv;
  }
  csync();
  float qr[6];
#pragma unroll
  for (int i = 0; i < 6; i++) {
    const float lo = d >= NROT / 2 && d < NROT ? sq[i * HD + d - NROT / 2] : 0.f;
    const float hi = d < NROT / 2 ? sq[i * HD + d + NROT / 2] : 0.f;
    qr[i] = rope_dim(qv[i], lo, hi, d, a.pos) * 0.0625f;
  }
  float kr = 0.f;
  if (last) {
    const float lo = d >= NROT / 2 && d < NROT ? skn[d - NROT / 2] : 0.f;
    const float hi = d < NROT / 2 ? skn[d + NROT / 2] : 0.f;
    kr = rope_dim(kv, lo, hi, d, a.pos);
  }
  csync();
#pragma unroll
  for (int i = 0; i < 6; i++) sq[i * HD + d] = qr[i];
  const size_t head_base = (((size_t)a.slot * NATTN + L.al) * NKV + hk) * (size_t)a.ctx;
  if (last) {
    const __half kh16 = __float2half(kr), vh16 = __float2half(vv);
    a.kc[(head_base + a.pos) * HD + d] = kh16;
    a.vc[(head_base + a.pos) * HD + d] = vh16;
    skn[d] = __half2float(kh16); svn[d] = __half2float(vh16);
  }
  csync();
  // scores: warp per key, lane over 8 dims
  float qreg[6][8];
#pragma unroll
  for (int i = 0; i < 6; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) qreg[i][j] = sq[i * HD + lane * 8 + j];
  for (int p = p0 + warp; p < p1; p += NCW) {
    float kk[8];
    if (p == a.pos) {
#pragma unroll
      for (int j = 0; j < 8; j++) kk[j] = skn[lane * 8 + j];
    } else {
      const uint4 raw = __ldcg((const uint4*)(a.kc + (head_base + p) * HD + lane * 8));
      const __half2* hp = (const __half2*)&raw;
#pragma unroll
      for (int j = 0; j < 4; j++) { const float2 f = __half22float2(hp[j]); kk[2 * j] = f.x; kk[2 * j + 1] = f.y; }
    }
#pragma unroll
    for (int i = 0; i < 6; i++) {
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++) s = fmaf(qreg[i][j], kk[j], s);
      s = warp_sum(s);
      if (lane == 0) ssc[i * Lp + (p - p0)] = s;
    }
  }
  csync();
  if (warp < 6) {
    float mx = -INFINITY;
    for (int p = lane; p < n; p += 32) mx = fmaxf(mx, ssc[warp * Lp + p]);
    mx = warp_max(mx);
    float l = 0.f;
    for (int p = lane; p < n; p += 32) { const float e = __expf(ssc[warp * Lp + p] - mx); ssc[warp * Lp + p] = e; l += e; }
    l = warp_sum(l);
    if (lane == 0) { red[warp * 8 + 0] = mx; red[warp * 8 + 1] = l; }
  }
  csync();
  // weighted values: thread d owns dimension d for all six heads; four keys' values in flight
  float o[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  {
    int p = p0;
    for (; p + 4 <= p1; p += 4) {
      float vd[4];
#pragma unroll
      for (int q = 0; q < 4; q++) vd[q] = p + q == a.pos ? svn[d] : __half2float(a.vc[(head_base + p + q) * HD + d]);
#pragma unroll
      for (int q = 0; q < 4; q++)
#pragma unroll
        for (int i = 0; i < 6; i++) o[i] = fmaf(ssc[i * Lp + (p + q - p0)], vd[q], o[i]);
    }
    for (; p < p1; p++) {
      const float vd = p == a.pos ? svn[d] : __half2float(a.vc[(head_base + p) * HD + d]);
#pragma unroll
      for (int i = 0; i < 6; i++) o[i] = fmaf(ssc[i * Lp + (p - p0)], vd, o[i]);
    }
  }
  if (S == 1) {   // one split: normalize, gate, encode directly (no partials, no merge round trip)
#pragma unroll
    for (int i = 0; i < 6; i++) {
      const int hq = hk * 6 + i;
      const float gate = __ldcg(a.att + (size_t)hq * 2 * HD + HD + d);
      const float yv = o[i] / red[i * 8 + 1] * bm_sigm(gate);
      a.y[hq * HD + d] = yv;
      sq[i * HD + d] = yv;
    }
    csync();
    for (int bl = warp; bl < 12; bl += NCW) {
      const float* src = sq + bl * 128;
      float mx = fmaxf(fabsf(src[lane]), fmaxf(fabsf(src[lane + 32]), fmaxf(fabsf(src[lane + 64]), fabsf(src[lane + 96]))));
      mx = warp_max(mx);
      const int blk = hk * 12 + bl;
      if (lane < 16) {
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = src[lane * 8 + j];
        encode_octet(v, mx > 0.f ? 127.f / mx : 0.f, enc, ab_frag(nb), blk * 16 + lane);
      }
      publish_block(nb, blk, mx / 127.f, gout, lane);
    }
    return;
  }
  float* part = a.part + ((size_t)(hk * MAX_SPLITS + sp) * 6) * PART_STRIDE;
#pragma unroll
  for (int i = 0; i < 6; i++) part[i * PART_STRIDE + d] = o[i];
  if (d < 6) {
    part[d * PART_STRIDE + HD] = n > 0 ? red[d * 8 + 0] : -INFINITY;
    part[d * PART_STRIDE + HD + 1] = n > 0 ? red[d * 8 + 1] : 0.f;
  }
  __threadfence();
  csync();
  if (d == 0) {
    const unsigned old = atomicAdd(a.att_cnt + hk, 1u);
    *flag = old == (unsigned)(S - 1);
  }
  csync();
  if (!*flag) return;
  __threadfence();
  // merge: per-(head, split) weights exp(m - M) and the normalizer staged in shared memory once,
  // then every thread streams its dimension of all partials with many loads in flight
  float* sm_w = ssc;                       // [6][MAX_SPLITS]
  float* sm_l = ssc + 6 * MAX_SPLITS;      // [6]
  if (d < 6) {
    const float* pb = a.part + ((size_t)(hk * MAX_SPLITS) * 6 + d) * PART_STRIDE;
    float M = -INFINITY;
    for (int sp2 = 0; sp2 < S; sp2++) M = fmaxf(M, __ldcg(pb + (size_t)sp2 * 6 * PART_STRIDE + HD));
    float Lsum = 0.f;
    for (int sp2 = 0; sp2 < S; sp2++) {
      const float ms = __ldcg(pb + (size_t)sp2 * 6 * PART_STRIDE + HD);
      const float wgt = ms == -INFINITY ? 0.f : __expf(ms - M);
      sm_w[d * MAX_SPLITS + sp2] = wgt;
      Lsum += __ldcg(pb + (size_t)sp2 * 6 * PART_STRIDE + HD + 1) * wgt;
    }
    sm_l[d] = Lsum;
  }
  csync();
  float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  {
    const float* pb = a.part + ((size_t)(hk * MAX_SPLITS) * 6) * PART_STRIDE + d;
    int sp2 = 0;
    for (; sp2 + 2 <= S; sp2 += 2) {
      float o[2][6];
#pragma unroll
      for (int q = 0; q < 2; q++)
#pragma unroll
        for (int i = 0; i < 6; i++) o[q][i] = __ldcg(pb + ((size_t)(sp2 + q) * 6 + i) * PART_STRIDE);
#pragma unroll
      for (int q = 0; q < 2; q++)
#pragma unroll
        for (int i = 0; i < 6; i++) acc[i] = fmaf(o[q][i], sm_w[i * MAX_SPLITS + sp2 + q], acc[i]);
    }
    for (; sp2 < S; sp2++)
#pragma unroll
      for (int i = 0; i < 6; i++) acc[i] = fmaf(__ldcg(pb + ((size_t)sp2 * 6 + i) * PART_STRIDE), sm_w[i * MAX_SPLITS + sp2], acc[i]);
  }
#pragma unroll
  for (int i = 0; i < 6; i++) {
    const int hq = hk * 6 + i;
    const float gate = __ldcg(a.att + (size_t)hq * 2 * HD + HD + d);
    const float yv = acc[i] / sm_l[i] * bm_sigm(gate);
    a.y[hq * HD + d] = yv;
    sq[i * HD + d] = yv;
  }
  if (d == 0) a.att_cnt[hk] = 0;
  csync();
  // encode the 12 K128 blocks of these six heads for the output projection
  for (int bl = warp; bl < 12; bl += NCW) {
    const float* src = sq + bl * 128;   // head i = bl/2, half = bl%2 in [6][256]
    float mx = fmaxf(fabsf(src[lane]), fmaxf(fabsf(src[lane + 32]), fmaxf(fabsf(src[lane + 64]), fabsf(src[lane + 96]))));
    mx = warp_max(mx);
    const int blk = hk * 12 + bl;
    if (lane < 16) {
      float v[8];
#pragma unroll
      for (int j = 0; j < 8; j++) v[j] = src[lane * 8 + j];
      encode_octet(v, mx > 0.f ? 127.f / mx : 0.f, enc, ab_frag(nb), blk * 16 + lane);
    }
    publish_block(nb, blk, mx / 127.f, gout, lane);
  }
}

__device__ __forceinline__ bool mbar_test(uint64_t* b, unsigned parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void l2pf(const void* p, size_t bytes) {
  const uintptr_t b0 = (uintptr_t)p & ~(uintptr_t)15, b1 = ((uintptr_t)p + bytes + 15) & ~(uintptr_t)15;
  for (uintptr_t q = b0; q < b1; q += 65536) {
    const unsigned n = (unsigned)min((uintptr_t)65536, b1 - q);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(q), "r"(n) : "memory");
  }
}
// State this CTA touches in layer l, prefetched into L2 about a phase ahead of use.
__device__ __forceinline__ void prefetch_layer_state(const Args& a, int l, int S) {
  const LayerP L = a.L[l];
  const int c = blockIdx.x;
  // norm slices this CTA reads as a block encoder (rows 32c.. lie in block c/4)
  const int blk = c >> 2;
  l2pf(L.attn_norm + blk * 128, 512); l2pf(L.post_norm + blk * 128, 512);
  if (l + 1 < NLAYER) l2pf(a.L[l + 1].attn_norm + blk * 128, 512); else l2pf(a.out_norm + blk * 128, 512);
  if (!L.attn) {
    l2pf(L.ssm_norm, 512); l2pf(L.ssm_a, 192); l2pf(L.ssm_dt, 192);
    if (c < GDN_ITEMS) {
      const int h = c >> 1, half = c & 1;
      l2pf(a.gdn_state + (((size_t)a.slot * NGDN + L.gl) * HV + h) * 128 * 128 + (size_t)half * 64 * 128, 64 * 128 * 2);
    }
    const int4 e = a.tab[(l * 4) * NCTA + c];
    const int r0 = e.x * 16, r1 = min(CONV_CH, (e.x + e.y) * 16);
    if (r0 < r1) {
      const float* st = a.conv_state + ((size_t)a.slot * NGDN + L.gl) * 3 * CONV_CH;
      for (int k = 0; k < 3; k++) l2pf(st + (size_t)k * CONV_CH + r0, (size_t)(r1 - r0) * 4);
      l2pf(L.conv_w + (size_t)r0 * 4, (size_t)(r1 - r0) * 16);
    }
  } else if (c < 4 * S) {
    l2pf(L.q_norm, HD * 4); l2pf(L.k_norm, HD * 4);
    const int hk = c & 3, sp = c >> 2, P = a.pos + 1, Lp = (P + S - 1) / S, p0 = min(P, sp * Lp), p1 = min(P, p0 + Lp);
    if (p1 > p0) {
      const size_t base = ((((size_t)a.slot * NATTN + L.al) * NKV + hk) * (size_t)a.ctx + p0) * HD;
      l2pf(a.kc + base, (size_t)(p1 - p0) * HD * 2);
      l2pf(a.vc + base, (size_t)(p1 - p0) * HD * 2);
    }
  }
}

// One thread streams the CTA's whole-token chunk sequence: L2 prefetch LOOKAHEAD chunks ahead of the
// shared ring, so latency-bound phases never leave DRAM idle; the ring wait never blocks the prefetch.
constexpr int PF_GROUP = 4;
__device__ __forceinline__ void producer(const Args& a, uint8_t* ring, uint64_t* full, uint64_t* empty, int S) {
  const int lane = threadIdx.x & 31;
  const bool lane_pf = a.pf_policy == 2;   // per-lane prefetch.global.L2: the TMA queue carries only ring copies
  if (!lane_pf && lane != 0) return;
  const int c = blockIdx.x;
  const long long o0 = a.stream_off[c], o1 = a.stream_off[c + 1];
  const uint8_t* src = a.stream + (size_t)o0 * CHUNK;
  const int total = (int)(o1 - o0);
  uint64_t pol, pkeep;
  asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %0, 1.0;" : "=l"(pkeep));
  int i = 0, pf = 0, nl = 0, mark = a.tab[c].z;
  bool stalled = false;
  while (i < total) {
    // Shallow lookahead while the consumers drain the ring (current-phase copies keep DRAM priority);
    // deep lookahead only while they are stalled in latency-bound work and DRAM would otherwise idle.
    const int want = min(total, i + (stalled ? a.lookahead_hi : a.lookahead));
    while (a.nowait < 3 && pf < want) {
      while (nl < NLAYER && pf >= mark) {   // per-layer state, about one phase ahead of use
        if (lane == 0) prefetch_layer_state(a, nl, S);
        nl++;
        mark = nl < NLAYER ? a.tab[(nl * 4) * NCTA + c].z : 0x7fffffff;
      }
      const int n = min(PF_GROUP, total - pf);
      if (lane_pf) {
        const uint8_t* base = src + (size_t)pf * CHUNK;
        for (int q = lane; q < n * (CHUNK / 128); q += 32) asm volatile("prefetch.global.L2 [%0];" :: "l"(base + q * 128));
      } else if (a.pf_policy) asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(src + (size_t)pf * CHUNK), "r"(n * CHUNK), "l"(pkeep) : "memory");
      else asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + (size_t)pf * CHUNK), "r"(n * CHUNK) : "memory");
      pf += n;
      if (stalled) break;   // pace deep prefetch: one group per stall poll
    }
    // one TMA copy and one barrier per group of GROUP consecutive chunks (i is group-aligned)
    const int slot = i % NSLOT, grp = (i / GROUP) % NGRP, ng = min(GROUP, total - i);
    if (a.spin_mode == 0) {
      if (i >= NSLOT && lane == 0) mbar_wait(empty + grp, (unsigned)((i / NSLOT) - 1) & 1u);
      if (lane_pf) __syncwarp();
    } else {
      bool ready = true;
      if (i >= NSLOT) {
        if (lane == 0) ready = mbar_test(empty + grp, (unsigned)((i / NSLOT) - 1) & 1u);
        if (lane_pf) ready = __shfl_sync(0xffffffffu, ready, 0);
      }
      if (!ready) { stalled = true; if (a.spin_mode == 1 && pf >= min(total, i + a.lookahead_hi)) __nanosleep(32); continue; }
      stalled = false;
    }
    if (lane == 0) {
      if (a.nowait >= 3) mbar_arrive(full + grp);   // diagnostic: no data movement, consumers compute on stale ring contents
      else {
        mbar_expect(full + grp, ng * CHUNK);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                     :: "r"(smem_u32(ring + (size_t)slot * CHUNK)), "l"(src + (size_t)i * CHUNK), "r"(ng * CHUNK), "r"(smem_u32(full + grp)), "l"(pol) : "memory");
      }
    }
    i += ng;
  }
}

__global__ void __launch_bounds__(THREADS, 1) k_bmega(const Args a) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint8_t* ring = sm;
  uint64_t* full = (uint64_t*)(sm + NSLOT * CHUNK);
  uint64_t* empty = full + NSLOT;
  uint32_t* enc = (uint32_t*)(empty + NSLOT);
  float* red = (float*)(enc + 256);      // 64 floats
  float* xs = red + 64;                  // this CTA's 32 residual rows
  int* pflag = (int*)(xs + 32);          // 32 piece flags
  uint8_t* work = (uint8_t*)(pflag + 32);
  const int tid = threadIdx.x, warp = tid >> 5;
  if (tid == 0) {
    for (int s = 0; s < NGRP; s++) { mbar_init(full + s, 1); mbar_init(empty + s, GROUP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 256) enc[tid] = exact_codes(tid - 128);
  if (tid < 32) pflag[tid] = -1;
  if (tid == 0 && blockIdx.x == 0) { g_nowait = a.nowait; g_wacc = a.arrive; }
  __syncthreads();
  const int P = a.pos + 1;
  const int S = min(MAX_SPLITS, max(1, (P + a.attn_keys - 1) / a.attn_keys));
  if (warp == NCW) { producer(a, ring, full, empty, S); return; }
  if (warp == NCW + 1) { fetcher(a, (volatile int*)(pflag + 31), (volatile int*)pflag, red + 48, work); return; }
  if (a.prof && blockIdx.x == 0 && tid == 0) a.prof[0] = 0;
  stamp(a, 1);
  // embedding: this CTA's residual rows, published for layer 0's input projection
  const int c = blockIdx.x, n0 = 32 * c;
  if (tid < 32) {
    const int k = n0 + tid;
    const uint8_t* blk = a.emb + ((size_t)a.tok * (D / 128) + (k >> 7)) * 18;
    const float sc = __half2float(*(const __half*)blk);
    xs[tid] = ((blk[2 + ((k & 127) >> 3)] >> (k & 7)) & 1) ? sc : -sc;
  }
  csync();
  residual_publish(a, n0, xs, a.L[0].attn_norm, abuf(a, 0), gen_of(a, 0), enc, red);
  csync();
  stamp(a, 2);
  for (int l = 0; l < NLAYER; l++) {
    const LayerP L = a.L[l];
    const int g = l * 4;
    if (L.attn) gemv<K_IN_ATT>(a, L, g, D, ring, full, empty, work, xs, nullptr, enc, red, pflag);
    else gemv<K_IN_GDN>(a, L, g, D, ring, full, empty, work, xs, nullptr, enc, red, pflag);
    stamp(a, 3);
    if (a.nowait >= 2) {}
    else if (L.attn) { if (blockIdx.x < 4 * S) attn_item(a, L, g, work, S, abuf(a, g + 1), gen_of(a, g + 1), enc); }
    else if (blockIdx.x < GDN_ITEMS) gdn_item(a, L, g, work, abuf(a, g + 1), gen_of(a, g + 1), enc, red);
    csync();
    if (g_wacc && tid == 0) g_wacc[(size_t)blockIdx.x * 1024 + 780 + l] = gtime();
    stamp(a, 4);
    gemv<K_OUT>(a, L, g + 1, VDIM, ring, full, empty, work, xs, L.post_norm, enc, red, pflag);
    stamp(a, 5);
    gemv<K_GU>(a, L, g + 2, D, ring, full, empty, work, xs, nullptr, enc, red, pflag);
    stamp(a, 6);
    gemv<K_DOWN>(a, L, g + 3, FF, ring, full, empty, work, xs, l + 1 < NLAYER ? a.L[l + 1].attn_norm : a.out_norm, enc, red, pflag);
    stamp(a, 7);
  }
  gemv<K_HEAD>(a, a.L[0], NPH - 1, D, ring, full, empty, work, xs, nullptr, enc, red, pflag);
  stamp(a, 8);
  if (tid == 0) {
    __threadfence();
    const unsigned old = atom_add_acqrel(a.head_cnt, 1u);
    if ((old + 1u) % NCTA == 0u) {
      const unsigned long long k = ld_acq64(a.amax + (a.step & 1u));
      *a.out_tok = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
      a.amax[(a.step + 1u) & 1u] = 0ull;   // next step's slot; this step's slot was reset by the previous one
    }
  }
}

int smem_bytes() { return SMEM; }

void arrive_dump(const Model& m) {
  static long steps = 0;
  if (++steps != 100) return;
  std::vector<unsigned long long> h((size_t)NCTA * 1024);
  if (cudaMemcpy(h.data(), m.arrive, h.size() * 8, cudaMemcpyDeviceToHost) != cudaSuccess) return;
  FILE* f = fopen(getenv("H5_BMEGA_ARRIVE"), "wb");
  if (f) { fwrite(h.data(), 8, h.size(), f); fclose(f); fprintf(stderr, "bmega: arrival times written\n"); }
}
void prof_collect(const Model& m) {
  static std::vector<unsigned long long> h(4096);
  static double acc[64] = {}; static long steps = 0;
  if (cudaMemcpy(h.data(), m.prof, 4096 * 8, cudaMemcpyDeviceToHost) != cudaSuccess) return;
  const long n = std::min<long>((long)h[0], 4095);
  for (long i = 1; i < n; i++) acc[h[1 + i] & 63] += ((h[1 + i] >> 8) - (h[i] >> 8)) * 1e-3;
  if (++steps % 64) return;
  double total = 0; for (int i = 0; i < 64; i++) total += acc[i];
  fprintf(stderr, "bmega us/token (%ld steps, total %.1f): barrier waits in %.1f out %.1f gu %.1f down %.1f head %.1f | embed %.1f mix+barrier %.1f\n",
          steps, total / steps, acc[3] / steps, acc[5] / steps, acc[6] / steps, acc[7] / steps, acc[8] / steps, acc[2] / steps, acc[4] / steps);
  fprintf(stderr, "  fetch rms %.1f gated %.1f plain %.1f\n", acc[16] / steps, acc[17] / steps, acc[18] / steps);
  static const char* kind[6] = {"inG", "inA", "out", "gu", "down", "head"};
  for (int k = 0; k < 6; k++) fprintf(stderr, "  %-4s chunks %.1f epilogue %.1f\n", kind[k], acc[32 + k] / steps, acc[48 + k] / steps);
}

cudaError_t launch(const Args& a, cudaStream_t st) {
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(k_bmega, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (e != cudaSuccess) return e;
    init = true;
  }
  if (a.pos + 1 > MAX_SPLITS * 1200) return cudaErrorInvalidValue;   // score scratch bound
  void* args[] = {(void*)&a};
  return cudaLaunchCooperativeKernel((void*)k_bmega, dim3(NCTA), dim3(THREADS), args, SMEM, st);
}

// ------------------------------------------------------------------------------------------------
// Host: static chunk streams
// ------------------------------------------------------------------------------------------------
namespace {
struct Phase { const RowSource* src; int ntiles; int pieces; };

uint32_t g_spread[256];
void init_spread() {
  for (int b = 0; b < 256; b++) { uint32_t v = 0; for (int j = 0; j < 8; j++) if ((b >> j) & 1) v |= 1u << (4 * j); g_spread[b] = v; }
}
inline const uint8_t* row_ptr(const RowSource& s, int tile, int r) {
  const size_t row_bytes = (size_t)(s.K / 128) * 18;
  if (s.gu) return s.base[r < 8 ? 0 : 1] + (size_t)(tile * 8 + (r & 7)) * row_bytes;
  int n = tile * 16 + r;
  for (size_t i = 0; i < s.base.size(); i++) {
    if (n < s.rows[i]) return s.base[i] + (size_t)n * row_bytes;
    n -= s.rows[i];
  }
  throw std::runtime_error("bmega row out of range");
}
void fill_chunk(uint8_t* dst, const RowSource& s, int tile, int piece) {
  const uint8_t* rp[16];
  for (int r = 0; r < 16; r++) rp[r] = row_ptr(s, tile, r);
  uint32_t* words = (uint32_t*)dst;
  for (int st = 0; st < 16; st++) {                   // k64 steps of the 1024-K piece
    const int b = piece * 8 + st / 2;
    for (int lane = 0; lane < 32; lane++) {
      const int g = lane >> 2, t = lane & 3;
      uint32_t w = 0;
      for (int r = 0; r < 8; r++) {                   // INT8 A registers: MMA m=r/4, register q=r%4
        const int m = r >> 2, q = r & 3;
        const int row = g + 8 * (q & 1);
        for (int j = 0; j < 4; j++) {
          const int kk = (st & 1) * 64 + m * 32 + (q >> 1) * 16 + 4 * t + j;   // within the K128 block
          const unsigned bit = (rp[row][(size_t)b * 18 + 2 + kk / 8] >> (kk % 8)) & 1u;
          if (!bit) w |= 1u << (8 * j + r);           // set = negative weight
        }
      }
      words[st * 32 + lane] = w;
    }
  }
  uint32_t* sc = (uint32_t*)(dst + 2048);
  for (int bb = 0; bb < 8; bb++) {
    const size_t b = (size_t)piece * 8 + bb;
    for (int g = 0; g < 8; g++) {
      uint16_t lo, hi;
      memcpy(&lo, rp[g] + b * 18, 2); memcpy(&hi, rp[g + 8] + b * 18, 2);
      sc[bb * 8 + g] = (uint32_t)lo | ((uint32_t)hi << 16);
    }
  }
}
template <typename T> T* dupload(const std::vector<T>& v) {
  T* d = nullptr;
  if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess) throw std::runtime_error("bmega allocation");
  if (cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("bmega upload");
  return d;
}
}  // namespace

size_t build(Model& m, const std::vector<HostLayer>& layers, const RowSource& head, int nthreads) {
  if ((int)layers.size() != NLAYER) throw std::runtime_error("bmega requires 64 layers");
  init_spread();
  std::vector<Phase> ph;
  for (int l = 0; l < NLAYER; l++) {
    const HostLayer& y = layers[l];
    int in_rows = 0; for (int r : y.in.rows) in_rows += r;
    if (in_rows % 16 || y.in.K != D) throw std::runtime_error("bmega input projection geometry");
    ph.push_back({&y.in, in_rows / 16, D / 1024});
    if (y.out.K != ATTN_OUT || y.out.rows[0] != D) throw std::runtime_error("bmega output projection geometry");
    ph.push_back({&y.out, D / 16, ATTN_OUT / 1024});
    if (!y.gu.gu || y.gu.K != D || y.gu.rows[0] != FF) throw std::runtime_error("bmega gate/up geometry");
    ph.push_back({&y.gu, FF / 8, D / 1024});
    if (y.down.K != FF || y.down.rows[0] != D) throw std::runtime_error("bmega down geometry");
    ph.push_back({&y.down, D / 16, FF / 1024});
  }
  ph.push_back({&head, VOCAB / 16, D / 1024});
  if ((int)ph.size() != NPH) throw std::runtime_error("bmega phase count");
  std::vector<int4> tab((size_t)NPH * NCTA);
  std::vector<long long> cnt(NCTA, 0);
  int rot = 0;
  for (int p = 0; p < NPH; p++) {
    const int nt = ph[p].ntiles, base = nt / NCTA, extra = nt % NCTA;
    int t0 = 0;
    for (int c = 0; c < NCTA; c++) {
      const int n = base + (((c - rot + NCTA) % NCTA) < extra ? 1 : 0);
      tab[(size_t)p * NCTA + c] = make_int4(t0, n, (int)cnt[c], 0);
      cnt[c] += (long long)n * ph[p].pieces; t0 += n;
    }
    if (t0 != nt) throw std::runtime_error("bmega tile partition");
    rot = (rot + extra) % NCTA;
  }
  std::vector<long long> off(NCTA + 1, 0);
  for (int c = 0; c < NCTA; c++) off[c + 1] = off[c] + cnt[c];
  const size_t bytes = (size_t)off[NCTA] * CHUNK;
  if (cudaMalloc(&m.stream, bytes) != cudaSuccess) throw std::runtime_error("bmega stream allocation");
  m.stream_bytes = bytes;
  std::atomic<int> next{0}; std::atomic<bool> failed{false};
  auto worker = [&]() {
    std::vector<uint8_t> buf;
    for (int c = next++; c < NCTA && !failed; c = next++) {
      buf.resize((size_t)cnt[c] * CHUNK);
      for (int p = 0; p < NPH; p++) {
        const int4 e = tab[(size_t)p * NCTA + c];
        for (int pc = 0; pc < ph[p].pieces; pc++)
          for (int tl = 0; tl < e.y; tl++)
            fill_chunk(buf.data() + ((size_t)e.z + (size_t)pc * e.y + tl) * CHUNK, *ph[p].src, e.x + tl, pc);
      }
      if (cudaMemcpy(m.stream + (size_t)off[c] * CHUNK, buf.data(), buf.size(), cudaMemcpyHostToDevice) != cudaSuccess) failed = true;
    }
  };
  std::vector<std::thread> pool;
  for (int i = 0; i < std::max(1, nthreads); i++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (failed) throw std::runtime_error("bmega stream upload");
  m.tab = dupload(tab);
  m.stream_off = dupload(off);
  auto falloc = [](size_t n) { float* p = nullptr; if (cudaMalloc(&p, n * 4) != cudaSuccess) throw std::runtime_error("bmega scratch"); cudaMemset(p, 0, n * 4); return p; };
  m.x = falloc(D); m.conv_out = falloc(CONV_CH); m.zbuf = falloc(VDIM); m.ab = falloc(2 * HV); m.att = falloc(ATT_IN);
  m.y = falloc(VDIM); m.h = falloc(FF); m.part = falloc((size_t)NKV * MAX_SPLITS * 6 * PART_STRIDE);
  if (cudaMalloc(&m.actbuf, 3 * AB_BYTES) != cudaSuccess || cudaMemset(m.actbuf, 0, 3 * AB_BYTES) != cudaSuccess) throw std::runtime_error("bmega activation buffers");
  m.ubuf = falloc(VDIM);
  if (cudaMalloc(&m.indone, NCTA * 8) != cudaSuccess || cudaMemset(m.indone, 0, NCTA * 8) != cudaSuccess) throw std::runtime_error("bmega flags");
  if (cudaMalloc(&m.head_cnt, 4) != cudaSuccess || cudaMemset(m.head_cnt, 0, 4) != cudaSuccess) throw std::runtime_error("bmega head counter");
  if (cudaMalloc(&m.att_cnt, NKV * 4) != cudaSuccess || cudaMemset(m.att_cnt, 0, NKV * 4) != cudaSuccess) throw std::runtime_error("bmega counters");
  if (cudaMalloc(&m.bar, 8) != cudaSuccess || cudaMemset(m.bar, 0, 8) != cudaSuccess) throw std::runtime_error("bmega barrier");
  if (cudaMalloc(&m.amax, 16) != cudaSuccess || cudaMemset(m.amax, 0, 16) != cudaSuccess || cudaMalloc(&m.out_tok, 4) != cudaSuccess) throw std::runtime_error("bmega output");
  if (getenv("H5_BMEGA_ARRIVE")) { if (cudaMalloc(&m.arrive, (size_t)NCTA * 1024 * 8) != cudaSuccess) throw std::runtime_error("bmega arrive"); cudaMemset(m.arrive, 0, (size_t)NCTA * 1024 * 8); }
  if (getenv("H5_BMEGA_PROF")) { if (cudaMalloc(&m.prof, 4096 * 8) != cudaSuccess) throw std::runtime_error("bmega profile"); cudaMemset(m.prof, 0, 4096 * 8); }
  return bytes;
}
}  // namespace bmega
