// Persistent work-grabbing weight GEMM for steps of at most eight rows (verifier, ordinary decode,
// drafter).
//
//   out[t][n] (op)= AG[t] * RS[n] * sum_k w[n][k] * a[t][k]
//
// The work of one projection is a list of items (tile, K chunk): a tile is FW*WG 32-feature groups, a
// chunk is `chunk` consecutive 128-K blocks. A launch has one CTA per resident slot; each CTA's producer
// lane takes items from the launch site's atomic counter (the last failing take rearms it to zero), so
// a CTA that starts late because its SM still runs the predecessor simply takes fewer items. A tile's
// K chunks meet either by atomics straight into the output (ATOMIC: output precleared or a residual
// sum) or through a self-cleaning workspace whose last-arriving piece writes the epilogue (STORE, SILU).
// Within a CTA the producer streams each item's weights and the K block's activations through a ring of
// mbarrier-guarded stages, tagging every stage with its (segment, tile, kb); consumers accumulate runs
// of consecutive K blocks of one tile and flush them at a tag change, so the weight stream never waits
// for an epilogue.
//
// Weights: WF 3 dense5 labels (832 B per 32x128 chunk) with the row-mean scale carrier (RS);
//          WF 1 NVFP4 (2048 B chunk + 256 B UE4M3 scales) with an FP32 row scale (RS).
// Activations: TOKW = 8 rows of the [tile64][kb][64 rows][64 B] NVFP4 operand (row offset 0).
#define H5_PDLCLOCK_MODULE sk
#define H5_LIT_MODULE sk
#define H5_LIT_MODULE_ID 7u
#include "kernels.cuh"
#include "compact.cuh"
#include "fp4_ops.cuh"
#include "sk.h"
#include <algorithm>
#include <cstdio>

namespace h5 {
namespace sk {

template <int WF> struct Fmt { static constexpr int CW = WF == 1 ? 2048 : 832, CS = WF == 1 ? 256 : 0; };
// Kernel variants (measurement): 0 normal, 1 copy only (no decode/MMA), 2 compute only (no weight or
// activation traffic: consumers decode stale shared memory), 3 normal with lane-private dense5 codebook
// copies (32 KB, conflict-free lookups), 4 compute only with lane-private codebooks.
enum Var : int { V_NORMAL = 0, V_COPY = 1, V_COMPUTE = 2, V_BOOK32 = 3, V_BOOK32_COMPUTE = 4 };
template <int VAR> constexpr bool book32() { return VAR == V_BOOK32 || VAR == V_BOOK32_COMPUTE; }
template <int VAR> constexpr bool compute_only() { return VAR == V_COMPUTE || VAR == V_BOOK32_COMPUTE; }
template <int FW, int WG, int WF, int STAGES, int VAR = V_NORMAL> struct Geom {
  static constexpr int NG = FW * WG, SW = NG * Fmt<WF>::CW, SWS = NG * Fmt<WF>::CS, SA = 8 * 64, SAS = 8 * 8;
  static constexpr int NG2 = (SW + SWS) / 2304 < 1 ? 1 : (SW + SWS) / 2304 > NG ? NG : (SW + SWS) / 2304;   // NVFP4 groups per segment-2 tile
  static constexpr int SW2 = NG2 * 2048, SWB = SW + SWS > NG2 * 2304 ? SW + SWS : NG2 * 2304;   // weight bytes of a stage
  static constexpr int STAGE = SWB + SA + SAS;
  static constexpr int BOOK = WF == 3 ? (book32<VAR>() ? 32768 : 1024) : 0;
  static constexpr int SMEM = STAGES * STAGE + BOOK + 2 * STAGES * 8 + STAGES * 16;
};


// One warp's 128-K block of WG groups starting at local group gw: decode the weight operand and accumulate.
template <bool B32>
__device__ __forceinline__ void dense5_regs(const uint8_t* chunk, int lane, const uint32_t* book, unsigned* regs) {
  if constexpr (!B32) { dense5_registers(chunk, lane, book, regs); return; }
  uint32_t labels[7];
#pragma unroll
  for (int p = 0; p < 6; p++) labels[p] = *(const uint32_t*)(chunk + p * 128 + lane * 4);
  labels[6] = *(const uint16_t*)(chunk + 768 + lane * 2);
  uint32_t e[26];
#pragma unroll
  for (int b = 0; b < 26; b++) e[b] = book[(((labels[b / 4] >> (8 * (b % 4))) & 255) << 5) | lane];
#pragma unroll
  for (int w = 0; w < 16; w++) {
    const int b = (8 * w) / 5, sh = ((8 * w) % 5) * 4;
    uint32_t v = (e[b] >> sh) | (e[b + 1] << (20 - sh));
    if (sh >= 12) v |= e[b + 2] << (40 - sh);
    regs[w] = v;
  }
}
template <int WF, int WG, int MT, bool B32 = false>
__device__ __forceinline__ void consume(const uint8_t* st, const uint8_t* as, int CSoff, int gw, int ng, const uint32_t* book, float (&acc)[MT][4]) {
  constexpr int CW = Fmt<WF>::CW, CS = Fmt<WF>::CS;
  const int lane = threadIdx.x & 31, gid = lane >> 2, tq = lane & 3;
  const int row = lane & 7, c = lane >> 3;
  unsigned b0, b1, b2, b3;
  ldsm4(b0, b1, b2, b3, as + row * 64 + ((c ^ ((row >> 1) & 3)) << 4));
  const uint2 bs = *(const uint2*)(as + 8 * 64 + gid * 8);
#pragma unroll
  for (int q = 0; q < WG; q++) {
    if (WF == 1 && gw + q >= ng) break;
    unsigned a[2][2][4], sca[2][2];
    if constexpr (WF == 3) {
      unsigned native[16];
      dense5_regs<B32>(st + (gw + q) * CW, lane, book, native);
#pragma unroll
      for (int mm = 0; mm < 2; mm++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int w = mm * 4 + h * 2;
          unsigned* o = a[mm][h];
          o[0] = native[w * 2]; o[2] = native[w * 2 + 1]; o[1] = native[(w + 1) * 2]; o[3] = native[(w + 1) * 2 + 1];
        }
    } else {
      const uint8_t* wc = st + (gw + q) * CW + lane * 16;
#pragma unroll
      for (int g4 = 0; g4 < 4; g4++) {
        const uint4 wv = *(const uint4*)(wc + g4 * 512);
        unsigned* o = a[g4 >> 1][g4 & 1];
        o[0] = wv.x; o[1] = wv.y; o[2] = wv.z; o[3] = wv.w;
      }
      const uint8_t* sc = st + CSoff + (gw + q) * CS;
#pragma unroll
      for (int mm = 0; mm < 2; mm++) {
        const uint2 v = *(const uint2*)(sc + (16 * mm + gid + 8 * (tq & 1)) * 8);
        sca[mm][0] = v.x; sca[mm][1] = v.y;
      }
    }
#pragma unroll
    for (int mm = 0; mm < 2; mm++) {
      float* d = acc[q * 2 + mm];
      mma_fp4(d, a[mm][0], b0, b1, bs.x, d, WF == 3 ? 0x38383838u : sca[mm][0]);
      mma_fp4(d, a[mm][1], b2, b3, bs.y, d, WF == 3 ? 0x38383838u : sca[mm][1]);
    }
  }
}
// acc[m][r*2+j] holds feature g*32 + 16(m&1) + 8r + gid of group g = gw + (m>>1), row 2tq + j.
template <int MT>
__device__ __forceinline__ void epilogue_atomic(const float (&acc)[MT][4], float* out, int ldo, const float* RS, const float* AG, int R, int gbase, int ngroups) {
  const int lane = threadIdx.x & 31, gid = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int r_ = tq * 2 + j;
    if (r_ >= R) continue;
    const float gt = activation_scale(AG + r_);
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if ((m >> 1) >= ngroups) continue;
#pragma unroll
      for (int r = 0; r < 2; r++) {
        const int f = (gbase + (m >> 1)) * 32 + (m & 1) * 16 + r * 8 + gid;
        atomicAdd(out + (size_t)r_ * ldo + f, acc[m][r * 2 + j] * gt * RS[f]);
      }
    }
  }
}

template <int FW, int WG, int WF, int STAGES, int MODE, int VAR, bool SEG2>
__global__ void __launch_bounds__((FW + 1) * 32) k_sk(SkArgs p) {
  H5_LIT((uint32_t)p.N * (uint32_t)(p.K / 128));
  using G = Geom<FW, WG, WF, STAGES, VAR>;
  constexpr bool COPY = VAR == V_COPY, NOLOAD = compute_only<VAR>(), B32 = book32<VAR>();
  constexpr int CW = Fmt<WF>::CW, CS = Fmt<WF>::CS, MT = 2 * WG;
  static_assert(MODE != SK_SILU || WG == 2, "SiLU tiles pair gate and up groups inside one warp");
  static_assert(!SEG2 || MODE == SK_ATOMIC, "a second segment accumulates atomically like the first");
  extern __shared__ __align__(128) uint8_t smem[];
  uint32_t* book = (uint32_t*)(smem + STAGES * G::STAGE);
  uint64_t* full = (uint64_t*)(smem + STAGES * G::STAGE + G::BOOK);
  uint64_t* empty = full + STAGES;
  int4* desc = (int4*)(empty + STAGES);   // per stage: {segment, tile, kb, groups}; tile < 0 = no more work
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nkb = p.K / 128, NGall = p.N / 32, NG2all = SEG2 ? p.N2 / 32 : 0;
  const int KC = p.chunk, ipt = (nkb + KC - 1) / KC;
  const int T1 = (NGall + G::NG - 1) / G::NG, T2 = SEG2 ? (NG2all + G::NG2 - 1) / G::NG2 : 0;
  const int items = (T1 + T2) * ipt;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], FW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if constexpr (WF == 3)
    for (int e = tid; e < (B32 ? 256 * 32 : 256); e += blockDim.x) {
      unsigned q = B32 ? e >> 5 : e, v = 0;
#pragma unroll
      for (int j = 0; j < 5; j++, q /= 3) v |= (q % 3 == 0 ? 10u : q % 3 == 2 ? 2u : 0u) << (4 * j);
      book[e] = v;
    }
  __syncthreads();

  if (warp == FW) {
    if (lane == 0) {
      const uint64_t policy = weight_tma_policy();
      int i = 0;   // ring position
      bool waited = false;
      auto stage = [&](int seg, int t, int kb, bool acts) {
        const int s = i % STAGES;
        if (i >= STAGES) mbar_wait(&empty[s], ((i / STAGES) & 1) ^ 1);
        uint8_t* st = smem + s * G::STAGE;
        if (t < 0) { desc[s] = make_int4(0, -1, 0, 0); mbar_arrive(&full[s]); i++; return; }
        const int g0 = t * (seg ? G::NG2 : G::NG), ng = seg ? min(G::NG2, NG2all - g0) : min(G::NG, NGall - g0);
        desc[s] = make_int4(seg, t, kb, ng);
        if (NOLOAD) { mbar_arrive(&full[s]); i++; return; }
        if (SEG2 && seg) {
          mbar_expect(&full[s], ng * 2304 + G::SA + G::SAS);
          bulk_weight_g2s(st, p.W2 + ((size_t)kb * NG2all + g0) * 2048, ng * 2048, &full[s], policy);
          bulk_g2s(st + G::SW2, p.S2 + ((size_t)kb * NG2all + g0) * 256, ng * 256, &full[s]);
        } else {
          mbar_expect(&full[s], ng * (CW + CS) + G::SA + G::SAS);
          bulk_weight_g2s(st, p.W + ((size_t)kb * NGall + g0) * CW, ng * CW, &full[s], policy);
          if constexpr (CS > 0) bulk_g2s(st + G::SW, p.S + ((size_t)kb * NGall + g0) * CS, ng * CS, &full[s]);
        }
        if (acts) {
          bulk_g2s(st + G::SWB, p.A + (size_t)kb * 64 * 64, G::SA, &full[s]);
          bulk_g2s(st + G::SWB + G::SA, p.AS + (size_t)kb * 64 * 8, G::SAS, &full[s]);
        }
        i++;
      };
      // Weights of the first stages are issued before the predecessor wait; their activations after it.
      int pend_kb[STAGES], npend = 0;
      for (;;) {
        const int item = atomicAdd(p.next, 1);
        if (item >= items) {
          if (item == items + (int)gridDim.x - 1) atomicExch(p.next, 0);   // last grab of this launch: rearm
          break;
        }
        const int seg = item >= T1 * ipt, t = (seg ? item - T1 * ipt : item) / ipt, kb0 = (item % ipt) * KC, kb1 = min(nkb, kb0 + KC);
        for (int kb = kb0; kb < kb1; kb++) {
          if (!waited && (i >= min(STAGES, p.prewait) || NOLOAD)) {
            pdl_wait(); pdl_launch(); waited = true;
            for (int j = 0; j < npend; j++) {
              uint8_t* st = smem + j * G::STAGE;
              bulk_g2s(st + G::SWB, p.A + (size_t)pend_kb[j] * 64 * 64, G::SA, &full[j]);
              bulk_g2s(st + G::SWB + G::SA, p.AS + (size_t)pend_kb[j] * 64 * 8, G::SAS, &full[j]);
            }
          }
          if (!waited) pend_kb[npend++] = kb;
          stage(seg, t, kb, waited);
        }
      }
      if (!waited) {
        pdl_wait(); pdl_launch();
        for (int j = 0; j < npend; j++) {
          uint8_t* st = smem + j * G::STAGE;
          bulk_g2s(st + G::SWB, p.A + (size_t)pend_kb[j] * 64 * 64, G::SA, &full[j]);
          bulk_g2s(st + G::SWB + G::SA, p.AS + (size_t)pend_kb[j] * 64 * 8, G::SAS, &full[j]);
        }
      }
      stage(0, -1, 0, false);
    }
    return;
  }

  pdl_wait();
  pdl_launch();
  const int gw = warp * WG;
  int i = 0;
  float acc[MT][4];
  int cseg = 0, ct = -1, ckb0 = 0, cnext = -1, cng = 0;   // current piece: segment, tile, first kb, next expected kb, groups
  auto flush = [&]() {
    if (ct < 0 || gw >= cng) return;
    const int nk = cnext - ckb0;
    const int g0 = ct * (cseg ? G::NG2 : G::NG);
    if (SEG2 && cseg) { epilogue_atomic<MT>(acc, p.out2, p.ldo2, p.RS2, p.AG, p.R, g0 + gw, min(WG, cng - gw)); return; }
    if constexpr (MODE == SK_ATOMIC) { epilogue_atomic<MT>(acc, p.out, p.ldo, p.RS, p.AG, p.R, g0 + gw, min(WG, cng - gw)); return; }
    if (nk < nkb) {
      float* ws = p.ws + ((size_t)ct * FW + warp) * (MT * 4 * 32) + lane;
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int j = 0; j < 4; j++) atomicAdd(ws + (m * 4 + j) * 32, acc[m][j]);
      __threadfence();
      __syncwarp();
      int last = 0;
      if (lane == 0) last = atomicAdd(p.cnt + (size_t)ct * FW + warp, nk) + nk == nkb;
      last = __shfl_sync(0xffffffffu, last, 0);
      if (!last) return;
      __threadfence();
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int j = 0; j < 4; j++) { acc[m][j] = __ldcg(ws + (m * 4 + j) * 32); __stcg(ws + (m * 4 + j) * 32, 0.f); }
      if (lane == 0) p.cnt[(size_t)ct * FW + warp] = 0;
    }
    const int gid = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int r_ = tq * 2 + j;
      if (r_ >= p.R) continue;
      const float gt = activation_scale(p.AG + r_);
      if constexpr (MODE == SK_SILU) {
        const int fg = (g0 + gw) >> 1;
        float* o = p.out + (size_t)r_ * p.ldo + fg * 32 + gid;
#pragma unroll
        for (int mm = 0; mm < 2; mm++)
#pragma unroll
          for (int r = 0; r < 2; r++) {
            const float gv = acc[mm][r * 2 + j] * gt * p.RS[(g0 + gw) * 32 + mm * 16 + r * 8 + gid];
            const float uv = acc[2 + mm][r * 2 + j] * gt * p.RS[(g0 + gw + 1) * 32 + mm * 16 + r * 8 + gid];
            o[mm * 16 + r * 8] = gv / (1.f + __expf(-gv)) * uv;
          }
      } else {
        float* o = p.out + (size_t)r_ * p.ldo + (g0 + gw) * 32 + gid;
#pragma unroll
        for (int m = 0; m < MT; m++) {
          if ((m >> 1) >= cng - gw) continue;
#pragma unroll
          for (int r = 0; r < 2; r++) {
            const int f = m * 16 + r * 8;
            o[f] = acc[m][r * 2 + j] * gt * p.RS[(g0 + gw) * 32 + gid + f];
          }
        }
      }
    }
  };
  for (;; i++) {
    const int s = i % STAGES;
    mbar_wait(&full[s], (i / STAGES) & 1);
    const int4 d = desc[s];
    if (d.y < 0) break;
    if (d.x != cseg || d.y != ct || d.z != cnext) {
      flush();
      cseg = d.x; ct = d.y; ckb0 = d.z; cng = d.w;
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[m][j] = 0.f;
    }
    cnext = d.z + 1;
    const uint8_t* st = smem + s * G::STAGE;
    if constexpr (COPY) acc[0][0] += (float)st[lane];
    else if (SEG2 && d.x) { if (gw < d.w) consume<1, WG, MT>(st, st + G::SWB, G::SW2, gw, d.w, book, acc); }
    else if (gw < d.w) consume<WF, WG, MT, B32>(st, st + G::SWB, G::SW, gw, d.w, book, acc);
        ring_release(&empty[s], lane == 0);
  }
  flush();
}

template <int FW, int WG, int WF, int STAGES, int MODE, int VAR, bool SEG2>
static cudaError_t launch_one(const SkArgs& a, int grid, cudaStream_t st) {
  using G = Geom<FW, WG, WF, STAGES, VAR>;
  if constexpr (G::SMEM > 99 * 1024) return cudaErrorInvalidValue;
  auto f = k_sk<FW, WG, WF, STAGES, MODE, VAR, SEG2>;
  static bool attr = false;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    if (e != cudaSuccess) return e;
    attr = true;
  }
  pdl(f, dim3(grid), dim3((FW + 1) * 32), G::SMEM, st, a);
  return cudaGetLastError();
}
template <int FW, int WG, int WF, int STAGES, int MODE, int VAR>
static int occupancy() {
  using G = Geom<FW, WG, WF, STAGES, VAR>;
  if constexpr (G::SMEM > 99 * 1024) return 0;
  auto f = k_sk<FW, WG, WF, STAGES, MODE, VAR, false>;
  if (cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM) != cudaSuccess) return -1;
  int b = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, f, (FW + 1) * 32, G::SMEM) != cudaSuccess) return -1;
  return b;
}

#define SK_SHAPES(X) X(8, 1, 8) X(8, 1, 10) X(4, 1, 12) X(8, 1, 4) X(4, 1, 6) X(12, 1, 6) X(6, 2, 6) X(8, 2, 6) X(8, 2, 4) X(4, 2, 8) X(4, 2, 10)

}  // namespace sk

static const SkShape kSkShapes[] = {
#define S(fw, wg, stages) {fw, wg, stages},
  SK_SHAPES(S)
#undef S
};
int sk_chunk(int shape, int wf, int N, int K, int N2, int grid, int items_per_cta) {
  if (shape < 0 || shape >= sk_nshapes() || grid < 1 || items_per_cta < 1 || K < 128 || N < 32) return -1;
  const SkShape s = kSkShapes[shape];
  const int ng = s.fw * s.wg, nkb = K / 128;
  const int ng2 = std::max(1, std::min(ng, ng * (wf == 1 ? 2304 : 832) / 2304));
  const long long units = (long long)((N / 32 + ng - 1) / ng + (N2 > 0 ? (N2 / 32 + ng2 - 1) / ng2 : 0)) * nkb;
  const long long want = (long long)grid * items_per_cta;
  return (int)std::max(1LL, std::min((long long)nkb, (units + want - 1) / want));
}
int sk_nshapes() { return (int)(sizeof(kSkShapes) / sizeof(kSkShapes[0])); }
SkShape sk_shape(int i) { return kSkShapes[i]; }

template <int WF, int MODE, int COPY>
static cudaError_t sk_dispatch(int shape, const SkArgs& a, int grid, cudaStream_t st) {
  int i = 0;
#define S(fw, wg, stages) \
  if (shape == i++) { \
    if constexpr (MODE == SK_SILU && wg != 2) return cudaErrorInvalidValue; \
    else if (a.N2 > 0) { if constexpr (MODE == SK_ATOMIC && WF == 3 && (COPY == sk::V_NORMAL || COPY == sk::V_BOOK32)) return sk::launch_one<fw, wg, WF, stages, MODE, COPY, true>(a, grid, st); else return cudaErrorInvalidValue; } \
    else return sk::launch_one<fw, wg, WF, stages, MODE, COPY, false>(a, grid, st); }
  SK_SHAPES(S)
#undef S
  return cudaErrorInvalidValue;
}
template <int WF, int MODE, int COPY>
static int sk_occ(int shape) {
  int i = 0;
#define S(fw, wg, stages) \
  if (shape == i++) { \
    if constexpr (MODE == SK_SILU && wg != 2) return -1; \
    else return sk::occupancy<fw, wg, WF, stages, MODE, COPY>(); }
  SK_SHAPES(S)
#undef S
  return -1;
}

static bool sk_args_valid(const SkArgs& a, int shape, int wf, int mode) {
  if (shape < 0 || shape >= sk_nshapes() || (wf != 1 && wf != 3) || mode < SK_ATOMIC || mode > SK_SILU) return false;
  const SkShape s = kSkShapes[shape];
  if (a.R < 1 || a.R > 8 || a.K < 128 || a.K % 128 || a.N < 32 || a.N % 32 || !a.A || !a.AS || !a.AG || !a.W || !a.out || !a.RS) return false;
  if (wf == 1 && !a.S) return false;
  if (mode == SK_SILU && (s.wg != 2 || a.N % 64 || a.ldo < a.N / 2)) return false;
  if (mode != SK_SILU && a.ldo < a.N) return false;
  if (mode != SK_ATOMIC && (!a.ws || !a.cnt)) return false;
  if (!a.next || a.chunk < 1) return false;
  if (a.prewait < 0) return false;
  if (a.N2 && (a.N2 < 0 || a.N2 % 32 || !a.W2 || !a.S2 || !a.RS2 || !a.out2 || a.ldo2 < a.N2 || mode != SK_ATOMIC || wf != 3)) return false;
  return true;
}

cudaError_t sk_launch(int shape, int wf, int mode, int variant, const SkArgs& a, int grid, cudaStream_t st) {
  if (!sk_args_valid(a, shape, wf, mode) || grid < 1 || variant < 0 || variant > 4 || (wf != 3 && variant)) return cudaErrorInvalidValue;
#define D(W, M) \
  if (wf == W && mode == M) switch (variant) { \
    case 0: return sk_dispatch<W, M, 0>(shape, a, grid, st); \
    case 1: if constexpr (W == 3) return sk_dispatch<W, M, 1>(shape, a, grid, st); else return cudaErrorInvalidValue; \
    case 2: if constexpr (W == 3) return sk_dispatch<W, M, 2>(shape, a, grid, st); else return cudaErrorInvalidValue; \
    case 3: if constexpr (W == 3) return sk_dispatch<W, M, 3>(shape, a, grid, st); else return cudaErrorInvalidValue; \
    default: if constexpr (W == 3) return sk_dispatch<W, M, 4>(shape, a, grid, st); else return cudaErrorInvalidValue; }
  D(3, SK_ATOMIC) D(3, SK_STORE) D(3, SK_SILU) D(1, SK_ATOMIC) D(1, SK_STORE)
#undef D
  return cudaErrorInvalidValue;
}
int sk_occupancy(int shape, int wf, int mode, int variant) {
  if (shape < 0 || shape >= sk_nshapes() || variant < 0 || variant > 4 || (wf != 3 && variant)) return -1;
  const bool b32 = variant == 3 || variant == 4;
  if (wf == 3 && mode == SK_ATOMIC) return b32 ? sk_occ<3, SK_ATOMIC, 3>(shape) : sk_occ<3, SK_ATOMIC, 0>(shape);
  if (wf == 3 && mode == SK_STORE) return b32 ? sk_occ<3, SK_STORE, 3>(shape) : sk_occ<3, SK_STORE, 0>(shape);
  if (wf == 3 && mode == SK_SILU) return b32 ? sk_occ<3, SK_SILU, 3>(shape) : sk_occ<3, SK_SILU, 0>(shape);
  if (wf == 1 && mode == SK_ATOMIC) return sk_occ<1, SK_ATOMIC, 0>(shape);
  if (wf == 1 && mode == SK_STORE) return sk_occ<1, SK_STORE, 0>(shape);
  return -1;
}

}  // namespace h5

H5_LIT_EXPORT
