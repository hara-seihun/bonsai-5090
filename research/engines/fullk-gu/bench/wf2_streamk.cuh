#pragma once
#include "fp4_ops.cuh"

namespace h5::wf2_streamk {
constexpr int FW = 8, WG = 1, TOK = 128, NG = FW * WG, COLS = NG * 32;
constexpr int SW = NG * 1024, SWS = NG * 64, SA = TOK * 64, SAS = TOK * 8;
constexpr int STAGE = SW + SWS + SA + SAS;
#ifndef H5_STAGE_KB
#define H5_STAGE_KB 96
#endif
constexpr int STAGES = H5_STAGE_KB * 1024 / STAGE;
static_assert(STAGES > 0 && STAGES <= 8);
constexpr int SMEM = STAGES * STAGE + STAGES * 16;
constexpr int THREADS = (FW + 1) * 32;

struct Args {
  const uint8_t *A, *AS, *W, *S;
  const float *AG, *RS;
  float *out, *scratch;
  int R, K, N, ldo;
};

__host__ __device__ inline int64_t begin(int64_t total, int worker, int workers) {
  return total * worker / workers;
}

// A worker's contiguous interval may cross tile boundaries. Only its first tile can be a
// continuation: write that tile to its private scratch slot; every tile starting at K=0 owns out.
// A second kernel merges continuation heads after all owners have stored out, without a clear.
template <bool ATOMIC>
__global__ void __launch_bounds__(THREADS) gemm(Args p) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* full = (uint64_t*)(smem + STAGES * STAGE);
  uint64_t* empty = full + STAGES;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nkb = p.K / 128, nt = (p.R + TOK - 1) / TOK, nf = (p.N + COLS - 1) / COLS;
  const int64_t total = (int64_t)nt * nf * nkb;
  const int64_t first = begin(total, blockIdx.x, gridDim.x), last = begin(total, blockIdx.x + 1, gridDim.x);
  const int work = int(last - first), npre = min(work, STAGES);
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, FW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == FW) {
    if (lane == 0) {
      const uint64_t policy = weight_tma_policy();
      auto weights = [&](int i) {
        const int64_t q = first + i;
        const int tile = int(q / nkb), kb = int(q % nkb), g0 = (tile / nt) * NG;
        const int ng = min(NG, p.N / 32 - g0), s = i % STAGES;
        uint8_t* st = smem + s * STAGE;
        mbar_expect(full + s, ng * (1024 + 64) + SA + SAS);
        bulk_weight_g2s(st, p.W + ((size_t)kb * (p.N / 32) + g0) * 1024, ng * 1024, full + s, policy);
        bulk_g2s(st + SW, p.S + ((size_t)kb * (p.N / 32) + g0) * 64, ng * 64, full + s);
      };
      if (l2pf_enabled()) for (int i = STAGES; i < work; ++i) {
        const int64_t q = first + i;
        const int tile = int(q / nkb), kb = int(q % nkb), g0 = (tile / nt) * NG;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::
                     "l"(p.W + ((size_t)kb * (p.N / 32) + g0) * 1024), "r"(min(NG, p.N / 32 - g0) * 1024) : "memory");
      }
      for (int i = 0; i < npre; ++i) weights(i);
      pdl_wait();
#ifdef H5_PDL_EARLY_GEMM_RELEASE
      pdl_launch();
#endif
      for (int i = 0; i < work; ++i) {
        const int s = i % STAGES;
        if (i >= npre) {
          mbar_wait(empty + s, ((i / STAGES) & 1) ^ 1);
          weights(i);
        }
        const int64_t q = first + i;
        const int tile = int(q / nkb), kb = int(q % nkb), t0 = (tile % nt) * TOK;
        uint8_t* st = smem + s * STAGE;
#pragma unroll
        for (int u = 0; u < TOK / 64; ++u) {
          const size_t rb = ((size_t)(t0 / 64 + u) * nkb + kb) * 64;
          bulk_g2s(st + SW + SWS + u * 64 * 64, p.A + rb * 64, 64 * 64, full + s);
          bulk_g2s(st + SW + SWS + SA + u * 64 * 8, p.AS + rb * 8, 64 * 8, full + s);
        }
      }
    }
    return;
  }
#ifdef H5_PDL_EARLY_GEMM_RELEASE
  pdl_wait(); pdl_launch();
#endif
  const int gid = lane >> 2, tq = lane & 3;
  int i = 0;
  while (i < work) {
    const int64_t q = first + i;
    const int tile = int(q / nkb), kb0 = int(q % nkb), count = min(work - i, nkb - kb0);
    const int g0 = (tile / nt) * NG, t0 = (tile % nt) * TOK;
    float acc[2][TOK / 8][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < TOK / 8; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[m][n][r] = 0.f;
    for (int k = 0; k < count; ++k, ++i) {
      const int s = i % STAGES;
      mbar_wait(full + s, (i / STAGES) & 1);
      const uint8_t* st = smem + s * STAGE;
      const uint8_t* wc = st + warp * 1024 + lane * 16;
      const uint4 w0 = *(const uint4*)wc, w1 = *(const uint4*)(wc + 512);
      const unsigned wv[8] = {w0.x,w0.y,w0.z,w0.w,w1.x,w1.y,w1.z,w1.w};
      unsigned a[2][2][4], sc[2];
#pragma unroll
      for (int mm = 0; mm < 2; ++mm)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const unsigned r0 = wv[mm * 4 + h * 2], r1 = wv[mm * 4 + h * 2 + 1];
          a[mm][h][0] = r0 & 0xAAAAAAAAu; a[mm][h][2] = (r0 << 1) & 0xAAAAAAAAu;
          a[mm][h][1] = r1 & 0xAAAAAAAAu; a[mm][h][3] = (r1 << 1) & 0xAAAAAAAAu;
        }
      const unsigned sb4 = *(const unsigned*)(st + SW + warp * 64 + gid * 4);
#pragma unroll
      for (int mm = 0; mm < 2; ++mm) {
#if defined(H5_WF2_SCALE_PRMT) && H5_WF2_SCALE_PRMT
        sc[mm] = __byte_perm(sb4, 0, (tq & 1) * 0x1111u + mm * 0x2222u);
#else
        sc[mm] = ((sb4 >> (8 * (2 * mm + (tq & 1)))) & 255) * 0x01010101u;
#endif
      }
      const uint8_t* as = st + SW + SWS;
#pragma unroll
      for (int n = 0; n < TOK / 8; ++n) {
        const int row = n * 8 + (lane & 7), c = lane >> 3;
        unsigned b0,b1,b2,b3;
        ldsm4(b0,b1,b2,b3, as + row * 64 + ((c ^ ((row >> 1) & 3)) << 4));
        const uint2 bs = *(const uint2*)(as + SA + (n * 8 + gid) * 8);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          mma_fp4(acc[m][n], a[m][0], b0,b1,bs.x, acc[m][n], sc[m]);
          mma_fp4(acc[m][n], a[m][1], b2,b3,bs.y, acc[m][n], sc[m]);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + s);
    }
    if (g0 + warp >= p.N / 32) continue;
#pragma unroll
    for (int n = 0; n < TOK / 8; ++n)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int tr = n * 8 + tq * 2 + j, t = t0 + tr;
        if (t >= p.R) continue;
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const int f = (g0 + warp) * 32 + gid + m * 16 + r * 8;
            const float v = acc[m][n][r * 2 + j] * p.AG[t] * p.RS[f];
            if constexpr (ATOMIC) atomicAdd(p.out + (size_t)t * p.ldo + f, v);
            else if (kb0 == 0) p.out[(size_t)t * p.ldo + f] = v;
            else p.scratch[((size_t)blockIdx.x * TOK + tr) * COLS + f - g0 * 32] = v;
          }
      }
  }
#ifndef H5_PDL_EARLY_GEMM_RELEASE
  pdl_wait(); pdl_launch();
#endif
}

template <bool EXCLUSIVE>
__global__ void merge_heads(Args p, int workers) {
  const int nkb = p.K / 128, nt = (p.R + TOK - 1) / TOK, nf = (p.N + COLS - 1) / COLS;
  const int64_t q = begin((int64_t)nt * nf * nkb, blockIdx.x, workers);
  if (q % nkb == 0) return;
  const int tile = int(q / nkb), t0 = (tile % nt) * TOK, f0 = (tile / nt) * COLS;
  for (int j = threadIdx.x; j < TOK * COLS; j += blockDim.x) {
    const int t = t0 + j / COLS, f = f0 + j % COLS;
    if (t < p.R && f < p.N) {
      float* out = p.out + (size_t)t * p.ldo + f;
      const float v = p.scratch[(size_t)blockIdx.x * TOK * COLS + j];
      if constexpr (EXCLUSIVE) *out += v;
      else atomicAdd(out, v);
    }
  }
}
}
