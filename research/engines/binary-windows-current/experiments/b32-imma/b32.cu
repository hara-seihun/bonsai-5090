#include "b32.cuh"
#include <algorithm>
#include <cstdlib>

namespace b32 {
#ifdef B32_CLOCKS
__device__ unsigned long long b32_clock_stats[6];
__device__ unsigned long long b32_timeline[4096][4];
void timeline(unsigned long long* out, int n) { cudaMemcpyFromSymbol(out, b32_timeline, sizeof(unsigned long long) * 4 * n); }
void clock_stats(unsigned long long* out) { cudaMemcpyFromSymbol(out, b32_clock_stats, sizeof(b32_clock_stats)); unsigned long long z[6] = {}; cudaMemcpyToSymbol(b32_clock_stats, z, sizeof(z)); }
#endif
namespace {
__device__ __forceinline__ unsigned su32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(su32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, unsigned tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(su32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}"
               :: "r"(su32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(bar)) : "memory");
}
__device__ __forceinline__ void mma_seed(int* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1, const int4& c) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
}
__device__ __forceinline__ void mma_acc(int* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned rot2(unsigned w, unsigned s) { return __funnelshift_r(w, w, s) & 0x02020202u; }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_relaxed(unsigned* p, unsigned v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_release(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ void consumer_sync(int threads) { asm volatile("bar.sync 1, %0;" :: "r"(threads) : "memory"); }
// Owner epilogue for the fused GU: gate g[nt][j], up u[nt][j] of features f = 16*warp + (lane>>2) + 8*(j>>1),
// tokens 8*nt + 2*(lane&3) + (j&1). All CW consumer warps of the CTA call this together.
template <int NT, int CW>
__device__ __forceinline__ void silu_epilogue(const float (&gt)[NT][4], const float (&ut)[NT][4], unsigned char* es, int warp, int lane, int r,
                                              uint32_t* __restrict__ out, uint32_t* __restrict__ error) {
  const int g = lane >> 2, q = lane & 3;
  uint8_t* stage = es;                                                    // [4096] q bytes in carrier order
  float* tmax = reinterpret_cast<float*>(es + CARRIER_BLOCK);             // [CW][32]
  int* tsum = reinterpret_cast<int*>(es + CARRIER_BLOCK + CW * 32 * 4);    // [CW][32]
  float h[NT][4]; float m[NT][2]; int bad[NT][2];
#pragma unroll
  for (int b = 0; b < NT; ++b) {
#pragma unroll
    for (int j = 0; j < 4; ++j) h[b][j] = silu_f(gt[b][j]) * ut[b][j];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      bad[b][jj] = !(isfinite(h[b][jj]) && isfinite(h[b][jj + 2]));
      m[b][jj] = fmaxf(fabsf(h[b][jj]), fabsf(h[b][jj + 2]));
    }
  }
#pragma unroll
  for (int o = 4; o < 32; o <<= 1)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        bad[b][jj] |= __shfl_xor_sync(0xffffffffu, bad[b][jj], o);
        m[b][jj] = fmaxf(m[b][jj], __shfl_xor_sync(0xffffffffu, m[b][jj], o));
      }
  if (g == 0) {
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) tmax[warp * 32 + b * 8 + 2 * q + jj] = bad[b][jj] ? __int_as_float(0x7fc00000) : m[b][jj];
  }
  consumer_sync(CW * 32);
  float inv[NT][2], sig[NT][2]; unsigned st[NT][2];
#pragma unroll
  for (int b = 0; b < NT; ++b)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      float mx = 0.f; bool nan = false;
#pragma unroll
      for (int w = 0; w < CW; ++w) { const float v = tmax[w * 32 + b * 8 + 2 * q + jj]; nan |= isnan(v); mx = fmaxf(mx, v); }
      st[b][jj] = nan ? 1u : (mx != 0.f && (mx < 1e-30f || mx > 1e30f)) ? 2u : 0u;
      sig[b][jj] = st[b][jj] ? __int_as_float(0x7fc00000) : sigma8(mx);
      inv[b][jj] = (st[b][jj] || mx == 0.f) ? 0.f : 1.f / sig[b][jj];
    }
  int qs[NT][4]; int s[NT][2];
#pragma unroll
  for (int b = 0; b < NT; ++b) {
#pragma unroll
    for (int j = 0; j < 4; ++j) qs[b][j] = __float2int_rn(h[b][j] * inv[b][j & 1]);
    s[b][0] = qs[b][0] + qs[b][2]; s[b][1] = qs[b][1] + qs[b][3];
  }
#pragma unroll
  for (int o = 4; o < 32; o <<= 1)
#pragma unroll
    for (int b = 0; b < NT; ++b) { s[b][0] += __shfl_xor_sync(0xffffffffu, s[b][0], o); s[b][1] += __shfl_xor_sync(0xffffffffu, s[b][1], o); }
  if (g == 0) {
#pragma unroll
    for (int b = 0; b < NT; ++b) { tsum[warp * 32 + b * 8 + 2 * q] = s[b][0]; tsum[warp * 32 + b * 8 + 2 * q + 1] = s[b][1]; }
  }
  // q bytes into carrier order: feature f -> chunk c = f>>5, K' = f&31 = jj' + 8i -> word slot.
#pragma unroll
  for (int b = 0; b < NT; ++b)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int f = 16 * warp + g + 8 * (j >> 1), tok = b * 8 + 2 * q + (j & 1);
      const int c = f >> 5, kp = f & 31, jw = kp & 7, i = kp >> 3;
      const int word = ((b * 2 + (c >> 1)) * 32 + 4 * (tok & 7) + (jw & 3)) * 4 + (c & 1) * 2 + (jw >> 2);
      stage[word * 4 + i] = uint8_t(qs[b][j] & 255);
    }
  consumer_sync(CW * 32);
  // copy the q block (NT KB), sigma and seeded MAGIC-DC entries
  const int t = warp * 32 + lane;
  for (int i = t; i < NT * 256; i += CW * 32) out[i] = reinterpret_cast<const uint32_t*>(stage)[i];
  if (t < NT * 8) {
    const int tok = t;
    int dc = 0; bool nan = false;
#pragma unroll
    for (int w = 0; w < CW; ++w) { dc += tsum[w * 32 + tok]; nan |= isnan(tmax[w * 32 + tok]); }
    float mx = 0.f;
#pragma unroll
    for (int w = 0; w < CW; ++w) mx = fmaxf(mx, tmax[w * 32 + tok]);
    const unsigned stt = nan ? 1u : (mx != 0.f && (mx < 1e-30f || mx > 1e30f)) ? 2u : 0u;
    const float sg8 = stt ? __int_as_float(0x7fc00000) : sigma8(mx);
    reinterpret_cast<float*>(out + 1024)[tok] = sg8;
    reinterpret_cast<float*>(out + 1024 + 96)[tok] = -MAGIC_F * sg8;
    int* md = reinterpret_cast<int*>(out + 1024 + 32) + ((tok >> 3) * 4 + ((tok & 7) >> 1)) * 4;
    md[tok & 1] = MAGIC - dc; md[(tok & 1) + 2] = MAGIC - dc;
    if (stt && error && tok < r) atomicMax(error, stt);
  }
  (void)sig;
  consumer_sync(CW * 32);   // staging/tmax/tsum reuse by the next tile's epilogue
}

template <int NT, int MT, int CW, int STAGES, int KPS> struct Cfg {
  static constexpr int TN = CW * MT * 16;
  static constexpr int WREC = 18 * TN;                       // one tiled weight record (signs + scales)
  static constexpr int W_OFF = 0, W_BYTES = KPS * WREC;
  static constexpr int A_OFF = (W_BYTES + 127) & ~127, A_BYTES = KPS * CARRIER_BLOCK;
  static constexpr int STAGE = (A_OFF + A_BYTES + 127) & ~127;
  static constexpr int BAR_OFF = STAGE * STAGES;
  static constexpr int EPI_OFF = BAR_OFF + 2 * STAGES * 8;           // SiLU epilogue staging (EPI 2 only)
  static constexpr int EPI_BYTES = CARRIER_BLOCK + 2 * CW * 32 * 4;   // carrier block + per-warp token max/sum
  static constexpr int SMEM = BAR_OFF + 2 * STAGES * 8;
  static constexpr int SMEM_EPI = EPI_OFF + EPI_BYTES;
  static constexpr int THREADS = (CW + 1) * 32;
  static constexpr int PART = MT * NT * 128;   // floats per consumer warp
};

__device__ __forceinline__ long long range_start(int c, long long iters, int ctas) { return (long long)c * iters / ctas; }

// EPI 0: y[tok*ldy+row] (+=) ; EPI 1: rows >= split go to y2[tok*ldy2 + row-split] ;
// EPI 2: tile t = 128 gate rows interleaved with their 128 up rows (per warp: m-tile 0 gate, m-tile 1 up);
//        h = silu(gate)*up, quantized per (token, K128 = tile) straight into the next carrier block t.
struct Epilogue { float* y2; int ldy2, split; uint32_t* carrier; uint32_t* error; };
template <int NT, int MT, int CW, int STAGES, int KPS, int MINB, int EPI>
__global__ void __launch_bounds__((CW + 1) * 32, MINB)
gemm(const uint8_t* __restrict__ tiled, const uint8_t* __restrict__ act, float* __restrict__ y, int ldy, float beta,
     int n, int kbs, int r, float* __restrict__ part, unsigned* __restrict__ counts, Epilogue ep) {
  using C = Cfg<NT, MT, CW, STAGES, KPS>;
  extern __shared__ __align__(128) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
  uint64_t* empty = full + STAGES;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#ifdef B32_TIMELINE
  if (threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); ::b32::b32_timeline[blockIdx.x][0] = t; }
#endif
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 33); mbar_init(&empty[s], CW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int tiles = (n + C::TN - 1) / C::TN;
  const long long iters = (long long)tiles * kbs;
  const int cta = blockIdx.x, ctas = gridDim.x;
  const long long s0 = range_start(cta, iters, ctas), e0 = range_start(cta + 1, iters, ctas);
  int left = int(e0 - s0);
  const int tile0 = int(s0 / kbs), kb0 = int(s0 - (long long)tile0 * kbs);
  if (warp == CW) {
    // Producer warp. Weights: one TMA bulk record per stage (lane 0). Activations: LSU cp.async by all 32
    // lanes (the TMA unit serializes bulk operations per SM, so activations stay off it). The full barrier
    // counts 1 expect_tx arrive + 32 cp.async completion arrives.
    // Programmatic dependent launch: weights are immutable and stream before the predecessor completes.
    {
      int tile = tile0, kb = kb0, l = left;
      for (int i = 0; i < STAGES && l > 0; ++i) {
        const int nb = min(KPS, min(kbs - kb, l));
        if (lane == 0) {
          unsigned char* st = smem + i * C::STAGE;
          mbar_expect(&full[i], unsigned(nb * C::WREC));
          bulk(st + C::W_OFF, tiled + ((size_t)tile * kbs + kb) * C::WREC, unsigned(nb * C::WREC), &full[i]);
        }
        l -= nb; kb += nb; if (kb == kbs) { kb = 0; ++tile; }
      }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    auto act_copy = [&](int stage, int kb, int nb) {
      const uint4* src = reinterpret_cast<const uint4*>(act + (size_t)kb * CARRIER_BLOCK);
      const unsigned dst = su32(smem + stage * C::STAGE + C::A_OFF);
      const int chunks = nb * (CARRIER_BLOCK / 16);
      for (int c = lane; c < chunks; c += 32)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + c * 16), "l"(src + c) : "memory");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(su32(&full[stage])) : "memory");
    };
    {
      int kb = kb0, l = left;
      for (int i = 0; i < STAGES && l > 0; ++i) {
        const int nb = min(KPS, min(kbs - kb, l));
        act_copy(i, kb, nb);
        l -= nb; kb += nb; if (kb == kbs) kb = 0;
      }
    }
    if (lane == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int tile = tile0, kb = kb0;
    for (int i = 0; left > 0; ++i) {
      const int nb = min(KPS, min(kbs - kb, left));
      const int stage = i % STAGES;
      if (i < STAGES) { left -= nb; kb += nb; if (kb == kbs) { kb = 0; ++tile; } continue; }
      mbar_wait(&empty[stage], ((i / STAGES) & 1) ^ 1);
      if (lane == 0) {
        mbar_expect(&full[stage], unsigned(nb * C::WREC));
        bulk(smem + stage * C::STAGE + C::W_OFF, tiled + ((size_t)tile * kbs + kb) * C::WREC, unsigned(nb * C::WREC), &full[stage]);
      }
      act_copy(stage, kb, nb);
      left -= nb; kb += nb;
      if (kb == kbs) { kb = 0; ++tile; }
    }
    return;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int g = lane >> 2, q = lane & 3;
  const unsigned sh0 = unsigned(q + 31) & 31u, sh1 = unsigned(q + 3);
  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[a][b][j] = 0.f;
  int tile = tile0, kb = kb0, seg_kb = kb0;
#ifdef B32_CLOCKS
  long long wait_cycles = 0, compute_cycles = 0; const long long t_start = clock64();
#endif
  for (int i = 0; left > 0; ++i) {
    const int nb = min(KPS, min(kbs - kb, left));
    const int stage = i % STAGES;
#ifdef B32_CLOCKS
    const long long c0 = clock64();
#endif
    mbar_wait(&full[stage], (i / STAGES) & 1);
#ifdef B32_TIMELINE
    if (i == 0 && threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); ::b32::b32_timeline[blockIdx.x][2] = t; }
#endif
#ifdef B32_CLOCKS
    const long long c1 = clock64(); wait_cycles += c1 - c0;
#endif
    const unsigned char* st = smem + stage * C::STAGE;
#pragma unroll
    for (int blk = 0; blk < KPS; ++blk) {
      if (blk >= nb) break;
      const unsigned char* wb = st + C::W_OFF + blk * C::WREC;
      const unsigned char* ab = st + C::A_OFF + blk * CARRIER_BLOCK;
      const uint4* A4 = reinterpret_cast<const uint4*>(ab);
      uint4 bf[NT][2];
#pragma unroll
      for (int b = 0; b < NT; ++b) { bf[b][0] = A4[(b * 2) * 32 + lane]; bf[b][1] = A4[(b * 2 + 1) * 32 + lane]; }
      const float* qf = reinterpret_cast<const float*>(ab + 4096);
      const int4* qi = reinterpret_cast<const int4*>(ab + 4096 + 128);
      const float* qn = reinterpret_cast<const float*>(ab + 4096 + 384);
      float2 sx[NT], nm[NT]; int4 md[NT];
#pragma unroll
      for (int b = 0; b < NT; ++b) {
        sx[b] = *reinterpret_cast<const float2*>(qf + b * 8 + 2 * q); md[b] = qi[b * 4 + q];
        nm[b] = *reinterpret_cast<const float2*>(qn + b * 8 + 2 * q);
      }
      const uint4* sW = reinterpret_cast<const uint4*>(wb);
      const __half2* sS = reinterpret_cast<const __half2*>(wb + 16 * C::TN);
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int mt = warp * MT + m;
        const uint4 lo = sW[(mt * 8 + g) * 2], hi = sW[(mt * 8 + g) * 2 + 1];
        const unsigned wg[4] = {lo.x, lo.y, lo.z, lo.w}, wh[4] = {hi.x, hi.y, hi.z, hi.w};
        const float2 sgh = __half22float2(sS[mt * 8 + g]);
        const float sg = sgh.x, shh = sgh.y;
        int ci[NT][4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const unsigned a0 = rot2(wg[c], sh0), a2 = rot2(wg[c], sh1), a1 = rot2(wh[c], sh0), a3 = rot2(wh[c], sh1);
#pragma unroll
          for (int b = 0; b < NT; ++b) {
            const uint4 v = bf[b][c >> 1];
            const unsigned b0 = (c & 1) ? v.z : v.x, b1 = (c & 1) ? v.w : v.y;
            if (c == 0) mma_seed(ci[b], a0, a1, a2, a3, b0, b1, md[b]);
            else mma_acc(ci[b], a0, a1, a2, a3, b0, b1);
          }
        }
#pragma unroll
        for (int b = 0; b < NT; ++b) {
          acc[m][b][0] = fmaf(fmaf(__int_as_float(ci[b][0]), sx[b].x, nm[b].x), sg, acc[m][b][0]);
          acc[m][b][1] = fmaf(fmaf(__int_as_float(ci[b][1]), sx[b].y, nm[b].y), sg, acc[m][b][1]);
          acc[m][b][2] = fmaf(fmaf(__int_as_float(ci[b][2]), sx[b].x, nm[b].x), shh, acc[m][b][2]);
          acc[m][b][3] = fmaf(fmaf(__int_as_float(ci[b][3]), sx[b].y, nm[b].y), shh, acc[m][b][3]);
        }
      }
    }
    __syncwarp();
#ifdef B32_CLOCKS
    compute_cycles += clock64() - c1;
#endif
    if (lane == 0) mbar_arrive(&empty[stage]);
    left -= nb; kb += nb - 1;
    const bool tile_end = kb == kbs - 1, last = left == 0;
    if (!tile_end && !last) { ++kb; continue; }
    // Segment [seg_kb, kb] of `tile` ends here.
    const bool head = seg_kb == 0, whole = head && tile_end;
    if (!head) {
      float* mine = part + ((size_t)cta * CW + warp) * C::PART;
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int b = 0; b < NT; ++b)
          __stcg(reinterpret_cast<float4*>(mine) + (m * NT + b) * 32 + lane, make_float4(acc[m][b][0], acc[m][b][1], acc[m][b][2], acc[m][b][3]));
      __threadfence();
      __syncwarp();
      if (lane == 0) red_release(&counts[(size_t)tile * CW + warp], 1u);
    } else {
      if (!whole) {
        // CTA owning the tile's last iteration: largest c with start(c) <= (tile+1)*kbs - 1.
        const long long tile_end_it = (long long)(tile + 1) * kbs;
        const int last_c = int((tile_end_it * ctas - 1) / iters);
        const unsigned expected = unsigned(last_c - cta);
        if (lane == 0) {
          unsigned* f = &counts[(size_t)tile * CW + warp];
          while (ld_acquire(f) != expected) {}
          st_relaxed(f, 0u);
        }
        __syncwarp();
        // Fixed ascending order (deterministic); two contributors' loads in flight at once.
        int c2 = cta + 1;
        for (; c2 + 1 <= last_c; c2 += 2) {
          const float4* o1 = reinterpret_cast<const float4*>(part + ((size_t)c2 * CW + warp) * C::PART);
          const float4* o2 = reinterpret_cast<const float4*>(part + ((size_t)(c2 + 1) * CW + warp) * C::PART);
          float4 v1[MT][NT], v2[MT][NT];
#pragma unroll
          for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int b = 0; b < NT; ++b) { v1[m][b] = __ldcg(o1 + (m * NT + b) * 32 + lane); v2[m][b] = __ldcg(o2 + (m * NT + b) * 32 + lane); }
#pragma unroll
          for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int b = 0; b < NT; ++b) {
              acc[m][b][0] += v1[m][b].x; acc[m][b][1] += v1[m][b].y; acc[m][b][2] += v1[m][b].z; acc[m][b][3] += v1[m][b].w;
              acc[m][b][0] += v2[m][b].x; acc[m][b][1] += v2[m][b].y; acc[m][b][2] += v2[m][b].z; acc[m][b][3] += v2[m][b].w;
            }
        }
        if (c2 <= last_c) {
          const float4* other = reinterpret_cast<const float4*>(part + ((size_t)c2 * CW + warp) * C::PART);
          float4 v[MT][NT];
#pragma unroll
          for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int b = 0; b < NT; ++b) v[m][b] = __ldcg(other + (m * NT + b) * 32 + lane);
#pragma unroll
          for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int b = 0; b < NT; ++b) { acc[m][b][0] += v[m][b].x; acc[m][b][1] += v[m][b].y; acc[m][b][2] += v[m][b].z; acc[m][b][3] += v[m][b].w; }
        }
      }
      const int row0 = tile * C::TN;
      if constexpr (EPI == 2) {
        // SiLU epilogue pairs m-tile 0 (gate) with m-tile 1 (up); launch() rejects other MT.
        if constexpr (MT == 2) silu_epilogue<NT, CW>(acc[0], acc[1], smem + C::EPI_OFF, warp, lane, r, ep.carrier + (size_t)tile * (CARRIER_BLOCK / 4), ep.error);
      } else {
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int b = 0; b < NT; ++b)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int row = row0 + (warp * MT + m) * 16 + g + 8 * (j >> 1), tok = b * 8 + 2 * q + (j & 1);
            if (row < n && tok < r) {
              float* p;
              if (EPI == 1 && row >= ep.split) p = ep.y2 + (size_t)tok * ep.ldy2 + (row - ep.split);
              else p = y + (size_t)tok * ldy + row;
              *p = beta != 0.f ? *p + acc[m][b][j] : acc[m][b][j];
            }
          }
      }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int b = 0; b < NT; ++b)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[m][b][j] = 0.f;
    if (tile_end) { kb = 0; ++tile; } else ++kb;
    seg_kb = kb;
  }
#ifdef B32_TIMELINE
  if (threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); ::b32::b32_timeline[blockIdx.x][1] = t; }
#endif
#ifdef B32_CLOCKS
  if (lane == 0) {
    atomicAdd(&::b32::b32_clock_stats[0], (unsigned long long)wait_cycles);
    atomicAdd(&::b32::b32_clock_stats[1], (unsigned long long)compute_cycles);
    atomicAdd(&::b32::b32_clock_stats[2], (unsigned long long)(clock64() - t_start));
    atomicAdd(&::b32::b32_clock_stats[3], 1ull);
    atomicAdd(&::b32::b32_clock_stats[4], (unsigned long long)(e0 - s0));
  }
#endif
}

__global__ void tile_kernel(const uint32_t* __restrict__ signs, const __half* __restrict__ scales, int n, int kbs, int tn, uint8_t* __restrict__ out) {
  const size_t total = (size_t)((n + tn - 1) / tn) * kbs * tn;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int row_in = int(i % tn); const size_t rec = i / tn; const int kb = int(rec % kbs), tile = int(rec / kbs);
    const int row = tile * tn + row_in;
    uint8_t* base = out + rec * (size_t)(18 * tn);
    uint32_t* w = reinterpret_cast<uint32_t*>(base);
    __half* s = reinterpret_cast<__half*>(base + 16 * tn);
    const int m16 = row_in >> 4, gg = row_in & 7, hi = (row_in >> 3) & 1;
    for (int c = 0; c < 4; ++c) w[((m16 * 8 + gg) * 2 + hi) * 4 + c] = row < n ? signs[((size_t)kb * 4 + c) * n + row] : 0u;
    s[(m16 * 8 + gg) * 2 + hi] = row < n ? scales[(size_t)kb * n + row] : __float2half(0.f);
  }
}

// One warp per (token, K128 block). Lane l holds natural K = 4l..4l+3 of the block.
__global__ void producer(const float* __restrict__ x, int ldx, int k, int r, int rows_padded, uint32_t* __restrict__ out,
                         uint32_t* __restrict__ error) {
  const int lane = threadIdx.x & 31, t = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5), b = blockIdx.x;
  if (t >= rows_padded) return;
  unsigned packed = 0; int dc = 0; float sigma = 0.f;
  if (t < r) {
    const float4 v = *reinterpret_cast<const float4*>(x + (size_t)t * ldx + (size_t)b * 128 + 4 * lane);
    const bool bad = !(isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && isfinite(v.w));
    float a = bad ? 0.f : fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int d = 16; d; d >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, d));
    const unsigned anybad = __any_sync(0xffffffffu, bad);
    const unsigned status = anybad ? 1u : (a != 0.f && (a < 1e-30f || a > 1e30f)) ? 2u : 0u;
    const float sg8 = status ? 0.f : sigma8(a);
    const float inv = (status || a == 0.f) ? 0.f : 1.f / sg8;
    const int q0 = __float2int_rn(v.x * inv), q1 = __float2int_rn(v.y * inv), q2 = __float2int_rn(v.z * inv), q3 = __float2int_rn(v.w * inv);
    packed = (unsigned(q0) & 255u) | ((unsigned(q1) & 255u) << 8) | ((unsigned(q2) & 255u) << 16) | ((unsigned(q3) & 255u) << 24);
    dc = q0 + q1 + q2 + q3;
#pragma unroll
    for (int d = 16; d; d >>= 1) dc += __shfl_xor_sync(0xffffffffu, dc, d);
    sigma = status ? __int_as_float(0x7fc00000) : sg8;
    if (status && lane == 0 && error) atomicMax(error, status);
  }
  // Lane l = 8c + j emits transposed word j of chunk c: bytes i <- natural K' = j + 8i of the chunk.
  const int c = lane >> 3, j = lane & 7;
  unsigned word = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned v = __shfl_sync(0xffffffffu, packed, 8 * c + 2 * i + (j >> 2));
    word |= ((v >> (8 * (j & 3))) & 255u) << (8 * i);
  }
  const int nt = t >> 3, gg = t & 7, qq = j & 3, h = j >> 2, cp = c >> 1, slot = (c & 1) * 2 + h;
  uint32_t* blk = out + (size_t)b * (CARRIER_BLOCK / 4);
  blk[((nt * 2 + cp) * 32 + (4 * gg + qq)) * 4 + slot] = word;
  if (lane == 0) {
    reinterpret_cast<float*>(blk + 1024)[t] = sigma;
    reinterpret_cast<float*>(blk + 1024 + 96)[t] = -MAGIC_F * sigma;
    int* md = reinterpret_cast<int*>(blk + 1024 + 32) + (nt * 4 + (gg >> 1)) * 4;
    md[t & 1] = MAGIC - dc; md[(t & 1) + 2] = MAGIC - dc;
  }
}

static bool pdl_enabled() { static const int on = getenv("H5_PDL") ? atoi(getenv("H5_PDL")) : 1; return on != 0; }
struct Variant { const char* name; int nt, mt, cw, stages, kps; };
#define B32_VARIANTS(X) \
  X(0, 4, 2, 8, 6, 1, 1, "nt4-tn256-cw8-s6") \
  X(1, 3, 2, 8, 6, 1, 1, "nt3-tn256-cw8-s6") \
  X(2, 2, 2, 8, 6, 1, 1, "nt2-tn256-cw8-s6") \
  X(3, 1, 2, 8, 6, 1, 1, "nt1-tn256-cw8-s6") \
  X(4, 4, 1, 8, 6, 1, 1, "nt4-tn128-cw8-s6") \
  X(5, 3, 1, 8, 6, 1, 1, "nt3-tn128-cw8-s6") \
  X(6, 2, 1, 8, 6, 1, 1, "nt2-tn128-cw8-s6") \
  X(7, 1, 1, 8, 6, 1, 1, "nt1-tn128-cw8-s6")
#define B32_ROW(I, NT, MT, CW, S, K, MB, NAME) {NAME, NT, MT, CW, S, K},
const Variant kVariants[] = { B32_VARIANTS(B32_ROW) };
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

template <int NT, int MT, int CW, int STAGES, int KPS, int MINB> struct Launcher {
  using C = Cfg<NT, MT, CW, STAGES, KPS>;
  template <int EPI> static constexpr int smem() { return EPI == 2 ? C::SMEM_EPI : C::SMEM; }
  template <int EPI> static int max_ctas() {
    static int cached = 0;
    if (!cached) {
      if (EPI == 2 && MT != 2) return 0;
      cudaFuncSetAttribute(gemm<NT, MT, CW, STAGES, KPS, MINB, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem<EPI>());
      int per_sm = 0, sms = 0, dev = 0;
      cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm<NT, MT, CW, STAGES, KPS, MINB, EPI>, C::THREADS, smem<EPI>());
      cached = per_sm * sms;
    }
    return cached;
  }
  static int max_ctas_for(int epi) { return epi == 0 ? max_ctas<0>() : epi == 1 ? max_ctas<1>() : max_ctas<2>(); }
  static int tn() { return C::TN; }
  static int part() { return C::PART * CW; }
  template <int EPI> static cudaError_t launch(const Plan& p, const uint8_t* tiled, Carrier in, float* y, int ldy, float beta, Epilogue ep, void* ws, cudaStream_t st) {
    if (EPI == 2 && MT != 2) return cudaErrorInvalidValue;
    unsigned* counts = reinterpret_cast<unsigned*>(ws);
    float* partials = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ws) + COUNTER_BYTES);
    if ((size_t)((p.n + C::TN - 1) / C::TN) * CW * sizeof(unsigned) > COUNTER_BYTES) return cudaErrorInvalidValue;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.ctas); cfg.blockDim = dim3(C::THREADS); cfg.dynamicSmemBytes = smem<EPI>(); cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = pdl_enabled() ? 1 : 0;
    cfg.attrs = at; cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, gemm<NT, MT, CW, STAGES, KPS, MINB, EPI>, tiled, reinterpret_cast<const uint8_t*>(in.q), y, ldy, beta,
                              p.n, p.k / 128, p.r, partials, counts, ep);
  }
};

#define B32_CASE(I, NT, MT, CW, S, K, MB, NAME) case I: de_ = dispatch_(Launcher<NT, MT, CW, S, K, MB>{}); break;
#define B32_DISPATCH(V, EXPR_) \
  { auto dispatch_ = [&](auto tag) -> cudaError_t { using L = decltype(tag); EXPR_; return cudaSuccess; }; \
    cudaError_t de_ = cudaErrorInvalidValue; \
    switch (V) { B32_VARIANTS(B32_CASE) } \
    if (de_ != cudaSuccess) return de_; }

__global__ void tile_gu_kernel(const uint32_t* __restrict__ signs, const __half* __restrict__ scales, int ff, int kbs, uint8_t* __restrict__ out) {
  const int n = 2 * ff, tn = 256;
  const size_t total = (size_t)(n / tn) * kbs * tn;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int rt = int(i % tn); const size_t rec = i / tn; const int kb = int(rec % kbs), tile = int(rec / kbs);
    const int w = rt >> 5, m = (rt >> 4) & 1, rr = rt & 15;
    const int row = (m ? ff : 0) + 128 * tile + 16 * w + rr;
    uint8_t* base = out + rec * (size_t)(18 * tn);
    uint32_t* wd = reinterpret_cast<uint32_t*>(base);
    __half* sc = reinterpret_cast<__half*>(base + 16 * tn);
    const int m16 = rt >> 4, gg = rt & 7, hi = (rt >> 3) & 1;
    for (int c = 0; c < 4; ++c) wd[((m16 * 8 + gg) * 2 + hi) * 4 + c] = signs[((size_t)kb * 4 + c) * n + row];
    sc[(m16 * 8 + gg) * 2 + hi] = scales[(size_t)kb * n + row];
  }
}
}  // namespace

int variants() { return kNumVariants; }
const char* variant_name(int v) { return v >= 0 && v < kNumVariants ? kVariants[v].name : "invalid"; }
size_t carrier_bytes(int k) { return (size_t)(k / 128) * CARRIER_BLOCK; }
int plan_tile_rows(int v) { return v >= 0 && v < kNumVariants ? kVariants[v].cw * kVariants[v].mt * 16 : 0; }
size_t tiled_bytes(int n, int k, int tn) { return (size_t)((n + tn - 1) / tn) * (k / 128) * 18 * tn; }
cudaError_t tile_from_packed1(const uint32_t* signs, const __half* scales, int n, int k, int tn, uint8_t* out, cudaStream_t st) {
  if (!signs || !scales || !out || n <= 0 || k <= 0 || k % 128 || tn <= 0 || tn % 16) return cudaErrorInvalidValue;
  tile_kernel<<<1024, 256, 0, st>>>(signs, scales, n, k / 128, tn, out);
  return cudaGetLastError();
}
cudaError_t tile_gu_from_packed1(const uint32_t* signs, const __half* scales, int ff, int k, uint8_t* out, cudaStream_t st) {
  if (!signs || !scales || !out || ff <= 0 || ff % 128 || k <= 0 || k % 128) return cudaErrorInvalidValue;
  tile_gu_kernel<<<1024, 256, 0, st>>>(signs, scales, ff, k / 128, out);
  return cudaGetLastError();
}

cudaError_t produce(const float* x, int ldx, int k, int r, Carrier out, cudaStream_t st) {
  if (!x || !out.q || k <= 0 || k % 128 || r < 1 || r > MAX_TOKENS || ldx < k || ldx % 4) return cudaErrorInvalidValue;
  const int padded = (r + 7) / 8 * 8;
  producer<<<dim3(k / 128, (padded + 3) / 4), 128, 0, st>>>(x, ldx, k, r, padded, out.q, out.error);
  return cudaGetLastError();
}

static int default_variant(int n, int r, int epi) {
  const int nt = (r + 7) / 8;
  const bool wide = epi == 2 || n >= 14336;
  return (wide ? 0 : 4) + (4 - nt);
}

cudaError_t select(int n, int k, int r, Plan* out, int variant, int epi) {
  if (!out || n <= 0 || n % 8 || k <= 0 || k % 128 || r < 1 || r > MAX_TOKENS || epi < 0 || epi > 2) return cudaErrorInvalidValue;
  if (variant < 0) variant = default_variant(n, r, epi);
  if (variant >= kNumVariants || kVariants[variant].nt * 8 < r) return cudaErrorInvalidValue;
  if (epi == 2 && (kVariants[variant].mt != 2 || plan_tile_rows(variant) != 256 || n % 256)) return cudaErrorInvalidValue;
  int maxc = 0, tn = 0;
  B32_DISPATCH(variant, (maxc = L::max_ctas_for(epi), tn = L::tn()));
  if (maxc <= 0) return cudaErrorInvalidConfiguration;
  const long long iters = (long long)((n + tn - 1) / tn) * (k / 128);
  long long ctas = std::min<long long>(maxc, std::max<long long>(1, iters / 4));
  *out = Plan{n, k, r, int(ctas), variant, epi};
  return cudaSuccess;
}

size_t workspace_bytes(const Plan& p) {
  int part = 0;
  auto get = [&]() -> cudaError_t { B32_DISPATCH(p.variant, (part = L::part())); return cudaSuccess; };
  if (get() != cudaSuccess) return 0;
  return COUNTER_BYTES + (size_t)p.ctas * part * sizeof(float);
}

static bool plan_ok(const Plan& p, int epi) {
  return p.variant >= 0 && p.variant < kNumVariants && kVariants[p.variant].nt * 8 >= p.r && p.epi == epi;
}
cudaError_t project(const Plan& p, const uint8_t* tiled, Carrier in, float* y, int ldy, float beta, void* ws, cudaStream_t st) {
  if (!tiled || !in.q || !y || !ws || ldy < p.n || (beta != 0.f && beta != 1.f) || !plan_ok(p, 0)) return cudaErrorInvalidValue;
  B32_DISPATCH(p.variant, return L::template launch<0>(p, tiled, in, y, ldy, beta, Epilogue{}, ws, st));
  return cudaSuccess;
}
cudaError_t project_split(const Plan& p, const uint8_t* tiled, Carrier in, float* y, int ldy, int split, float* y2, int ldy2, void* ws, cudaStream_t st) {
  if (!tiled || !in.q || !y || !y2 || !ws || split <= 0 || split >= p.n || ldy < split || ldy2 < p.n - split || !plan_ok(p, 1)) return cudaErrorInvalidValue;
  B32_DISPATCH(p.variant, return L::template launch<1>(p, tiled, in, y, ldy, 0.f, Epilogue{y2, ldy2, split, nullptr, nullptr}, ws, st));
  return cudaSuccess;
}
cudaError_t project_silu(const Plan& p, const uint8_t* tiled_gu, Carrier in, Carrier out, void* ws, cudaStream_t st) {
  if (!tiled_gu || !in.q || !out.q || !ws || !plan_ok(p, 2)) return cudaErrorInvalidValue;
  B32_DISPATCH(p.variant, return L::template launch<2>(p, tiled_gu, in, nullptr, 0, 0.f, Epilogue{nullptr, 0, 0, out.q, out.error}, ws, st));
  return cudaSuccess;
}
}  // namespace b32
