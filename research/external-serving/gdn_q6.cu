#include "kernels.cuh"
#include <cmath>
#include <type_traits>
#include <stdexcept>
namespace gdn_bits_0 {using namespace h5;
__device__ __forceinline__ float silu(float x) {return x/(1.f+__expf(-x));}
__device__ __forceinline__ float sigm(float x) {return 1.f/(1.f+__expf(-x));}
__device__ __forceinline__ float st_ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void st_st(__half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ float4 ld4(const __half* p) {
  const uint2 u = *(const uint2*)p; const __half2* b = (const __half2*)&u;
  const float2 f0 = __half22float2(b[0]), f1 = __half22float2(b[1]);
  return make_float4(f0.x, f0.y, f1.x, f1.y);
}
__device__ __forceinline__ float st_ld(const float* p) { return *p; }
__device__ __forceinline__ float st_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void st_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ float4 ld4(const float* p) { return *(const float4*)p; }
__device__ __forceinline__ float4 ld4(const __nv_bfloat16* p) {
  const uint2 u = *(const uint2*)p; const __nv_bfloat162* b = (const __nv_bfloat162*)&u;
  const float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]);
  return make_float4(f0.x, f0.y, f1.x, f1.y);
}
// PTX bulk prefetch accepts a uint32 byte size, multiple of 16; 16 KiB is a chosen
// request granularity, not an ISA limit. These are best-effort hints, not completed copies.
__device__ __forceinline__ void chain_l2pf(const uint8_t* p, unsigned bytes, h5::PfChainGranularity granularity) {
  if (!p || !bytes) return;
  const unsigned ncta = gridDim.x * gridDim.y, cta = blockIdx.y * gridDim.x + blockIdx.x;
  const h5::PfChainSlice slice = h5::pf_chain_slice(bytes, ncta, cta);
  if (granularity == h5::PfChainGranularity::CtaShare) {
    if (threadIdx.x == 0 && slice.bytes)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + slice.begin), "r"(slice.bytes) : "memory");
    return;
  }
  const unsigned b1 = slice.begin + slice.bytes;
  for (unsigned o = slice.begin + threadIdx.x * 16384u; o < b1; o += blockDim.x * 16384u) {
    const unsigned n = min(16384u, b1 - o);
    if (n) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + o), "r"(n) : "memory");
  }
}
__device__ __forceinline__ void st_load32(const float* p, float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) { const float4 v = *(const float4*)(p + 16 * i); m[4 * i] = v.x; m[4 * i + 1] = v.y; m[4 * i + 2] = v.z; m[4 * i + 3] = v.w; }
}
__device__ __forceinline__ void st_load32(const __nv_bfloat16* p, float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint2 u = *(const uint2*)(p + 16 * i); const __nv_bfloat162* b = (const __nv_bfloat162*)&u;
    const float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]);
    m[4 * i] = f0.x; m[4 * i + 1] = f0.y; m[4 * i + 2] = f1.x; m[4 * i + 3] = f1.y;
  }
}
__device__ __forceinline__ void st_store32(float* p, const float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) *(float4*)(p + 16 * i) = make_float4(m[4 * i], m[4 * i + 1], m[4 * i + 2], m[4 * i + 3]);
}
__device__ __forceinline__ void st_store32(__nv_bfloat16* p, const float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint2 u; __nv_bfloat162* b = (__nv_bfloat162*)&u;
    b[0] = __floats2bfloat162_rn(m[4 * i], m[4 * i + 1]); b[1] = __floats2bfloat162_rn(m[4 * i + 2], m[4 * i + 3]);
    *(uint2*)(p + 16 * i) = u;
  }
}
__device__ __forceinline__ void st_load32(const __half* p, float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float4 v = ld4(p + 16 * i);
    m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
  }
}
__device__ __forceinline__ void st_store32(__half* p, const float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint2 u; __half2* b = (__half2*)&u;
    b[0] = __floats2half2_rn(m[4*i], m[4*i+1]); b[1] = __floats2half2_rn(m[4*i+2], m[4*i+3]);
    *(uint2*)(p + 16*i) = u;
  }
}
template <int CG>
__device__ __forceinline__ void gdn_load_q8_row(const GdnQ8Layer* layer, int h, int j, int cg, float* m) {
  const float scale = layer->scales[h][j];
#pragma unroll
  for (int i = 0; i < 128 / CG / 4; i++) {
    const unsigned word = *(const unsigned*)&layer->data[h][j][4*cg + CG*4*i];
#pragma unroll
    for (int k = 0; k < 4; k++) m[4*i+k] = (float)(int8_t)(word >> (8*k)) * scale;
  }
}
template <typename ST, int CG, bool Q8>
__device__ __forceinline__ void gdn_store_row(ST* M, GdnQ8Layer* layer, int h, int j, int cg, const float* m, Q8ScaleMode mode) {
  constexpr int MC = 128 / CG;
  if constexpr (Q8) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < MC; i++) amax = fmaxf(amax, fabsf(m[i]));
#pragma unroll
    for (int sh = 1; sh < CG; sh <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, sh));
    const float scale = amax * (1.f / 127.f), inv = scale > 0.f ? 1.f / scale : 0.f;
    const bool ls = mode == Q8ScaleMode::LeastSquares;
    float xq[4] = {0.f, 0.f, 0.f, 0.f};
    int qq = 0;
#pragma unroll
    for (int i = 0; i < MC / 4; i++) {
      unsigned word = 0;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const int q = max(-127, min(127, __float2int_rn(m[4*i+k] * inv)));
        word |= (unsigned)(uint8_t)(int8_t)q << (8*k);
        if (ls) xq[k] = fmaf(m[4*i+k], (float)q, xq[k]);
      }
      if (ls) qq = __dp4a((int)word, (int)word, qq);
      *(unsigned*)&layer->data[h][j][4*cg + CG*4*i] = word;
    }
    float fitted = scale;
    if (ls) {
      float dot = (xq[0] + xq[1]) + (xq[2] + xq[3]);
#pragma unroll
      for (int sh = 1; sh < CG; sh <<= 1) {
        dot += __shfl_xor_sync(0xffffffff, dot, sh);
        qq += __shfl_xor_sync(0xffffffff, qq, sh);
      }
      fitted = qq == 0 ? 0.f : dot / (float)qq;
    }
    if (cg == 0) layer->scales[h][j] = fitted;
  } else if constexpr (CG == 4) st_store32(M, m);
  else {
#pragma unroll
    for (int i = 0; i < MC; i++) st_st(M + (i / 4) * CG * 4 + i % 4, m[i]);
  }
}
constexpr int GDN_T = 2 * PMAX;   // max tokens per segment visit (replay + new); longer segments loop in chunks
// Two CTAs per head (64 state rows each, 256 threads). Output is the raw o / sqrt(128); the gated
// RMSNorm runs in the ssm_out prep (P_GDN_NORM).
constexpr int GDN_SMEM = GDN_T * (128 + 128 + 64 + 2) * 4;
#include "state_fusion.cuh"
struct GdnPrefetch { const void* next = nullptr; uint64_t* bar = nullptr; void* buf = nullptr; unsigned bytes = 0; };
template <typename ST> struct GdnPacked { using Pair = float2; };
template <> struct GdnPacked<__half> {
  using Pair = __half2;
  __device__ static Pair pack(float x, float y) { return __floats2half2_rn(x, y); }
  __device__ static float2 unpack(Pair x) { return __half22float2(x); }
  __device__ static Pair mul(Pair x, Pair y) { return __hmul2(x, y); }
  __device__ static Pair fma(Pair x, Pair y, Pair z) { return __hfma2(x, y, z); }
};
template <> struct GdnPacked<__nv_bfloat16> {
  using Pair = __nv_bfloat162;
  __device__ static Pair pack(float x, float y) { return __floats2bfloat162_rn(x, y); }
  __device__ static float2 unpack(Pair x) { return __bfloat1622float2(x); }
  __device__ static Pair mul(Pair x, Pair y) { return __hmul2(x, y); }
  __device__ static Pair fma(Pair x, Pair y, Pair z) { return __hfma2(x, y, z); }
};
constexpr int GDN_PACKED_SMEM = GDN_T * 2 * 128 * 2;
static unsigned gdn_host_variant = 0;
template <typename ST, int CG = 4, bool FUSED = false, bool PDL = false, bool SCALED = false, bool PACKED = false, bool Q8 = false>
__device__ __forceinline__ void gdn_body(const GdnArgs& a, const Seg sg, int bx, uint8_t* smraw, ConvArgs ca = {}, const ST* Mload = nullptr, const GdnPrefetch pf = {}) {
  float (*sk)[128] = (float (*)[128])smraw;
  float (*sq)[128] = (float (*)[128])(smraw + GDN_T * 128 * 4);
  float (*sv)[64] = (float (*)[64])(smraw + GDN_T * 256 * 4);
  float* sg_ = (float*)(smraw + GDN_T * 320 * 4);
  float* sb_ = sg_ + GDN_T;
  constexpr int JR = 256 / CG, MC = 128 / CG, CH = 64 + JR / 4;
  const int h = bx / (CG / 2), half = bx % (CG / 2), kh = h % HK;
  const int tid = threadIdx.x, lane = tid & 31;
  const int jl = tid / CG, j = JR * half + jl, cg = lane & (CG - 1);
  ST* M = Q8 ? nullptr : (ST*)a.state + (((size_t)sg.slot * NGDN + a.gl) * HV + h) * 128 * 128 + j * 128 + 4 * cg;
  GdnQ8Layer* q8 = Q8 ? (GdnQ8Layer*)a.state + (size_t)sg.slot * NGDN + a.gl : nullptr;
  static_assert(!Q8 || (!PACKED && !SCALED && sizeof(ST) == 2));
  ST* pbase = (ST*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * 2 * a.pending_rows * PEND_GDN;
  const ST* prd = pbase + (size_t)(sg.par ^ 1) * a.pending_rows * PEND_GDN;
  ST* pwr = pbase + (size_t)sg.par * a.pending_rows * PEND_GDN;
  static_assert(!PACKED || (sizeof(ST) == 2 && !SCALED));
  using Ops = GdnPacked<ST>;
  using Pair = typename Ops::Pair;
  Pair mp[MC / 2];
  Pair (*pk)[64] = (Pair (*)[64])(smraw + GDN_SMEM);
  Pair (*pq)[64] = pk + GDN_T;
  float m[MC];
  // Mload / pf (k_gdn_pipe): this CTA's state rows were bulk-copied into shared memory ahead of time.
  const ST* const ML = Mload ? Mload + jl * 128 + 4 * cg : M;
  if constexpr (PACKED) {
#pragma unroll
    for (int i = 0; i < MC / 2; i++) {
#ifdef EXP_GDN_NOMEM
      mp[i] = Ops::pack(0.f, 0.f);
#else
      mp[i] = *(const Pair*)(ML + (i / 2) * CG * 4 + (i % 2) * 2);
#endif
    }
  } else {
#ifdef EXP_GDN_NOMEM
#pragma unroll
    for (int i = 0; i < MC; i++) m[i] = 0.f;
#else
    if constexpr (Q8) gdn_load_q8_row<CG>(q8, h, j, cg, m);
    else {
#pragma unroll
      for (int i = 0; i < MC / 4; i++) {
        const float4 v = ld4(ML + CG * 4 * i);
        m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
      }
    }
#endif
  }
  if (pf.buf) {   // state is in registers: the shared buffer may now receive the next stream's rows
    __syncthreads();
    if (threadIdx.x == 0 && pf.next) {
      const unsigned b = (unsigned)__cvta_generic_to_shared(pf.bar), d = (unsigned)__cvta_generic_to_shared(pf.buf);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(pf.bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d), "l"(pf.next), "r"(pf.bytes), "r"(b) : "memory");
    }
  }
  bool prestaged = false;
  [[maybe_unused]] CgCarry carry;
#ifdef H5_PDL_GDN_STATE_OVERLAP
  // Conv has acquired its predecessor before releasing us, and never writes GDN state.
  // Only state reads move ahead of the fence; conv outputs and every store stay behind it.
  // Variant bit 0x10: the first chunk's replayed rows (pending k, v, g, beta of the previous step) are staged
  // into shared memory before the wait as well.
  if constexpr (FUSED && PDL) conv_gdn_pre<ST, CG>(ca, sg, h, half, carry);
  if constexpr (PDL) if (FUSED || (g_variant & 16u)) {
    const auto c0 = replay_chunk(sg.commit, sg.len, 0, 0, GDN_T);
    for (int i = tid; i < c0.replay_rows * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      if (c < 32) continue;
      const ST* pr = prd + (size_t)t * PEND_GDN;
      const float4 v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      *(float4*)(c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4]) = v4;
    }
    if (tid < c0.replay_rows) { const ST* pr = prd + (size_t)tid * PEND_GDN; sg_[tid] = st_ld(pr + KDIM + VDIM + h); sb_[tid] = st_ld(pr + KDIM + VDIM + HV + h); }
    prestaged = true;
  }
  if constexpr (PDL) { pdl_enter_e(a.early); }
#endif
  H5_LIT_MARKX(0);   // phase marks: 0 wait returned, 1 first chunk staged, 2 recurrence done, 3 before final save
  // SCALED (H5_GDN_SCALED=1): m holds M / Gs, Gs the decay accumulated since the last renormalisation (uniform per
  // CTA: one head). The per-token update is then one FMA per element instead of a multiply and an FMA.
  float Gs = 1.f;
  auto save = [&]() {
    if constexpr (PACKED) {
#pragma unroll
      for (int i = 0; i < MC / 4; i++) {
        const Pair p[2] = {mp[2*i], mp[2*i+1]};
        *(uint2*)(M + CG * 4 * i) = *(const uint2*)p;
      }
    } else {
    float mt[MC];
#pragma unroll
    for (int i = 0; i < MC; i++) mt[i] = SCALED ? Gs * m[i] : m[i];
    gdn_store_row<ST, CG, Q8>(M, q8, h, j, cg, mt, a.q8_scale);
    }
  };
  int done = 0, replayed = 0;
  bool first = true;
  while (first || replayed < sg.commit || done < sg.len) {
    const auto chunk = replay_chunk(sg.commit, sg.len, replayed, done, GDN_T);
    const int nrep = chunk.replay_rows, nnew = chunk.new_rows;
    const int T = nrep + nnew;
    const int t_first = prestaged ? nrep : 0;   // replayed rows of the first chunk are already staged
    prestaged = false;
    __syncthreads();
    for (int i = tid + t_first * CH; i < T * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      float4 v4; bool have = true;
      if (t < nrep) {
        const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN;
        if (c < 32) have = false;
        else v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      } else if constexpr (FUSED) {
        have = false;
      } else {
        const float* co = a.conv_out + (size_t)(sg.start + done + t - nrep) * CONV_CH;
        v4 = *(const float4*)(c < 32 ? co + kh * 128 + c * 4 : c < 64 ? co + KDIM + kh * 128 + (c - 32) * 4 : co + 2 * KDIM + h * 128 + JR * half + (c - 64) * 4);
      }
      if (have) {
        float* dst = c < 32 ? &sq[t][c * 4] : c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4];
        *(float4*)dst = v4;
      }
    }
    if (tid >= t_first && tid < T) {
      const int t = tid;
      if (t < nrep) { const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN; sg_[t] = st_ld(pr + KDIM + VDIM + h); sb_[t] = st_ld(pr + KDIM + VDIM + HV + h); }
      else if constexpr (!FUSED) { const float* gr = a.gates + (size_t)(sg.start + done + t - nrep) * 96; sg_[t] = gr[h]; sb_[t] = gr[HV + h]; }
    }
    if constexpr (FUSED && PDL) conv_gdn_stage_b<ST, CG>(ca, sg, h, half, first, done, nrep, nnew, sq, sk, sv, sg_, sb_, carry);
    else if constexpr (FUSED) conv_gdn_stage<ST, CG>(ca, sg, h, half, done, nrep, nnew, sq, sk, sv, sg_, sb_);
    __syncthreads();
    if (first) H5_LIT_MARKX(1);
    if (sg.keep) {   // new speculative rows -> pending[par] (disjoint from the replay buffer)
      for (int i = tid; i < nnew * 128; i += 256) {
        const int t = nrep + (i >> 7), d = i & 127;
        ST* pw = pwr + (size_t)(done + t - nrep) * PEND_GDN;
        if (h < HK && half == 0) st_st(pw + kh * 128 + d, sk[t][d]);
        if (d < JR && !a.replay_delta) st_st(pw + KDIM + h * 128 + JR * half + d, sv[t][d]);
        if (d == 0 && half == 0) { st_st(pw + KDIM + VDIM + h, sg_[t]); st_st(pw + KDIM + VDIM + HV + h, sb_[t]); }
      }
    }
    // Q8 must leave an uncommitted checkpoint byte-identical, not re-quantize it.
    if constexpr (!Q8) if (first && sg.keep && nrep == 0) save();
    if constexpr (PACKED) {
      for (int i = tid; i < T * 64; i += 256) {
        const int t = i / 64, c = (i % 64) * 2;
        pk[t][i % 64] = Ops::pack(sk[t][c], sk[t][c+1]);
        if (t >= nrep) pq[t][i % 64] = Ops::pack(sq[t][c], sq[t][c+1]);
      }
      __syncthreads();
      const Pair zero = Ops::pack(0.f, 0.f);
      for (int t = 0; t < T; t++) {
        const float g = sg_[t], beta = sb_[t];
        float delta;
        if (a.replay_delta && t < nrep) delta = sv[t][jl];
        else {
          Pair s0 = zero, s1 = zero;
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            s0 = Ops::fma(mp[2*i], pk[t][2*cg + CG*2*i], s0);
            s1 = Ops::fma(mp[2*i+1], pk[t][2*cg + CG*2*i+1], s1);
          }
          const float2 f0 = Ops::unpack(s0), f1 = Ops::unpack(s1);
          float s = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
          delta = beta * (sv[t][jl] - g * s);
          if (a.replay_delta && sg.keep && cg == 0)
            st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
        }
        const Pair g2 = Ops::pack(g, g), d2 = Ops::pack(delta, delta);
#pragma unroll
        for (int i = 0; i < MC / 2; i++)
          mp[i] = Ops::fma(d2, pk[t][2*cg + (i/2)*CG*2 + i%2], Ops::mul(g2, mp[i]));
        if (t >= nrep) {
          Pair o0 = zero, o1 = zero;
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            o0 = Ops::fma(mp[2*i], pq[t][2*cg + CG*2*i], o0);
            o1 = Ops::fma(mp[2*i+1], pq[t][2*cg + CG*2*i+1], o1);
          }
          const float2 f0 = Ops::unpack(o0), f1 = Ops::unpack(o1);
          float o = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
          if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
        } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
      }
    } else
    // Variant bit 0x20: closed-form chunk recurrence. With M the chunk's initial state, all dot products with
    // M (a_t = M k_t, p_t = M q_t) and the key Gram tables (KK = k_u.k_t, KQ = k_u.q_t) are independent and
    // computed in parallel; the per-token recurrence is then scalar per state row:
    //   s_t = Gam a_t + sum_{u<t} w_u KK[u][t];  delta_t = beta_t (v_t - g_t s_t);  w_u *= g_t; w_t = delta_t; Gam *= g_t
    //   o_t = Gam p_t + sum_{u<=t} w_u KQ[u][t];  M_end = Gam M + sum_u w_u k_u
    // (w_u = delta_u times the decay since token u, Gam = the decay since the chunk start). Same algebra as the
    // per-token loop below; the summation order differs.
    if (g_variant & 32u) {
      __shared__ float KK[GDN_T][GDN_T], KQ[GDN_T][GDN_T];
      const int warp = tid >> 5;
      for (int pi = warp; pi < 2 * T * T; pi += 8) {
        const bool isq = pi >= T * T;
        const int u = (pi % (T * T)) / T, t = pi % T;
        if (isq ? (u > t || t < nrep) : (u >= t)) continue;
        const float4 x = *(const float4*)&sk[u][lane * 4], y = *(const float4*)(isq ? &sq[t][lane * 4] : &sk[t][lane * 4]);
        float d = fmaf(x.x, y.x, fmaf(x.y, y.y, fmaf(x.z, y.z, x.w * y.w)));
#pragma unroll
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (lane == 0) (isq ? KQ : KK)[u][t] = d;
      }
      float a_[GDN_T], p_[GDN_T];
#pragma unroll
      for (int t = 0; t < GDN_T; t++) {
        a_[t] = 0.f; p_[t] = 0.f;
        if (t < T) {
          float s4[4] = {0.f, 0.f, 0.f, 0.f}, q4s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            const float4 k4 = *(const float4*)(&sk[t][4 * cg] + CG * 4 * i);
            s4[0] = fmaf(m[4*i], k4.x, s4[0]); s4[1] = fmaf(m[4*i+1], k4.y, s4[1]); s4[2] = fmaf(m[4*i+2], k4.z, s4[2]); s4[3] = fmaf(m[4*i+3], k4.w, s4[3]);
            if (t >= nrep) {
              const float4 q4 = *(const float4*)(&sq[t][4 * cg] + CG * 4 * i);
              q4s[0] = fmaf(m[4*i], q4.x, q4s[0]); q4s[1] = fmaf(m[4*i+1], q4.y, q4s[1]); q4s[2] = fmaf(m[4*i+2], q4.z, q4s[2]); q4s[3] = fmaf(m[4*i+3], q4.w, q4s[3]);
            }
          }
          a_[t] = (s4[0] + s4[1]) + (s4[2] + s4[3]); p_[t] = (q4s[0] + q4s[1]) + (q4s[2] + q4s[3]);
        }
      }
#pragma unroll
      for (int sh = 1; sh < CG; sh <<= 1)
#pragma unroll
        for (int t = 0; t < GDN_T; t++) { a_[t] += __shfl_xor_sync(0xffffffff, a_[t], sh); p_[t] += __shfl_xor_sync(0xffffffff, p_[t], sh); }
      __syncthreads();   // KK, KQ complete
      float w[GDN_T], Gam = 1.f;
      auto materialize = [&](int upto, float* out) {   // out = Gam M + sum_{u<=upto} w_u k_u (lane's columns)
#pragma unroll
        for (int i = 0; i < MC; i++) out[i] = Gam * m[i];
#pragma unroll
        for (int u = 0; u < GDN_T; u++) if (u <= upto) {
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            const float4 k4 = *(const float4*)(&sk[u][4 * cg] + CG * 4 * i);
            out[4*i] = fmaf(w[u], k4.x, out[4*i]); out[4*i+1] = fmaf(w[u], k4.y, out[4*i+1]);
            out[4*i+2] = fmaf(w[u], k4.z, out[4*i+2]); out[4*i+3] = fmaf(w[u], k4.w, out[4*i+3]);
          }
        }
      };
#pragma unroll
      for (int t = 0; t < GDN_T; t++) {
        w[t] = 0.f;
        if (t < T) {
          const float g = sg_[t], beta = sb_[t];
          float delta;
          if (a.replay_delta && t < nrep) delta = sv[t][jl];
          else {
            float s0 = Gam * a_[t], s1 = 0.f;
#pragma unroll
            for (int u = 0; u < t; u++) { if (u & 1) s1 = fmaf(w[u], KK[u][t], s1); else s0 = fmaf(w[u], KK[u][t], s0); }
            delta = beta * (sv[t][jl] - g * (s0 + s1));
            if (a.replay_delta && sg.keep && cg == 0)
              st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
          }
#pragma unroll
          for (int u = 0; u < t; u++) w[u] *= g;
          Gam *= g; w[t] = delta;
          if (t >= nrep) {
            float o0 = Gam * p_[t], o1 = 0.f;
#pragma unroll
            for (int u = 0; u <= t; u++) { if (u & 1) o1 = fmaf(w[u], KQ[u][t], o1); else o0 = fmaf(w[u], KQ[u][t], o0); }
            if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = (o0 + o1) * 0.08838834764831845f;
          } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) {
            float mt[MC];
            materialize(t, mt);
            gdn_store_row<ST, CG, Q8>(M, q8, h, j, cg, mt, a.q8_scale);
          }
        }
      }
      { float mt[MC]; materialize(T - 1, mt);
#pragma unroll
        for (int i = 0; i < MC; i++) m[i] = mt[i]; }
    } else if (g_variant & 64u) {
      // Variant bit 0x40: o_t = M_t q_t = g_t (M_{t-1} q_t) + delta_t (k_t . q_t). Both dot products with M_{t-1}
      // are taken together and reduced in the same shuffle rounds, so each token carries one reduction
      // chain instead of two; k_t . q_t is state-independent (one dot per token per CTA, staged once).
      __shared__ float kq_[GDN_T];
      {
        const int warp = tid >> 5;
        for (int t = nrep + warp; t < T; t += 8) {
          const float4 x = *(const float4*)&sk[t][lane * 4], y = *(const float4*)&sq[t][lane * 4];
          float d = fmaf(x.x, y.x, fmaf(x.y, y.y, fmaf(x.z, y.z, x.w * y.w)));
#pragma unroll
          for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
          if (lane == 0) kq_[t] = d;
        }
      }
      __syncthreads();
      for (int t = 0; t < T; t++) {
        const float g = sg_[t], beta = sb_[t];
        const bool out = t >= nrep;
        const float* kp = &sk[t][4 * cg];
        const float* qp = &sq[t][4 * cg];
        float kr[MC];
        float s4[4] = {0.f, 0.f, 0.f, 0.f}, p4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) {
          const float4 k4 = *(const float4*)(kp + CG * 4 * i);
          kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w;
          s4[0] = fmaf(m[4*i], k4.x, s4[0]); s4[1] = fmaf(m[4*i+1], k4.y, s4[1]); s4[2] = fmaf(m[4*i+2], k4.z, s4[2]); s4[3] = fmaf(m[4*i+3], k4.w, s4[3]);
          if (out) {
            const float4 q4 = *(const float4*)(qp + CG * 4 * i);
            p4[0] = fmaf(m[4*i], q4.x, p4[0]); p4[1] = fmaf(m[4*i+1], q4.y, p4[1]); p4[2] = fmaf(m[4*i+2], q4.z, p4[2]); p4[3] = fmaf(m[4*i+3], q4.w, p4[3]);
          }
        }
        float sv_ = (s4[0] + s4[1]) + (s4[2] + s4[3]), pv = (p4[0] + p4[1]) + (p4[2] + p4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) { sv_ += __shfl_xor_sync(0xffffffff, sv_, sh); pv += __shfl_xor_sync(0xffffffff, pv, sh); }
        float delta;
        if (a.replay_delta && t < nrep) delta = sv[t][jl];
        else {
          delta = beta * (sv[t][jl] - g * sv_);
          if (a.replay_delta && sg.keep && cg == 0)
            st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
        }
#pragma unroll
        for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
        if (out) {
          if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = fmaf(g, pv, delta * kq_[t]) * 0.08838834764831845f;
        } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
      }
    } else if constexpr (SCALED) {
      for (int t = 0; t < T; t++) {
        const float g = sg_[t], beta = sb_[t];
        const float* kp = &sk[t][4 * cg];
        float kr[MC];
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
        float delta;
        if (a.replay_delta && t < nrep) delta = sv[t][jl];
        else {
          float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
          float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
          delta = beta * (sv[t][jl] - g * (Gs * s));
          if (a.replay_delta && sg.keep && cg == 0)
            st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
        }
        Gs *= g;
        if (Gs < 1e-18f) {   // uniform: g is per head
#pragma unroll
          for (int i = 0; i < MC; i++) m[i] *= Gs;
          Gs = 1.f;
        }
        const float c = delta / Gs;
#pragma unroll
        for (int i = 0; i < MC; i++) m[i] = fmaf(c, kr[i], m[i]);
        if (t >= nrep) {
          const float* qp = &sq[t][4 * cg];
          float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
          float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
          if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = (Gs * o) * 0.08838834764831845f;
        } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
      }
    } else
#ifdef EXP_GDN_NOCOMP
    if (T > 1000)
#endif
    for (int t = 0; t < T; t++) {
      const float g = sg_[t], beta = sb_[t];
      const float* kp = &sk[t][4 * cg];
      float kr[MC];
#pragma unroll
      for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
      float delta;
      if (a.replay_delta && t < nrep) delta = sv[t][jl];
      else {
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
        float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
        delta = beta * (sv[t][jl] - g * s);
        if (a.replay_delta && sg.keep && cg == 0)
          st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
      }
#pragma unroll
      for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
      if (t >= nrep) {
        const float* qp = &sq[t][4 * cg];
        float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
        float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
        #pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
        if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
      } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
    }
    replayed += nrep; done += nnew; first = false;
  }
  H5_LIT_MARKX(2);
#ifndef EXP_GDN_NOMEM
  if constexpr (!FUSED) if (a.sig) {
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(a.sig) : "memory");
  }
  if (!sg.keep) save();
#endif
}
template <typename ST, int CG = 4, int OCC = 2, bool SCALED = false, bool PACKED = false, bool Q8 = false>
__global__ void __launch_bounds__(256, OCC) k_gdn(GdnArgs a) { H5_LIT(0);
#ifndef H5_PDL_GDN_STATE_OVERLAP
  // Segment metadata is also upstream of conv, but retain the control's entry fence.
  pdl_enter();
#endif
  __shared__ __align__(16) uint8_t sm[GDN_SMEM + (PACKED ? GDN_PACKED_SMEM : 0)];
#ifdef H5_PDL_GDN_STATE_OVERLAP
  gdn_body<ST, CG, false, true, SCALED, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm);
#else
  gdn_body<ST, CG, false, false, SCALED, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm);
#endif
}
// Included in kernels.cu after the shared storage and launch-selection helpers.
template <int CG> struct alignas(16) GdnSingleShared {
  float q[128], k[128], v[256 / CG], gates[2], red[8];
};

template <int CG>
__device__ __forceinline__ void single_conv_stage(const ConvArgs& c, Seg sg, int h, int part,
    GdnSingleShared<CG>& sm, CgCarry carry) {
  constexpr int JR = 256 / CG;
  const int t = threadIdx.x, d = t & 127, kh = h % HK;
  const int channel = (t < 128 ? 0 : KDIM) + kh * 128 + d;
  const size_t layer = (size_t)sg.slot * NGDN + c.gl;
  float* dst = c.state + (2 * layer + (sg.conv_par ^ 1)) * 3 * CONV_CH;
  const float x = c.proj[(size_t)sg.start * GDN_IN + channel];
  const float4 w = carry.wk;
  float v = silu(w.x * carry.s0 + w.y * carry.s1 + w.z * carry.s2 + w.w * x);
  float ss = v * v;
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  if ((t & 31) == 0) sm.red[t >> 5] = ss;
  if (h < HK && part == 0) {
    dst[channel] = carry.s1; dst[CONV_CH + channel] = carry.s2; dst[2 * CONV_CH + channel] = x;
  }
  if (t < JR) {
    const int vc = 2 * KDIM + h * 128 + JR * part + t;
    const float vx = c.proj[(size_t)sg.start * GDN_IN + vc];
    const float4 vw = carry.vw;
    sm.v[t] = silu(vw.x * carry.v0 + vw.y * carry.v1 + vw.z * carry.v2 + vw.w * vx);
    dst[vc] = carry.v1; dst[CONV_CH + vc] = carry.v2; dst[2 * CONV_CH + vc] = vx;
  }
  if (t == 0) {
    const float g = c.ab[(size_t)sg.start * 96 + h] + c.dt_bias[h];
    const float sp = g > 20.f ? g : log1pf(__expf(g));
    sm.gates[0] = __expf(c.ssm_a[h] * sp);
    sm.gates[1] = sigm(c.ab[(size_t)sg.start * 96 + HV + h]);
  }
  __syncthreads();
  const int b = t < 128 ? 0 : 4;
  ss = sm.red[b] + sm.red[b + 1] + sm.red[b + 2] + sm.red[b + 3];
  (t < 128 ? sm.q : sm.k)[d] = v * (1.f / fmaxf(sqrtf(ss), NORM_EPS));
}

template <typename ST, int CG, bool FUSED, bool Q8 = false>
__global__ void __launch_bounds__(256, 2) k_gdn_single(GdnArgs a, ConvArgs c) { H5_LIT_SITE(GdnSingle, 0);
  constexpr int JR = 256 / CG, MC = 128 / CG;
  __shared__ GdnSingleShared<CG> sm;
#ifndef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  const Seg sg = a.segs[blockIdx.y];
  if (sg.len != 1 || sg.commit || sg.keep) { asm volatile("trap;"); return; }
  const int h = blockIdx.x / (CG / 2), part = blockIdx.x % (CG / 2), kh = h % HK;
  const int t = threadIdx.x, cg = t & (CG - 1), jl = t / CG, j = part * JR + jl;
  static_assert(!Q8 || !FUSED);
  ST* M = Q8 ? nullptr : (ST*)a.state + (((size_t)sg.slot * NGDN + a.gl) * HV + h) * 128 * 128 + j * 128 + 4 * cg;
  GdnQ8Layer* q8 = Q8 ? (GdnQ8Layer*)a.state + (size_t)sg.slot * NGDN + a.gl : nullptr;
  float m[MC];
  if constexpr (Q8) gdn_load_q8_row<CG>(q8, h, j, cg, m);
  else {
#pragma unroll
    for (int i = 0; i < MC / 4; ++i) {
      const float4 v = ld4(M + CG * 4 * i);
      m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
    }
  }
  CgCarry carry;
  if constexpr (FUSED) {
    const int channel = (t < 128 ? 0 : KDIM) + kh * 128 + (t & 127);
    const float* src = c.state + (2 * ((size_t)sg.slot * NGDN + c.gl) + sg.conv_par) * 3 * CONV_CH;
    carry.s0 = src[channel]; carry.s1 = src[CONV_CH + channel]; carry.s2 = src[2 * CONV_CH + channel];
    carry.wk = *(const float4*)(c.w + channel * 4);
    if (t < JR) {
      const int vc = 2 * KDIM + h * 128 + part * JR + t;
      carry.v0 = src[vc]; carry.v1 = src[CONV_CH + vc]; carry.v2 = src[2 * CONV_CH + vc];
      carry.vw = *(const float4*)(c.w + vc * 4);
    }
  }
#ifdef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  H5_LIT_MARKX(0);
  if constexpr (FUSED) {
    if (c.pf_placement == h5::PfChainPlacement::Prologue) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
    single_conv_stage<CG>(c, sg, h, part, sm, carry);
  } else {
    const float* co = a.conv_out + (size_t)sg.start * CONV_CH;
    if (t < 32) {
      *(float4*)(sm.q + t * 4) = *(const float4*)(co + kh * 128 + t * 4);
      *(float4*)(sm.k + t * 4) = *(const float4*)(co + KDIM + kh * 128 + t * 4);
    }
    if (t < JR / 4) *(float4*)(sm.v + t * 4) = *(const float4*)(co + 2 * KDIM + h * 128 + part * JR + t * 4);
    if (t == 0) {
      sm.gates[0] = a.gates[(size_t)sg.start * 96 + h];
      sm.gates[1] = a.gates[(size_t)sg.start * 96 + HV + h];
    }
  }
  __syncthreads();
  H5_LIT_MARKX(1);
  float kr[MC], dot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < MC / 4; ++i) {
    const float4 k = *(const float4*)(sm.k + CG * 4 * i + 4 * cg);
    kr[4*i] = k.x; kr[4*i+1] = k.y; kr[4*i+2] = k.z; kr[4*i+3] = k.w;
  }
#pragma unroll
  for (int i = 0; i < MC; ++i) dot[i & 3] = fmaf(m[i], kr[i], dot[i & 3]);
  float s = (dot[0] + dot[1]) + (dot[2] + dot[3]);
#pragma unroll
  for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
  const float g = sm.gates[0], delta = sm.gates[1] * (sm.v[jl] - g * s);
#pragma unroll
  for (int i = 0; i < MC; ++i) m[i] = fmaf(g, m[i], delta * kr[i]);
  float out[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < MC / 4; ++i) {
    const float4 q = *(const float4*)(sm.q + CG * 4 * i + 4 * cg);
    out[0] = fmaf(m[4*i], q.x, out[0]); out[1] = fmaf(m[4*i+1], q.y, out[1]);
    out[2] = fmaf(m[4*i+2], q.z, out[2]); out[3] = fmaf(m[4*i+3], q.w, out[3]);
  }
  float o = (out[0] + out[1]) + (out[2] + out[3]);
#pragma unroll
  for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
  if (cg == 0) a.y[(size_t)sg.start * VDIM + h * 128 + j] = o * 0.08838834764831845f;
  if (a.sig) prep_signal(a.sig);
  H5_LIT_MARKX(2);
  if constexpr (Q8) gdn_store_row<ST, CG, true>(M, q8, h, j, cg, m, a.q8_scale);
  else {
#pragma unroll
    for (int i = 0; i < MC / 4; ++i) {
      if constexpr (sizeof(ST) == 2) {
        uint2 v;
#pragma unroll
        for (int k = 0; k < 4; ++k) st_st((ST*)&v + k, m[4*i+k]);
        *(uint2*)(M + CG * 4 * i) = v;
      } else *(float4*)(M + CG * 4 * i) = make_float4(m[4*i], m[4*i+1], m[4*i+2], m[4*i+3]);
    }
  }
  if constexpr (FUSED) {
    H5_LIT_MARKX(3);
    if (c.pf_placement == h5::PfChainPlacement::Tail) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
  }
}


}
namespace gdn_bits_1 {using namespace h5;
__device__ __forceinline__ float silu(float x) {return x/(1.f+__expf(-x));}
__device__ __forceinline__ float sigm(float x) {return 1.f/(1.f+__expf(-x));}
__device__ __forceinline__ float st_ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void st_st(__half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ float4 ld4(const __half* p) {
  const uint2 u = *(const uint2*)p; const __half2* b = (const __half2*)&u;
  const float2 f0 = __half22float2(b[0]), f1 = __half22float2(b[1]);
  return make_float4(f0.x, f0.y, f1.x, f1.y);
}
__device__ __forceinline__ float st_ld(const float* p) { return *p; }
__device__ __forceinline__ float st_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void st_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ float4 ld4(const float* p) { return *(const float4*)p; }
__device__ __forceinline__ float4 ld4(const __nv_bfloat16* p) {
  const uint2 u = *(const uint2*)p; const __nv_bfloat162* b = (const __nv_bfloat162*)&u;
  const float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]);
  return make_float4(f0.x, f0.y, f1.x, f1.y);
}
// PTX bulk prefetch accepts a uint32 byte size, multiple of 16; 16 KiB is a chosen
// request granularity, not an ISA limit. These are best-effort hints, not completed copies.
__device__ __forceinline__ void chain_l2pf(const uint8_t* p, unsigned bytes, h5::PfChainGranularity granularity) {
  if (!p || !bytes) return;
  const unsigned ncta = gridDim.x * gridDim.y, cta = blockIdx.y * gridDim.x + blockIdx.x;
  const h5::PfChainSlice slice = h5::pf_chain_slice(bytes, ncta, cta);
  if (granularity == h5::PfChainGranularity::CtaShare) {
    if (threadIdx.x == 0 && slice.bytes)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + slice.begin), "r"(slice.bytes) : "memory");
    return;
  }
  const unsigned b1 = slice.begin + slice.bytes;
  for (unsigned o = slice.begin + threadIdx.x * 16384u; o < b1; o += blockDim.x * 16384u) {
    const unsigned n = min(16384u, b1 - o);
    if (n) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + o), "r"(n) : "memory");
  }
}
__device__ __forceinline__ void st_load32(const float* p, float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) { const float4 v = *(const float4*)(p + 16 * i); m[4 * i] = v.x; m[4 * i + 1] = v.y; m[4 * i + 2] = v.z; m[4 * i + 3] = v.w; }
}
__device__ __forceinline__ void st_load32(const __nv_bfloat16* p, float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint2 u = *(const uint2*)(p + 16 * i); const __nv_bfloat162* b = (const __nv_bfloat162*)&u;
    const float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]);
    m[4 * i] = f0.x; m[4 * i + 1] = f0.y; m[4 * i + 2] = f1.x; m[4 * i + 3] = f1.y;
  }
}
__device__ __forceinline__ void st_store32(float* p, const float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) *(float4*)(p + 16 * i) = make_float4(m[4 * i], m[4 * i + 1], m[4 * i + 2], m[4 * i + 3]);
}
__device__ __forceinline__ void st_store32(__nv_bfloat16* p, const float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint2 u; __nv_bfloat162* b = (__nv_bfloat162*)&u;
    b[0] = __floats2bfloat162_rn(m[4 * i], m[4 * i + 1]); b[1] = __floats2bfloat162_rn(m[4 * i + 2], m[4 * i + 3]);
    *(uint2*)(p + 16 * i) = u;
  }
}
__device__ __forceinline__ void st_load32(const __half* p, float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float4 v = ld4(p + 16 * i);
    m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
  }
}
__device__ __forceinline__ void st_store32(__half* p, const float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint2 u; __half2* b = (__half2*)&u;
    b[0] = __floats2half2_rn(m[4*i], m[4*i+1]); b[1] = __floats2half2_rn(m[4*i+2], m[4*i+3]);
    *(uint2*)(p + 16*i) = u;
  }
}
// CG8 signed six-bit carrier: sixteen coordinates -> three aligned words.
// Existing Q8 row/slot allocation is retained, but only96 of128 payload bytes
// are touched. Padding is never interpreted; zero-filled allocation encodes0.
__device__ __forceinline__ unsigned q6_load4(const unsigned* words,int lane4){
 const unsigned w=lane4<3?words[lane4]:0;
 const int bit=24*lane4,word=bit>>5;
 const unsigned lo=__shfl_sync(0xffffffff,w,word,4),hi=__shfl_sync(0xffffffff,w,word+1,4);
 return __funnelshift_r(lo,hi,bit&31)&0xffffffu;
}
__device__ __forceinline__ void q6_store4(unsigned* words,unsigned packed,int lane4){
 const unsigned next=__shfl_down_sync(0xffffffff,packed,1,4);
 if(lane4<3)words[lane4]=(packed>>(8*lane4))|(next<<(24-8*lane4));
}
template<int CG>
__device__ __forceinline__ void gdn_load_q8_row(const GdnQ8Layer*layer,int h,int j,int cg,float*m){
 static_assert(CG==8,"Q6 planes require CG8");const float scale=layer->scales[h][j];
 const unsigned*p=(const unsigned*)layer->data[h][j];const unsigned words[3]={p[cg],p[8+cg],p[16+cg]};
#pragma unroll
 for(int i=0;i<16;i++){
  constexpr unsigned mask=63u;const int bit=6*i,w=bit/32,shift=bit%32;
  const unsigned v=shift>26?__funnelshift_r(words[w],words[w+1],shift):words[w]>>shift;
  m[i]=float(int((v&mask)<<26)>>26)*scale;
 }
}
template<typename ST,int CG,bool Q8>
__device__ __forceinline__ void gdn_store_row(ST*,GdnQ8Layer*layer,int h,int j,int cg,const float*m,Q8ScaleMode mode){
 static_assert(Q8&&CG==8,"Q6 save currently requires Q8 container and CG8");
 float amax=0;
#pragma unroll
 for(int i=0;i<128/CG;i++)amax=fmaxf(amax,fabsf(m[i]));
#pragma unroll
 for(int sh=1;sh<CG;sh<<=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,sh));
 const float scale=amax*(1.f/31.f),inv=scale>0?1.f/scale:0;float xq[4]={0,0,0,0};int qq=0;unsigned packs[4];
#pragma unroll
 for(int i=0;i<128/CG/4;i++){
  unsigned packed=0,bytes=0;
#pragma unroll
  for(int k=0;k<4;k++){
   const int q=max(-31,min(31,__float2int_rn(m[4*i+k]*inv)));
   packed|=(unsigned(q)&63u)<<(6*k);bytes|=unsigned(uint8_t(int8_t(q)))<<(8*k);
   xq[k]=fmaf(m[4*i+k],float(q),xq[k]);
  }
  qq=__dp4a(int(bytes),int(bytes),qq);
  packs[i]=packed;
 }
 unsigned*p=(unsigned*)layer->data[h][j];
 p[cg]=packs[0]|(packs[1]<<24);
 p[8+cg]=(packs[1]>>8)|(packs[2]<<16);
 p[16+cg]=(packs[2]>>16)|(packs[3]<<8);
 float dot=(xq[0]+xq[1])+(xq[2]+xq[3]);
#pragma unroll
 for(int sh=1;sh<CG;sh<<=1){dot+=__shfl_xor_sync(0xffffffff,dot,sh);qq+=__shfl_xor_sync(0xffffffff,qq,sh);}
 if(cg==0)layer->scales[h][j]=mode==Q8ScaleMode::LeastSquares?(qq==0?0.f:dot/float(qq)):scale;
}

constexpr int GDN_T = 2 * PMAX;   // max tokens per segment visit (replay + new); longer segments loop in chunks
// Two CTAs per head (64 state rows each, 256 threads). Output is the raw o / sqrt(128); the gated
// RMSNorm runs in the ssm_out prep (P_GDN_NORM).
constexpr int GDN_SMEM = GDN_T * (128 + 128 + 64 + 2) * 4;
#include "state_fusion.cuh"
struct GdnPrefetch { const void* next = nullptr; uint64_t* bar = nullptr; void* buf = nullptr; unsigned bytes = 0; };
template <typename ST> struct GdnPacked { using Pair = float2; };
template <> struct GdnPacked<__half> {
  using Pair = __half2;
  __device__ static Pair pack(float x, float y) { return __floats2half2_rn(x, y); }
  __device__ static float2 unpack(Pair x) { return __half22float2(x); }
  __device__ static Pair mul(Pair x, Pair y) { return __hmul2(x, y); }
  __device__ static Pair fma(Pair x, Pair y, Pair z) { return __hfma2(x, y, z); }
};
template <> struct GdnPacked<__nv_bfloat16> {
  using Pair = __nv_bfloat162;
  __device__ static Pair pack(float x, float y) { return __floats2bfloat162_rn(x, y); }
  __device__ static float2 unpack(Pair x) { return __bfloat1622float2(x); }
  __device__ static Pair mul(Pair x, Pair y) { return __hmul2(x, y); }
  __device__ static Pair fma(Pair x, Pair y, Pair z) { return __hfma2(x, y, z); }
};
constexpr int GDN_PACKED_SMEM = GDN_T * 2 * 128 * 2;
static unsigned gdn_host_variant = 0;
template <typename ST, int CG = 4, bool FUSED = false, bool PDL = false, bool SCALED = false, bool PACKED = false, bool Q8 = false>
__device__ __forceinline__ void gdn_body(const GdnArgs& a, const Seg sg, int bx, uint8_t* smraw, ConvArgs ca = {}, const ST* Mload = nullptr, const GdnPrefetch pf = {}) {
  float (*sk)[128] = (float (*)[128])smraw;
  float (*sq)[128] = (float (*)[128])(smraw + GDN_T * 128 * 4);
  float (*sv)[64] = (float (*)[64])(smraw + GDN_T * 256 * 4);
  float* sg_ = (float*)(smraw + GDN_T * 320 * 4);
  float* sb_ = sg_ + GDN_T;
  constexpr int JR = 256 / CG, MC = 128 / CG, CH = 64 + JR / 4;
  const int h = bx / (CG / 2), half = bx % (CG / 2), kh = h % HK;
  const int tid = threadIdx.x, lane = tid & 31;
  const int jl = tid / CG, j = JR * half + jl, cg = lane & (CG - 1);
  ST* M = Q8 ? nullptr : (ST*)a.state + (((size_t)sg.slot * NGDN + a.gl) * HV + h) * 128 * 128 + j * 128 + 4 * cg;
  GdnQ8Layer* q8 = Q8 ? (GdnQ8Layer*)a.state + (size_t)sg.slot * NGDN + a.gl : nullptr;
  static_assert(!Q8 || (!PACKED && !SCALED && sizeof(ST) == 2));
  ST* pbase = (ST*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * 2 * a.pending_rows * PEND_GDN;
  const ST* prd = pbase + (size_t)(sg.par ^ 1) * a.pending_rows * PEND_GDN;
  ST* pwr = pbase + (size_t)sg.par * a.pending_rows * PEND_GDN;
  static_assert(!PACKED || (sizeof(ST) == 2 && !SCALED));
  using Ops = GdnPacked<ST>;
  using Pair = typename Ops::Pair;
  Pair mp[MC / 2];
  Pair (*pk)[64] = (Pair (*)[64])(smraw + GDN_SMEM);
  Pair (*pq)[64] = pk + GDN_T;
  float m[MC];
  // Mload / pf (k_gdn_pipe): this CTA's state rows were bulk-copied into shared memory ahead of time.
  const ST* const ML = Mload ? Mload + jl * 128 + 4 * cg : M;
  if constexpr (PACKED) {
#pragma unroll
    for (int i = 0; i < MC / 2; i++) {
#ifdef EXP_GDN_NOMEM
      mp[i] = Ops::pack(0.f, 0.f);
#else
      mp[i] = *(const Pair*)(ML + (i / 2) * CG * 4 + (i % 2) * 2);
#endif
    }
  } else {
#ifdef EXP_GDN_NOMEM
#pragma unroll
    for (int i = 0; i < MC; i++) m[i] = 0.f;
#else
    if constexpr (Q8) gdn_load_q8_row<CG>(q8, h, j, cg, m);
    else {
#pragma unroll
      for (int i = 0; i < MC / 4; i++) {
        const float4 v = ld4(ML + CG * 4 * i);
        m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
      }
    }
#endif
  }
  if (pf.buf) {   // state is in registers: the shared buffer may now receive the next stream's rows
    __syncthreads();
    if (threadIdx.x == 0 && pf.next) {
      const unsigned b = (unsigned)__cvta_generic_to_shared(pf.bar), d = (unsigned)__cvta_generic_to_shared(pf.buf);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(pf.bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d), "l"(pf.next), "r"(pf.bytes), "r"(b) : "memory");
    }
  }
  bool prestaged = false;
  [[maybe_unused]] CgCarry carry;
#ifdef H5_PDL_GDN_STATE_OVERLAP
  // Conv has acquired its predecessor before releasing us, and never writes GDN state.
  // Only state reads move ahead of the fence; conv outputs and every store stay behind it.
  // Variant bit 0x10: the first chunk's replayed rows (pending k, v, g, beta of the previous step) are staged
  // into shared memory before the wait as well.
  if constexpr (FUSED && PDL) conv_gdn_pre<ST, CG>(ca, sg, h, half, carry);
  if constexpr (PDL) if (FUSED || (g_variant & 16u)) {
    const auto c0 = replay_chunk(sg.commit, sg.len, 0, 0, GDN_T);
    for (int i = tid; i < c0.replay_rows * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      if (c < 32) continue;
      const ST* pr = prd + (size_t)t * PEND_GDN;
      const float4 v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      *(float4*)(c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4]) = v4;
    }
    if (tid < c0.replay_rows) { const ST* pr = prd + (size_t)tid * PEND_GDN; sg_[tid] = st_ld(pr + KDIM + VDIM + h); sb_[tid] = st_ld(pr + KDIM + VDIM + HV + h); }
    prestaged = true;
  }
  if constexpr (PDL) { pdl_enter_e(a.early); }
#endif
  H5_LIT_MARKX(0);   // phase marks: 0 wait returned, 1 first chunk staged, 2 recurrence done, 3 before final save
  // SCALED (H5_GDN_SCALED=1): m holds M / Gs, Gs the decay accumulated since the last renormalisation (uniform per
  // CTA: one head). The per-token update is then one FMA per element instead of a multiply and an FMA.
  float Gs = 1.f;
  auto save = [&]() {
    if constexpr (PACKED) {
#pragma unroll
      for (int i = 0; i < MC / 4; i++) {
        const Pair p[2] = {mp[2*i], mp[2*i+1]};
        *(uint2*)(M + CG * 4 * i) = *(const uint2*)p;
      }
    } else {
    float mt[MC];
#pragma unroll
    for (int i = 0; i < MC; i++) mt[i] = SCALED ? Gs * m[i] : m[i];
    gdn_store_row<ST, CG, Q8>(M, q8, h, j, cg, mt, a.q8_scale);
    }
  };
  int done = 0, replayed = 0;
  bool first = true;
  while (first || replayed < sg.commit || done < sg.len) {
    const auto chunk = replay_chunk(sg.commit, sg.len, replayed, done, GDN_T);
    const int nrep = chunk.replay_rows, nnew = chunk.new_rows;
    const int T = nrep + nnew;
    const int t_first = prestaged ? nrep : 0;   // replayed rows of the first chunk are already staged
    prestaged = false;
    __syncthreads();
    for (int i = tid + t_first * CH; i < T * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      float4 v4; bool have = true;
      if (t < nrep) {
        const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN;
        if (c < 32) have = false;
        else v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      } else if constexpr (FUSED) {
        have = false;
      } else {
        const float* co = a.conv_out + (size_t)(sg.start + done + t - nrep) * CONV_CH;
        v4 = *(const float4*)(c < 32 ? co + kh * 128 + c * 4 : c < 64 ? co + KDIM + kh * 128 + (c - 32) * 4 : co + 2 * KDIM + h * 128 + JR * half + (c - 64) * 4);
      }
      if (have) {
        float* dst = c < 32 ? &sq[t][c * 4] : c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4];
        *(float4*)dst = v4;
      }
    }
    if (tid >= t_first && tid < T) {
      const int t = tid;
      if (t < nrep) { const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN; sg_[t] = st_ld(pr + KDIM + VDIM + h); sb_[t] = st_ld(pr + KDIM + VDIM + HV + h); }
      else if constexpr (!FUSED) { const float* gr = a.gates + (size_t)(sg.start + done + t - nrep) * 96; sg_[t] = gr[h]; sb_[t] = gr[HV + h]; }
    }
    if constexpr (FUSED && PDL) conv_gdn_stage_b<ST, CG>(ca, sg, h, half, first, done, nrep, nnew, sq, sk, sv, sg_, sb_, carry);
    else if constexpr (FUSED) conv_gdn_stage<ST, CG>(ca, sg, h, half, done, nrep, nnew, sq, sk, sv, sg_, sb_);
    __syncthreads();
    if (first) H5_LIT_MARKX(1);
    if (sg.keep) {   // new speculative rows -> pending[par] (disjoint from the replay buffer)
      for (int i = tid; i < nnew * 128; i += 256) {
        const int t = nrep + (i >> 7), d = i & 127;
        ST* pw = pwr + (size_t)(done + t - nrep) * PEND_GDN;
        if (h < HK && half == 0) st_st(pw + kh * 128 + d, sk[t][d]);
        if (d < JR && !a.replay_delta) st_st(pw + KDIM + h * 128 + JR * half + d, sv[t][d]);
        if (d == 0 && half == 0) { st_st(pw + KDIM + VDIM + h, sg_[t]); st_st(pw + KDIM + VDIM + HV + h, sb_[t]); }
      }
    }
    // Q8 must leave an uncommitted checkpoint byte-identical, not re-quantize it.
    if constexpr (!Q8) if (first && sg.keep && nrep == 0) save();
    if constexpr (PACKED) {
      for (int i = tid; i < T * 64; i += 256) {
        const int t = i / 64, c = (i % 64) * 2;
        pk[t][i % 64] = Ops::pack(sk[t][c], sk[t][c+1]);
        if (t >= nrep) pq[t][i % 64] = Ops::pack(sq[t][c], sq[t][c+1]);
      }
      __syncthreads();
      const Pair zero = Ops::pack(0.f, 0.f);
      for (int t = 0; t < T; t++) {
        const float g = sg_[t], beta = sb_[t];
        float delta;
        if (a.replay_delta && t < nrep) delta = sv[t][jl];
        else {
          Pair s0 = zero, s1 = zero;
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            s0 = Ops::fma(mp[2*i], pk[t][2*cg + CG*2*i], s0);
            s1 = Ops::fma(mp[2*i+1], pk[t][2*cg + CG*2*i+1], s1);
          }
          const float2 f0 = Ops::unpack(s0), f1 = Ops::unpack(s1);
          float s = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
          delta = beta * (sv[t][jl] - g * s);
          if (a.replay_delta && sg.keep && cg == 0)
            st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
        }
        const Pair g2 = Ops::pack(g, g), d2 = Ops::pack(delta, delta);
#pragma unroll
        for (int i = 0; i < MC / 2; i++)
          mp[i] = Ops::fma(d2, pk[t][2*cg + (i/2)*CG*2 + i%2], Ops::mul(g2, mp[i]));
        if (t >= nrep) {
          Pair o0 = zero, o1 = zero;
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            o0 = Ops::fma(mp[2*i], pq[t][2*cg + CG*2*i], o0);
            o1 = Ops::fma(mp[2*i+1], pq[t][2*cg + CG*2*i+1], o1);
          }
          const float2 f0 = Ops::unpack(o0), f1 = Ops::unpack(o1);
          float o = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
          if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
        } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
      }
    } else
    // Variant bit 0x20: closed-form chunk recurrence. With M the chunk's initial state, all dot products with
    // M (a_t = M k_t, p_t = M q_t) and the key Gram tables (KK = k_u.k_t, KQ = k_u.q_t) are independent and
    // computed in parallel; the per-token recurrence is then scalar per state row:
    //   s_t = Gam a_t + sum_{u<t} w_u KK[u][t];  delta_t = beta_t (v_t - g_t s_t);  w_u *= g_t; w_t = delta_t; Gam *= g_t
    //   o_t = Gam p_t + sum_{u<=t} w_u KQ[u][t];  M_end = Gam M + sum_u w_u k_u
    // (w_u = delta_u times the decay since token u, Gam = the decay since the chunk start). Same algebra as the
    // per-token loop below; the summation order differs.
    if (g_variant & 32u) {
      __shared__ float KK[GDN_T][GDN_T], KQ[GDN_T][GDN_T];
      const int warp = tid >> 5;
      for (int pi = warp; pi < 2 * T * T; pi += 8) {
        const bool isq = pi >= T * T;
        const int u = (pi % (T * T)) / T, t = pi % T;
        if (isq ? (u > t || t < nrep) : (u >= t)) continue;
        const float4 x = *(const float4*)&sk[u][lane * 4], y = *(const float4*)(isq ? &sq[t][lane * 4] : &sk[t][lane * 4]);
        float d = fmaf(x.x, y.x, fmaf(x.y, y.y, fmaf(x.z, y.z, x.w * y.w)));
#pragma unroll
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (lane == 0) (isq ? KQ : KK)[u][t] = d;
      }
      float a_[GDN_T], p_[GDN_T];
#pragma unroll
      for (int t = 0; t < GDN_T; t++) {
        a_[t] = 0.f; p_[t] = 0.f;
        if (t < T) {
          float s4[4] = {0.f, 0.f, 0.f, 0.f}, q4s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            const float4 k4 = *(const float4*)(&sk[t][4 * cg] + CG * 4 * i);
            s4[0] = fmaf(m[4*i], k4.x, s4[0]); s4[1] = fmaf(m[4*i+1], k4.y, s4[1]); s4[2] = fmaf(m[4*i+2], k4.z, s4[2]); s4[3] = fmaf(m[4*i+3], k4.w, s4[3]);
            if (t >= nrep) {
              const float4 q4 = *(const float4*)(&sq[t][4 * cg] + CG * 4 * i);
              q4s[0] = fmaf(m[4*i], q4.x, q4s[0]); q4s[1] = fmaf(m[4*i+1], q4.y, q4s[1]); q4s[2] = fmaf(m[4*i+2], q4.z, q4s[2]); q4s[3] = fmaf(m[4*i+3], q4.w, q4s[3]);
            }
          }
          a_[t] = (s4[0] + s4[1]) + (s4[2] + s4[3]); p_[t] = (q4s[0] + q4s[1]) + (q4s[2] + q4s[3]);
        }
      }
#pragma unroll
      for (int sh = 1; sh < CG; sh <<= 1)
#pragma unroll
        for (int t = 0; t < GDN_T; t++) { a_[t] += __shfl_xor_sync(0xffffffff, a_[t], sh); p_[t] += __shfl_xor_sync(0xffffffff, p_[t], sh); }
      __syncthreads();   // KK, KQ complete
      float w[GDN_T], Gam = 1.f;
      auto materialize = [&](int upto, float* out) {   // out = Gam M + sum_{u<=upto} w_u k_u (lane's columns)
#pragma unroll
        for (int i = 0; i < MC; i++) out[i] = Gam * m[i];
#pragma unroll
        for (int u = 0; u < GDN_T; u++) if (u <= upto) {
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            const float4 k4 = *(const float4*)(&sk[u][4 * cg] + CG * 4 * i);
            out[4*i] = fmaf(w[u], k4.x, out[4*i]); out[4*i+1] = fmaf(w[u], k4.y, out[4*i+1]);
            out[4*i+2] = fmaf(w[u], k4.z, out[4*i+2]); out[4*i+3] = fmaf(w[u], k4.w, out[4*i+3]);
          }
        }
      };
#pragma unroll
      for (int t = 0; t < GDN_T; t++) {
        w[t] = 0.f;
        if (t < T) {
          const float g = sg_[t], beta = sb_[t];
          float delta;
          if (a.replay_delta && t < nrep) delta = sv[t][jl];
          else {
            float s0 = Gam * a_[t], s1 = 0.f;
#pragma unroll
            for (int u = 0; u < t; u++) { if (u & 1) s1 = fmaf(w[u], KK[u][t], s1); else s0 = fmaf(w[u], KK[u][t], s0); }
            delta = beta * (sv[t][jl] - g * (s0 + s1));
            if (a.replay_delta && sg.keep && cg == 0)
              st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
          }
#pragma unroll
          for (int u = 0; u < t; u++) w[u] *= g;
          Gam *= g; w[t] = delta;
          if (t >= nrep) {
            float o0 = Gam * p_[t], o1 = 0.f;
#pragma unroll
            for (int u = 0; u <= t; u++) { if (u & 1) o1 = fmaf(w[u], KQ[u][t], o1); else o0 = fmaf(w[u], KQ[u][t], o0); }
            if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = (o0 + o1) * 0.08838834764831845f;
          } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) {
            float mt[MC];
            materialize(t, mt);
            gdn_store_row<ST, CG, Q8>(M, q8, h, j, cg, mt, a.q8_scale);
          }
        }
      }
      { float mt[MC]; materialize(T - 1, mt);
#pragma unroll
        for (int i = 0; i < MC; i++) m[i] = mt[i]; }
    } else if (g_variant & 64u) {
      // Variant bit 0x40: o_t = M_t q_t = g_t (M_{t-1} q_t) + delta_t (k_t . q_t). Both dot products with M_{t-1}
      // are taken together and reduced in the same shuffle rounds, so each token carries one reduction
      // chain instead of two; k_t . q_t is state-independent (one dot per token per CTA, staged once).
      __shared__ float kq_[GDN_T];
      {
        const int warp = tid >> 5;
        for (int t = nrep + warp; t < T; t += 8) {
          const float4 x = *(const float4*)&sk[t][lane * 4], y = *(const float4*)&sq[t][lane * 4];
          float d = fmaf(x.x, y.x, fmaf(x.y, y.y, fmaf(x.z, y.z, x.w * y.w)));
#pragma unroll
          for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
          if (lane == 0) kq_[t] = d;
        }
      }
      __syncthreads();
      for (int t = 0; t < T; t++) {
        const float g = sg_[t], beta = sb_[t];
        const bool out = t >= nrep;
        const float* kp = &sk[t][4 * cg];
        const float* qp = &sq[t][4 * cg];
        float kr[MC];
        float s4[4] = {0.f, 0.f, 0.f, 0.f}, p4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) {
          const float4 k4 = *(const float4*)(kp + CG * 4 * i);
          kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w;
          s4[0] = fmaf(m[4*i], k4.x, s4[0]); s4[1] = fmaf(m[4*i+1], k4.y, s4[1]); s4[2] = fmaf(m[4*i+2], k4.z, s4[2]); s4[3] = fmaf(m[4*i+3], k4.w, s4[3]);
          if (out) {
            const float4 q4 = *(const float4*)(qp + CG * 4 * i);
            p4[0] = fmaf(m[4*i], q4.x, p4[0]); p4[1] = fmaf(m[4*i+1], q4.y, p4[1]); p4[2] = fmaf(m[4*i+2], q4.z, p4[2]); p4[3] = fmaf(m[4*i+3], q4.w, p4[3]);
          }
        }
        float sv_ = (s4[0] + s4[1]) + (s4[2] + s4[3]), pv = (p4[0] + p4[1]) + (p4[2] + p4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) { sv_ += __shfl_xor_sync(0xffffffff, sv_, sh); pv += __shfl_xor_sync(0xffffffff, pv, sh); }
        float delta;
        if (a.replay_delta && t < nrep) delta = sv[t][jl];
        else {
          delta = beta * (sv[t][jl] - g * sv_);
          if (a.replay_delta && sg.keep && cg == 0)
            st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
        }
#pragma unroll
        for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
        if (out) {
          if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = fmaf(g, pv, delta * kq_[t]) * 0.08838834764831845f;
        } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
      }
    } else if constexpr (SCALED) {
      for (int t = 0; t < T; t++) {
        const float g = sg_[t], beta = sb_[t];
        const float* kp = &sk[t][4 * cg];
        float kr[MC];
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
        float delta;
        if (a.replay_delta && t < nrep) delta = sv[t][jl];
        else {
          float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
          float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
          delta = beta * (sv[t][jl] - g * (Gs * s));
          if (a.replay_delta && sg.keep && cg == 0)
            st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
        }
        Gs *= g;
        if (Gs < 1e-18f) {   // uniform: g is per head
#pragma unroll
          for (int i = 0; i < MC; i++) m[i] *= Gs;
          Gs = 1.f;
        }
        const float c = delta / Gs;
#pragma unroll
        for (int i = 0; i < MC; i++) m[i] = fmaf(c, kr[i], m[i]);
        if (t >= nrep) {
          const float* qp = &sq[t][4 * cg];
          float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
          float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
          if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = (Gs * o) * 0.08838834764831845f;
        } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
      }
    } else
#ifdef EXP_GDN_NOCOMP
    if (T > 1000)
#endif
    for (int t = 0; t < T; t++) {
      const float g = sg_[t], beta = sb_[t];
      const float* kp = &sk[t][4 * cg];
      float kr[MC];
#pragma unroll
      for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
      float delta;
      if (a.replay_delta && t < nrep) delta = sv[t][jl];
      else {
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
        float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
        delta = beta * (sv[t][jl] - g * s);
        if (a.replay_delta && sg.keep && cg == 0)
          st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
      }
#pragma unroll
      for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
      if (t >= nrep) {
        const float* qp = &sq[t][4 * cg];
        float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
        float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
        #pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
        if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
      } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
    }
    replayed += nrep; done += nnew; first = false;
  }
  H5_LIT_MARKX(2);
#ifndef EXP_GDN_NOMEM
  if constexpr (!FUSED) if (a.sig) {
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(a.sig) : "memory");
  }
  if (!sg.keep) save();
#endif
}
template <typename ST, int CG = 4, int OCC = 2, bool SCALED = false, bool PACKED = false, bool Q8 = false>
__global__ void __launch_bounds__(256, OCC) k_gdn(GdnArgs a) { H5_LIT(0);
#ifndef H5_PDL_GDN_STATE_OVERLAP
  // Segment metadata is also upstream of conv, but retain the control's entry fence.
  pdl_enter();
#endif
  __shared__ __align__(16) uint8_t sm[GDN_SMEM + (PACKED ? GDN_PACKED_SMEM : 0)];
#ifdef H5_PDL_GDN_STATE_OVERLAP
  gdn_body<ST, CG, false, true, SCALED, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm);
#else
  gdn_body<ST, CG, false, false, SCALED, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm);
#endif
}
// Included in kernels.cu after the shared storage and launch-selection helpers.
template <int CG> struct alignas(16) GdnSingleShared {
  float q[128], k[128], v[256 / CG], gates[2], red[8];
};

template <int CG>
__device__ __forceinline__ void single_conv_stage(const ConvArgs& c, Seg sg, int h, int part,
    GdnSingleShared<CG>& sm, CgCarry carry) {
  constexpr int JR = 256 / CG;
  const int t = threadIdx.x, d = t & 127, kh = h % HK;
  const int channel = (t < 128 ? 0 : KDIM) + kh * 128 + d;
  const size_t layer = (size_t)sg.slot * NGDN + c.gl;
  float* dst = c.state + (2 * layer + (sg.conv_par ^ 1)) * 3 * CONV_CH;
  const float x = c.proj[(size_t)sg.start * GDN_IN + channel];
  const float4 w = carry.wk;
  float v = silu(w.x * carry.s0 + w.y * carry.s1 + w.z * carry.s2 + w.w * x);
  float ss = v * v;
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  if ((t & 31) == 0) sm.red[t >> 5] = ss;
  if (h < HK && part == 0) {
    dst[channel] = carry.s1; dst[CONV_CH + channel] = carry.s2; dst[2 * CONV_CH + channel] = x;
  }
  if (t < JR) {
    const int vc = 2 * KDIM + h * 128 + JR * part + t;
    const float vx = c.proj[(size_t)sg.start * GDN_IN + vc];
    const float4 vw = carry.vw;
    sm.v[t] = silu(vw.x * carry.v0 + vw.y * carry.v1 + vw.z * carry.v2 + vw.w * vx);
    dst[vc] = carry.v1; dst[CONV_CH + vc] = carry.v2; dst[2 * CONV_CH + vc] = vx;
  }
  if (t == 0) {
    const float g = c.ab[(size_t)sg.start * 96 + h] + c.dt_bias[h];
    const float sp = g > 20.f ? g : log1pf(__expf(g));
    sm.gates[0] = __expf(c.ssm_a[h] * sp);
    sm.gates[1] = sigm(c.ab[(size_t)sg.start * 96 + HV + h]);
  }
  __syncthreads();
  const int b = t < 128 ? 0 : 4;
  ss = sm.red[b] + sm.red[b + 1] + sm.red[b + 2] + sm.red[b + 3];
  (t < 128 ? sm.q : sm.k)[d] = v * (1.f / fmaxf(sqrtf(ss), NORM_EPS));
}

template <typename ST, int CG, bool FUSED, bool Q8 = false>
__global__ void __launch_bounds__(256, 2) k_gdn_single(GdnArgs a, ConvArgs c) { H5_LIT_SITE(GdnSingle, 0);
  constexpr int JR = 256 / CG, MC = 128 / CG;
  __shared__ GdnSingleShared<CG> sm;
#ifndef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  const Seg sg = a.segs[blockIdx.y];
  if (sg.len != 1 || sg.commit || sg.keep) { asm volatile("trap;"); return; }
  const int h = blockIdx.x / (CG / 2), part = blockIdx.x % (CG / 2), kh = h % HK;
  const int t = threadIdx.x, cg = t & (CG - 1), jl = t / CG, j = part * JR + jl;
  static_assert(!Q8 || !FUSED);
  ST* M = Q8 ? nullptr : (ST*)a.state + (((size_t)sg.slot * NGDN + a.gl) * HV + h) * 128 * 128 + j * 128 + 4 * cg;
  GdnQ8Layer* q8 = Q8 ? (GdnQ8Layer*)a.state + (size_t)sg.slot * NGDN + a.gl : nullptr;
  float m[MC];
  if constexpr (Q8) gdn_load_q8_row<CG>(q8, h, j, cg, m);
  else {
#pragma unroll
    for (int i = 0; i < MC / 4; ++i) {
      const float4 v = ld4(M + CG * 4 * i);
      m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
    }
  }
  CgCarry carry;
  if constexpr (FUSED) {
    const int channel = (t < 128 ? 0 : KDIM) + kh * 128 + (t & 127);
    const float* src = c.state + (2 * ((size_t)sg.slot * NGDN + c.gl) + sg.conv_par) * 3 * CONV_CH;
    carry.s0 = src[channel]; carry.s1 = src[CONV_CH + channel]; carry.s2 = src[2 * CONV_CH + channel];
    carry.wk = *(const float4*)(c.w + channel * 4);
    if (t < JR) {
      const int vc = 2 * KDIM + h * 128 + part * JR + t;
      carry.v0 = src[vc]; carry.v1 = src[CONV_CH + vc]; carry.v2 = src[2 * CONV_CH + vc];
      carry.vw = *(const float4*)(c.w + vc * 4);
    }
  }
#ifdef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  H5_LIT_MARKX(0);
  if constexpr (FUSED) {
    if (c.pf_placement == h5::PfChainPlacement::Prologue) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
    single_conv_stage<CG>(c, sg, h, part, sm, carry);
  } else {
    const float* co = a.conv_out + (size_t)sg.start * CONV_CH;
    if (t < 32) {
      *(float4*)(sm.q + t * 4) = *(const float4*)(co + kh * 128 + t * 4);
      *(float4*)(sm.k + t * 4) = *(const float4*)(co + KDIM + kh * 128 + t * 4);
    }
    if (t < JR / 4) *(float4*)(sm.v + t * 4) = *(const float4*)(co + 2 * KDIM + h * 128 + part * JR + t * 4);
    if (t == 0) {
      sm.gates[0] = a.gates[(size_t)sg.start * 96 + h];
      sm.gates[1] = a.gates[(size_t)sg.start * 96 + HV + h];
    }
  }
  __syncthreads();
  H5_LIT_MARKX(1);
  float kr[MC], dot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < MC / 4; ++i) {
    const float4 k = *(const float4*)(sm.k + CG * 4 * i + 4 * cg);
    kr[4*i] = k.x; kr[4*i+1] = k.y; kr[4*i+2] = k.z; kr[4*i+3] = k.w;
  }
#pragma unroll
  for (int i = 0; i < MC; ++i) dot[i & 3] = fmaf(m[i], kr[i], dot[i & 3]);
  float s = (dot[0] + dot[1]) + (dot[2] + dot[3]);
#pragma unroll
  for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
  const float g = sm.gates[0], delta = sm.gates[1] * (sm.v[jl] - g * s);
#pragma unroll
  for (int i = 0; i < MC; ++i) m[i] = fmaf(g, m[i], delta * kr[i]);
  float out[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < MC / 4; ++i) {
    const float4 q = *(const float4*)(sm.q + CG * 4 * i + 4 * cg);
    out[0] = fmaf(m[4*i], q.x, out[0]); out[1] = fmaf(m[4*i+1], q.y, out[1]);
    out[2] = fmaf(m[4*i+2], q.z, out[2]); out[3] = fmaf(m[4*i+3], q.w, out[3]);
  }
  float o = (out[0] + out[1]) + (out[2] + out[3]);
#pragma unroll
  for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
  if (cg == 0) a.y[(size_t)sg.start * VDIM + h * 128 + j] = o * 0.08838834764831845f;
  if (a.sig) prep_signal(a.sig);
  H5_LIT_MARKX(2);
  if constexpr (Q8) gdn_store_row<ST, CG, true>(M, q8, h, j, cg, m, a.q8_scale);
  else {
#pragma unroll
    for (int i = 0; i < MC / 4; ++i) {
      if constexpr (sizeof(ST) == 2) {
        uint2 v;
#pragma unroll
        for (int k = 0; k < 4; ++k) st_st((ST*)&v + k, m[4*i+k]);
        *(uint2*)(M + CG * 4 * i) = v;
      } else *(float4*)(M + CG * 4 * i) = make_float4(m[4*i], m[4*i+1], m[4*i+2], m[4*i+3]);
    }
  }
  if constexpr (FUSED) {
    H5_LIT_MARKX(3);
    if (c.pf_placement == h5::PfChainPlacement::Tail) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
  }
}


}
namespace gdn_bits_striped {using namespace h5;
__device__ __forceinline__ float silu(float x) {return x/(1.f+__expf(-x));}
__device__ __forceinline__ float sigm(float x) {return 1.f/(1.f+__expf(-x));}
__device__ __forceinline__ float st_ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void st_st(__half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ float4 ld4(const __half* p) {
  const uint2 u = *(const uint2*)p; const __half2* b = (const __half2*)&u;
  const float2 f0 = __half22float2(b[0]), f1 = __half22float2(b[1]);
  return make_float4(f0.x, f0.y, f1.x, f1.y);
}
__device__ __forceinline__ float st_ld(const float* p) { return *p; }
__device__ __forceinline__ float st_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void st_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ float4 ld4(const float* p) { return *(const float4*)p; }
__device__ __forceinline__ float4 ld4(const __nv_bfloat16* p) {
  const uint2 u = *(const uint2*)p; const __nv_bfloat162* b = (const __nv_bfloat162*)&u;
  const float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]);
  return make_float4(f0.x, f0.y, f1.x, f1.y);
}
// PTX bulk prefetch accepts a uint32 byte size, multiple of 16; 16 KiB is a chosen
// request granularity, not an ISA limit. These are best-effort hints, not completed copies.
__device__ __forceinline__ void chain_l2pf(const uint8_t* p, unsigned bytes, h5::PfChainGranularity granularity) {
  if (!p || !bytes) return;
  const unsigned ncta = gridDim.x * gridDim.y, cta = blockIdx.y * gridDim.x + blockIdx.x;
  const h5::PfChainSlice slice = h5::pf_chain_slice(bytes, ncta, cta);
  if (granularity == h5::PfChainGranularity::CtaShare) {
    if (threadIdx.x == 0 && slice.bytes)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + slice.begin), "r"(slice.bytes) : "memory");
    return;
  }
  const unsigned b1 = slice.begin + slice.bytes;
  for (unsigned o = slice.begin + threadIdx.x * 16384u; o < b1; o += blockDim.x * 16384u) {
    const unsigned n = min(16384u, b1 - o);
    if (n) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + o), "r"(n) : "memory");
  }
}
__device__ __forceinline__ void st_load32(const float* p, float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) { const float4 v = *(const float4*)(p + 16 * i); m[4 * i] = v.x; m[4 * i + 1] = v.y; m[4 * i + 2] = v.z; m[4 * i + 3] = v.w; }
}
__device__ __forceinline__ void st_load32(const __nv_bfloat16* p, float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint2 u = *(const uint2*)(p + 16 * i); const __nv_bfloat162* b = (const __nv_bfloat162*)&u;
    const float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]);
    m[4 * i] = f0.x; m[4 * i + 1] = f0.y; m[4 * i + 2] = f1.x; m[4 * i + 3] = f1.y;
  }
}
__device__ __forceinline__ void st_store32(float* p, const float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) *(float4*)(p + 16 * i) = make_float4(m[4 * i], m[4 * i + 1], m[4 * i + 2], m[4 * i + 3]);
}
__device__ __forceinline__ void st_store32(__nv_bfloat16* p, const float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint2 u; __nv_bfloat162* b = (__nv_bfloat162*)&u;
    b[0] = __floats2bfloat162_rn(m[4 * i], m[4 * i + 1]); b[1] = __floats2bfloat162_rn(m[4 * i + 2], m[4 * i + 3]);
    *(uint2*)(p + 16 * i) = u;
  }
}
__device__ __forceinline__ void st_load32(const __half* p, float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float4 v = ld4(p + 16 * i);
    m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
  }
}
__device__ __forceinline__ void st_store32(__half* p, const float* m) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint2 u; __half2* b = (__half2*)&u;
    b[0] = __floats2half2_rn(m[4*i], m[4*i+1]); b[1] = __floats2half2_rn(m[4*i+2], m[4*i+3]);
    *(uint2*)(p + 16*i) = u;
  }
}
// CG8 signed six-bit carrier: sixteen coordinates -> three aligned words.
// Existing Q8 row/slot allocation is retained, but only96 of128 payload bytes
// are touched. Padding is never interpreted; zero-filled allocation encodes0.
__device__ __forceinline__ unsigned q6_load4(const unsigned* words,int lane4){
 const unsigned w=lane4<3?words[lane4]:0;
 const int bit=24*lane4,word=bit>>5;
 const unsigned lo=__shfl_sync(0xffffffff,w,word,4),hi=__shfl_sync(0xffffffff,w,word+1,4);
 return __funnelshift_r(lo,hi,bit&31)&0xffffffu;
}
__device__ __forceinline__ void q6_store4(unsigned* words,unsigned packed,int lane4){
 const unsigned next=__shfl_down_sync(0xffffffff,packed,1,4);
 if(lane4<3)words[lane4]=(packed>>(8*lane4))|(next<<(24-8*lane4));
}
template<int CG>
__device__ __forceinline__ void gdn_load_q8_row(const GdnQ8Layer*layer,int h,int j,int cg,float*m){
 static_assert(CG==8,"Q6 carrier currently requires CG8");const float scale=layer->scales[h][j];
#pragma unroll
 for(int i=0;i<128/CG/4;i++){
  const auto*p=(const unsigned*)&layer->data[h][j][12*(2*i+cg/4)];const unsigned word=q6_load4(p,cg&3);
#pragma unroll
  for(int k=0;k<4;k++)m[4*i+k]=float(int(word<<(26-6*k))>>26)*scale;
 }
}
template<typename ST,int CG,bool Q8>
__device__ __forceinline__ void gdn_store_row(ST*,GdnQ8Layer*layer,int h,int j,int cg,const float*m,Q8ScaleMode mode){
 static_assert(Q8&&CG==8,"Q6 save currently requires Q8 container and CG8");
 float amax=0;
#pragma unroll
 for(int i=0;i<128/CG;i++)amax=fmaxf(amax,fabsf(m[i]));
#pragma unroll
 for(int sh=1;sh<CG;sh<<=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,sh));
 const float scale=amax*(1.f/31.f),inv=scale>0?1.f/scale:0;float xq[4]={0,0,0,0};int qq=0;
#pragma unroll
 for(int i=0;i<128/CG/4;i++){
  unsigned packed=0,bytes=0;
#pragma unroll
  for(int k=0;k<4;k++){
   const int q=max(-31,min(31,__float2int_rn(m[4*i+k]*inv)));
   packed|=(unsigned(q)&63u)<<(6*k);bytes|=unsigned(uint8_t(int8_t(q)))<<(8*k);
   xq[k]=fmaf(m[4*i+k],float(q),xq[k]);
  }
  qq=__dp4a(int(bytes),int(bytes),qq);
  q6_store4((unsigned*)&layer->data[h][j][12*(2*i+cg/4)],packed,cg&3);
 }
 float dot=(xq[0]+xq[1])+(xq[2]+xq[3]);
#pragma unroll
 for(int sh=1;sh<CG;sh<<=1){dot+=__shfl_xor_sync(0xffffffff,dot,sh);qq+=__shfl_xor_sync(0xffffffff,qq,sh);}
 if(cg==0)layer->scales[h][j]=mode==Q8ScaleMode::LeastSquares?(qq==0?0.f:dot/float(qq)):scale;
}

constexpr int GDN_T = 2 * PMAX;   // max tokens per segment visit (replay + new); longer segments loop in chunks
// Two CTAs per head (64 state rows each, 256 threads). Output is the raw o / sqrt(128); the gated
// RMSNorm runs in the ssm_out prep (P_GDN_NORM).
constexpr int GDN_SMEM = GDN_T * (128 + 128 + 64 + 2) * 4;
#include "state_fusion.cuh"
struct GdnPrefetch { const void* next = nullptr; uint64_t* bar = nullptr; void* buf = nullptr; unsigned bytes = 0; };
template <typename ST> struct GdnPacked { using Pair = float2; };
template <> struct GdnPacked<__half> {
  using Pair = __half2;
  __device__ static Pair pack(float x, float y) { return __floats2half2_rn(x, y); }
  __device__ static float2 unpack(Pair x) { return __half22float2(x); }
  __device__ static Pair mul(Pair x, Pair y) { return __hmul2(x, y); }
  __device__ static Pair fma(Pair x, Pair y, Pair z) { return __hfma2(x, y, z); }
};
template <> struct GdnPacked<__nv_bfloat16> {
  using Pair = __nv_bfloat162;
  __device__ static Pair pack(float x, float y) { return __floats2bfloat162_rn(x, y); }
  __device__ static float2 unpack(Pair x) { return __bfloat1622float2(x); }
  __device__ static Pair mul(Pair x, Pair y) { return __hmul2(x, y); }
  __device__ static Pair fma(Pair x, Pair y, Pair z) { return __hfma2(x, y, z); }
};
constexpr int GDN_PACKED_SMEM = GDN_T * 2 * 128 * 2;
static unsigned gdn_host_variant = 0;
template <typename ST, int CG = 4, bool FUSED = false, bool PDL = false, bool SCALED = false, bool PACKED = false, bool Q8 = false>
__device__ __forceinline__ void gdn_body(const GdnArgs& a, const Seg sg, int bx, uint8_t* smraw, ConvArgs ca = {}, const ST* Mload = nullptr, const GdnPrefetch pf = {}) {
  float (*sk)[128] = (float (*)[128])smraw;
  float (*sq)[128] = (float (*)[128])(smraw + GDN_T * 128 * 4);
  float (*sv)[64] = (float (*)[64])(smraw + GDN_T * 256 * 4);
  float* sg_ = (float*)(smraw + GDN_T * 320 * 4);
  float* sb_ = sg_ + GDN_T;
  constexpr int JR = 256 / CG, MC = 128 / CG, CH = 64 + JR / 4;
  const int h = bx / (CG / 2), half = bx % (CG / 2), kh = h % HK;
  const int tid = threadIdx.x, lane = tid & 31;
  const int jl = tid / CG, j = JR * half + jl, cg = lane & (CG - 1);
  ST* M = Q8 ? nullptr : (ST*)a.state + (((size_t)sg.slot * NGDN + a.gl) * HV + h) * 128 * 128 + j * 128 + 4 * cg;
  GdnQ8Layer* q8 = Q8 ? (GdnQ8Layer*)a.state + (size_t)sg.slot * NGDN + a.gl : nullptr;
  static_assert(!Q8 || (!PACKED && !SCALED && sizeof(ST) == 2));
  ST* pbase = (ST*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * 2 * a.pending_rows * PEND_GDN;
  const ST* prd = pbase + (size_t)(sg.par ^ 1) * a.pending_rows * PEND_GDN;
  ST* pwr = pbase + (size_t)sg.par * a.pending_rows * PEND_GDN;
  static_assert(!PACKED || (sizeof(ST) == 2 && !SCALED));
  using Ops = GdnPacked<ST>;
  using Pair = typename Ops::Pair;
  Pair mp[MC / 2];
  Pair (*pk)[64] = (Pair (*)[64])(smraw + GDN_SMEM);
  Pair (*pq)[64] = pk + GDN_T;
  float m[MC];
  // Mload / pf (k_gdn_pipe): this CTA's state rows were bulk-copied into shared memory ahead of time.
  const ST* const ML = Mload ? Mload + jl * 128 + 4 * cg : M;
  if constexpr (PACKED) {
#pragma unroll
    for (int i = 0; i < MC / 2; i++) {
#ifdef EXP_GDN_NOMEM
      mp[i] = Ops::pack(0.f, 0.f);
#else
      mp[i] = *(const Pair*)(ML + (i / 2) * CG * 4 + (i % 2) * 2);
#endif
    }
  } else {
#ifdef EXP_GDN_NOMEM
#pragma unroll
    for (int i = 0; i < MC; i++) m[i] = 0.f;
#else
    if constexpr (Q8) gdn_load_q8_row<CG>(q8, h, j, cg, m);
    else {
#pragma unroll
      for (int i = 0; i < MC / 4; i++) {
        const float4 v = ld4(ML + CG * 4 * i);
        m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
      }
    }
#endif
  }
  if (pf.buf) {   // state is in registers: the shared buffer may now receive the next stream's rows
    __syncthreads();
    if (threadIdx.x == 0 && pf.next) {
      const unsigned b = (unsigned)__cvta_generic_to_shared(pf.bar), d = (unsigned)__cvta_generic_to_shared(pf.buf);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(pf.bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d), "l"(pf.next), "r"(pf.bytes), "r"(b) : "memory");
    }
  }
  bool prestaged = false;
  [[maybe_unused]] CgCarry carry;
#ifdef H5_PDL_GDN_STATE_OVERLAP
  // Conv has acquired its predecessor before releasing us, and never writes GDN state.
  // Only state reads move ahead of the fence; conv outputs and every store stay behind it.
  // Variant bit 0x10: the first chunk's replayed rows (pending k, v, g, beta of the previous step) are staged
  // into shared memory before the wait as well.
  if constexpr (FUSED && PDL) conv_gdn_pre<ST, CG>(ca, sg, h, half, carry);
  if constexpr (PDL) if (FUSED || (g_variant & 16u)) {
    const auto c0 = replay_chunk(sg.commit, sg.len, 0, 0, GDN_T);
    for (int i = tid; i < c0.replay_rows * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      if (c < 32) continue;
      const ST* pr = prd + (size_t)t * PEND_GDN;
      const float4 v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      *(float4*)(c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4]) = v4;
    }
    if (tid < c0.replay_rows) { const ST* pr = prd + (size_t)tid * PEND_GDN; sg_[tid] = st_ld(pr + KDIM + VDIM + h); sb_[tid] = st_ld(pr + KDIM + VDIM + HV + h); }
    prestaged = true;
  }
  if constexpr (PDL) { pdl_enter_e(a.early); }
#endif
  H5_LIT_MARKX(0);   // phase marks: 0 wait returned, 1 first chunk staged, 2 recurrence done, 3 before final save
  // SCALED (H5_GDN_SCALED=1): m holds M / Gs, Gs the decay accumulated since the last renormalisation (uniform per
  // CTA: one head). The per-token update is then one FMA per element instead of a multiply and an FMA.
  float Gs = 1.f;
  auto save = [&]() {
    if constexpr (PACKED) {
#pragma unroll
      for (int i = 0; i < MC / 4; i++) {
        const Pair p[2] = {mp[2*i], mp[2*i+1]};
        *(uint2*)(M + CG * 4 * i) = *(const uint2*)p;
      }
    } else {
    float mt[MC];
#pragma unroll
    for (int i = 0; i < MC; i++) mt[i] = SCALED ? Gs * m[i] : m[i];
    gdn_store_row<ST, CG, Q8>(M, q8, h, j, cg, mt, a.q8_scale);
    }
  };
  int done = 0, replayed = 0;
  bool first = true;
  while (first || replayed < sg.commit || done < sg.len) {
    const auto chunk = replay_chunk(sg.commit, sg.len, replayed, done, GDN_T);
    const int nrep = chunk.replay_rows, nnew = chunk.new_rows;
    const int T = nrep + nnew;
    const int t_first = prestaged ? nrep : 0;   // replayed rows of the first chunk are already staged
    prestaged = false;
    __syncthreads();
    for (int i = tid + t_first * CH; i < T * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      float4 v4; bool have = true;
      if (t < nrep) {
        const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN;
        if (c < 32) have = false;
        else v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      } else if constexpr (FUSED) {
        have = false;
      } else {
        const float* co = a.conv_out + (size_t)(sg.start + done + t - nrep) * CONV_CH;
        v4 = *(const float4*)(c < 32 ? co + kh * 128 + c * 4 : c < 64 ? co + KDIM + kh * 128 + (c - 32) * 4 : co + 2 * KDIM + h * 128 + JR * half + (c - 64) * 4);
      }
      if (have) {
        float* dst = c < 32 ? &sq[t][c * 4] : c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4];
        *(float4*)dst = v4;
      }
    }
    if (tid >= t_first && tid < T) {
      const int t = tid;
      if (t < nrep) { const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN; sg_[t] = st_ld(pr + KDIM + VDIM + h); sb_[t] = st_ld(pr + KDIM + VDIM + HV + h); }
      else if constexpr (!FUSED) { const float* gr = a.gates + (size_t)(sg.start + done + t - nrep) * 96; sg_[t] = gr[h]; sb_[t] = gr[HV + h]; }
    }
    if constexpr (FUSED && PDL) conv_gdn_stage_b<ST, CG>(ca, sg, h, half, first, done, nrep, nnew, sq, sk, sv, sg_, sb_, carry);
    else if constexpr (FUSED) conv_gdn_stage<ST, CG>(ca, sg, h, half, done, nrep, nnew, sq, sk, sv, sg_, sb_);
    __syncthreads();
    if (first) H5_LIT_MARKX(1);
    if (sg.keep) {   // new speculative rows -> pending[par] (disjoint from the replay buffer)
      for (int i = tid; i < nnew * 128; i += 256) {
        const int t = nrep + (i >> 7), d = i & 127;
        ST* pw = pwr + (size_t)(done + t - nrep) * PEND_GDN;
        if (h < HK && half == 0) st_st(pw + kh * 128 + d, sk[t][d]);
        if (d < JR && !a.replay_delta) st_st(pw + KDIM + h * 128 + JR * half + d, sv[t][d]);
        if (d == 0 && half == 0) { st_st(pw + KDIM + VDIM + h, sg_[t]); st_st(pw + KDIM + VDIM + HV + h, sb_[t]); }
      }
    }
    // Q8 must leave an uncommitted checkpoint byte-identical, not re-quantize it.
    if constexpr (!Q8) if (first && sg.keep && nrep == 0) save();
    if constexpr (PACKED) {
      for (int i = tid; i < T * 64; i += 256) {
        const int t = i / 64, c = (i % 64) * 2;
        pk[t][i % 64] = Ops::pack(sk[t][c], sk[t][c+1]);
        if (t >= nrep) pq[t][i % 64] = Ops::pack(sq[t][c], sq[t][c+1]);
      }
      __syncthreads();
      const Pair zero = Ops::pack(0.f, 0.f);
      for (int t = 0; t < T; t++) {
        const float g = sg_[t], beta = sb_[t];
        float delta;
        if (a.replay_delta && t < nrep) delta = sv[t][jl];
        else {
          Pair s0 = zero, s1 = zero;
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            s0 = Ops::fma(mp[2*i], pk[t][2*cg + CG*2*i], s0);
            s1 = Ops::fma(mp[2*i+1], pk[t][2*cg + CG*2*i+1], s1);
          }
          const float2 f0 = Ops::unpack(s0), f1 = Ops::unpack(s1);
          float s = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
          delta = beta * (sv[t][jl] - g * s);
          if (a.replay_delta && sg.keep && cg == 0)
            st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
        }
        const Pair g2 = Ops::pack(g, g), d2 = Ops::pack(delta, delta);
#pragma unroll
        for (int i = 0; i < MC / 2; i++)
          mp[i] = Ops::fma(d2, pk[t][2*cg + (i/2)*CG*2 + i%2], Ops::mul(g2, mp[i]));
        if (t >= nrep) {
          Pair o0 = zero, o1 = zero;
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            o0 = Ops::fma(mp[2*i], pq[t][2*cg + CG*2*i], o0);
            o1 = Ops::fma(mp[2*i+1], pq[t][2*cg + CG*2*i+1], o1);
          }
          const float2 f0 = Ops::unpack(o0), f1 = Ops::unpack(o1);
          float o = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
          if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
        } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
      }
    } else
    // Variant bit 0x20: closed-form chunk recurrence. With M the chunk's initial state, all dot products with
    // M (a_t = M k_t, p_t = M q_t) and the key Gram tables (KK = k_u.k_t, KQ = k_u.q_t) are independent and
    // computed in parallel; the per-token recurrence is then scalar per state row:
    //   s_t = Gam a_t + sum_{u<t} w_u KK[u][t];  delta_t = beta_t (v_t - g_t s_t);  w_u *= g_t; w_t = delta_t; Gam *= g_t
    //   o_t = Gam p_t + sum_{u<=t} w_u KQ[u][t];  M_end = Gam M + sum_u w_u k_u
    // (w_u = delta_u times the decay since token u, Gam = the decay since the chunk start). Same algebra as the
    // per-token loop below; the summation order differs.
    if (g_variant & 32u) {
      __shared__ float KK[GDN_T][GDN_T], KQ[GDN_T][GDN_T];
      const int warp = tid >> 5;
      for (int pi = warp; pi < 2 * T * T; pi += 8) {
        const bool isq = pi >= T * T;
        const int u = (pi % (T * T)) / T, t = pi % T;
        if (isq ? (u > t || t < nrep) : (u >= t)) continue;
        const float4 x = *(const float4*)&sk[u][lane * 4], y = *(const float4*)(isq ? &sq[t][lane * 4] : &sk[t][lane * 4]);
        float d = fmaf(x.x, y.x, fmaf(x.y, y.y, fmaf(x.z, y.z, x.w * y.w)));
#pragma unroll
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
        if (lane == 0) (isq ? KQ : KK)[u][t] = d;
      }
      float a_[GDN_T], p_[GDN_T];
#pragma unroll
      for (int t = 0; t < GDN_T; t++) {
        a_[t] = 0.f; p_[t] = 0.f;
        if (t < T) {
          float s4[4] = {0.f, 0.f, 0.f, 0.f}, q4s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            const float4 k4 = *(const float4*)(&sk[t][4 * cg] + CG * 4 * i);
            s4[0] = fmaf(m[4*i], k4.x, s4[0]); s4[1] = fmaf(m[4*i+1], k4.y, s4[1]); s4[2] = fmaf(m[4*i+2], k4.z, s4[2]); s4[3] = fmaf(m[4*i+3], k4.w, s4[3]);
            if (t >= nrep) {
              const float4 q4 = *(const float4*)(&sq[t][4 * cg] + CG * 4 * i);
              q4s[0] = fmaf(m[4*i], q4.x, q4s[0]); q4s[1] = fmaf(m[4*i+1], q4.y, q4s[1]); q4s[2] = fmaf(m[4*i+2], q4.z, q4s[2]); q4s[3] = fmaf(m[4*i+3], q4.w, q4s[3]);
            }
          }
          a_[t] = (s4[0] + s4[1]) + (s4[2] + s4[3]); p_[t] = (q4s[0] + q4s[1]) + (q4s[2] + q4s[3]);
        }
      }
#pragma unroll
      for (int sh = 1; sh < CG; sh <<= 1)
#pragma unroll
        for (int t = 0; t < GDN_T; t++) { a_[t] += __shfl_xor_sync(0xffffffff, a_[t], sh); p_[t] += __shfl_xor_sync(0xffffffff, p_[t], sh); }
      __syncthreads();   // KK, KQ complete
      float w[GDN_T], Gam = 1.f;
      auto materialize = [&](int upto, float* out) {   // out = Gam M + sum_{u<=upto} w_u k_u (lane's columns)
#pragma unroll
        for (int i = 0; i < MC; i++) out[i] = Gam * m[i];
#pragma unroll
        for (int u = 0; u < GDN_T; u++) if (u <= upto) {
#pragma unroll
          for (int i = 0; i < MC / 4; i++) {
            const float4 k4 = *(const float4*)(&sk[u][4 * cg] + CG * 4 * i);
            out[4*i] = fmaf(w[u], k4.x, out[4*i]); out[4*i+1] = fmaf(w[u], k4.y, out[4*i+1]);
            out[4*i+2] = fmaf(w[u], k4.z, out[4*i+2]); out[4*i+3] = fmaf(w[u], k4.w, out[4*i+3]);
          }
        }
      };
#pragma unroll
      for (int t = 0; t < GDN_T; t++) {
        w[t] = 0.f;
        if (t < T) {
          const float g = sg_[t], beta = sb_[t];
          float delta;
          if (a.replay_delta && t < nrep) delta = sv[t][jl];
          else {
            float s0 = Gam * a_[t], s1 = 0.f;
#pragma unroll
            for (int u = 0; u < t; u++) { if (u & 1) s1 = fmaf(w[u], KK[u][t], s1); else s0 = fmaf(w[u], KK[u][t], s0); }
            delta = beta * (sv[t][jl] - g * (s0 + s1));
            if (a.replay_delta && sg.keep && cg == 0)
              st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
          }
#pragma unroll
          for (int u = 0; u < t; u++) w[u] *= g;
          Gam *= g; w[t] = delta;
          if (t >= nrep) {
            float o0 = Gam * p_[t], o1 = 0.f;
#pragma unroll
            for (int u = 0; u <= t; u++) { if (u & 1) o1 = fmaf(w[u], KQ[u][t], o1); else o0 = fmaf(w[u], KQ[u][t], o0); }
            if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = (o0 + o1) * 0.08838834764831845f;
          } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) {
            float mt[MC];
            materialize(t, mt);
            gdn_store_row<ST, CG, Q8>(M, q8, h, j, cg, mt, a.q8_scale);
          }
        }
      }
      { float mt[MC]; materialize(T - 1, mt);
#pragma unroll
        for (int i = 0; i < MC; i++) m[i] = mt[i]; }
    } else if (g_variant & 64u) {
      // Variant bit 0x40: o_t = M_t q_t = g_t (M_{t-1} q_t) + delta_t (k_t . q_t). Both dot products with M_{t-1}
      // are taken together and reduced in the same shuffle rounds, so each token carries one reduction
      // chain instead of two; k_t . q_t is state-independent (one dot per token per CTA, staged once).
      __shared__ float kq_[GDN_T];
      {
        const int warp = tid >> 5;
        for (int t = nrep + warp; t < T; t += 8) {
          const float4 x = *(const float4*)&sk[t][lane * 4], y = *(const float4*)&sq[t][lane * 4];
          float d = fmaf(x.x, y.x, fmaf(x.y, y.y, fmaf(x.z, y.z, x.w * y.w)));
#pragma unroll
          for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
          if (lane == 0) kq_[t] = d;
        }
      }
      __syncthreads();
      for (int t = 0; t < T; t++) {
        const float g = sg_[t], beta = sb_[t];
        const bool out = t >= nrep;
        const float* kp = &sk[t][4 * cg];
        const float* qp = &sq[t][4 * cg];
        float kr[MC];
        float s4[4] = {0.f, 0.f, 0.f, 0.f}, p4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) {
          const float4 k4 = *(const float4*)(kp + CG * 4 * i);
          kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w;
          s4[0] = fmaf(m[4*i], k4.x, s4[0]); s4[1] = fmaf(m[4*i+1], k4.y, s4[1]); s4[2] = fmaf(m[4*i+2], k4.z, s4[2]); s4[3] = fmaf(m[4*i+3], k4.w, s4[3]);
          if (out) {
            const float4 q4 = *(const float4*)(qp + CG * 4 * i);
            p4[0] = fmaf(m[4*i], q4.x, p4[0]); p4[1] = fmaf(m[4*i+1], q4.y, p4[1]); p4[2] = fmaf(m[4*i+2], q4.z, p4[2]); p4[3] = fmaf(m[4*i+3], q4.w, p4[3]);
          }
        }
        float sv_ = (s4[0] + s4[1]) + (s4[2] + s4[3]), pv = (p4[0] + p4[1]) + (p4[2] + p4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) { sv_ += __shfl_xor_sync(0xffffffff, sv_, sh); pv += __shfl_xor_sync(0xffffffff, pv, sh); }
        float delta;
        if (a.replay_delta && t < nrep) delta = sv[t][jl];
        else {
          delta = beta * (sv[t][jl] - g * sv_);
          if (a.replay_delta && sg.keep && cg == 0)
            st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
        }
#pragma unroll
        for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
        if (out) {
          if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = fmaf(g, pv, delta * kq_[t]) * 0.08838834764831845f;
        } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
      }
    } else if constexpr (SCALED) {
      for (int t = 0; t < T; t++) {
        const float g = sg_[t], beta = sb_[t];
        const float* kp = &sk[t][4 * cg];
        float kr[MC];
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
        float delta;
        if (a.replay_delta && t < nrep) delta = sv[t][jl];
        else {
          float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
          float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
          delta = beta * (sv[t][jl] - g * (Gs * s));
          if (a.replay_delta && sg.keep && cg == 0)
            st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
        }
        Gs *= g;
        if (Gs < 1e-18f) {   // uniform: g is per head
#pragma unroll
          for (int i = 0; i < MC; i++) m[i] *= Gs;
          Gs = 1.f;
        }
        const float c = delta / Gs;
#pragma unroll
        for (int i = 0; i < MC; i++) m[i] = fmaf(c, kr[i], m[i]);
        if (t >= nrep) {
          const float* qp = &sq[t][4 * cg];
          float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
          float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
#pragma unroll
          for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
          if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = (Gs * o) * 0.08838834764831845f;
        } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
      }
    } else
#ifdef EXP_GDN_NOCOMP
    if (T > 1000)
#endif
    for (int t = 0; t < T; t++) {
      const float g = sg_[t], beta = sb_[t];
      const float* kp = &sk[t][4 * cg];
      float kr[MC];
#pragma unroll
      for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
      float delta;
      if (a.replay_delta && t < nrep) delta = sv[t][jl];
      else {
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
        float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
        delta = beta * (sv[t][jl] - g * s);
        if (a.replay_delta && sg.keep && cg == 0)
          st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
      }
#pragma unroll
      for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
      if (t >= nrep) {
        const float* qp = &sq[t][4 * cg];
        float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
        float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
        #pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
        if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
      } else if (t == nrep - 1 && replayed + nrep == sg.commit && sg.keep) save();
    }
    replayed += nrep; done += nnew; first = false;
  }
  H5_LIT_MARKX(2);
#ifndef EXP_GDN_NOMEM
  if constexpr (!FUSED) if (a.sig) {
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(a.sig) : "memory");
  }
  if (!sg.keep) save();
#endif
}
template <typename ST, int CG = 4, int OCC = 2, bool SCALED = false, bool PACKED = false, bool Q8 = false>
__global__ void __launch_bounds__(256, OCC) k_gdn(GdnArgs a) { H5_LIT(0);
#ifndef H5_PDL_GDN_STATE_OVERLAP
  // Segment metadata is also upstream of conv, but retain the control's entry fence.
  pdl_enter();
#endif
  __shared__ __align__(16) uint8_t sm[GDN_SMEM + (PACKED ? GDN_PACKED_SMEM : 0)];
#ifdef H5_PDL_GDN_STATE_OVERLAP
  gdn_body<ST, CG, false, true, SCALED, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm);
#else
  gdn_body<ST, CG, false, false, SCALED, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm);
#endif
}
// Included in kernels.cu after the shared storage and launch-selection helpers.
template <int CG> struct alignas(16) GdnSingleShared {
  float q[128], k[128], v[256 / CG], gates[2], red[8];
};

template <int CG>
__device__ __forceinline__ void single_conv_stage(const ConvArgs& c, Seg sg, int h, int part,
    GdnSingleShared<CG>& sm, CgCarry carry) {
  constexpr int JR = 256 / CG;
  const int t = threadIdx.x, d = t & 127, kh = h % HK;
  const int channel = (t < 128 ? 0 : KDIM) + kh * 128 + d;
  const size_t layer = (size_t)sg.slot * NGDN + c.gl;
  float* dst = c.state + (2 * layer + (sg.conv_par ^ 1)) * 3 * CONV_CH;
  const float x = c.proj[(size_t)sg.start * GDN_IN + channel];
  const float4 w = carry.wk;
  float v = silu(w.x * carry.s0 + w.y * carry.s1 + w.z * carry.s2 + w.w * x);
  float ss = v * v;
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  if ((t & 31) == 0) sm.red[t >> 5] = ss;
  if (h < HK && part == 0) {
    dst[channel] = carry.s1; dst[CONV_CH + channel] = carry.s2; dst[2 * CONV_CH + channel] = x;
  }
  if (t < JR) {
    const int vc = 2 * KDIM + h * 128 + JR * part + t;
    const float vx = c.proj[(size_t)sg.start * GDN_IN + vc];
    const float4 vw = carry.vw;
    sm.v[t] = silu(vw.x * carry.v0 + vw.y * carry.v1 + vw.z * carry.v2 + vw.w * vx);
    dst[vc] = carry.v1; dst[CONV_CH + vc] = carry.v2; dst[2 * CONV_CH + vc] = vx;
  }
  if (t == 0) {
    const float g = c.ab[(size_t)sg.start * 96 + h] + c.dt_bias[h];
    const float sp = g > 20.f ? g : log1pf(__expf(g));
    sm.gates[0] = __expf(c.ssm_a[h] * sp);
    sm.gates[1] = sigm(c.ab[(size_t)sg.start * 96 + HV + h]);
  }
  __syncthreads();
  const int b = t < 128 ? 0 : 4;
  ss = sm.red[b] + sm.red[b + 1] + sm.red[b + 2] + sm.red[b + 3];
  (t < 128 ? sm.q : sm.k)[d] = v * (1.f / fmaxf(sqrtf(ss), NORM_EPS));
}

template <typename ST, int CG, bool FUSED, bool Q8 = false>
__global__ void __launch_bounds__(256, 2) k_gdn_single(GdnArgs a, ConvArgs c) { H5_LIT_SITE(GdnSingle, 0);
  constexpr int JR = 256 / CG, MC = 128 / CG;
  __shared__ GdnSingleShared<CG> sm;
#ifndef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  const Seg sg = a.segs[blockIdx.y];
  if (sg.len != 1 || sg.commit || sg.keep) { asm volatile("trap;"); return; }
  const int h = blockIdx.x / (CG / 2), part = blockIdx.x % (CG / 2), kh = h % HK;
  const int t = threadIdx.x, cg = t & (CG - 1), jl = t / CG, j = part * JR + jl;
  static_assert(!Q8 || !FUSED);
  ST* M = Q8 ? nullptr : (ST*)a.state + (((size_t)sg.slot * NGDN + a.gl) * HV + h) * 128 * 128 + j * 128 + 4 * cg;
  GdnQ8Layer* q8 = Q8 ? (GdnQ8Layer*)a.state + (size_t)sg.slot * NGDN + a.gl : nullptr;
  float m[MC];
  if constexpr (Q8) gdn_load_q8_row<CG>(q8, h, j, cg, m);
  else {
#pragma unroll
    for (int i = 0; i < MC / 4; ++i) {
      const float4 v = ld4(M + CG * 4 * i);
      m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
    }
  }
  CgCarry carry;
  if constexpr (FUSED) {
    const int channel = (t < 128 ? 0 : KDIM) + kh * 128 + (t & 127);
    const float* src = c.state + (2 * ((size_t)sg.slot * NGDN + c.gl) + sg.conv_par) * 3 * CONV_CH;
    carry.s0 = src[channel]; carry.s1 = src[CONV_CH + channel]; carry.s2 = src[2 * CONV_CH + channel];
    carry.wk = *(const float4*)(c.w + channel * 4);
    if (t < JR) {
      const int vc = 2 * KDIM + h * 128 + part * JR + t;
      carry.v0 = src[vc]; carry.v1 = src[CONV_CH + vc]; carry.v2 = src[2 * CONV_CH + vc];
      carry.vw = *(const float4*)(c.w + vc * 4);
    }
  }
#ifdef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  H5_LIT_MARKX(0);
  if constexpr (FUSED) {
    if (c.pf_placement == h5::PfChainPlacement::Prologue) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
    single_conv_stage<CG>(c, sg, h, part, sm, carry);
  } else {
    const float* co = a.conv_out + (size_t)sg.start * CONV_CH;
    if (t < 32) {
      *(float4*)(sm.q + t * 4) = *(const float4*)(co + kh * 128 + t * 4);
      *(float4*)(sm.k + t * 4) = *(const float4*)(co + KDIM + kh * 128 + t * 4);
    }
    if (t < JR / 4) *(float4*)(sm.v + t * 4) = *(const float4*)(co + 2 * KDIM + h * 128 + part * JR + t * 4);
    if (t == 0) {
      sm.gates[0] = a.gates[(size_t)sg.start * 96 + h];
      sm.gates[1] = a.gates[(size_t)sg.start * 96 + HV + h];
    }
  }
  __syncthreads();
  H5_LIT_MARKX(1);
  float kr[MC], dot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < MC / 4; ++i) {
    const float4 k = *(const float4*)(sm.k + CG * 4 * i + 4 * cg);
    kr[4*i] = k.x; kr[4*i+1] = k.y; kr[4*i+2] = k.z; kr[4*i+3] = k.w;
  }
#pragma unroll
  for (int i = 0; i < MC; ++i) dot[i & 3] = fmaf(m[i], kr[i], dot[i & 3]);
  float s = (dot[0] + dot[1]) + (dot[2] + dot[3]);
#pragma unroll
  for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
  const float g = sm.gates[0], delta = sm.gates[1] * (sm.v[jl] - g * s);
#pragma unroll
  for (int i = 0; i < MC; ++i) m[i] = fmaf(g, m[i], delta * kr[i]);
  float out[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < MC / 4; ++i) {
    const float4 q = *(const float4*)(sm.q + CG * 4 * i + 4 * cg);
    out[0] = fmaf(m[4*i], q.x, out[0]); out[1] = fmaf(m[4*i+1], q.y, out[1]);
    out[2] = fmaf(m[4*i+2], q.z, out[2]); out[3] = fmaf(m[4*i+3], q.w, out[3]);
  }
  float o = (out[0] + out[1]) + (out[2] + out[3]);
#pragma unroll
  for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
  if (cg == 0) a.y[(size_t)sg.start * VDIM + h * 128 + j] = o * 0.08838834764831845f;
  if (a.sig) prep_signal(a.sig);
  H5_LIT_MARKX(2);
  if constexpr (Q8) gdn_store_row<ST, CG, true>(M, q8, h, j, cg, m, a.q8_scale);
  else {
#pragma unroll
    for (int i = 0; i < MC / 4; ++i) {
      if constexpr (sizeof(ST) == 2) {
        uint2 v;
#pragma unroll
        for (int k = 0; k < 4; ++k) st_st((ST*)&v + k, m[4*i+k]);
        *(uint2*)(M + CG * 4 * i) = v;
      } else *(float4*)(M + CG * 4 * i) = make_float4(m[4*i], m[4*i+1], m[4*i+2], m[4*i+3]);
    }
  }
  if constexpr (FUSED) {
    H5_LIT_MARKX(3);
    if (c.pf_placement == h5::PfChainPlacement::Tail) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
  }
}


}
extern "C" void gdn_bits_launch(const h5::GdnArgs&a,int n,int mode,bool single,cudaStream_t stream){
 if(mode==0){if(single)h5::pdl(gdn_bits_0::k_gdn_single<__half,8,false,true>,dim3(4*h5::HV,n),dim3(256),0,stream,a,h5::ConvArgs{});else h5::pdl(gdn_bits_0::k_gdn<__half,8,2,false,false,true>,dim3(4*h5::HV,n),dim3(256),0,stream,a);}
 else if(mode==1){if(single)h5::pdl(gdn_bits_1::k_gdn_single<__half,8,false,true>,dim3(4*h5::HV,n),dim3(256),0,stream,a,h5::ConvArgs{});else h5::pdl(gdn_bits_1::k_gdn<__half,8,2,false,false,true>,dim3(4*h5::HV,n),dim3(256),0,stream,a);}
 else throw std::runtime_error("invalid GDN bits mode");
}
extern "C" void gdn_bits_resources(int mode,bool single){cudaFuncAttributes a{};int active=0;
 const void*k=single?(mode?(const void*)gdn_bits_1::k_gdn_single<__half,8,false,true>:(const void*)gdn_bits_0::k_gdn_single<__half,8,false,true>):(mode?(const void*)gdn_bits_1::k_gdn<__half,8,2,false,false,true>:(const void*)gdn_bits_0::k_gdn<__half,8,2,false,false,true>);
 cudaFuncGetAttributes(&a,k);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,k,256,0);
 printf("{\"kind\":\"resource\",\"bits\":%d,\"single\":%d,\"registers\":%d,\"shared_bytes\":%zu,\"local_bytes\":%zu,\"ctas_per_sm\":%d}\n",mode?6:8,single,a.numRegs,a.sharedSizeBytes,a.localSizeBytes,active);
}
__global__ void q6_pack_probe(const unsigned*input,unsigned*packed,unsigned*decoded,int groups){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=groups)return;
 unsigned a=input[4*i],b=input[4*i+1],c=input[4*i+2],d=input[4*i+3];
 unsigned w0=a|(b<<24),w1=(b>>8)|(c<<16),w2=(c>>16)|(d<<8);
 packed[3*i]=w0;packed[3*i+1]=w1;packed[3*i+2]=w2;
 decoded[4*i]=w0&0xffffff;decoded[4*i+1]=__funnelshift_r(w0,w1,24)&0xffffff;
 decoded[4*i+2]=__funnelshift_r(w1,w2,16)&0xffffff;decoded[4*i+3]=w2>>8;
}
extern "C" void gdn_bits_pack(const unsigned*i,unsigned*p,unsigned*d,int groups,cudaStream_t s){q6_pack_probe<<<(groups*4+255)/256,256,0,s>>>(i,p,d,groups);}

extern "C" void gdn_striped_launch(const h5::GdnArgs&a,int n,bool single,cudaStream_t stream){
 if(single)h5::pdl(gdn_bits_striped::k_gdn_single<__half,8,false,true>,dim3(4*h5::HV,n),dim3(256),0,stream,a,h5::ConvArgs{});
 else h5::pdl(gdn_bits_striped::k_gdn<__half,8,2,false,false,true>,dim3(4*h5::HV,n),dim3(256),0,stream,a);
}
