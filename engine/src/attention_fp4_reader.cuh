#pragma once
#include <cuda_fp4.h>
// Tensor-core split-KV decode attention for blocks of consecutive rows of one slot.
//
// A query block (QBlk) is up to QB_ROWS consecutive rows of one slot; with GQA 6:1 that is up to
// 48 query vectors per KV head, so the KV stream of a block is read once for all its rows.
// Grid (qblock, kv head, split). Each CTA walks its key range in 32-key tiles (double-buffered
// cp.async), S = Q K^T and O += P V on mma.m16n8k16 (fp16 in, fp32 accumulate), online softmax.
// Warp w computes S for keys 8w..8w+7 of the tile and owns O dims 64w..64w+63.
// Partials (unnormalised O, m, l) are merged by k_attn_combine, which also applies the gate.
#include "kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdlib>

namespace p12kv4tc {

using namespace h5;
constexpr int AT_KT = 32;                  // keys per tile
constexpr int AT_PLD = AT_KT + 8;          // padded P row (80 B)
template <int HDT, int GT> struct AtG {
  static constexpr int QM = ((QB_ROWS * GT + 15) / 16) * 16;   // query vectors per block (48 target, 32 drafter)
  static constexpr int LD = HDT + 8;                           // padded fp16 row
  static constexpr int SMEM16 = (QM * LD + 4 * AT_KT * LD + QM * AT_PLD) * 2 + 2 * 4 * QM * 4;
  static constexpr int STG8 = 2 * AT_KT * (HDT / 2) + 2 * AT_KT * (HDT / 16);  // one fp8 stage: K, V bytes + K, V scales
  static constexpr int SMEM8 = (QM * LD + 2 * AT_KT * LD + QM * AT_PLD) * 2 + 2 * 4 * QM * 4 + 2 * STG8;
};

__device__ __forceinline__ void a_cp16(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void a_ldsm4(unsigned* r, const void* p) {
  unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void a_ldsm4t(unsigned* r, const void* p) {
  unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void attn_output_pair(const AttnTcArgs& a, float* dst, int row,
    int hq, int d, float x, float y, float sum) {
  if (a.direct) {
    x /= sum; y /= sum;
    if (a.proj) {
      const float* gate = a.proj + (size_t)row * ATT_IN + hq * 2 * h5::HD + h5::HD + d;
      x /= 1.f + __expf(-gate[0]); y /= 1.f + __expf(-gate[1]);
    }
  }
  *(float2*)dst = make_float2(x, y);
}

template <int MT, bool KV8, int HDT, int GT>
__global__ void __launch_bounds__(128) k_attn_tc(AttnTcArgs a) { H5_LIT(0);
  pdl_enter();
  using Gm = AtG<HDT, GT>;
  constexpr int AT_QM = Gm::QM, AT_LD = Gm::LD, AT_STG8 = Gm::STG8, HD = HDT, NH = 0 + (GT == 0 ? 0 : 1);   // NH unused; heads come from a.nh
  (void)NH;
  extern __shared__ __align__(16) uint8_t smem[];
  __half* Qs = (__half*)smem;                         // [AT_QM][AT_LD]
  __half* KVs = Qs + AT_QM * AT_LD;                   // fp16: [2 stages][K,V][AT_KT][AT_LD]; fp8: [K,V][AT_KT][AT_LD]
  __half* Ps = KVs + (KV8 ? 2 : 4) * AT_KT * AT_LD;   // [AT_QM][AT_PLD]
  float* rmax = (float*)(Ps + AT_QM * AT_PLD);        // [4][AT_QM]
  float* rsum = rmax + 4 * AT_QM;                     // [4][AT_QM]
  const QBlk qb = a.qb[blockIdx.x];
  const int hk = blockIdx.y, sp = blockIdx.z, nsp = gridDim.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tq = lane & 3;
  constexpr int G = GT;
  const int nkv = a.nkv, nhq = a.nkv * G;
  const int nq = qb.len * G;
  const int kend = a.draft_block_rows ? draft_key_end(a.row_pos[qb.start], qb.start, a.draft_block_rows) : qb.pos_last + 1;
  const int ntile = (kend + AT_KT - 1) / AT_KT;
  const int kmin = a.window ? max(0, a.row_pos[qb.start] - a.window + 1) : 0;
  const int TPS = a.tps;   // tiles per split: split boundaries are absolute key positions
  const int t0 = kmin / AT_KT + sp * TPS, t1 = min(ntile, t0 + TPS);
  const size_t kvbase = (((size_t)qb.slot * a.nlayer + a.al) * nkv + hk) * (size_t)a.ctx * HD;
  const __half* Kg = (const __half*)a.kc + kvbase;
  const __half* Vg = (const __half*)a.vc + kvbase;
  const uint8_t* K8 = (const uint8_t*)a.kc + kvbase / 2;
  const uint8_t* V8 = (const uint8_t*)a.vc + kvbase / 2;
  const size_t sbase = kvbase / HD;
  uint8_t* stg8 = (uint8_t*)(rsum + 4 * AT_QM);      // fp8 staging [2][K bytes | V bytes | K sc | V sc]

  // Q -> smem fp16 (rows >= nq zero)
  for (int i = tid; i < MT * 16 * (HD / 8); i += 128) {
    const int qi = i / (HD / 8), c = i % (HD / 8);
    __half2 h[4];
    if (qi < nq) {
      const int row = qb.start + qi / G, hq = hk * G + qi % G;
      const float4* src = (const float4*)(a.qrot + (size_t)row * nhq * HD + hq * HD + c * 8);
      const float4 v0 = src[0], v1 = src[1];
      h[0] = __floats2half2_rn(v0.x, v0.y); h[1] = __floats2half2_rn(v0.z, v0.w);
      h[2] = __floats2half2_rn(v1.x, v1.y); h[3] = __floats2half2_rn(v1.z, v1.w);
    } else { h[0] = h[1] = h[2] = h[3] = __floats2half2_rn(0.f, 0.f); }
    *(uint4*)(Qs + qi * AT_LD + c * 8) = *(uint4*)h;
  }
  // per-thread row positions for masking: rows (mt, gid) and (mt, gid+8)
  int qpos[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int qi = mt * 16 + gid + 8 * h;
      qpos[mt][h] = qi < nq ? a.row_pos[qb.start + qi / G] : -1;
    }

  auto load_tile = [&](int t, int stg) {
    if (KV8) {
      uint8_t* b = stg8 + stg * AT_STG8;
      for (int i = tid; i < AT_KT * (HD / 32); i += 128) {
        const int k = i / (HD / 32), c = i % (HD / 32), key = t * AT_KT + k;
        const bool p = key < kend;
        const size_t off = (size_t)(p ? key : 0) * (HD / 2) + c * 16;
        a_cp16(b + k * (HD / 2) + c * 16, K8 + off, p);
        a_cp16(b + AT_KT * (HD / 2) + k * (HD / 2) + c * 16, V8 + off, p);
      }
      if (tid < 2 * AT_KT) {
        const int which = tid / AT_KT, k = tid % AT_KT, key = t * AT_KT + k;
        const bool p = key < kend;
        a_cp16(b + AT_KT * HD + tid * (HD / 16),
          (const uint8_t*)(which ? a.vsc : a.ksc) + (sbase + (p ? key : 0)) * (HD / 16), p);
      }
      asm volatile("cp.async.commit_group;\n");
      return;
    }
    __half* Kd = KVs + stg * 2 * AT_KT * AT_LD;
    __half* Vd = Kd + AT_KT * AT_LD;
    for (int i = tid; i < AT_KT * (HD / 8); i += 128) {
      const int k = i / (HD / 8), c = i % (HD / 8), key = t * AT_KT + k;
      const bool p = key < kend && key >= kmin;
      const int kk = a.ring ? key % a.ring : key;
      const size_t off = (size_t)(p ? kk : 0) * HD + c * 8;
      a_cp16(Kd + k * AT_LD + c * 8, Kg + off, p);
      a_cp16(Vd + k * AT_LD + c * 8, Vg + off, p);
    }
    asm volatile("cp.async.commit_group;\n");
  };

  constexpr int ON = HD / 32;   // n-tiles of 8 dims per warp (warp owns HD/4 dims)
  float o[MT][ON][4];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int n = 0; n < ON; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) o[mt][n][i] = 0.f;
  float m_run[MT][2], l_run[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) { m_run[mt][0] = m_run[mt][1] = -INFINITY; l_run[mt][0] = l_run[mt][1] = 0.f; }

  if (t0 < t1) load_tile(t0, 0);
  for (int t = t0; t < t1; t++) {
    const int stg = (t - t0) & 1;
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    if (t + 1 < t1) load_tile(t + 1, stg ^ 1);
    const __half* Ks = KVs + (KV8 ? 0 : stg * 2 * AT_KT * AT_LD);
    const __half* Vs = Ks + AT_KT * AT_LD;
#ifdef EXP_A_NOCONV
    if (false) {
#else
    if (KV8) {   // fp8 staging -> scaled fp16 tile
#endif
      const uint8_t* b = stg8 + stg * AT_STG8;
      const uint8_t* sc = b + AT_KT * HD;
      for (int i = tid; i < 2 * AT_KT * (HD / 16); i += 128) {
        const int which = i / (AT_KT * (HD / 16)), j = i % (AT_KT * (HD / 16));
        const int k = j / (HD / 16), c = j % (HD / 16);
        const unsigned long long raw = *(const unsigned long long*)(b + which * AT_KT * (HD / 2) + k * (HD / 2) + c * 8);
        const float scl = __half2float(__half(__nv_cvt_fp8_to_halfraw(sc[i], __NV_E4M3)));
        __half2 h[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const float2 f = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2((uint8_t)(raw >> (e * 8)), __NV_E2M1)));
          h[e] = __floats2half2_rn(f.x * scl, f.y * scl);
        }
        __half* dst = (__half*)(which ? Vs : Ks) + k * AT_LD + c * 16;
        *(uint4*)dst = *(uint4*)&h[0]; *(uint4*)(dst + 8) = *(uint4*)&h[4];
      }
      __syncthreads();
    }
    // S for keys 8w..8w+7
    float s[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) s[mt][0] = s[mt][1] = s[mt][2] = s[mt][3] = 0.f;
#pragma unroll
#ifdef EXP_A_NOS
    for (int kk = 0; kk < 1; kk++) {
#else
    for (int kk = 0; kk < HD / 32; kk++) {   // two k-steps (32 dims) per B load
#endif
      unsigned b[4];
      a_ldsm4(b, Ks + (warp * 8 + (lane & 7)) * AT_LD + kk * 32 + (lane >> 3) * 8);
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        unsigned af[4];
        a_ldsm4(af, Qs + (mt * 16 + (lane & 15)) * AT_LD + kk * 32 + (lane >> 4) * 8);
        mma16816(s[mt], af, b[0], b[1]);
        a_ldsm4(af, Qs + (mt * 16 + (lane & 15)) * AT_LD + kk * 32 + 16 + (lane >> 4) * 8);
        mma16816(s[mt], af, b[2], b[3]);
      }
    }
    const int key0 = t * AT_KT + warp * 8 + 2 * tq;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int qp = qpos[mt][h];
        const int nest_end = kend - a.draft_block_rows + a.draft_nest;   // first block key hidden from the nested prefix
        const int hi = qp < 0 ? -1 : (a.noncausal ? (a.draft_nest && qp < nest_end ? nest_end - 1 : kend - 1) : qp), lo = a.window ? qp - a.window + 1 : 0;
        if (key0 > hi || key0 < lo) s[mt][2 * h] = -INFINITY;
        if (key0 + 1 > hi || key0 + 1 < lo) s[mt][2 * h + 1] = -INFINITY;
        float mx = fmaxf(s[mt][2 * h], s[mt][2 * h + 1]);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        if (tq == 0) rmax[warp * AT_QM + mt * 16 + gid + 8 * h] = mx;
      }
    __syncthreads();
    float corr[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int qi = mt * 16 + gid + 8 * h;
        float mt_ = fmaxf(fmaxf(rmax[qi], rmax[AT_QM + qi]), fmaxf(rmax[2 * AT_QM + qi], rmax[3 * AT_QM + qi]));
        const float mnew = fmaxf(m_run[mt][h], mt_);
        corr[mt][h] = mnew == -INFINITY ? 1.f : __expf(m_run[mt][h] - mnew);
        m_run[mt][h] = mnew;
        const float p0 = mnew == -INFINITY ? 0.f : __expf(s[mt][2 * h] - mnew);
        const float p1 = mnew == -INFINITY ? 0.f : __expf(s[mt][2 * h + 1] - mnew);
        *(__half2*)(Ps + qi * AT_PLD + warp * 8 + 2 * tq) = __floats2half2_rn(p0, p1);
        float sm = p0 + p1;
        sm += __shfl_xor_sync(0xffffffff, sm, 1);
        sm += __shfl_xor_sync(0xffffffff, sm, 2);
        if (tq == 0) rsum[warp * AT_QM + qi] = sm;
      }
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int qi = mt * 16 + gid + 8 * h;
        l_run[mt][h] = l_run[mt][h] * corr[mt][h] + rsum[qi] + rsum[AT_QM + qi] + rsum[2 * AT_QM + qi] + rsum[3 * AT_QM + qi];
#pragma unroll
        for (int n = 0; n < ON; n++) { o[mt][n][2 * h] *= corr[mt][h]; o[mt][n][2 * h + 1] *= corr[mt][h]; }
      }
    // O[:, 64w..64w+63] += P V
#ifdef EXP_A_NOPV
    if (t < 0)
#endif
#pragma unroll
    for (int ks = 0; ks < AT_KT / 16; ks++) {
      unsigned pf[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; mt++) a_ldsm4(pf[mt], Ps + (mt * 16 + (lane & 15)) * AT_PLD + ks * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int np = 0; np < ON / 2; np++) {   // pairs of n-tiles (16 dims)
        unsigned b[4];
        const int j = lane >> 3;
        a_ldsm4t(b, Vs + (ks * 16 + (j & 1) * 8 + (lane & 7)) * AT_LD + warp * (HD / 4) + np * 16 + (j >> 1) * 8);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          mma16816(o[mt][np * 2], pf[mt], b[0], b[1]);
          mma16816(o[mt][np * 2 + 1], pf[mt], b[2], b[3]);
        }
      }
    }
  }
  // write partials
  float* Op = a.part_o;
  float2* ML = a.part_ml;
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int qi = mt * 16 + gid + 8 * h;
      if (qi >= nq) continue;
      const int row = qb.start + qi / G, hq = hk * G + qi % G;
      const size_t pidx = ((size_t)row * nhq + hq) * nsp + sp;
      const int d = warp * (HD / 4) + 2 * tq;
      float* dst = (a.direct ? a.y + ((size_t)row * nhq + hq) * HD : Op + pidx * HD) + d;
#pragma unroll
      for (int n = 0; n < ON; n++) attn_output_pair(a, dst + n * 8, row, hq, d + n * 8,
          o[mt][n][2 * h], o[mt][n][2 * h + 1], l_run[mt][h]);
      if (!a.direct && warp == 0 && tq == 0) ML[pidx] = make_float2(m_run[mt][h], l_run[mt][h]);
    }
}

// Merge split partials, normalise, apply sigmoid(gate). Grid (rows, NH), HD threads.
__device__ __forceinline__ void attn_combine_body(const AttnTcArgs& a, int nsp, int hd, int row, int hq, int nhq) {
  const int d = threadIdx.x;
  if (d >= hd) return;
  const size_t base = ((size_t)row * nhq + hq) * nsp;
  float mx = -INFINITY;
  for (int s = 0; s < nsp; s++) mx = fmaxf(mx, a.part_ml[base + s].x);
  float L = 0.f, acc = 0.f;
  for (int s = 0; s < nsp; s++) {
    const float2 ml = a.part_ml[base + s];
    const float w = ml.x == -INFINITY ? 0.f : __expf(ml.x - mx);
    L += ml.y * w;
    acc += a.part_o[(base + s) * hd + d] * w;
  }
  float v = acc / L;
  if (a.proj) v /= 1.f + __expf(-a.proj[(size_t)row * ATT_IN + hq * 2 * HD + HD + d]);
  a.y[(size_t)row * nhq * hd + hq * hd + d] = v;
}
__global__ void __launch_bounds__(256) k_attn_combine(AttnTcArgs a, int nsp, int hd) { H5_LIT(0);
  pdl_enter_e(a.early);
  attn_combine_body(a, nsp, hd, blockIdx.x, blockIdx.y, gridDim.y);
  if (a.sig) {
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(a.sig) : "memory");
  }
}


template <int MT, bool KV8, int HDT, int GT> static void attn_launch(const AttnTcArgs& a, dim3 grid, cudaStream_t st) {
  constexpr int sm = KV8 ? AtG<HDT, GT>::SMEM8 : AtG<HDT, GT>::SMEM16;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(k_attn_tc<MT, KV8, HDT, GT>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); init = true; }
  pdl(k_attn_tc<MT, KV8, HDT, GT>, dim3(grid), dim3(128), sm, st, a);
}

inline void launch(const h5::AttnTcArgs& a, int rows, int splits, cudaStream_t st, bool f16=false, bool optimized=false) {
  dim3 grid(rows,4,splits);
  if (f16) attn_launch<1,false,256,6>(a,grid,st); else attn_launch<1,true,256,6>(a,grid,st);
  if (!a.direct) h5::pdl(k_attn_combine,dim3(rows,24),dim3(256),0,st,a,splits,256);
}
}
