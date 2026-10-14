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

namespace p12kv4direct {
using namespace h5;

constexpr int AT_KT = 32;                  // keys per tile
constexpr int AT_PLD = AT_KT + 8;          // padded P row (80 B)
template <int HDT, int GT> struct AtG {
  static constexpr int QM = ((QB_ROWS * GT + 15) / 16) * 16;   // query vectors per block (48 target, 32 drafter)
  static constexpr int LD = HDT + 8;                           // padded fp16 row
  static constexpr int SMEM16 = (QM * LD + 4 * AT_KT * LD + QM * AT_PLD) * 2 + 2 * 4 * QM * 4;
  static constexpr int STG8 = 2 * AT_KT * HDT + 2 * AT_KT * 4;  // one fp8 stage: K, V bytes + K, V scales
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


constexpr int A8_KLD=144,A8_VLD=144,A8_STAGES=3,A8_QLD=HD+8;
constexpr int A8_STG=AT_KT*(A8_KLD+A8_VLD)+2*AT_KT*16;
template<int MT> constexpr int a4_smem(){return A8_STAGES*A8_STG+(MT*16*A8_QLD+MT*16*AT_PLD)*2+2*4*MT*16*4;}
__device__ __forceinline__ unsigned f4_scaled(uint8_t q,uint8_t scale){
 __half2 a=__half2(__nv_cvt_fp4x2_to_halfraw2(q,__NV_E2M1));
 __half b=__half(__nv_cvt_fp8_to_halfraw(scale,__NV_E4M3));
 __half2 c=__hmul2(a,__half2half2(b));__half2_raw r=c;return (unsigned)r.x|((unsigned)r.y<<16);
}
__device__ __forceinline__ unsigned f4_v(const uint8_t*v,const uint8_t*sc,int d,int key){
 unsigned lo=(v[key*A8_VLD+d/2]>>((d&1)*4))&15,hi=(v[(key+1)*A8_VLD+d/2]>>((d&1)*4))&15;
 unsigned scales=sc[key*16+d/16]|(unsigned(sc[(key+1)*16+d/16])<<8);
 __half2 a=__half2(__nv_cvt_fp4x2_to_halfraw2(lo|(hi<<4),__NV_E2M1));
 __half2 b=__half2(__nv_cvt_fp8x2_to_halfraw2(scales,__NV_E4M3));__half2_raw r=__hmul2(a,b);return (unsigned)r.x|((unsigned)r.y<<16);
}
template <int MT>
__device__ __forceinline__ void attn8_body(const AttnTcArgs& a, int bx, int hk, int sp, int nsp, uint8_t* smem) {
  constexpr int A8_QM=MT*16;
  uint8_t* stg = smem;
  __half* Qs = (__half*)(smem + A8_STAGES * A8_STG);
  __half* Ps = Qs + A8_QM * A8_QLD;
  float* rmax = (float*)(Ps + A8_QM * AT_PLD);
  float* rsum = rmax + 4 * A8_QM;
  const QBlk qb = a.qb[bx];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tq = lane & 3;
  constexpr int G = 6, MH = (MT + 1) / 2;
  const int nt = warp & 3;                                  // S: keys 8nt..8nt+7
  const int mlo = warp < 4 ? 0 : MH, mhi = warp < 4 ? MH : MT;   // S: owned m-tiles
  const int nq = qb.len * G, nhq = NKV * G;
  const int kend = qb.pos_last + 1;
  const int ntile = (kend + AT_KT - 1) / AT_KT;
  const int t0 = sp * a.tps, t1 = min(ntile, t0 + a.tps);
  const size_t hreg = (((size_t)qb.slot * a.nlayer + a.al) * NKV + hk) * (size_t)a.ctx;
  const uint8_t* K8 = (const uint8_t*)a.kc + hreg * (HD / 2);
  const uint8_t* V8 = (const uint8_t*)a.vc + hreg * (HD / 2);
  for (int i = tid; i < MT * 16 * (HD / 8); i += 256) {
    const int qi = i / (HD / 8), c = i % (HD / 8);
    __half2 h[4];
    if (qi < nq) {
      const int row = qb.start + qi / G, hq = hk * G + qi % G;
      const float4* src = (const float4*)(a.qrot + (size_t)row * nhq * HD + hq * HD + c * 8);
      const float4 v0 = src[0], v1 = src[1];
      h[0] = __floats2half2_rn(v0.x, v0.y); h[1] = __floats2half2_rn(v0.z, v0.w);
      h[2] = __floats2half2_rn(v1.x, v1.y); h[3] = __floats2half2_rn(v1.z, v1.w);
    } else { h[0] = h[1] = h[2] = h[3] = __floats2half2_rn(0.f, 0.f); }
    *(uint4*)(Qs + qi * A8_QLD + c * 8) = *(uint4*)h;
  }
  int qpos[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int h = 0; h < 2; h++) { const int qi = mt * 16 + gid + 8 * h; qpos[mt][h] = qi < nq ? a.row_pos[qb.start + qi / G] : -1; }

  auto load_tile = [&](int t, int sidx) {
    uint8_t* b = stg + sidx * A8_STG;
    if(tid<AT_KT*8){int k=tid/8,c=tid%8,key=t*AT_KT+k;bool valid=key<kend;size_t off=(size_t)(valid?key:0)*128+c*16;
      a_cp16(b+k*A8_KLD+c*16,K8+off,valid);
      a_cp16(b+AT_KT*A8_KLD+k*A8_VLD+c*16,V8+off,valid);
    }
    if(tid<2*AT_KT){int which=tid/AT_KT,k=tid%AT_KT,key=t*AT_KT+k;bool valid=key<kend;
      a_cp16(b+AT_KT*(A8_KLD+A8_VLD)+tid*16,(const uint8_t*)(which?a.vsc:a.ksc)+(hreg+(valid?key:0))*16,valid);
    }
    asm volatile("cp.async.commit_group;\n");
  };

  float o[MT][4][4];   // dims 32w .. 32w+31
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int n = 0; n < 4; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) o[mt][n][i] = 0.f;
  float m_run[MT][2], l_run[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) { m_run[mt][0] = m_run[mt][1] = -INFINITY; l_run[mt][0] = l_run[mt][1] = 0.f; }

#pragma unroll
  for (int i = 0; i < A8_STAGES - 1; i++) { if (t0 + i < t1) load_tile(t0 + i, i); else asm volatile("cp.async.commit_group;\n"); }
  for (int t = t0; t < t1; t++) {
    const int si = (t - t0) % A8_STAGES;
    asm volatile("cp.async.wait_group %0;\n" :: "n"(A8_STAGES - 2));
    __syncthreads();
    if (t + A8_STAGES - 1 < t1) load_tile(t + A8_STAGES - 1, (t - t0 + A8_STAGES - 1) % A8_STAGES);
    else asm volatile("cp.async.commit_group;\n");
    const uint8_t* Kt = stg + si * A8_STG;
    const uint8_t* Vt = Kt + AT_KT * A8_KLD;
    const uint8_t* ksc=Vt+AT_KT*A8_VLD;
    const uint8_t* vsc=ksc+AT_KT*16;
    float s[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) s[mt][0] = s[mt][1] = s[mt][2] = s[mt][3] = 0.f;
    const uint8_t* kr=Kt+(nt*8+gid)*A8_KLD+tq;
    const uint8_t* krsc=ksc+(nt*8+gid)*16;
    if (mlo < mhi) {
#pragma unroll
      for (int kk = 0; kk < HD / 32; kk++) {
        unsigned b[4];
        b[0]=f4_scaled(kr[kk*16+0],krsc[kk*2+0]);
        b[1]=f4_scaled(kr[kk*16+4],krsc[kk*2+0]);
        b[2]=f4_scaled(kr[kk*16+8],krsc[kk*2+1]);
        b[3]=f4_scaled(kr[kk*16+12],krsc[kk*2+1]);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          if (mt < mlo || mt >= mhi) continue;
          unsigned af[4];
          a_ldsm4(af, Qs + (mt * 16 + (lane & 15)) * A8_QLD + kk * 32 + (lane >> 4) * 8);
          mma16816(s[mt], af, b[0], b[1]);
          a_ldsm4(af, Qs + (mt * 16 + (lane & 15)) * A8_QLD + kk * 32 + 16 + (lane >> 4) * 8);
          mma16816(s[mt], af, b[2], b[3]);
        }
      }
    }
    const int kl0 = nt * 8 + 2 * tq, key0 = t * AT_KT + kl0;

#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      if (mt < mlo || mt >= mhi) continue;
#pragma unroll
      for (int h = 0; h < 2; h++) {

        if (key0 > qpos[mt][h]) s[mt][2 * h] = -INFINITY;
        if (key0 + 1 > qpos[mt][h]) s[mt][2 * h + 1] = -INFINITY;
        float mx = fmaxf(s[mt][2 * h], s[mt][2 * h + 1]);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        if (tq == 0) rmax[nt * A8_QM + mt * 16 + gid + 8 * h] = mx;
      }
    }
    __syncthreads();

    float corr[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int qi = mt * 16 + gid + 8 * h;
        const float mt_ = fmaxf(fmaxf(rmax[qi], rmax[A8_QM + qi]), fmaxf(rmax[2 * A8_QM + qi], rmax[3 * A8_QM + qi]));
        const float mnew = fmaxf(m_run[mt][h], mt_);
        corr[mt][h] = mnew == -INFINITY ? 1.f : __expf(m_run[mt][h] - mnew);
        m_run[mt][h] = mnew;
        if (mt >= mlo && mt < mhi) {
          const float p0 = mnew == -INFINITY ? 0.f : __expf(s[mt][2 * h] - mnew);
          const float p1 = mnew == -INFINITY ? 0.f : __expf(s[mt][2 * h + 1] - mnew);
          *(__half2*)(Ps + qi * AT_PLD + kl0) = __floats2half2_rn(p0,p1);
          float sm = p0 + p1;
          sm += __shfl_xor_sync(0xffffffff, sm, 1);
          sm += __shfl_xor_sync(0xffffffff, sm, 2);
          if (tq == 0) rsum[nt * A8_QM + qi] = sm;
        }
      }
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int qi = mt * 16 + gid + 8 * h;
        l_run[mt][h] = l_run[mt][h] * corr[mt][h] + rsum[qi] + rsum[A8_QM + qi] + rsum[2 * A8_QM + qi] + rsum[3 * A8_QM + qi];
#pragma unroll
        for (int n = 0; n < 4; n++) { o[mt][n][2 * h] *= corr[mt][h]; o[mt][n][2 * h + 1] *= corr[mt][h]; }
      }
#pragma unroll
    for (int ks = 0; ks < AT_KT / 16; ks++) {
      unsigned pf[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; mt++) a_ldsm4(pf[mt], Ps + (mt * 16 + (lane & 15)) * AT_PLD + ks * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int n = 0; n < 4; n++) {
        const int d=warp*32+n*8+gid,key=ks*16+2*tq;
        const unsigned b0=f4_v(Vt,vsc,d,key),b1=f4_v(Vt,vsc,d,key+8);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) mma16816(o[mt][n], pf[mt], b0, b1);
      }
    }
  }
  asm volatile("cp.async.wait_group 0;\n");
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int qi = mt * 16 + gid + 8 * h;
      if (qi >= nq) continue;
      const int row = qb.start + qi / G, hq = hk * G + qi % G;
      const size_t pidx = ((size_t)row * nhq + hq) * nsp + sp;
      const int d = warp * 32 + 2 * tq;
      float* dst = (a.direct ? a.y + ((size_t)row * nhq + hq) * HD : a.part_o + pidx * HD) + d;
#pragma unroll
      for (int n = 0; n < 4; n++) attn_output_pair(a, dst + n * 8, row, hq, d + n * 8,
          o[mt][n][2 * h], o[mt][n][2 * h + 1], l_run[mt][h]);
      if (!a.direct && warp == 0 && tq == 0) a.part_ml[pidx] = make_float2(m_run[mt][h], l_run[mt][h]);
    }
}
template <int MT>
__global__ void __launch_bounds__(256) k_attn8(AttnTcArgs a) { H5_LIT(0);
  pdl_enter_e(a.early);
  extern __shared__ __align__(16) uint8_t smem[];
  attn8_body<MT>(a, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z, smem);
}
template <int MT> static void attn8_launch(const AttnTcArgs& a, dim3 grid, cudaStream_t st) {
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(k_attn8<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, a4_smem<MT>()); init = true; }
  pdl(k_attn8<MT>, dim3(grid), dim3(256), a4_smem<MT>(), st, a);
}


inline void launch(const AttnTcArgs&a,int rows,int splits,cudaStream_t st){
 attn8_launch<1>(a,dim3(rows,4,splits),st);
 if(!a.direct)h5::pdl(k_attn_combine,dim3(rows,24),dim3(256),0,st,a,splits,256);
}
}
