// NVFP4 tensor-core ternary GEMM and the FP4 activation quantiser.
//
//   out[t][n] (+)= g[t] * sum_kb ws[n][kb] * sum_{k in kb} trit[n][k] * a[t][k]
//
// Weights: trits stored at two bits each so that one 32-bit word becomes two E2M1 operand
// registers with one AND and one SHIFT+AND (bit 1 of a nibble = |trit|, bit 3 = sign; +1 = 0x2,
// -1 = 0xA, 0 = 0x0). The FP16 group scale (per 128 K) is applied exactly in FP32 after each
// 128-K block, so the only loss against the INT8 path is the FP4 activation.
// Activations: E2M1 values, UE4M3 scale per 16 K, FP32 scale per token (g).
//
// Weight chunk (g, kb) = 32 features x 128 K = 1024 bytes; lane l owns bytes [32l, 32l+32) = 8 words.
// lane: gid = l>>2, tq = l&3. word w = m*4 + h*2 + r  (m = m-tile, h = k64 half, r = row half)
//   row  = 32g + 16m + 8r + gid
//   nibble j of the word: bits {1,3} = trit(row, 128kb + 64h + 8tq + j),
//                         bits {0,2} = trit(row, 128kb + 64h + 32 + 8tq + j)
// Scales: half [g][kb][gid][4] = rows (m=0,r=0), (0,1), (1,0), (1,1).
#include "kernels.cuh"
#include <cuda_fp8.h>
#include <cstdio>
#include "f1_epilogue.cuh"
#include "f1_swar.cuh"

namespace f1 {
using h5::pdl; using h5::pdl_wait; using h5::pdl_launch; using h5::l2pf_enabled;
__device__ __constant__ uint16_t h4_table[256];

__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp8(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 8 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
  unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void mma_fp4(float* d, const unsigned* a, unsigned b0, unsigned b1, unsigned sb, const float* c, unsigned sa = 0x38383838u) {
  const uint16_t z = 0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},%14,{%16,%16},%15,{%16,%16};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sa), "r"(sb), "h"(z));
}

// Warp-specialised GEMM. FW consumer warps (32*WG features each) x TOKW tokens; warp FW is the
// producer: lane 0 streams each 128-K stage with 1D bulk copies (TMA) into a STAGES-deep ring
// guarded by mbarriers, so no CTA-wide barrier runs in the K loop.
//
// Global layouts (all produced by this engine):
//   weights  [kb][g] chunks of 1024 B; inside, word (lane l, w) at (w>>2)*512 + l*16 + (w&3)*4
//   wscale   [kb][g][gid][4] halves (64 B per chunk)
//   acts     [tile64][kb][64 rows][64 B], 16-B chunk c of row r stored at chunk c ^ ((r>>1)&3)
//   ascale   [tile64][kb][64 rows][8 B] UE4M3, one per 16 K; ag [row] FP32
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, unsigned count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(count)); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, unsigned tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" :: "r"(smem_u32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

// WF 0: ternary trits (1024 B chunk + 64 B fp16 group scales); WF 1: NVFP4 weights (2048 B chunk of
// E2M1 nibbles, word (lane l, w) at (w>>2)*512 + l*16 + (w&3)*4 holding a[m=w>>3][h=(w>>2)&1][w&3],
// + 256 B UE4M3 scales [32 rows][8 k16-blocks]) with an FP32 scale per output row.
// WF 2: trits as WF 0, but the group scale is E4M3 [gid][4 rows (m,r)] in the MMA's scale-A operand
// (relative to an FP32 row scale RS), so the MMAs accumulate directly with no per-block epilogue.
template <int WF> struct WFmt { static constexpr int CW = (WF == 1 || WF == 4) ? 2048 : 1024, CS = WF == 1 ? 256 : 64; };
template <int FW, int WG, int TOKW, int WF = 0> struct WsGeom {
  static constexpr int NG = FW * WG;                        // 32-feature groups per CTA
  static constexpr int SW = NG * WFmt<WF>::CW, SWS = NG * WFmt<WF>::CS, SA = TOKW * 64, SAS = TOKW * 8;
  static constexpr int STAGE = SW + SWS + SA + SAS;
#ifndef H5_STAGE_KB
#define H5_STAGE_KB 96
#endif
  static constexpr int STAGES = (H5_STAGE_KB * 1024) / STAGE > 8 ? 8 : (H5_STAGE_KB * 1024) / STAGE;
  static constexpr int SMEM = STAGES * STAGE + 2 * STAGES * 8;
};

template <int FW, int WG, int TOKW, int MODE, int WF>   // MODE 0 store, 1 add, 2 atomic add, 3 silu(gate)*up (WG 2, interleaved groups)
__global__ void __launch_bounds__((FW + 1) * 32) k_gemm_fp4(const uint8_t* __restrict__ A, const uint8_t* __restrict__ AS, const float* __restrict__ AG,
                                                              int R, int K, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N,
                                                              float* __restrict__ out, int ldo, const float* __restrict__ RS,
                                                              uint8_t* Q,uint8_t* QS,float* QG,const float* signs,float fixed) {
  using G = WsGeom<FW, WG, TOKW, WF>;
  constexpr int CW = WFmt<WF>::CW, CS = WFmt<WF>::CS;
  constexpr int NT = TOKW / 8, MT = 2 * WG, STAGES = G::STAGES;
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* full = (uint64_t*)(smem + STAGES * G::STAGE);
  uint64_t* empty = full + STAGES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nkb_all = K / 128;
  const int kb0 = (int)((long)nkb_all * blockIdx.z / gridDim.z), kb1 = (int)((long)nkb_all * (blockIdx.z + 1) / gridDim.z);
  const int nkb = kb1 - kb0;
  const int NGall = N / 32;
  const int g0 = blockIdx.y * G::NG;
  const int t0 = blockIdx.x * TOKW;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], FW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
#ifdef F1_USE_LUT
  __shared__ uint16_t local_h4[WF==3?256:1];
  if constexpr(WF==3) for(int i=tid;i<256;i+=(FW+1)*32) local_h4[i]=h4_table[i];
#endif
  __syncthreads();

  if (warp == FW) {   // producer
#ifdef EXP_NOWAIT
    return;
#endif
    if (lane == 0) {
      const int ng = min(G::NG, NGall - g0);
      const int tile = t0 / 64, toff = t0 % 64;
      const unsigned tx = ng * CW + ng * CS + TOKW * 64 + TOKW * 8;
      // weights do not depend on the previous kernel: start streaming this CTA's whole weight slice
      // into L2 and issue the first ring of it into shared memory, then wait for the producer kernel
      if (l2pf_enabled()) for (int kb = STAGES; kb < nkb; kb++) {
        const size_t kk = kb0 + kb;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(W + (kk * NGall + g0) * CW), "r"(ng * CW) : "memory");
      }
      const int npre = min(nkb, STAGES);
      for (int kb = 0; kb < npre; kb++) {
        uint8_t* st = smem + kb * G::STAGE;
        mbar_expect(&full[kb], tx);
        const size_t kk = kb0 + kb;
        bulk_g2s(st, W + (kk * NGall + g0) * CW, ng * CW, &full[kb]);
        bulk_g2s(st + G::SW, S + (kk * NGall + g0) * CS, ng * CS, &full[kb]);
      }
      pdl_wait();
      for (int kb = 0; kb < nkb; kb++) {
        const int s = kb % STAGES;
        uint8_t* st = smem + s * G::STAGE;
        const size_t kk = kb0 + kb;
        if (kb >= npre) {
          mbar_wait(&empty[s], ((kb / STAGES) & 1) ^ 1);
          mbar_expect(&full[s], tx);
          bulk_g2s(st, W + (kk * NGall + g0) * CW, ng * CW, &full[s]);
          bulk_g2s(st + G::SW, S + (kk * NGall + g0) * CS, ng * CS, &full[s]);
        }
        constexpr int SUB = TOKW < 64 ? TOKW : 64;
#pragma unroll
        for (int u = 0; u < TOKW / SUB; u++) {
          const size_t rb = ((size_t)(tile + u) * nkb_all + kk) * 64 + toff;
          bulk_g2s(st + G::SW + G::SWS + u * SUB * 64, A + rb * 64, SUB * 64, &full[s]);
          bulk_g2s(st + G::SW + G::SWS + G::SA + u * SUB * 8, AS + rb * 8, SUB * 8, &full[s]);
        }
      }
    }
    if constexpr(MODE==4) { __syncthreads(); __syncthreads(); emit<TOKW>((float*)smem,R,N/2,signs,fixed,Q,QS,QG); }
    return;
  }

  const int gid = lane >> 2, tq = lane & 3;
  const int gw = warp * WG;                       // first group of this warp inside the CTA
  const bool live = g0 + gw < NGall;
  float acc[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) acc[m][n][i] = 0.f;
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};

  for (int kb = 0; kb < nkb; kb++) {
    const int s = kb % STAGES;
#ifdef EXP_NOWAIT
    const uint8_t* st = smem + (kb & 1) * G::STAGE;   // compute on stale smem; no producer dependency
#else
    mbar_wait(&full[s], (kb / STAGES) & 1);
    const uint8_t* st = smem + s * G::STAGE;
#endif
    unsigned a[MT][2][4];
    float wsc[MT][2];
    unsigned sca[MT][2];
#pragma unroll
    for (int q = 0; q < WG; q++) {
      if (WF != 1 && WF != 4) {
        const uint8_t* wc = st + (gw + q) * CW + lane * 16;
        const uint4 w0 = *(const uint4*)wc, w1 = *(const uint4*)(wc + 512);
        const unsigned wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int mm = 0; mm < 2; mm++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const unsigned r0 = wv[mm * 4 + h * 2], r1 = wv[mm * 4 + h * 2 + 1];
            unsigned* o = a[q * 2 + mm][h];
            if constexpr(WF!=3) {
              o[0] = r0 & 0xAAAAAAAAu; o[2] = (r0 << 1) & 0xAAAAAAAAu;
              o[1] = r1 & 0xAAAAAAAAu; o[3] = (r1 << 1) & 0xAAAAAAAAu;
            }
          }
        if constexpr(WF==3) {
          #pragma unroll
          for(int mm=0;mm<2;mm++)
            #pragma unroll
            for(int rh=0;rh<2;rh++) {
              unsigned x=wv[mm*4+rh],y=wv[mm*4+2+rh];
              unsigned r[4]={0,0,0,0};
#ifndef F1_USE_LUT
              h4_swar(x,y,r);
#else
              #pragma unroll
              for(int j=0;j<8;j++) {
                unsigned v0=(x>>(4*j+1)&1)*(1+(x>>(4*j+3)&1));
                unsigned v1=(x>>(4*j)&1)*(1+(x>>(4*j+2)&1));
                unsigned v2=(y>>(4*j+1)&1)*(1+(y>>(4*j+3)&1));
                unsigned v3=(y>>(4*j)&1)*(1+(y>>(4*j+2)&1));
                unsigned z=local_h4[v0|(v1<<2)|(v2<<4)|(v3<<6)];
                #pragma unroll
                for(int p=0;p<4;p++) r[p]|=((z>>(4*p))&15)<<(4*j);
              }
#endif
              a[q*2+mm][0][rh]=r[0];a[q*2+mm][0][rh+2]=r[1];
              a[q*2+mm][1][rh]=r[2];a[q*2+mm][1][rh+2]=r[3];
            }
          }
        if (WF == 2) {
          const uint32_t sb4 = *(const uint32_t*)(st + G::SW + (gw + q) * CS + gid * 4);
#pragma unroll
          for (int mm = 0; mm < 2; mm++) {
            const uint32_t b = (sb4 >> (8 * (2 * mm + (tq & 1)))) & 0xff;
            sca[q * 2 + mm][0] = sca[q * 2 + mm][1] = b * 0x01010101u;
          }
        } else {
          const uint2 sc = *(const uint2*)(st + G::SW + (gw + q) * CS + gid * 8);
          const __half2* sh = (const __half2*)&sc;
          const float2 s01 = __half22float2(sh[0]), s23 = __half22float2(sh[1]);
          wsc[q * 2][0] = s01.x; wsc[q * 2][1] = s01.y; wsc[q * 2 + 1][0] = s23.x; wsc[q * 2 + 1][1] = s23.y;
        }
      } else {
        const uint8_t* wc = st + (gw + q) * CW + lane * 16;
#pragma unroll
        for (int g4 = 0; g4 < 4; g4++) {
          const uint4 wv = *(const uint4*)(wc + g4 * 512);
          unsigned* o = a[q * 2 + (g4 >> 1)][g4 & 1];
          o[0] = wv.x; o[1] = wv.y; o[2] = wv.z; o[3] = wv.w;
        }
        const uint8_t* sc = st + G::SW + (gw + q) * CS;
        if constexpr(WF==4) {
          const uint2 raw=*(const uint2*)(sc+gid*8);
          const __half2* sh=(const __half2*)&raw;
          float2 x=__half22float2(sh[0]),y=__half22float2(sh[1]);
          wsc[q*2][0]=x.x;wsc[q*2][1]=x.y;wsc[q*2+1][0]=y.x;wsc[q*2+1][1]=y.y;
        } else {
          #pragma unroll
          for (int mm = 0; mm < 2; mm++) {
            const uint2 v = *(const uint2*)(sc + (16 * mm + gid + 8 * (tq & 1)) * 8);
            sca[q * 2 + mm][0] = v.x; sca[q * 2 + mm][1] = v.y;
          }
        }
      }
    }
    const uint8_t* as = st + G::SW + G::SWS;
    const uint8_t* ssc = as + G::SA;
#ifdef EXP_NOMMA
    acc[0][0][0] += __int_as_float(a[0][0][0] ^ a[MT - 1][1][3]) + wsc[0][0] + (float)sca[0][0] + as[lane] + ssc[lane];
    __syncwarp(); if (lane == 0) mbar_arrive(&empty[s]);
    continue;
#endif
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int row = n * 8 + (lane & 7), c = lane >> 3;
      unsigned b0, b1, b2, b3;
      ldsm4(b0, b1, b2, b3, as + row * 64 + ((c ^ ((row >> 1) & 3)) << 4));
      const uint2 bs = *(const uint2*)(ssc + (n * 8 + gid) * 8);
#pragma unroll
      for (int m = 0; m < MT; m++) {
        if (WF == 1 || WF == 2) {
          mma_fp4(acc[m][n], a[m][0], b0, b1, bs.x, acc[m][n], sca[m][0]);
          mma_fp4(acc[m][n], a[m][1], b2, b3, bs.y, acc[m][n], sca[m][1]);
          continue;
        }
        float d[4];
        mma_fp4(d, a[m][0], b0, b1, bs.x, zero);
        mma_fp4(d, a[m][1], b2, b3, bs.y, d);
        acc[m][n][0] = fmaf(d[0], wsc[m][0], acc[m][n][0]);
        acc[m][n][1] = fmaf(d[1], wsc[m][0], acc[m][n][1]);
        acc[m][n][2] = fmaf(d[2], wsc[m][1], acc[m][n][2]);
        acc[m][n][3] = fmaf(d[3], wsc[m][1], acc[m][n][3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[s]);
  }
  pdl_wait();
  pdl_launch();
  if constexpr(MODE==4) {
    static_assert(FW==8 && WG==2,"F1 owns256 hidden outputs");
    __syncthreads();
    if(live) {
      #pragma unroll
      for(int n=0;n<NT;n++)
        #pragma unroll
        for(int j=0;j<2;j++) {
          int t=t0+n*8+tq*2+j;
          if(t>=R) continue;
          #pragma unroll
          for(int mm=0;mm<2;mm++)
            #pragma unroll
            for(int r=0;r<2;r++) {
              float gv=acc[mm][n][r*2+j]*AG[t],uv=acc[2+mm][n][r*2+j]*AG[t];
              ((float*)smem)[(t-t0)*256+warp*32+mm*16+r*8+gid]=gv/(1.f+__expf(-gv))*uv;
            }
        }
    }
    __syncthreads();
    emit<TOKW>((float*)smem,R,N/2,signs,fixed,Q,QS,QG);
    return;
  }
  if (!live) return;
  if constexpr (MODE == 3) {   // gate group gw and up group gw+1 of the same 32 features: write silu(gate) * up
    if constexpr (WG == 2) {
      const int fg = (g0 + gw) >> 1;   // feature group in the hidden (FF) coordinate
#pragma unroll
      for (int n = 0; n < NT; n++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const int t = t0 + n * 8 + tq * 2 + j;
          if (t >= R) continue;
          const float gt = AG[t];
          float* o = out + (size_t)t * ldo + fg * 32 + gid;
#pragma unroll
          for (int mm = 0; mm < 2; mm++)
#pragma unroll
            for (int r = 0; r < 2; r++) {
              float gv = acc[mm][n][r * 2 + j] * gt, uv = acc[2 + mm][n][r * 2 + j] * gt;
              if (WF == 1 || WF == 2) {   // NVFP4 weight row scales
                gv *= RS[(g0 + gw) * 32 + mm * 16 + r * 8 + gid];
                uv *= RS[(g0 + gw + 1) * 32 + mm * 16 + r * 8 + gid];
              }
              o[mm * 16 + r * 8] = gv / (1.f + __expf(-gv)) * uv;
            }
        }
    }
    return;
  }
#pragma unroll
  for (int n = 0; n < NT; n++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int t = t0 + n * 8 + tq * 2 + j;
      if (t >= R) continue;
      const float gt = AG[t];
      float* o = out + (size_t)t * ldo + (g0 + gw) * 32 + gid;
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          if (g0 + gw + (m >> 1) >= NGall) continue;   // second group of a WG=2 warp past N
          float v = acc[m][n][r * 2 + j] * gt;
          if (WF == 1 || WF == 2) v *= RS[(g0 + gw) * 32 + gid + m * 16 + r * 8];
          if (MODE == 2) atomicAdd(o + m * 16 + r * 8, v);
          else { if (MODE == 1) v += o[m * 16 + r * 8]; o[m * 16 + r * 8] = v; }
        }
    }
  }
}

template<int T,int WF,int MODE=0> void run_down(const uint8_t* A,const uint8_t* AS,const float* AG,int R,int K,const void* W,const void* S,int N,float* Y,cudaStream_t st,int split=1) {
  auto f=k_gemm_fp4<8,1,T,MODE,WF>;
  constexpr int sm=WsGeom<8,1,T,WF>::SMEM;
  static bool attr=false;if(!attr){cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);attr=true;}
  if constexpr(MODE==2) cudaMemsetAsync(Y,0,(size_t)R*N*4,st);
  pdl(f,dim3((R+T-1)/T,(N/32+7)/8,split),dim3(288),sm,st,A,AS,AG,R,K,(const uint8_t*)W,(const uint8_t*)S,N,Y,N,(const float*)nullptr,(uint8_t*)nullptr,(uint8_t*)nullptr,(float*)nullptr,(const float*)nullptr,0.f);
}
void down(int T,bool compact,const uint8_t* A,const uint8_t* AS,const float* AG,int R,int K,const void* W,const void* S,int N,float* Y,cudaStream_t st) {
  static bool init=false;
  if(!init) {
    uint16_t tab[256];
    for(int i=0;i<256;i++) {
      int v[4];bool valid=true;for(int j=0;j<4;j++){int c=i>>(2*j)&3;v[j]=c==2?-1:c;valid&=c!=3;}
      unsigned z=0;for(int p=0;p<4;p++){int s=0;for(int j=0;j<4;j++)s+=((__builtin_parity((unsigned)(p&j)))?-1:1)*v[j];unsigned c=abs(s)|(s<0?8:0);z|=c<<(4*p);}tab[i]=valid?z:0xffff;
    }
    cudaMemcpyToSymbol(h4_table,tab,sizeof tab);init=true;
  }
  if(R<=8) {if(compact)run_down<16,3,2>(A,AS,AG,R,K,W,S,N,Y,st,16);else run_down<16,4,2>(A,AS,AG,R,K,W,S,N,Y,st,16);}
  else if(T==32) {if(compact)run_down<32,3>(A,AS,AG,R,K,W,S,N,Y,st);else run_down<32,4>(A,AS,AG,R,K,W,S,N,Y,st);}
  else {if(compact)run_down<64,3>(A,AS,AG,R,K,W,S,N,Y,st);else run_down<64,4>(A,AS,AG,R,K,W,S,N,Y,st);}
}
template<int T> void run_gu(const uint8_t* A,const uint8_t* AS,const float* AG,int R,int K,const void* W,const void* S,int N,const float* signs,float fixed,uint8_t* Q,uint8_t* QS,float* QG,cudaStream_t st) {
  auto f=k_gemm_fp4<8,2,T,4,0>;
  constexpr int sm=WsGeom<8,2,T,0>::SMEM;
  static_assert(sm>=T*256*4,"ring must fit epilogue tile");
  static bool attr=false;if(!attr){cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);attr=true;}
  pdl(f,dim3((R+T-1)/T,N/512),dim3(288),sm,st,A,AS,AG,R,K,(const uint8_t*)W,(const uint8_t*)S,N,(float*)nullptr,N/2,(const float*)nullptr,Q,QS,QG,signs,fixed);
}
void gu_fused(int T,const uint8_t* A,const uint8_t* AS,const float* AG,int R,int K,const void* W,const void* S,int N,const float* signs,float fixed,uint8_t* Q,uint8_t* QS,float* QG,cudaStream_t st) {
  if(T==16)run_gu<16>(A,AS,AG,R,K,W,S,N,signs,fixed,Q,QS,QG,st);
  else if(T==32)run_gu<32>(A,AS,AG,R,K,W,S,N,signs,fixed,Q,QS,QG,st);
  else run_gu<64>(A,AS,AG,R,K,W,S,N,signs,fixed,Q,QS,QG,st);
}
} // namespace f1
