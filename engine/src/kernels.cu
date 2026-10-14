#define launch_gdn qc_original_launch_gdn
#define launch_gdn_single qc_original_launch_gdn_single
#define launch_attn_pre qc_original_launch_attn_pre
#define launch_attn_tc qc_original_launch_attn_tc
#define H5_PDLCLOCK_MODULE kernels
#define H5_LIT_MODULE kernels
#define H5_LIT_MODULE_ID 1u
#include "kernels.cuh"
#include "gdn_single.h"
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <initializer_list>
#include <type_traits>
#include <string>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

namespace h5 {

__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o)); return v; }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float block_sum_256(float v, float* red) {
  v = warp_sum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) t += red[i];
  return t;
}

// ------------------------------------------------------------------------------------------
// Ternary GEMM on INT8 tensor cores.
//   out[tok][n] (+)= sum_k W[n][k] * x[tok][k]
// Activations: int8 [R][K] with per-(row,128) scale xs.  Weights: frag2 layout (see loader.cpp):
// each trit sits as a signed 2-bit field and is lifted to (+-64 or 0) bytes by one shift+mask,
// so the int32 accumulator is 64x the trit dot and S already carries scale/64.
// CTA = 128 tokens x 128 features, 8 warps = 2 (tokens) x 4 (features), warp = 64 x 32.
// ------------------------------------------------------------------------------------------
constexpr int GM = 128, GN = 128, GSTAGES = 3;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  int sz = pred ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
  unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void mma_s8(int* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int MODE>   // 0 = store, 1 = add into out
__global__ void __launch_bounds__(256, 1) k_gemm_t2(const int8_t* __restrict__ A, const float* __restrict__ xs, int R, int K,
                                                    const uint4* __restrict__ W, const float* __restrict__ S, int N,
                                                    float* __restrict__ out, int ldo) { H5_LIT(0);
  extern __shared__ __align__(16) uint8_t smem[];
  int8_t* As = (int8_t*)smem;                                   // [GSTAGES][GM][128]
  float* Xs = (float*)(smem + GSTAGES * GM * 128);               // [GSTAGES][GM]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp & 1, wn = warp >> 1;
  const int gid = lane >> 2, tq = lane & 3;
  const int m0 = blockIdx.y * GM, n0 = blockIdx.x * GN;
  const int nkb = K / 128;
  const int g = (n0 >> 5) + wn;                                   // 32-feature group of this warp
  const uint4* wchunk = W + (size_t)g * nkb * 64 + lane * 2;

  auto load_stage = [&](int kb, int st) {
    int8_t* dst = As + st * GM * 128;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int idx = tid + i * 256, r = idx >> 3, c = idx & 7;
      bool p = (m0 + r) < R;
      const int8_t* src = A + (size_t)(p ? m0 + r : 0) * K + kb * 128 + c * 16;
      cp_async16(dst + r * 128 + ((c ^ (r & 7)) << 4), src, p);
    }
    if (tid < GM) {
      bool p = (m0 + tid) < R;
      cp_async4(Xs + st * GM + tid, xs + (size_t)(p ? m0 + tid : 0) * nkb + kb, p);
    }
  };

  float accf[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int b = 0; b < 4; b++)
#pragma unroll
      for (int c = 0; c < 4; c++) accf[a][b][c] = 0.f;

  load_stage(0, 0); cp_commit();
  if (nkb > 1) load_stage(1, 1);
  cp_commit();
  uint4 wa = wchunk[0], wb = wchunk[1];

  for (int kb = 0; kb < nkb; kb++) {
    const int st = kb % GSTAGES;
    cp_wait<1>();
    __syncthreads();
    if (kb + 2 < nkb) load_stage(kb + 2, (kb + 2) % GSTAGES);
    cp_commit();
    uint4 na = wa, nb = wb;
    if (kb + 1 < nkb) { na = wchunk[(kb + 1) * 64]; nb = wchunk[(kb + 1) * 64 + 1]; }
    const unsigned wv[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
    const int8_t* as = As + st * GM * 128;

    int acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
      for (int b = 0; b < 4; b++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[a][b][c] = 0;

#pragma unroll
    for (int s = 0; s < 4; s++) {
      unsigned af[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; mt++) {
        const int row = wm * 64 + mt * 16 + (lane & 15);
        const int kc = s * 2 + (lane >> 4);
        ldsm_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3], as + row * 128 + ((kc ^ (row & 7)) << 4));
      }
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const unsigned w = wv[s * 2 + (nt >> 1)];
        const int f0 = (nt & 1) * 2;
        const unsigned b0 = (w << (6 - 2 * f0)) & 0xC0C0C0C0u;
        const unsigned b1 = (w << (4 - 2 * f0)) & 0xC0C0C0C0u;
#pragma unroll
        for (int mt = 0; mt < 4; mt++) mma_s8(acc[mt][nt], af[mt][0], af[mt][1], af[mt][2], af[mt][3], b0, b1);
      }
    }
    // dequantize this 128-block
    const float* sk = S + (size_t)kb * N + n0 + wn * 32 + tq * 2;
    float2 ws[4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) ws[nt] = *(const float2*)(sk + nt * 8);
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
      const float x0 = Xs[st * GM + wm * 64 + mt * 16 + gid], x1 = Xs[st * GM + wm * 64 + mt * 16 + gid + 8];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        accf[mt][nt][0] = fmaf((float)acc[mt][nt][0], x0 * ws[nt].x, accf[mt][nt][0]);
        accf[mt][nt][1] = fmaf((float)acc[mt][nt][1], x0 * ws[nt].y, accf[mt][nt][1]);
        accf[mt][nt][2] = fmaf((float)acc[mt][nt][2], x1 * ws[nt].x, accf[mt][nt][2]);
        accf[mt][nt][3] = fmaf((float)acc[mt][nt][3], x1 * ws[nt].y, accf[mt][nt][3]);
      }
    }
    wa = na; wb = nb;
  }
  cp_wait<0>();
#pragma unroll
  for (int mt = 0; mt < 4; mt++) {
    const int r0 = m0 + wm * 64 + mt * 16 + gid;
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const int col = n0 + wn * 32 + nt * 8 + tq * 2;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int r = r0 + h * 8;
        if (r < R) {
          float2* o = (float2*)(out + (size_t)r * ldo + col);
          float2 v = make_float2(accf[mt][nt][2 * h], accf[mt][nt][2 * h + 1]);
          if (MODE == 1) { float2 p = *o; v.x += p.x; v.y += p.y; }
          *o = v;
        }
      }
    }
  }
}

void launch_gemm(const int8_t* A, const float* xs, int R, int K, const uint4* W, const float* S, int N, float* out, int ldo, bool add, cudaStream_t st) {
  static bool init = false;
  const int smem = GSTAGES * GM * 128 + GSTAGES * GM * 4;
  if (!init) {
    cudaFuncSetAttribute(k_gemm_t2<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(k_gemm_t2<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    init = true;
  }
  dim3 grid(N / GN, (R + GM - 1) / GM);
  if (add) k_gemm_t2<1><<<grid, 256, smem, st>>>(A, xs, R, K, W, S, N, out, ldo);
  else k_gemm_t2<0><<<grid, 256, smem, st>>>(A, xs, R, K, W, S, N, out, ldo);
}

// ------------------------------------------------------------------------------------------
// Prep, one CTA per row: norm / silu*mul / gdn permute / sign / 1024-Hadamard / quant.
// Quant is INT8 per 128 (xq, xs) or NVFP4 (q4 nibbles, s4 UE4M3 per 16, g4 FP32 per row).
// ------------------------------------------------------------------------------------------
// Register prep: one warp per 1024-chunk, lane holds elements 32*lane .. +31 of its chunk.
template<bool CLEAR>
__global__ void __launch_bounds__(544) k_prep2(PrepArgs a) { H5_LIT(0);
  __shared__ float red[2][17];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5, row = blockIdx.x, n = a.n;
  // constants (norm weights, sign vectors) do not depend on the previous kernel: load them before waiting
  float4 cn[8], cs[8];
  {
    const int e0c = w * 1024 + lane * 32;
    if (a.flags & P_NORM) {
#pragma unroll
      for (int i = 0; i < 8; i++) cn[i] = *(const float4*)(a.norm_w + e0c + 4 * i);
    }
    if (a.flags & (P_SIGN | P_SIGN_AFTER)) {
#pragma unroll
      for (int i = 0; i < 8; i++) cs[i] = *(const float4*)(a.signs + e0c + 4 * i);
    }
  }
  if (a.early || (g_variant & 2u)) pdl_launch(); pdl_wait();
  if constexpr (CLEAR) {
    for (int i = threadIdx.x; i < a.clear_a_stride / 4; i += blockDim.x)
      reinterpret_cast<float4*>(a.clear_a + (size_t)row * a.clear_a_stride)[i] = make_float4(0, 0, 0, 0);
    for (int i = threadIdx.x; i < a.clear_b_stride / 4; i += blockDim.x)
      reinterpret_cast<float4*>(a.clear_b + (size_t)row * a.clear_b_stride)[i] = make_float4(0, 0, 0, 0);
  }
  pdl_launch();
  const int irow = a.rowmap ? a.rowmap[row] : row;
  const float* x = a.x + (size_t)irow * a.ldx;
  const float* x2 = a.x2 ? a.x2 + (size_t)irow * a.ldx : nullptr;
  const int e0 = w * 1024 + lane * 32;
  float v[32];
  if (a.flags & P_PERMUTE_GDN) {
#pragma unroll
    for (int r = 0; r < 32; r++) { const int gi = e0 + r, hd = gi & 127, q = gi >> 7, rr = q % 3, kh = q / 3; v[r] = x[hd + 128 * (kh + 16 * rr)]; }
  } else {
#pragma unroll
    for (int i = 0; i < 8; i++) { const float4 t = *(const float4*)(x + e0 + 4 * i); v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w; }
  }
  H5_LIT_MARK(0);
  if (a.flags & P_GDN_NORM) {   // one head = 128 consecutive (permuted) elements = 4 lanes
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < 32; r++) ss = fmaf(v[r], v[r], ss);
    ss += __shfl_xor_sync(0xffffffff, ss, 1); ss += __shfl_xor_sync(0xffffffff, ss, 2);
    const float inv = rsqrtf(ss / 128.f + NORM_EPS);
    const int q = e0 >> 7, rr = q % 3, kh = q / 3, head = kh + 16 * rr, hd0 = e0 & 127;
    const float* z = a.gz + (size_t)irow * a.gz_ld + head * 128 + hd0;
#pragma unroll
    for (int r = 0; r < 32; r++) v[r] = v[r] * inv * a.gnorm_w[hd0 + r] * silu(z[r]);
  }
  if (a.flags & P_NORM) {
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < 32; r++) ss = fmaf(v[r], v[r], ss);
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    if (lane == 0) red[0][w] = ss;
    __syncthreads();
    ss = 0.f; for (int i = 0; i < nw; i++) ss += red[0][i];
    const float inv = rsqrtf(ss / (float)n + NORM_EPS);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 nwt = cn[i];
      v[4 * i] *= inv * nwt.x; v[4 * i + 1] *= inv * nwt.y; v[4 * i + 2] *= inv * nwt.z; v[4 * i + 3] *= inv * nwt.w;
    }
    if (a.flags & P_OUT_NORM) {
#pragma unroll
      for (int i = 0; i < 8; i++) *(float4*)(a.out_norm + (size_t)row * a.ldo + e0 + 4 * i) = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
    }
  }
  H5_LIT_MARK(1);
  if (a.flags & P_SILU_MUL) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 u = *(const float4*)(x2 + e0 + 4 * i);
      v[4 * i] = silu(v[4 * i]) * u.x; v[4 * i + 1] = silu(v[4 * i + 1]) * u.y; v[4 * i + 2] = silu(v[4 * i + 2]) * u.z; v[4 * i + 3] = silu(v[4 * i + 3]) * u.w;
    }
  }
  if (a.flags & P_SIGN) {
#pragma unroll
    for (int i = 0; i < 8; i++) { const float4 sg = cs[i]; v[4 * i] *= sg.x; v[4 * i + 1] *= sg.y; v[4 * i + 2] *= sg.z; v[4 * i + 3] *= sg.w; }
  }
  if (a.flags & P_HADAMARD) {
#pragma unroll
    for (int len = 1; len < 32; len <<= 1)
#pragma unroll
      for (int r = 0; r < 32; r++)
        if (!(r & len)) { const float p = v[r], q = v[r + len]; v[r] = p + q; v[r + len] = p - q; }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
#pragma unroll
      for (int r = 0; r < 32; r++) { const float q = __shfl_xor_sync(0xffffffff, v[r], o); v[r] = (lane & o) ? q - v[r] : v[r] + q; }
#pragma unroll
    for (int r = 0; r < 32; r++) v[r] *= 0.03125f;
  }
  H5_LIT_MARK(2);
  if (a.flags & P_SIGN_AFTER) {
#pragma unroll
    for (int i = 0; i < 8; i++) { const float4 sg = cs[i]; v[4 * i] *= sg.x; v[4 * i + 1] *= sg.y; v[4 * i + 2] *= sg.z; v[4 * i + 3] *= sg.w; }
  }
  if (a.flags & P_STORE_F32) {
#pragma unroll
    for (int i = 0; i < 8; i++) *(float4*)(a.out_f32 + (size_t)row * a.ldo + e0 + 4 * i) = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
  }
  H5_LIT_MARK(3);
  if (!(a.flags & P_QUANT)) return;
  float bm0 = 0.f, bm1 = 0.f;
#pragma unroll
  for (int r = 0; r < 16; r++) { bm0 = fmaxf(bm0, fabsf(v[r])); bm1 = fmaxf(bm1, fabsf(v[r + 16])); }
  if (a.q4) {
    float am = fmaxf(bm0, bm1);
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
    if (lane == 0) red[1][w] = am;
    __syncthreads();
    am = 0.f; for (int i = 0; i < nw; i++) am = fmaxf(am, red[1][i]);
    const float g = fmaxf(am / (6.f * 448.f), 1e-30f), ig = 1.f / g;
    if (threadIdx.x == 0) a.g4[row] = g;
    uint32_t words[4]; uint8_t sbs[2];
#pragma unroll
    for (int b = 0; b < 2; b++) {
      const float target = (b ? bm1 : bm0) / 6.f * ig;
      __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
      float sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
      if (sv < target && sb < 0x7e) { sb++; sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3))); }
      const float rs = sv > 0.f ? ig / sv : 0.f;
      sbs[b] = sb;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint32_t wd = 0;
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const int r = 16 * b + 8 * h + 2 * q;
          wd |= (uint32_t)__nv_cvt_float2_to_fp4x2(make_float2(v[r] * rs, v[r + 1] * rs), __NV_E2M1, cudaRoundNearest) << (8 * q);
        }
        words[2 * b + h] = wd;
      }
    }
    const int tile = row >> 6, rr = row & 63, kb = (e0 >> 7), c = lane & 3;
    const size_t base = ((size_t)tile * (n / 128) + kb) * 64 + rr;
    *(uint4*)(a.q4 + base * 64 + ((c ^ ((rr >> 1) & 3)) << 4)) = make_uint4(words[0], words[1], words[2], words[3]);
    *(uint16_t*)(a.s4 + base * 8 + 2 * c) = (uint16_t)sbs[0] | ((uint16_t)sbs[1] << 8);
  } else {
    float bm = fmaxf(bm0, bm1);
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, 1)); bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, 2));
    const float is = bm > 0.f ? 127.f / bm : 0.f;
    uint32_t pk[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      pk[i] = 0;
#pragma unroll
      for (int e = 0; e < 4; e++) pk[i] |= (uint32_t)(__float2int_rn(v[4 * i + e] * is) & 0xff) << (8 * e);
    }
    uint4* dst = (uint4*)(a.xq + (size_t)row * n + e0);
    dst[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]); dst[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
    if ((lane & 3) == 0) a.xs[(size_t)row * (n / 128) + (e0 >> 7)] = bm / 127.f;
  }
}
#include "prep_chunk.cuh"


template<int SPLITS, bool FENCE>
__device__ __forceinline__ void ordered_fold8_load4(const float* p, size_t plane, float4& lo, float4& hi) {
  // Loads may move ahead; the RN additions must remain an ascending left fold.
#pragma unroll
  for (int first = 1; first < SPLITS; first += 4) {
    float4 next_lo[4], next_hi[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (first + j < SPLITS) {
        next_lo[j] = *(const float4*)(p + size_t(first + j) * plane);
        next_hi[j] = *(const float4*)(p + size_t(first + j) * plane + 4);
      }
    }
    // CUDA12.8 removes the physical warp barrier but changes scheduling; see bench/ordered_prep.md.
    if constexpr (FENCE) __syncwarp();
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (first + j < SPLITS) {
        const float4 u = next_lo[j], z = next_hi[j];
        lo.x = __fadd_rn(lo.x, u.x); lo.y = __fadd_rn(lo.y, u.y); lo.z = __fadd_rn(lo.z, u.z); lo.w = __fadd_rn(lo.w, u.w);
        hi.x = __fadd_rn(hi.x, z.x); hi.y = __fadd_rn(hi.y, z.y); hi.z = __fadd_rn(hi.z, z.z); hi.w = __fadd_rn(hi.w, z.w);
      }
    }
  }
}

// Wide row preparation: four warps per 1024-feature chunk, eight features per lane.
// Feature bits 0..2 live in registers, 3..7 in lanes, 8..9 in the warp index. Hadamard stages run
// in the same bit order with the same operand order as k_prep2, so the transform is bit-identical;
// only the RMS sum association differs. Supports the FP4 row-norm path (norm/sign/hadamard/quant,
// optional out_norm/sign_after/store_f32 and output clearing).
template<bool CLEAR, bool ORDERED, int SPLITS = 0, bool FENCE = false>
__device__ __forceinline__ void prep2w_body(PrepArgs a, OrderedResidual ordered) {
  __shared__ float xch[8][4][256];      // [chunk][warp-in-chunk][lane*8+j]
  __shared__ float red[2][32];
  const int t = threadIdx.x, lane = t & 31, w = t >> 5, nw = blockDim.x >> 5;
  const int c = t >> 7, wl = (t >> 5) & 3, row = blockIdx.x, n = a.n;
  const int e0 = c * 1024 + wl * 256 + lane * 8;
  float4 cn[2], cs[2];
  if (a.flags & P_NORM) { cn[0] = *(const float4*)(a.norm_w + e0); cn[1] = *(const float4*)(a.norm_w + e0 + 4); }
  if (a.flags & (P_SIGN | P_SIGN_AFTER)) { cs[0] = *(const float4*)(a.signs + e0); cs[1] = *(const float4*)(a.signs + e0 + 4); }
  if (a.early || (g_variant & 2u)) pdl_launch(); pdl_wait();
  if constexpr (CLEAR) {
    for (int i = t; i < a.clear_a_stride / 4; i += blockDim.x)
      reinterpret_cast<float4*>(a.clear_a + (size_t)row * a.clear_a_stride)[i] = make_float4(0, 0, 0, 0);
    for (int i = t; i < a.clear_b_stride / 4; i += blockDim.x)
      reinterpret_cast<float4*>(a.clear_b + (size_t)row * a.clear_b_stride)[i] = make_float4(0, 0, 0, 0);
  }
  pdl_launch();
  const float* x = a.x + (size_t)row * a.ldx;
  float v[8];
  { const float4 p = *(const float4*)(x + e0), q = *(const float4*)(x + e0 + 4);
    v[0] = p.x; v[1] = p.y; v[2] = p.z; v[3] = p.w; v[4] = q.x; v[5] = q.y; v[6] = q.z; v[7] = q.w; }
  if constexpr (ORDERED) {
    const size_t plane = size_t(ordered.rows) * n;
    const float* p = ordered.partials + size_t(row) * n + e0;
    float4 lo = *(const float4*)p, hi = *(const float4*)(p + 4);
    if constexpr (SPLITS > 0) ordered_fold8_load4<SPLITS, FENCE>(p, plane, lo, hi);
    else {
      for (int s = 1; s < ordered.splits; s++) {
        const float4 u = *(const float4*)(p + size_t(s) * plane), z = *(const float4*)(p + size_t(s) * plane + 4);
        lo.x = __fadd_rn(lo.x, u.x); lo.y = __fadd_rn(lo.y, u.y); lo.z = __fadd_rn(lo.z, u.z); lo.w = __fadd_rn(lo.w, u.w);
        hi.x = __fadd_rn(hi.x, z.x); hi.y = __fadd_rn(hi.y, z.y); hi.z = __fadd_rn(hi.z, z.z); hi.w = __fadd_rn(hi.w, z.w);
      }
    }
    v[0] = __fadd_rn(v[0], lo.x); v[1] = __fadd_rn(v[1], lo.y); v[2] = __fadd_rn(v[2], lo.z); v[3] = __fadd_rn(v[3], lo.w);
    v[4] = __fadd_rn(v[4], hi.x); v[5] = __fadd_rn(v[5], hi.y); v[6] = __fadd_rn(v[6], hi.z); v[7] = __fadd_rn(v[7], hi.w);
    float* dst = ordered.residual + size_t(row) * a.ldx + e0;
    *(float4*)dst = make_float4(v[0], v[1], v[2], v[3]);
    *(float4*)(dst + 4) = make_float4(v[4], v[5], v[6], v[7]);
  }
  if (a.flags & P_NORM) {
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < 8; r++) ss = fmaf(v[r], v[r], ss);
    ss = warp_sum(ss);
    H5_LIT_MARKX(0);   // marks: 0 row loaded, 1 norm barrier, 2 Hadamard exchange, 3 amax barrier
    if (lane == 0) red[0][w] = ss;
    __syncthreads();
    H5_LIT_MARKX(1);
    ss = 0.f; for (int i = 0; i < nw; i++) ss += red[0][i];
    const float inv = rsqrtf(ss / (float)n + NORM_EPS);
    v[0] *= inv * cn[0].x; v[1] *= inv * cn[0].y; v[2] *= inv * cn[0].z; v[3] *= inv * cn[0].w;
    v[4] *= inv * cn[1].x; v[5] *= inv * cn[1].y; v[6] *= inv * cn[1].z; v[7] *= inv * cn[1].w;
    if (a.flags & P_OUT_NORM) {
      float* o = a.out_norm + (size_t)row * a.ldo + e0;
      *(float4*)o = make_float4(v[0], v[1], v[2], v[3]); *(float4*)(o + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
  }
  if (a.flags & P_SIGN) {
    v[0] *= cs[0].x; v[1] *= cs[0].y; v[2] *= cs[0].z; v[3] *= cs[0].w;
    v[4] *= cs[1].x; v[5] *= cs[1].y; v[6] *= cs[1].z; v[7] *= cs[1].w;
  }
  if (a.flags & P_HADAMARD) {
#pragma unroll
    for (int len = 1; len < 8; len <<= 1)
#pragma unroll
      for (int r = 0; r < 8; r++)
        if (!(r & len)) { const float p = v[r], q = v[r + len]; v[r] = p + q; v[r + len] = p - q; }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
#pragma unroll
      for (int r = 0; r < 8; r++) { const float q = __shfl_xor_sync(0xffffffff, v[r], o); v[r] = (lane & o) ? q - v[r] : v[r] + q; }
#pragma unroll
    for (int r = 0; r < 8; r++) xch[c][wl][lane * 8 + r] = v[r];
    __syncthreads();
    H5_LIT_MARKX(2);
    const int b8 = wl & 1, b9 = wl >> 1;
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const float A = xch[c][0][lane * 8 + r], B = xch[c][1][lane * 8 + r], C = xch[c][2][lane * 8 + r], Dd = xch[c][3][lane * 8 + r];
      const float t0 = b8 ? A - B : A + B, t1 = b8 ? C - Dd : C + Dd;   // stage bit 8 for b9 = 0, 1
      v[r] = (b9 ? t0 - t1 : t0 + t1) * 0.03125f;                        // stage bit 9, then scale
    }
  }
  if (a.flags & P_SIGN_AFTER) {
    v[0] *= cs[0].x; v[1] *= cs[0].y; v[2] *= cs[0].z; v[3] *= cs[0].w;
    v[4] *= cs[1].x; v[5] *= cs[1].y; v[6] *= cs[1].z; v[7] *= cs[1].w;
  }
  if (a.flags & P_STORE_F32) {
    float* o = a.out_f32 + (size_t)row * a.ldo + e0;
    *(float4*)o = make_float4(v[0], v[1], v[2], v[3]); *(float4*)(o + 4) = make_float4(v[4], v[5], v[6], v[7]);
  }
  if (!(a.flags & P_QUANT)) return;
  float bm = 0.f;
#pragma unroll
  for (int r = 0; r < 8; r++) bm = fmaxf(bm, fabsf(v[r]));
  bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, 1));   // 16-feature block = lane pair
  // a.fixed_g > 0 (H5_PREP_ROW_FIXEDG=1): the chunk route's fixed NVFP4 row scale, no row-max barrier.
  float g;
  if (a.fixed_g > 0.f) g = a.fixed_g;
  else {
    float am = warp_max(bm);
    if (lane == 0) red[1][w] = am;
    __syncthreads();
    H5_LIT_MARKX(3);
    am = 0.f; for (int i = 0; i < nw; i++) am = fmaxf(am, red[1][i]);
    g = fmaxf(am / (6.f * 448.f), 1e-30f);
  }
  const float ig = 1.f / g;
  if (t == 0) a.g4[row] = g;
  const float target = bm / 6.f * ig;
  __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
  float sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
  if (sv < target && sb < 0x7e) { sb++; sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3))); }
  const float rs = sv > 0.f ? ig / sv : 0.f;
  uint32_t wd = 0;
#pragma unroll
  for (int q = 0; q < 4; q++)
    wd |= (uint32_t)__nv_cvt_float2_to_fp4x2(make_float2(v[2 * q] * rs, v[2 * q + 1] * rs), __NV_E2M1, cudaRoundNearest) << (8 * q);
  const int tile = row >> 6, rr = row & 63, kb = e0 >> 7, o = e0 & 127, seg = o >> 5;
  const size_t base = ((size_t)tile * (n / 128) + kb) * 64 + rr;
  *(uint32_t*)(a.q4 + base * 64 + ((seg ^ ((rr >> 1) & 3)) << 4) + ((o & 31) >> 1)) = wd;
  if (!(lane & 1)) a.s4[base * 8 + (o >> 4)] = sb;
  prep_signal(a.done_sig);
}
template<bool CLEAR>
__global__ void __launch_bounds__(1024) k_prep2w(PrepArgs a) { H5_LIT(0);
  prep2w_body<CLEAR, false>(a, OrderedResidual{});
}
template<int SPLITS, bool FENCE>
__global__ void __launch_bounds__(640) k_prep2w_ordered_load4(PrepArgs a, OrderedResidual ordered) { H5_LIT(0);
  prep2w_body<false, true, SPLITS, FENCE>(a, ordered);
}
__global__ void __launch_bounds__(640) k_prep2w_ordered_bound640(PrepArgs a, OrderedResidual ordered) { H5_LIT(0);
  prep2w_body<false, true>(a, ordered);
}
template<bool CLEAR>
__global__ void __launch_bounds__(1024) k_prep2w_ordered(PrepArgs a, OrderedResidual ordered) { H5_LIT_SITE(Prep2wOrdered, 0);
  prep2w_body<CLEAR, true>(a, ordered);
}
__global__ void __launch_bounds__(544) k_prep_rms(PrepArgs a) { H5_LIT(0);
  __shared__ float partials[17];
  const int lane = threadIdx.x & 31, chunk = threadIdx.x >> 5, row = blockIdx.x;
  pdl_enter();
  const int irow = a.rowmap ? a.rowmap[row] : row;
  const float* x = a.x + (size_t)irow * a.ldx + chunk * 1024 + lane * 32;
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float4 t = *(const float4*)(x + i * 4);
    ss = fmaf(t.x,t.x,ss); ss = fmaf(t.y,t.y,ss);
    ss = fmaf(t.z,t.z,ss); ss = fmaf(t.w,t.w,ss);
  }
  ss = warp_sum(ss);
  if (!lane) partials[chunk] = ss;
  __syncthreads();
  if (!threadIdx.x) {
    ss = 0.f;
    for (int i = 0; i < a.n / 1024; i++) ss += partials[i];
    a.rms_inv[row] = rsqrtf(ss / (float)a.n + NORM_EPS);
  }
}
void launch_prep_rms(const PrepArgs& a, int rows, float fixed_g, cudaStream_t st) {
  if (rows < 1 || a.n < 1024 || a.n > 17408 || a.n % 1024 || a.ldx < a.n || !a.x || !a.norm_w ||
      !(a.flags & P_NORM) || (a.flags & (P_PERMUTE_GDN | P_GDN_NORM)) ||
      !a.rms_inv || a.rms_rows < rows || a.rms_inv == a.g4 ||
      !std::isfinite(fixed_g) || fixed_g <= 0.f || !std::isfinite(1.f / fixed_g))
    throw std::runtime_error("RMS prep requires ordinary row norm, dedicated row-capacity scratch, aligned width and finite positive scale");
  pdl(k_prep_rms, dim3(rows), dim3(32 * (a.n / 1024)), 0, st, a);
  pdl(k_prep_chunk<true>, dim3(rows, a.n / 1024), dim3(32), 0, st, a, fixed_g);
}

void launch_prep_row_norm(const PrepArgs& a, int rows, float fixed_g, cudaStream_t st) {
  if (rows < 1 || a.n < 1024 || a.n > 17408 || a.n % 1024 || a.ldx < a.n || !a.x || !a.norm_w ||
      !(a.flags & P_NORM) || (a.flags & (P_PERMUTE_GDN | P_GDN_NORM)) ||
      !std::isfinite(fixed_g) || fixed_g <= 0.f || !std::isfinite(1.f / fixed_g))
    throw std::runtime_error("coalesced row norm requires ordinary aligned row input and finite positive scale");
  constexpr unsigned norm_quant = P_NORM | P_SIGN | P_HADAMARD | P_QUANT;
  const dim3 grid(rows), block(32 * (a.n / 1024));
  if (a.n <= 6144 && a.flags == norm_quant)
    pdl(k_prep_chunk<false, true, norm_quant>, grid, block, 0, st, a, fixed_g);
  else if (a.n <= 6144 && a.flags == (norm_quant | P_OUT_NORM))
    pdl(k_prep_chunk<false, true, norm_quant | P_OUT_NORM>, grid, block, 0, st, a, fixed_g);
  else
    pdl(k_prep_chunk<false, true>, grid, block, 0, st, a, fixed_g);
}

struct PrepMapping {
  int mode;
  float g;
};
// Wide row-norm preparation selection (H5_PREP_WIDE=0|1|multi, or per engine via h5_set_prep_wide 0|1|2):
// 0 never, 1 every step, multi (2) only on multi-row steps. Eight-row verifier replay: wide -0.66%/-0.78%
// (both arm orders); one-row ordinary step: +0.12%/+0.08%.
static int& prep_wide_slot() {
  static int v = [] { const char* c = getenv("H5_PREP_WIDE"); if (!c) return 0;
    if (!strcmp(c, "0")) return 0; if (!strcmp(c, "1")) return 1; if (!strcmp(c, "multi")) return 2;
    throw std::runtime_error("H5_PREP_WIDE must be 0, 1 or multi"); }();
  return v;
}
static bool prep_wide(int rows) { const int v = prep_wide_slot(); return v == 1 || (v == 2 && rows > 1); }
static PrepMapping prep_mapping() {
  const char* mode = getenv("H5_PREP_CHUNK");
  if (!mode || !strcmp(mode, "0")) {
    if (getenv("H5_PREP_G4")) throw std::runtime_error("H5_PREP_G4 requires H5_PREP_CHUNK=1");
    return {0, 0.f};
  }
  if (strcmp(mode, "1") && strcmp(mode, "2") && strcmp(mode, "3") && strcmp(mode, "4")) throw std::runtime_error("H5_PREP_CHUNK must be 0 (row), 1 (all chunks), 2 (nonnorm chunks), 3 (RMS prepass), or 4 (coalesced row norm)");
  const char* scale = getenv("H5_PREP_G4");
  if (!scale) throw std::runtime_error("chunk preparation requires explicit H5_PREP_G4");
  char* end; errno = 0;
  const float g = strtof(scale, &end);
  if (errno || end == scale || *end || !std::isfinite(g) || g <= 0.f || !std::isfinite(1.f / g))
    throw std::runtime_error("H5_PREP_G4 must be a finite positive scale with finite reciprocal");
  return {atoi(mode), g};
}
float prep_fixed_g() {
  const PrepMapping m = prep_mapping();
  if (m.mode == 0) throw std::runtime_error("producer-tail prep requires chunk preparation (H5_PREP_CHUNK != 0) and its H5_PREP_G4");
  return m.g;
}
void launch_prep(const PrepArgs& a, int rows, cudaStream_t st) {
  // One-row steps only: on eight-row verifier rounds any explicit carveout measured ~1.2% slower.
  struct Scope { bool on; explicit Scope(bool o) : on(o) { if (on) carveout_prep_depth()++; } ~Scope() { if (on) carveout_prep_depth()--; } } carveout_scope_guard(rows == 1 || carveout_scope() == 2);
  static const PrepMapping mapping = prep_mapping();
  const bool clear = a.clear_a != nullptr || a.clear_b != nullptr;
  // Clearing runs in the row kernels (mapping 0, or row-norm preps under mapping 2) or in the chunk
  // kernel (chunk-routed preps); never in the RMS-prepass or coalesced row-norm routes.
  const bool chunk_route = mapping.mode == 1 || ((mapping.mode == 2 || mapping.mode == 3 || mapping.mode == 4) && !(a.flags & P_NORM));
  const bool clear_route_ok = chunk_route || mapping.mode == 0 || ((a.flags & P_NORM) && mapping.mode == 2);
  if (clear && (!clear_route_ok || rows < 1 ||
                a.n < 1024 || a.n > 17408 || a.n % 1024 || a.ldx < a.n ||
                !a.clear_a || a.clear_a_stride <= 0 || a.clear_a_stride % 4 || uintptr_t(a.clear_a) % 16 || a.rowmap ||
                (a.clear_b ? a.clear_b_stride <= 0 || a.clear_b_stride % 4 || uintptr_t(a.clear_b) % 16 : a.clear_b_stride != 0)))
    throw std::runtime_error("prep output clearing requires a row-norm (mapping 0/2) or chunk route and aligned declared outputs");
  if (!clear && (a.clear_a_stride != 0 || a.clear_b_stride != 0))
    throw std::runtime_error("prep output clear stride requires its pointer");
  if (clear) {
    const auto overlap = [](const void* p, size_t bytes, const void* q, size_t qbytes) {
      if (!p || !q || !bytes || !qbytes) return false;
      const uintptr_t x = uintptr_t(p), y = uintptr_t(q);
      return x <= y ? y - x < bytes : x - y < qbytes;
    };
    const size_t ba = size_t(rows) * a.clear_a_stride * sizeof(float);
    const size_t bb = size_t(rows) * a.clear_b_stride * sizeof(float);
    if (overlap(a.clear_a, ba, a.clear_b, bb))
      throw std::runtime_error("prep clear outputs overlap");
    const auto reject = [&](const void* p, size_t bytes) {
      if (overlap(a.clear_a, ba, p, bytes) || overlap(a.clear_b, bb, p, bytes))
        throw std::runtime_error("prep clear output aliases a live prep operand");
    };
    reject(a.x, size_t(rows) * a.ldx * sizeof(float));
    reject(a.x2, size_t(rows) * a.ldx * sizeof(float));
    reject(a.norm_w, size_t(a.n) * sizeof(float));
    reject(a.signs, size_t(a.n) * sizeof(float));
    reject(a.out_f32, size_t(rows) * a.ldo * sizeof(float));
    reject(a.out_norm, size_t(rows) * a.ldo * sizeof(float));
    reject(a.q4, size_t(rows) * a.n / 2);
    reject(a.s4, size_t(rows) * a.n / 16);
    reject(a.g4, size_t(rows) * sizeof(float));
    reject(a.xq, size_t(rows) * a.n);
    reject(a.xs, size_t(rows) * a.n / 128 * sizeof(float));
  }
  if (mapping.mode == 4 && (a.flags & P_NORM)) {
    launch_prep_row_norm(a, rows, mapping.g, st);
    return;
  }
  if (mapping.mode == 3 && (a.flags & P_NORM)) {
    launch_prep_rms(a, rows, mapping.g, st);
    return;
  }
  if ((a.flags & P_CONSUME_CLEAR) && !(a.flags & P_SILU_IL)) throw std::runtime_error("P_CONSUME_CLEAR requires P_SILU_IL");
  if ((a.flags & P_SILU_IL) && (!chunk_route || a.rowmap || a.x2 || a.ldx < 2 * a.n || (a.flags & (P_NORM | P_SILU_MUL | P_PERMUTE_GDN | P_GDN_NORM | P_DCONV)) || a.ldx != 2 * a.n))
    throw std::runtime_error("P_SILU_IL requires the chunk route, unmapped rows of interleaved gate/up (ldx >= 2n) and no other input transform");
  if ((a.flags & P_DCONV) && (!chunk_route || a.rowmap || a.n != D || a.ldx != D || (a.flags & ~(P_DCONV | P_QUANT)) ||
                              !a.dc_dyn || !a.dc_base || a.dc_which < 0 || a.dc_which > 1 || a.dc_block_rows < 1))
    throw std::runtime_error("P_DCONV requires the chunk route, D-wide unmapped rows, quantization only and conv operands");
  const bool chunk = chunk_route;
  if ((a.sig != nullptr) != (a.sig_expect > 0) || (a.sig && !chunk)) throw std::runtime_error("flag-started prep requires the chunk route and a positive expected count");
  // Chunk CTAs are independent per 1024 features, so the chunk route also takes the drafter's
  // concatenated capture rows (5 x 5120); the caller owns NVFP4 buffer capacity for that width.
  // H5_PREP_NORM_CHUNK=1 (read per launch): D-wide row-norm preps with NVFP4 output and nothing else run
  // chunk-parallel with the bound-based row scale (k_prep_normc).
  const bool norm_chunk = [] { const char* c = getenv("H5_PREP_NORM_CHUNK"); if (!c || !strcmp(c, "0")) return false;
    if (strcmp(c, "1")) throw std::runtime_error("H5_PREP_NORM_CHUNK must be 0 or 1"); return true; }();
  if (norm_chunk && !chunk && a.q4 && a.flags == (P_NORM | P_SIGN | P_HADAMARD | P_QUANT) && a.n == D && a.ldx >= D &&
      !a.rowmap && !a.x2 && !clear && a.norm_w && a.signs) {
    pdl(k_prep_normc, dim3(rows, D / 1024), dim3(128), 0, st, a);
    return;
  }
  if (chunk && (rows < 1 || a.n < 1024 || a.n > 25600 || a.n % 1024 || (a.n > 17408 && (a.flags & ~P_QUANT))))
    throw std::runtime_error("chunk prep requires positive rows and 1024-aligned width in [1024,17408] (25600 for quantize-only)");
  if (chunk && (a.flags & P_NORM) && (a.flags & (P_PERMUTE_GDN | P_GDN_NORM)))
    throw std::runtime_error("chunk prep requires ordinary input order for row RMSNorm");
  // H5_PREP_CHUNK_WIDE=1: four warps per (row, chunk) for the verifier's non-norm chunk flags with NVFP4 output.
  // (read per launch, i.e. at graph capture, so two engines of one process can differ)
  const int chunk_wide = [] { const char* c = getenv("H5_PREP_CHUNK_WIDE"); if (!c || !strcmp(c, "0")) return 0;
    if (strcmp(c, "1")) throw std::runtime_error("H5_PREP_CHUNK_WIDE must be 0 or 1"); return 1; }();
  constexpr unsigned wide_ok = P_PERMUTE_GDN | P_GDN_NORM | P_SIGN | P_HADAMARD | P_QUANT;
  if (a.done_sig && !(chunk && chunk_wide) && !(!chunk && prep_wide(rows)))
    throw std::runtime_error("prep done-signal requires the wide chunk or wide row route");
  if (chunk && chunk_wide && a.q4 && !a.rowmap && !a.x2 && !a.clear_a && !a.clear_b && (a.flags & P_HADAMARD) && (a.flags & P_QUANT) &&
      !(a.flags & ~wide_ok) && ((a.flags & P_GDN_NORM) != 0) == ((a.flags & P_PERMUTE_GDN) != 0))
    pdl(k_prep_chunkw, dim3(rows, a.n / 1024), dim3(128), 0, st, a, mapping.g);
  else if (chunk)
    pdl(k_prep_chunk<false>, dim3(rows, a.n / 1024), dim3(32), 0, st, a, mapping.g);
  else if (prep_wide(rows) && a.q4 && a.n <= 8192 && !a.rowmap && !a.x2 &&
           !(a.flags & ~(P_NORM | P_OUT_NORM | P_SIGN | P_HADAMARD | P_SIGN_AFTER | P_STORE_F32 | P_QUANT)) && (a.flags & P_QUANT)) {
    PrepArgs w = a;
    if (const char* fg = getenv("H5_PREP_ROW_FIXEDG")) {   // read per launch (graph capture)
      if (strcmp(fg, "0") && strcmp(fg, "1")) throw std::runtime_error("H5_PREP_ROW_FIXEDG must be 0 or 1");
      if (fg[0] == '1') { if (mapping.mode == 0) throw std::runtime_error("H5_PREP_ROW_FIXEDG requires H5_PREP_CHUNK and H5_PREP_G4"); w.fixed_g = mapping.g; }
    }
    if (clear) pdl(k_prep2w<true>, dim3(rows), dim3(128 * (a.n / 1024)), 0, st, w);
    else pdl(k_prep2w<false>, dim3(rows), dim3(128 * (a.n / 1024)), 0, st, w);
  }
  else if (clear)
    pdl(k_prep2<true>, dim3(rows), dim3(32 * (a.n / 1024)), 0, st, a);
  else
    pdl(k_prep2<false>, dim3(rows), dim3(32 * (a.n / 1024)), 0, st, a);
}

cudaError_t launch_prep_ordered(const PrepArgs& a, const OrderedResidual& ordered, cudaStream_t st) {
  const auto mapping = prep_mapping();
  if (!ordered.partials || !ordered.residual || ordered.residual != a.x || ordered.rows < 2 || ordered.rows > 16 ||
      ordered.splits < 2 || a.n != D || a.ldx != D || a.rowmap || a.x2 || !a.norm_w || !a.signs ||
      !a.q4 || !a.s4 || !a.g4 || a.flags != (P_NORM | P_SIGN | P_HADAMARD | P_QUANT) ||
      mapping.mode != 2 || !prep_wide(ordered.rows) || a.sig || a.done_sig || a.clear_a || a.clear_b ||
      ordered.partials == a.x || (getenv("H5_PREP_NORM_CHUNK") && strcmp(getenv("H5_PREP_NORM_CHUNK"), "0")))
    return cudaErrorInvalidValue;
  PrepArgs w = a;
  if (const char* fg = getenv("H5_PREP_ROW_FIXEDG")) {
    if (strcmp(fg, "0") && strcmp(fg, "1")) return cudaErrorInvalidValue;
    if (fg[0] == '1') w.fixed_g = mapping.g;
  }
  const char* pipe = getenv("H5_PREP_ORDERED_PIPE");
  if (pipe && strcmp(pipe, "0") && strcmp(pipe, "1") && strcmp(pipe, "2") && strcmp(pipe, "3")) return cudaErrorInvalidValue;
  const dim3 grid(ordered.rows), block(128 * (D / 1024));
  if (pipe && !strcmp(pipe, "1")) {
    if (ordered.splits == 6) pdl(k_prep2w_ordered_load4<6, false>, grid, block, 0, st, w, ordered);
    else if (ordered.splits == 17) pdl(k_prep2w_ordered_load4<17, false>, grid, block, 0, st, w, ordered);
    else return cudaErrorInvalidValue;
  } else if (pipe && !strcmp(pipe, "2")) {
    if (ordered.splits == 6) pdl(k_prep2w_ordered_load4<6, true>, grid, block, 0, st, w, ordered);
    else if (ordered.splits == 17) pdl(k_prep2w_ordered_load4<17, true>, grid, block, 0, st, w, ordered);
    else return cudaErrorInvalidValue;
  } else if (pipe && !strcmp(pipe, "3")) pdl(k_prep2w_ordered_bound640, grid, block, 0, st, w, ordered);
  else pdl(k_prep2w_ordered<false>, grid, block, 0, st, w, ordered);
  return cudaGetLastError();
}

// ------------------------------------------------------------------------------------------
// Embedding: decode one GGUF PTQ1_0 row (natural element order) per token. Grid rows, 64 threads.
// ------------------------------------------------------------------------------------------
// One thread per eight elements (rows x 640 threads): element e of a 128-block takes byte idx(e) after n(e)
// base-3 shifts, in GGUF PTQ1_0 natural order (bytes 0..15: 5 trits, 16..23: 5 trits, 24..25: 4 trits); the
// arithmetic per element is the previous serial decoder's.
__global__ void __launch_bounds__(640) k_embed(const uint8_t* __restrict__ w, const int* __restrict__ tok, float* __restrict__ out) { H5_LIT(0);
  pdl_enter();
  const int row = blockIdx.x, e0 = threadIdx.x * 8, b = e0 >> 7;
  float* o = out + (size_t)row * D + e0;
  const int t = tok[row];
  if (t < 0) { *(float4*)o = make_float4(0.f, 0.f, 0.f, 0.f); *(float4*)(o + 4) = make_float4(0.f, 0.f, 0.f, 0.f); return; }
  const uint8_t* blk = w + ((size_t)t * (D / 128) + b) * 28;
  const float scale = __half2float(*(const __half*)(blk + 26));
  float v[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const int e = (e0 + k) & 127;
    int idx, n;
    if (e < 80) { n = e >> 4; idx = e & 15; } else if (e < 120) { n = (e - 80) >> 3; idx = 16 + ((e - 80) & 7); } else { n = (e - 120) >> 1; idx = 24 + ((e - 120) & 1); }
    unsigned q = blk[idx];
    for (int r = 0; r < n; r++) q = (q * 3u) & 0xff;
    v[k] = (float)((int)((q * 3u) >> 8) - 1) * scale;
  }
  *(float4*)o = make_float4(v[0], v[1], v[2], v[3]); *(float4*)(o + 4) = make_float4(v[4], v[5], v[6], v[7]);
}
void launch_embed(const uint8_t* w, const int* tok, float* out, int rows, cudaStream_t st) { pdl(k_embed, dim3(rows), dim3(D / 8), 0, st, w, tok, out); }

// ------------------------------------------------------------------------------------------
__global__ void k_visual_input(const int* tok, const float* visual, float* x) { H5_LIT(0);
  pdl_enter();
  const int row = blockIdx.x;
  if (tok[row] != -1) return;
  for (int j = threadIdx.x; j < D; j += blockDim.x) x[(size_t)row * D + j] = visual[(size_t)row * D + j];
}
void launch_visual_input(const int* tok, const float* visual, float* x, int rows, cudaStream_t st) {
  pdl(k_visual_input, dim3(rows), dim3(256), 0, st, tok, visual, x);
}

// alpha/beta: out[row][96] = Wab[96][5120] (bf16) . xn[row]. Grid (12, rows), warp per output.
// ------------------------------------------------------------------------------------------
__global__ void k_ab(const __nv_bfloat16* __restrict__ w, const float* __restrict__ xn, float* __restrict__ out) { H5_LIT(0);
  pdl_enter();
  const int row = blockIdx.y, o = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const __nv_bfloat162* wr = (const __nv_bfloat162*)(w + (size_t)o * D);
  const float2* xr = (const float2*)(xn + (size_t)row * D);
  float acc = 0.f;
  for (int i = lane; i < D / 2; i += 32) { float2 wf = __bfloat1622float2(wr[i]); float2 xv = xr[i]; acc = fmaf(wf.x, xv.x, fmaf(wf.y, xv.y, acc)); }
  acc = warp_sum(acc);
  if (lane == 0) out[(size_t)row * 96 + o] = acc;
}
void launch_ab(const __nv_bfloat16* w, const float* xn, float* out, int rows, cudaStream_t st) { pdl(k_ab, dim3(dim3(12, rows)), dim3(256), 0, st, w, xn, out); }

// ------------------------------------------------------------------------------------------
// GDN conv + silu, and L2 normalisation of q and k heads. Grid (CONV_CH/128, nseg), 128 threads.
// Block b < 32 is a q (b < 16) or k head and is normalised per row; v blocks pass through.
// ------------------------------------------------------------------------------------------
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
template <typename PT>
__global__ void __launch_bounds__(128) k_conv(ConvArgs a) { H5_LIT(0);
  __shared__ float abv[16];   // fused alpha/beta: [row j][alpha, beta] for this v head
  const bool fused_ab = a.ab_w != nullptr;
  if (fused_ab && blockIdx.x >= 32) {
    // Before the wait: xn rows were written by the row prep two grids upstream, whose completion the
    // in-projection observed before triggering this launch; segments come from the step's first copy.
    __shared__ float part[4][16];
    const Seg s0g = a.segs[blockIdx.y];
    const int hv = blockIdx.x - 32, nb = s0g.len, t = threadIdx.x;
    const __nv_bfloat162* wa = (const __nv_bfloat162*)(a.ab_w + (size_t)hv * D);
    const __nv_bfloat162* wb = (const __nv_bfloat162*)(a.ab_w + (size_t)(HV + hv) * D);
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.f;
    for (int i = t; i < D / 2; i += 128) {
      const float2 fa = __bfloat1622float2(wa[i]), fb = __bfloat1622float2(wb[i]);
#pragma unroll
      for (int j = 0; j < 8; j++) if (j < nb) {
        const float2 xv = ((const float2*)(a.xn + (size_t)(s0g.start + j) * D))[i];
        acc[2 * j] = fmaf(fa.x, xv.x, fmaf(fa.y, xv.y, acc[2 * j]));
        acc[2 * j + 1] = fmaf(fb.x, xv.x, fmaf(fb.y, xv.y, acc[2 * j + 1]));
      }
    }
#pragma unroll
    for (int j = 0; j < 16; j++) { const float v = warp_sum(acc[j]); if ((t & 31) == 0) part[t >> 5][j] = v; }
    __syncthreads();
    if (t < 16) abv[t] = part[0][t] + part[1][t] + part[2][t] + part[3][t];
    __syncthreads();
    if (t < 2 * nb) a.ab_out[(size_t)(s0g.start + (t >> 1)) * 96 + (t & 1) * HV + hv] = abv[t];
  }
  // Variant bit 0x10: the conv state, the replayed pending inputs and the weights are loaded before the wait.
  // All of them were written by the previous step (segments by the step's first copy), which every grid of
  // this step's chain has acquired before triggering; stores stay behind the wait.
  const bool hoist = g_variant & 16u;
  if (!hoist) pdl_enter_e(a.early);
  __shared__ float red[2][4];
  const Seg sg = a.segs[blockIdx.y];
  const int c = blockIdx.x * 128 + threadIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const bool qk = blockIdx.x < 32;
  float* st = a.state + ((size_t)sg.slot * NGDN + a.gl) * 3 * CONV_CH;
  float s0 = st[c], s1 = st[CONV_CH + c], s2 = st[2 * CONV_CH + c];
  PT* pend = (PT*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * a.pending_rows * CONV_CH + c;
  for (int i = 0; i < sg.commit; i++) { s0 = s1; s1 = s2; s2 = st_ld(pend + (size_t)i * CONV_CH); }
  const float4 wk = *(const float4*)(a.w + c * 4);
  if (hoist) pdl_enter_e(a.early);
  if (a.pf_placement == h5::PfChainPlacement::Prologue) chain_l2pf(a.pf, a.pf_bytes, a.pf_granularity);
  H5_LIT_MARKX(0);   // after state setup/prologue hints; 3 body complete before optional tail hints
  if (sg.keep) { st[c] = s0; st[CONV_CH + c] = s1; st[2 * CONV_CH + c] = s2; }
  if (sg.len > 1) {
    // Batched rows (verifier rounds; measured -2.3% eight-row step): every input row is known before the recurrence, so issue all loads together and
    // reduce the q/k norms of up to eight rows behind one barrier. Arithmetic order is unchanged.
    __shared__ float redb[8][4];
    for (int i0 = 0; i0 < sg.len; i0 += 8) {
      const int nb = min(8, sg.len - i0);
      float xc[8], v[8], xg[8], xb[8];
#pragma unroll
      for (int j = 0; j < 8; j++) if (j < nb) xc[j] = a.proj[(size_t)(sg.start + i0 + j) * GDN_IN + c];
      const bool gate = blockIdx.x >= 32 && threadIdx.x == 0;
      const int hv = blockIdx.x - 32;
      if (gate) {
#pragma unroll
        for (int j = 0; j < 8; j++) if (j < nb) {
          if (fused_ab) { xg[j] = abv[2 * j]; xb[j] = abv[2 * j + 1]; }
          else { const size_t r = (size_t)(sg.start + i0 + j) * 96; xg[j] = a.ab[r + hv]; xb[j] = a.ab[r + HV + hv]; }
        }
      }
#pragma unroll
      for (int j = 0; j < 8; j++) if (j < nb) { v[j] = silu(wk.x * s0 + wk.y * s1 + wk.z * s2 + wk.w * xc[j]); s0 = s1; s1 = s2; s2 = xc[j]; }
      if (i0 == 0) H5_LIT_MARKX(1);
      if (qk) {
#pragma unroll
        for (int j = 0; j < 8; j++) if (j < nb) {
          float ss = v[j] * v[j];
          for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
          if (lane == 0) redb[j][w] = ss;
        }
        __syncthreads();
#pragma unroll
        if (i0 == 0) H5_LIT_MARKX(2);
        for (int j = 0; j < 8; j++) if (j < nb) {
          const float ss = redb[j][0] + redb[j][1] + redb[j][2] + redb[j][3];
          v[j] *= 1.f / fmaxf(sqrtf(ss), NORM_EPS);
        }
        if (i0 + 8 < sg.len) __syncthreads();
      }
#pragma unroll
      for (int j = 0; j < 8; j++) if (j < nb) {
        const int row = sg.start + i0 + j;
        a.out[(size_t)row * CONV_CH + c] = v[j];
        if (sg.keep) st_st(pend + (size_t)(i0 + j) * CONV_CH, xc[j]);
        if (gate) {
          const float x = xg[j] + a.dt_bias[hv];
          const float sp = x > 20.f ? x : log1pf(__expf(x));
          a.gates[(size_t)row * 96 + hv] = __expf(a.ssm_a[hv] * sp);
          a.gates[(size_t)row * 96 + HV + hv] = sigm(xb[j]);
        }
      }
    }
    if (!sg.keep) { st[c] = s0; st[CONV_CH + c] = s1; st[2 * CONV_CH + c] = s2; }
    H5_LIT_MARKX(3);
    if (a.pf_placement == h5::PfChainPlacement::Tail) chain_l2pf(a.pf, a.pf_bytes, a.pf_granularity);
    return;
  }
  for (int i = 0; i < sg.len; i++) {
    const int row = sg.start + i;
    const float xc = a.proj[(size_t)row * GDN_IN + c];
    float v = silu(wk.x * s0 + wk.y * s1 + wk.z * s2 + wk.w * xc);
    if (qk) {
      float ss = v * v;
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      if (lane == 0) red[i & 1][w] = ss;
      __syncthreads();
      ss = red[i & 1][0] + red[i & 1][1] + red[i & 1][2] + red[i & 1][3];
      v *= 1.f / fmaxf(sqrtf(ss), NORM_EPS);
    }
    a.out[(size_t)row * CONV_CH + c] = v;
    if (sg.keep) st_st(pend + (size_t)i * CONV_CH, xc);
    if (blockIdx.x >= 32 && threadIdx.x == 0) {   // v head: decay and beta gates for this row
      const int hv = blockIdx.x - 32;
      const float xg = (fused_ab ? abv[2 * i] : a.ab[(size_t)row * 96 + hv]) + a.dt_bias[hv];
      const float sp = xg > 20.f ? xg : log1pf(__expf(xg));
      a.gates[(size_t)row * 96 + hv] = __expf(a.ssm_a[hv] * sp);
      a.gates[(size_t)row * 96 + HV + hv] = sigm(fused_ab ? abv[2 * i + 1] : a.ab[(size_t)row * 96 + HV + hv]);
    }
    s0 = s1; s1 = s2; s2 = xc;
  }
  if (!sg.keep) { st[c] = s0; st[CONV_CH + c] = s1; st[2 * CONV_CH + c] = s2; }
  H5_LIT_MARKX(3);
  if (a.pf_placement == h5::PfChainPlacement::Tail) chain_l2pf(a.pf, a.pf_bytes, a.pf_granularity);
}
void launch_conv(const ConvArgs& a, int nseg, StateFormat format, cudaStream_t st) {
  if (a.ab_w && (!a.xn || !a.ab_out)) throw std::runtime_error("fused alpha/beta conv requires xn and ab_out");
  if (format == StateFormat::F16 || format == StateFormat::Q8) pdl(k_conv<__half>, dim3(CONV_CH / 128, nseg), dim3(128), 0, st, a);
  else if (format == StateFormat::BF16) pdl(k_conv<__nv_bfloat16>, dim3(dim3(CONV_CH / 128, nseg)), dim3(128), 0, st, a);
  else pdl(k_conv<float>, dim3(dim3(CONV_CH / 128, nseg)), dim3(128), 0, st, a);
}

// ------------------------------------------------------------------------------------------
// GDN recurrence + gated RMSNorm. Grid (HV, nseg), 512 threads.
// State rows are independent: m_j <- g m_j + beta (v_j - g m_j.k) k ;  o_j = m_j.q / sqrt(128).
// Lane (warp w, lane l) owns row j = 8w + l/4, columns 32(l%4) .. +31; a row's dot products reduce
// over 4 lanes. All tokens of the segment (replayed + new) are staged in shared memory first.
// ------------------------------------------------------------------------------------------
// p points at column 4*cg of the row; register group i holds columns 16 i + 4 cg .. +3
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
// Pipelined GDN for many streams (H5_GDN_PIPE=SEGS, read per launch): each CTA owns one head half and walks
// several streams; the next stream's state rows (16 KB) are bulk-copied into shared memory while the current
// stream's recurrence runs, so DRAM streams state continuously instead of in per-CTA load/compute phases.
__device__ __forceinline__ unsigned gp_smem(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
template <typename ST, int CG>
__global__ void __launch_bounds__(256, 2) k_gdn_pipe(GdnArgs a, int nseg) { H5_LIT(0);
  constexpr int JR = 256 / CG;
  constexpr unsigned BYTES = JR * 128 * sizeof(ST);
  __shared__ __align__(16) uint8_t sm[GDN_SMEM];
  __shared__ __align__(128) ST sbuf[JR * 128];   // one buffer: refilled as soon as the state is in registers
  __shared__ __align__(8) uint64_t bar;
  const int h = blockIdx.x / (CG / 2), half = blockIdx.x % (CG / 2);
  auto src = [&](int s) { return (const ST*)a.state + (((size_t)a.segs[s].slot * NGDN + a.gl) * HV + h) * 128 * 128 + (size_t)JR * half * 128; };
  int s = blockIdx.y;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(gp_smem(&bar)));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (s < nseg) {   // GDN state is written only by this kernel (another layer/stream): safe before the wait
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(gp_smem(&bar)), "r"(BYTES) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(gp_smem(sbuf)), "l"(src(s)), "r"(BYTES), "r"(gp_smem(&bar)) : "memory");
    }
  }
  __syncthreads();
  for (int it = 0; s < nseg; s += gridDim.y, it++) {
    asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(gp_smem(&bar)), "r"((unsigned)it & 1u) : "memory");
    const int ns = s + gridDim.y;
    GdnPrefetch pf; pf.buf = sbuf; pf.bar = &bar; pf.bytes = BYTES; pf.next = ns < nseg ? (const void*)src(ns) : nullptr;
    gdn_body<ST, CG, false, true, false>(a, a.segs[s], blockIdx.x, sm, ConvArgs{}, sbuf, pf);
  }
}
// GDN with the ssm_out prep in its tail: the CTA that completes the last of the 8 heads of a Hadamard
// chunk (permuted head order q = 3 (h % 16) + h / 16) runs the gated RMSNorm, sign, Hadamard and NVFP4
// for every row of that chunk.
template <typename ST, int CG, bool PACKED = false, bool Q8 = false>
__global__ void __launch_bounds__(256, 2) k_gdn_tail(GdnArgs a, TailArgs ta) { H5_LIT(0);
#ifndef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter();
#endif
  __shared__ __align__(16) uint8_t sm[GDN_SMEM + (PACKED ? GDN_PACKED_SMEM : 0)];
#ifdef H5_PDL_GDN_STATE_OVERLAP
  gdn_body<ST, CG, false, true, false, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm);
#else
  gdn_body<ST, CG, false, false, false, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm);
#endif
  __shared__ unsigned tail_word;
  const int h = blockIdx.x / (CG / 2), c = (3 * (h % 16) + h / 16) >> 3;
  const unsigned long long t_arr = ta.dbg ? tail_clock() : 0ull;
  TailPre<1> pre;
  tail_preload<1, 8>(ta, c, pre);
  if (tail_arrive(ta, c, 8u * (CG / 2) * gridDim.y, threadIdx.x, 256, &tail_word)) {
    if (ta.dbg && threadIdx.x == 0) ta.dbg[c * TAIL_DBG_STAMPS] = t_arr;
    tail_finalise<TAIL_CHUNK, 8, 1>(ta, c, threadIdx.x, 256, pre);
  }
}
static bool gdn_packed(StateFormat format);
static int gdn_cg(int nseg);
template <typename ST, bool PACKED = false, bool Q8 = false>
void gdn_tail_dispatch(const GdnArgs& a, int nseg, const TailArgs& ta, cudaStream_t st) {
  static int ordinary_cg = getenv("H5_GDN_CG") ? atoi(getenv("H5_GDN_CG")) : 4;
  const int cg = (PACKED || Q8) ? gdn_cg(nseg) : ordinary_cg;
  if (cg == 8) pdl(k_gdn_tail<ST,8,PACKED,Q8>, dim3(4 * HV, nseg), dim3(256), 0, st, a, ta);
  else if (cg == 16) pdl(k_gdn_tail<ST,16,PACKED,Q8>, dim3(8 * HV, nseg), dim3(256), 0, st, a, ta);
  else pdl(k_gdn_tail<ST,4,PACKED,Q8>, dim3(2 * HV, nseg), dim3(256), 0, st, a, ta);
}
void launch_gdn_tail(const GdnArgs& a, int nseg, StateFormat format, const TailArgs& ta, cudaStream_t st) {
  validate_q8_scale_mode(a.q8_scale);
  constexpr unsigned flags = P_PERMUTE_GDN | P_GDN_NORM | P_SIGN | P_HADAMARD | P_QUANT;
  if (ta.x != a.y || ta.ldx != VDIM || ta.n != VDIM || ta.flags != flags || ta.R < 1 || ta.R > TAIL_MAX_ROWS ||
      !ta.mul || !ta.gnorm_w || !ta.gz || ta.gz_ld < GDN_IN || !ta.q4 || !ta.s4 || !ta.g4 || !ta.cnt || !(ta.fixed_g > 0.f) ||
      ta.clear_a || ta.clear_b)
    throw std::runtime_error("GDN tail prep requires the y rows, the gated-RMSNorm chunk flags, at most 8 rows and its operands");
  const bool packed = gdn_packed(format);
  if (format == StateFormat::Q8) {
    gdn_tail_dispatch<__half,false,true>(a, nseg, ta, st);
  } else if (format == StateFormat::F16) {
    if (packed) gdn_tail_dispatch<__half,true>(a, nseg, ta, st);
    else gdn_tail_dispatch<__half>(a, nseg, ta, st);
  } else if (format == StateFormat::BF16) {
    if (packed) gdn_tail_dispatch<__nv_bfloat16,true>(a, nseg, ta, st);
    else gdn_tail_dispatch<__nv_bfloat16>(a, nseg, ta, st);
  }
  else gdn_tail_dispatch<float>(a, nseg, ta, st);
}
// H5_GDN_CG=4|8|16, H5_GDN_OCC=2|3 (minimum resident CTAs per SM in the launch bounds; 3 caps registers at
// 85) and H5_GDN_SCALED=0|1, all read per launch (at graph capture) so engines of one process can differ.
static int gdn_env(const char* name, int unset, std::initializer_list<int> ok) {
  const char* c = getenv(name); if (!c) return unset;
  char* e = nullptr; const long v = strtol(c, &e, 10);
  for (int o : ok) if (e && !*e && e != c && v == o) return o;
  throw std::runtime_error(std::string(name) + " has an unsupported value");
}
static bool gdn_packed(StateFormat format) {
  if (format == StateFormat::Q8) {
    if (gdn_env("H5_GDN_SCALED", 0, {0, 1}) ||
        gdn_env("H5_GDN_PIPE", 0, {0, 2, 4, 8, 16, 24, 32, 48, 64}) ||
        gdn_env("H5_GDN_OCC", 2, {2, 3}) != 2)
      throw std::runtime_error("Q8 GDN requires SCALED=0, PIPE=0, OCC=2");
#if defined(EXP_GDN_NOMEM) || defined(EXP_GDN_NOCOMP)
    throw std::runtime_error("Q8 GDN requires state memory and recurrence computation");
#endif
  }
  const char* value = getenv("H5_GDN_ARITH");
  if (!value || !strcmp(value, "fp32")) return false;
  if (strcmp(value, "fp16x2") && strcmp(value, "bf16x2"))
    throw std::runtime_error("H5_GDN_ARITH requires fp32|fp16x2|bf16x2");
  const StateFormat required = !strcmp(value, "fp16x2") ? StateFormat::F16 : StateFormat::BF16;
  if (format != required) throw std::runtime_error("H5_GDN_ARITH packed arithmetic requires matching 16-bit state format");
  if (gdn_env("H5_GDN_SCALED", 0, {0, 1}) ||
      gdn_env("H5_GDN_PIPE", 0, {0, 2, 4, 8, 16, 24, 32, 48, 64}) ||
      gdn_env("H5_GDN_OCC", 2, {2, 3}) != 2 || (gdn_host_variant & 96u))
    throw std::runtime_error("packed GDN requires SCALED=0, PIPE=0, OCC=2 and no closed-form/output-identity variants (0x20/0x40)");
#if defined(EXP_GDN_NOMEM) || defined(EXP_GDN_NOCOMP)
  throw std::runtime_error("packed GDN requires state memory and recurrence computation");
#endif
  return true;
}
// H5_GDN_CG_SEGS=N (read per launch): steps with at least N segments (streams) use CG4 whatever H5_GDN_CG says;
// many streams fill the GPU with CTAs already, and four lanes per state row halve the shuffle rounds.
static int gdn_cg(int nseg) {
  const int segs = gdn_env("H5_GDN_CG_SEGS", 0, {0, 2, 4, 8, 12, 16, 24, 32, 48, 64});
  return segs && nseg >= segs ? 4 : gdn_env("H5_GDN_CG", 4, {4, 8, 16});
}
template <typename ST, bool PACKED = false>
void gdn_dispatch(const GdnArgs& a, int nseg, cudaStream_t st) {
  if constexpr (PACKED) {
    const int cg = gdn_cg(nseg);
    if (cg == 8) pdl(k_gdn<ST,8,2,false,true>, dim3(4 * HV, nseg), dim3(256), 0, st, a);
    else if (cg == 16) pdl(k_gdn<ST,16,2,false,true>, dim3(8 * HV, nseg), dim3(256), 0, st, a);
    else pdl(k_gdn<ST,4,2,false,true>, dim3(2 * HV, nseg), dim3(256), 0, st, a);
    return;
  }
  // H5_GDN_PIPE=N: steps with at least N streams run the pipelined CG4 kernel, streams spread over
  // ceil(nseg / H5_GDN_PIPE_PER) CTA rows (default 4 streams per CTA).
  const int pipe = gdn_env("H5_GDN_PIPE", 0, {0, 2, 4, 8, 16, 24, 32, 48, 64});
  if constexpr (sizeof(ST) == 2)   // f32 state rows do not fit beside the staging buffers in static shared memory
  if (pipe && nseg >= pipe && !a.sig && !a.replay_delta) {
    const int per = gdn_env("H5_GDN_PIPE_PER", 4, {1, 2, 3, 4, 6, 8, 16});
    pdl(k_gdn_pipe<ST, 4>, dim3(2 * HV, (nseg + per - 1) / per), dim3(256), 0, st, a, nseg);
    return;
  }
  const int cg = gdn_cg(nseg), occ = gdn_env("H5_GDN_OCC", 2, {2, 3}), sc = gdn_env("H5_GDN_SCALED", 0, {0, 1});
  const dim3 g8(4 * HV, nseg), g16(8 * HV, nseg), b(256);
  if (cg == 4 && nseg >= 2 && (occ != 2 || sc) && gdn_env("H5_GDN_CG_SEGS", 0, {0, 2, 4, 8, 12, 16, 24, 32, 48, 64})) { pdl(k_gdn<ST,4>, dim3(2 * HV, nseg), b, 0, st, a); return; }
  if (cg == 8) {
    if (occ == 3) { if (sc) pdl(k_gdn<ST,8,3,true>, g8, b, 0, st, a); else pdl(k_gdn<ST,8,3,false>, g8, b, 0, st, a); }
    else { if (sc) pdl(k_gdn<ST,8,2,true>, g8, b, 0, st, a); else pdl(k_gdn<ST,8,2,false>, g8, b, 0, st, a); }
  } else if (cg == 16) {
    if (occ == 3) { if (sc) pdl(k_gdn<ST,16,3,true>, g16, b, 0, st, a); else pdl(k_gdn<ST,16,3,false>, g16, b, 0, st, a); }
    else { if (sc) pdl(k_gdn<ST,16,2,true>, g16, b, 0, st, a); else pdl(k_gdn<ST,16,2,false>, g16, b, 0, st, a); }
  } else {
    if (occ != 2 || sc) throw std::runtime_error("H5_GDN_OCC=3 and H5_GDN_SCALED=1 require H5_GDN_CG=8 or 16");
    pdl(k_gdn<ST,4>, dim3(2 * HV, nseg), b, 0, st, a);
  }
}
int gdn_grid_ctas(int nseg) {
  const int cg = gdn_cg(nseg);
  return (cg == 8 ? 4 * HV : cg == 16 ? 8 * HV : 2 * HV) * nseg;
}
void launch_gdn(const GdnArgs& a, int nseg, StateFormat format, cudaStream_t st) {
  validate_q8_scale_mode(a.q8_scale);
  const bool packed = gdn_packed(format);
  if (format == StateFormat::Q8) {
    const int cg = gdn_cg(nseg);
    if (cg == 8) pdl(k_gdn<__half,8,2,false,false,true>, dim3(4 * HV, nseg), dim3(256), 0, st, a);
    else if (cg == 16) pdl(k_gdn<__half,16,2,false,false,true>, dim3(8 * HV, nseg), dim3(256), 0, st, a);
    else pdl(k_gdn<__half,4,2,false,false,true>, dim3(2 * HV, nseg), dim3(256), 0, st, a);
  } else if (format == StateFormat::F16) {
    if (packed) gdn_dispatch<__half,true>(a, nseg, st);
    else gdn_dispatch<__half>(a, nseg, st);
  } else if (format == StateFormat::BF16) {
    if (packed) gdn_dispatch<__nv_bfloat16,true>(a, nseg, st);
    else gdn_dispatch<__nv_bfloat16>(a, nseg, st);
  }
  else gdn_dispatch<float>(a, nseg, st);
}

template <typename ST, int CG, bool PACKED = false, bool Q8 = false>
__global__ void __launch_bounds__(256, 2) k_conv_gdn(ConvArgs c, GdnArgs a) { H5_LIT(0);
  __shared__ __align__(16) uint8_t sm[GDN_SMEM + (PACKED ? GDN_PACKED_SMEM : 0)];
#ifdef H5_PDL_GDN_STATE_OVERLAP
  // GDN state, conv carry and replayed rows were all written by the previous step: loaded before the wait.
  gdn_body<ST, CG, true, true, false, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm, c);
#else
  pdl_enter();
  gdn_body<ST, CG, true, false, false, PACKED, Q8>(a, a.segs[blockIdx.y], blockIdx.x, sm, c);
#endif
}
template <typename ST, bool PACKED = false, bool Q8 = false>
void conv_gdn_dispatch(const ConvArgs& c, const GdnArgs& a, int nseg, cudaStream_t st) {
  const char* value = getenv("H5_GDN_CG");
  const int cg = (PACKED || Q8) ? gdn_cg(nseg) : (value ? atoi(value) : 4);
  if (cg == 4) pdl(k_conv_gdn<ST,4,PACKED,Q8>, dim3(2 * HV, nseg), dim3(256), 0, st, c, a);
  else if (cg == 8) pdl(k_conv_gdn<ST,8,PACKED,Q8>, dim3(4 * HV, nseg), dim3(256), 0, st, c, a);
  else if (cg == 16) pdl(k_conv_gdn<ST,16,PACKED,Q8>, dim3(8 * HV, nseg), dim3(256), 0, st, c, a);
  else { fprintf(stderr, "fused GDN requires H5_GDN_CG=4|8|16\n"); abort(); }
}
void launch_conv_gdn(const ConvArgs& c, const GdnArgs& a, int nseg, StateFormat format, cudaStream_t st) {
  validate_q8_scale_mode(a.q8_scale);
  const bool packed = gdn_packed(format);
  if (format == StateFormat::Q8) {
    conv_gdn_dispatch<__half,false,true>(c, a, nseg, st);
  } else if (format == StateFormat::F16) {
    if (packed) conv_gdn_dispatch<__half,true>(c, a, nseg, st);
    else conv_gdn_dispatch<__half>(c, a, nseg, st);
  } else if (format == StateFormat::BF16) {
    if (packed) conv_gdn_dispatch<__nv_bfloat16,true>(c, a, nseg, st);
    else conv_gdn_dispatch<__nv_bfloat16>(c, a, nseg, st);
  }
  else conv_gdn_dispatch<float>(c, a, nseg, st);
}
#include "gdn_single.cuh"
#include "gdn_verify.cuh"
#include "gdn_verify_fused.cuh"

// ------------------------------------------------------------------------------------------
// Attention pre: q/k RMSNorm, RoPE on dims 0..63 (neox pairs), K/V append. Grid (NH+NKV, rows).
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ void attn_pre_body(const AttnArgs& a, int bx, int row) {
  __shared__ float s[HD];
  __shared__ float red[8];
  const int d = threadIdx.x;
  const int pos = a.row_pos[row], slot = a.row_slot[row];
  const bool is_q = bx < NH;
  const int hq = bx, hk = bx - NH;
  const float* pr = a.proj + (size_t)row * ATT_IN;
  float v = is_q ? pr[hq * 2 * HD + d] : pr[Q_OUT + hk * HD + d];
  const float ss = block_sum_256(v * v, red);
  v = v * rsqrtf(ss / (float)HD + NORM_EPS) * (is_q ? a.q_norm[d] : a.k_norm[d]);
  s[d] = v;
  __syncthreads();
  float outv = v;
  if (d < NROT) {
    const int i = d < NROT / 2 ? d : d - NROT / 2;
    int rp = pos;
    if (a.rope_pos) {
      // Qwen35 IMRoPE interleaves t/h/w frequencies, not contiguous M-RoPE sections.
      const int axis = i % 3 == 1 && i < 3 * a.rope_sections[1] ? 1 :
                       i % 3 == 2 && i < 3 * a.rope_sections[2] ? 2 :
                       i % 3 == 0 && i < 3 * a.rope_sections[0] ? 0 : 3;
      rp = a.rope_pos[4 * row + axis];
    }
    const float theta = (float)rp * powf(ROPE_BASE, -(float)i / (float)(NROT / 2));
    float c, sn; sincosf(theta, &sn, &c);
    outv = d < NROT / 2 ? s[d] * c - s[d + NROT / 2] * sn : s[d - NROT / 2] * sn + s[d] * c;
  }
  if (is_q) { a.qrot[(size_t)row * NH * HD + hq * HD + d] = outv * 0.0625f; return; }
  const size_t pidx = (((size_t)slot * NATTN + a.al) * NKV + hk) * a.ctx + pos;
  const float vv = pr[Q_OUT + KV_OUT + hk * HD + d];
  if (!a.ksc) {
    ((__half*)a.kc)[pidx * HD + d] = __float2half(outv);
    ((__half*)a.vc)[pidx * HD + d] = __float2half(vv);
    return;
  }
  __syncthreads();
  float km = fabsf(outv), vm = fabsf(vv);
  for (int o = 16; o; o >>= 1) { km = fmaxf(km, __shfl_xor_sync(0xffffffff, km, o)); vm = fmaxf(vm, __shfl_xor_sync(0xffffffff, vm, o)); }
  __shared__ float rk[8], rv[8];
  if ((d & 31) == 0) { rk[d >> 5] = km; rv[d >> 5] = vm; }
  __syncthreads();
  km = 0.f; vm = 0.f;
  for (int i = 0; i < 8; i++) { km = fmaxf(km, rk[i]); vm = fmaxf(vm, rv[i]); }
  const float ks = fmaxf(km, 1e-20f) / 448.f, vs = fmaxf(vm, 1e-20f) / 448.f;
  ((uint8_t*)a.kc)[pidx * HD + d] = __nv_cvt_float_to_fp8(outv / ks, __NV_SATFINITE, __NV_E4M3);
  if (a.vt) {   // V transposed per 32-key block: [pos/32][dim][pos%32]
    const size_t hbase = (pidx - pos) * HD;   // start of this (slot, layer, head) region
    ((uint8_t*)a.vc)[hbase + (size_t)(pos >> 5) * HD * 32 + d * 32 + (pos & 31)] = __nv_cvt_float_to_fp8(vv / vs, __NV_SATFINITE, __NV_E4M3);
  } else ((uint8_t*)a.vc)[pidx * HD + d] = __nv_cvt_float_to_fp8(vv / vs, __NV_SATFINITE, __NV_E4M3);
  if (d == 0) { a.ksc[pidx] = ks; a.vsc[pidx] = vs; }
}
__global__ void __launch_bounds__(256) k_attn_pre(AttnArgs a) { H5_LIT(0);
  pdl_enter_e(a.early);
  if (a.pf_placement == h5::PfChainPlacement::Prologue) chain_l2pf(a.pf, a.pf_bytes, a.pf_granularity);
  H5_LIT_MARKX(0);
  attn_pre_body(a, blockIdx.x, blockIdx.y);
  H5_LIT_MARKX(1);
  if (a.pf_placement == h5::PfChainPlacement::Tail) chain_l2pf(a.pf, a.pf_bytes, a.pf_granularity);
  H5_LIT_MARKX(2);
}
void launch_attn_pre(const AttnArgs& a, int rows, cudaStream_t st) { pdl(k_attn_pre, dim3(dim3(NH + NKV, rows)), dim3(256), 0, st, a); }

// Argmax over each logits row. Grid rows, 1024 threads.
// Top-2 merge for the relaxed argmax statistics: (b1,i1) > (b2,i2) in (value desc, index asc) order.
__device__ __forceinline__ bool argmax_better(float v, int i, float w, int j) { return v > w || (v == w && i < j); }
__device__ __forceinline__ void top2_merge(float& b1, int& i1, float& b2, int& i2, float o1, int oi1, float o2, int oi2) {
  if (argmax_better(o1, oi1, b1, i1)) {
    // other's first wins: second is the better of our first and other's second
    if (argmax_better(b1, i1, o2, oi2)) { b2 = b1; i2 = i1; } else { b2 = o2; i2 = oi2; }
    b1 = o1; i1 = oi1;
  } else if (argmax_better(o1, oi1, b2, i2)) { b2 = o1; i2 = oi1; }
}
template<bool RELAX = false>
__global__ void __launch_bounds__(1024) k_argmax(const float* __restrict__ logits, int ld, int* __restrict__ out,
    const int* tokens = nullptr, const Seg* segments = nullptr, int nseg = 0, int row0 = 0, float margin = 0.f,
    const int* rowmap = nullptr, RowStat* stats = nullptr, int stats_mode = 0) { H5_LIT(0);
  pdl_enter();
  __shared__ float bv[32], bv2[32], bs[32], bt[32]; __shared__ int bi[32], bi2[32];
  const float* l = logits + (size_t)blockIdx.x * ld;
  float best = -INFINITY; int idx = 0;
  float sec = -INFINITY; int sidx = 0x7fffffff;
  // stats_mode >= 1: also the runner-up; 2: online softmax s = sum exp(v - best), t = sum exp(v - best) * v.
  float sm = 0.f, tm = 0.f;
  if (RELAX && stats_mode == 2) {
    for (int i = threadIdx.x; i < VOCAB; i += 1024) {
      const float v = l[i];
      if (v > best) { const float c = __expf(best - v); sm = sm * c + 1.f; tm = tm * c + v; sec = best; sidx = idx; best = v; idx = i; }
      else { const float c = __expf(v - best); sm += c; tm += c * v; if (v > sec) { sec = v; sidx = i; } }
    }
  } else if (RELAX && stats_mode == 1) {
    for (int i = threadIdx.x; i < VOCAB; i += 1024) {
      const float v = l[i];
      if (v > best) { sec = best; sidx = idx; best = v; idx = i; } else if (v > sec) { sec = v; sidx = i; }
    }
  } else
    for (int i = threadIdx.x; i < VOCAB; i += 1024) { float v = l[i]; if (v > best) { best = v; idx = i; } }
  auto merge_soft = [&](float ob, float os, float ot) {
    // (best, sm, tm) is still this lane's pre-merge maximum here.
    const float m = fmaxf(best, ob);
    const float ca = best == -INFINITY ? 0.f : __expf(best - m), cb = ob == -INFINITY ? 0.f : __expf(ob - m);
    sm = sm * ca + os * cb; tm = tm * ca + ot * cb;
  };
  auto reduce = [&]() {
    for (int o = 16; o; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, best, o); int oi = __shfl_xor_sync(0xffffffff, idx, o);
      if (RELAX && stats_mode) {
        const float ov2 = __shfl_xor_sync(0xffffffff, sec, o); const int oi2 = __shfl_xor_sync(0xffffffff, sidx, o);
        if (stats_mode == 2) {
          const float os = __shfl_xor_sync(0xffffffff, sm, o), ot = __shfl_xor_sync(0xffffffff, tm, o);
          merge_soft(ov, os, ot);
        }
        top2_merge(best, idx, sec, sidx, ov, oi, ov2, oi2);
      } else if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
  };
  reduce();
  if ((threadIdx.x & 31) == 0) { const int w = threadIdx.x >> 5; bv[w] = best; bi[w] = idx; bv2[w] = sec; bi2[w] = sidx; bs[w] = sm; bt[w] = tm; }
  __syncthreads();
  if (threadIdx.x < 32) {
    best = bv[threadIdx.x]; idx = bi[threadIdx.x]; sec = bv2[threadIdx.x]; sidx = bi2[threadIdx.x]; sm = bs[threadIdx.x]; tm = bt[threadIdx.x];
    reduce();
    if (threadIdx.x == 0) {
      if constexpr (RELAX) {
        const int compact_row = row0 + blockIdx.x;
        const int row = rowmap ? rowmap[compact_row] : compact_row;
        const int greedy = idx;
        float score = __int_as_float(0x7fc00000);
        for (int s = 0; s < nseg; ++s) {
          const Seg seg = segments[s];
          if (!seg.keep || row < seg.start || row + 1 >= seg.start + seg.len) continue;
          const int proposed = tokens[row + 1];
          if (proposed >= 0 && proposed < VOCAB) score = l[proposed];
          // Never relax EOS or a non-finite target; the final bonus row remains greedy.
          if (proposed >= 0 && proposed < VOCAB && relaxed_draft_allowed(idx, proposed, best, l[proposed], margin, VOCAB)) idx = proposed;
          break;
        }
        if (stats) {
          const float nan = __int_as_float(0x7fc00000);
          const float lse = stats_mode == 2 ? best + __logf(sm) : nan;
          stats[compact_row] = RowStat{greedy, best, score, lse, stats_mode == 2 ? lse - tm / sm : nan, sidx, sec};
        }
      }
      out[blockIdx.x] = idx;
    }
  }
}
// Sparse per-row logit adjustment (H5_LOGIT_ADJ=K): row r of the head chunk adds adj[full_row * K + j].y (float
// bits) to token adj[...].x for every entry with x >= 0, before the (relaxed) argmax reads the row. Host-built
// repetition penalties (relax_policy.h, dry=...) use it; tokens within a row are unique.
__global__ void k_logit_adj(float* __restrict__ logits, int ld, const int2* __restrict__ adj, int K, int row0, const int* rowmap) {
  H5_LIT(0);
  pdl_enter();
  const int compact = row0 + blockIdx.x;
  const int row = rowmap ? rowmap[compact] : compact;
  for (int j = threadIdx.x; j < K; j += blockDim.x) {
    const int2 a = adj[(size_t)row * K + j];
    if (a.x >= 0 && a.x < VOCAB) logits[(size_t)blockIdx.x * ld + a.x] += __int_as_float(a.y);
  }
}
void launch_logit_adj(float* logits, int ld, int rows, const int2* adj, int K, int row0, const int* rowmap, cudaStream_t st) {
  pdl(k_logit_adj, dim3(rows), dim3(32), 0, st, logits, ld, adj, K, row0, rowmap);
}
__global__ void k_grammar_mask(float* logits, int ld, const uint32_t* masks, int row0, const int* rowmap) {
  H5_LIT(0);
  pdl_enter();
  const int compact = row0 + blockIdx.y;
  const int row = rowmap ? rowmap[compact] : compact;
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token < ld && !(masks[(size_t)row * GRAMMAR_WORDS + token / 32] & (1u << (token % 32))))
    logits[(size_t)blockIdx.y * ld + token] = -INFINITY;
}
void launch_grammar_mask(float* logits, int ld, int rows, const uint32_t* masks, int row0, const int* rowmap, cudaStream_t st) {
  pdl(k_grammar_mask, dim3((ld + 255) / 256, rows), dim3(256), 0, st, logits, ld, masks, row0, rowmap);
}

// Split greedy argmax (H5_ARGMAX_SPLIT=1): ARGMAX_SLICES CTAs per row each reduce a vocabulary slice with
// float4 loads, then combine by a 64-bit atomicMax on (orderable value, ~index), so ties keep the lowest
// index as k_argmax does. The last slice CTA of a row (arrival counter) writes the token and resets the key
// and counter, so the scratch is all-zero between steps.
constexpr int ARGMAX_SLICES = 16;
__device__ __forceinline__ unsigned long long argmax_key(float v, int i) {
  unsigned u = __float_as_uint(v); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)u << 32) | (unsigned)(~(unsigned)i);
}
__global__ void __launch_bounds__(256) k_argmax_split(const float* __restrict__ logits, int ld, int* __restrict__ out,
                                                      unsigned long long* keys, unsigned* cnt) { H5_LIT(0);
  pdl_enter();
  constexpr int N4 = VOCAB / 4, PER = (N4 + ARGMAX_SLICES - 1) / ARGMAX_SLICES;
  const int row = blockIdx.x, sl = blockIdx.y, b = sl * PER, e = min(N4, b + PER);
  const float4* l = reinterpret_cast<const float4*>(logits + (size_t)row * ld);
  float best = -INFINITY; int idx = 0x7fffffff;
  for (int i = b + threadIdx.x; i < e; i += 256) {
    const float4 v = __ldcg(l + i);
    if (v.x > best) { best = v.x; idx = 4 * i; }
    if (v.y > best) { best = v.y; idx = 4 * i + 1; }
    if (v.z > best) { best = v.z; idx = 4 * i + 2; }
    if (v.w > best) { best = v.w; idx = 4 * i + 3; }
  }
  unsigned long long k = idx == 0x7fffffff ? 0ull : argmax_key(best, idx);
  for (int o = 16; o; o >>= 1) { const unsigned long long ok = __shfl_xor_sync(0xffffffff, k, o); k = ok > k ? ok : k; }
  __shared__ unsigned long long wk[8];
  __shared__ bool last;
  if ((threadIdx.x & 31) == 0) wk[threadIdx.x >> 5] = k;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < 8; w++) k = wk[w] > k ? wk[w] : k;
    k = k > wk[0] ? k : wk[0];
    atomicMax(keys + row, k);
    __threadfence();
    last = atomicAdd(cnt + row, 1u) == ARGMAX_SLICES - 1;
    if (last) {
      __threadfence();
      const unsigned long long fk = atomicExch(keys + row, 0ull);
      cnt[row] = 0;
      out[row] = fk ? (int)(~(unsigned)(fk & 0xffffffffu)) : 0;   // all -inf/NaN rows: index 0, as k_argmax
    }
  }
}
void launch_argmax_split(const float* logits, int ld, int* out, int rows, unsigned long long* keys, unsigned* cnt, cudaStream_t st) {
  if (rows < 1 || ld % 4 || uintptr_t(logits) % 16 || !keys || !cnt || !out) throw std::runtime_error("split argmax requires aligned rows and scratch");
  pdl(k_argmax_split, dim3(rows, ARGMAX_SLICES), dim3(256), 0, st, logits, ld, out, keys, cnt);
}
void launch_argmax(const float* logits, int ld, int* out, int rows, cudaStream_t st) { pdl(k_argmax<false>, dim3(rows), dim3(1024), 0, st, logits, ld, out, (const int*)nullptr, (const Seg*)nullptr, 0, 0, 0.f, (const int*)nullptr, (RowStat*)nullptr, 0); }
void launch_relaxed_argmax(const float* logits, int ld, int* out, int rows, cudaStream_t st,
    const int* tokens, const Seg* segments, int nseg, int row0, float margin, const int* rowmap, RowStat* stats, int stats_mode) {
  pdl(k_argmax<true>, dim3(rows), dim3(1024), 0, st, logits, ld, out, tokens, segments, nseg, row0, margin, rowmap, stats, stats_mode);
}

}  // namespace h5

// One translation unit: attention bodies and the megakernel use the device bodies above.
#include "attn.cu"
#include "mega.cu"

H5_LIT_EXPORT

extern "C" void h5_set_variant_kernels(unsigned v) {
  if (cudaMemcpyToSymbol(h5::g_variant, &v, sizeof(v)) != cudaSuccess) abort();
  h5::gdn_host_variant = v;
}

extern "C" void h5_set_launch_carveout(int v) { if (v < -1 || v > 100) abort(); h5::launch_carveout_override() = v; }

extern "C" void h5_set_prep_wide(int v) { if (v < 0 || v > 2) abort(); h5::prep_wide_slot() = v; }
