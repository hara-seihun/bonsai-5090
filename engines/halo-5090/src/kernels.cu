#define H5_PDLCLOCK_MODULE kernels
#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>
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
                                                    float* __restrict__ out, int ldo) {
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
__global__ void __launch_bounds__(544) k_prep2(PrepArgs a) {
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
  pdl_wait(); pdl_launch();
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
  if (a.flags & P_SIGN_AFTER) {
#pragma unroll
    for (int i = 0; i < 8; i++) { const float4 sg = cs[i]; v[4 * i] *= sg.x; v[4 * i + 1] *= sg.y; v[4 * i + 2] *= sg.z; v[4 * i + 3] *= sg.w; }
  }
  if (a.flags & P_STORE_F32) {
#pragma unroll
    for (int i = 0; i < 8; i++) *(float4*)(a.out_f32 + (size_t)row * a.ldo + e0 + 4 * i) = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
  }
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
void launch_prep(const PrepArgs& a, int rows, cudaStream_t st) {
  pdl(k_prep2, dim3(rows), dim3(32 * (a.n / 1024)), 0, st, a);
}

// ------------------------------------------------------------------------------------------
// Embedding: decode one GGUF PTQ1_0 row (natural element order) per token. Grid rows, 64 threads.
// ------------------------------------------------------------------------------------------
__global__ void k_embed(const uint8_t* __restrict__ w, const int* __restrict__ tok, float* __restrict__ out) {
  pdl_wait(); pdl_launch();
  const int row = blockIdx.x, b = threadIdx.x;
  if (b >= D / 128) return;
  if (tok[row] < 0) {
    for (int j = 0; j < 128; j++) out[(size_t)row * D + b * 128 + j] = 0.f;
    return;
  }
  const uint8_t* blk = w + ((size_t)tok[row] * (D / 128) + b) * 28;
  const float scale = __half2float(*(const __half*)(blk + 26));
  float* o = out + (size_t)row * D + b * 128;
  int e = 0;
  for (int stage = 0; stage < 2; stage++) {
    const int j0 = stage ? 16 : 0, c = stage ? 8 : 16;
    for (int n = 0; n < 5; n++)
      for (int m = 0; m < c; m++) {
        unsigned q = blk[j0 + m];
        for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
        o[e++] = (float)((int)((q * 3u) >> 8) - 1) * scale;
      }
  }
  for (int n = 0; n < 4; n++)
    for (int h = 0; h < 2; h++) {
      unsigned q = blk[24 + h];
      for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
      o[e++] = (float)((int)((q * 3u) >> 8) - 1) * scale;
    }
}
void launch_embed(const uint8_t* w, const int* tok, float* out, int rows, cudaStream_t st) { pdl(k_embed, dim3(rows), dim3(64), 0, st, w, tok, out); }

// ------------------------------------------------------------------------------------------
__global__ void k_visual_input(const int* tok, const float* visual, float* x) {
  pdl_wait(); pdl_launch();
  const int row = blockIdx.x;
  if (tok[row] != -1) return;
  for (int j = threadIdx.x; j < D; j += blockDim.x) x[(size_t)row * D + j] = visual[(size_t)row * D + j];
}
void launch_visual_input(const int* tok, const float* visual, float* x, int rows, cudaStream_t st) {
  pdl(k_visual_input, dim3(rows), dim3(256), 0, st, tok, visual, x);
}

// alpha/beta: out[row][96] = Wab[96][5120] (bf16) . xn[row]. Grid (12, rows), warp per output.
// ------------------------------------------------------------------------------------------
__global__ void k_ab(const __nv_bfloat16* __restrict__ w, const float* __restrict__ xn, float* __restrict__ out) {
  pdl_wait(); pdl_launch();
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
template <typename PT>
__global__ void __launch_bounds__(128) k_conv(ConvArgs a) {
  pdl_wait(); pdl_launch();
  __shared__ float red[2][4];
  const Seg sg = a.segs[blockIdx.y];
  const int c = blockIdx.x * 128 + threadIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const bool qk = blockIdx.x < 32;
  float* st = a.state + ((size_t)sg.slot * NGDN + a.gl) * 3 * CONV_CH;
  float s0 = st[c], s1 = st[CONV_CH + c], s2 = st[2 * CONV_CH + c];
  PT* pend = (PT*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * a.pending_rows * CONV_CH + c;
  for (int i = 0; i < sg.commit; i++) { s0 = s1; s1 = s2; s2 = st_ld(pend + (size_t)i * CONV_CH); }
  if (sg.keep) { st[c] = s0; st[CONV_CH + c] = s1; st[2 * CONV_CH + c] = s2; }
  const float4 wk = *(const float4*)(a.w + c * 4);
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
      const float xg = a.ab[(size_t)row * 96 + hv] + a.dt_bias[hv];
      const float sp = xg > 20.f ? xg : log1pf(__expf(xg));
      a.gates[(size_t)row * 96 + hv] = __expf(a.ssm_a[hv] * sp);
      a.gates[(size_t)row * 96 + HV + hv] = sigm(a.ab[(size_t)row * 96 + HV + hv]);
    }
    s0 = s1; s1 = s2; s2 = xc;
  }
  if (!sg.keep) { st[c] = s0; st[CONV_CH + c] = s1; st[2 * CONV_CH + c] = s2; }
}
void launch_conv(const ConvArgs& a, int nseg, StateFormat format, cudaStream_t st) {
  if (format == StateFormat::F16) pdl(k_conv<__half>, dim3(CONV_CH / 128, nseg), dim3(128), 0, st, a);
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
constexpr int GDN_T = 2 * PMAX;   // max tokens per segment visit (replay + new); longer segments loop in chunks
// Two CTAs per head (64 state rows each, 256 threads). Output is the raw o / sqrt(128); the gated
// RMSNorm runs in the ssm_out prep (P_GDN_NORM).
constexpr int GDN_SMEM = GDN_T * (128 + 128 + 64 + 2) * 4;
template <typename ST, int CG = 4>
__device__ __forceinline__ void gdn_body(const GdnArgs& a, const Seg sg, int bx, uint8_t* smraw) {
  float (*sk)[128] = (float (*)[128])smraw;
  float (*sq)[128] = (float (*)[128])(smraw + GDN_T * 128 * 4);
  float (*sv)[64] = (float (*)[64])(smraw + GDN_T * 256 * 4);
  float* sg_ = (float*)(smraw + GDN_T * 320 * 4);
  float* sb_ = sg_ + GDN_T;
  constexpr int JR = 256 / CG, MC = 128 / CG, CH = 64 + JR / 4;
  const int h = bx / (CG / 2), half = bx % (CG / 2), kh = h % HK;
  const int tid = threadIdx.x, lane = tid & 31;
  const int jl = tid / CG, j = JR * half + jl, cg = lane & (CG - 1);
  ST* M = (ST*)a.state + (((size_t)sg.slot * NGDN + a.gl) * HV + h) * 128 * 128 + j * 128 + 4 * cg;
  ST* pbase = (ST*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * 2 * a.pending_rows * PEND_GDN;
  const ST* prd = pbase + (size_t)(sg.par ^ 1) * a.pending_rows * PEND_GDN;
  ST* pwr = pbase + (size_t)sg.par * a.pending_rows * PEND_GDN;
  float m[MC];
#ifdef EXP_GDN_NOMEM
#pragma unroll
  for (int i = 0; i < MC; i++) m[i] = 0.f;
#else
#pragma unroll
  for (int i = 0; i < MC / 4; i++) {
    const float4 v = ld4(M + CG * 4 * i);
    m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
  }
#endif
  auto save = [&]() {
    if constexpr (CG == 4) st_store32(M, m);
    else {
#pragma unroll
      for (int i = 0; i < MC; i++) st_st(M + (i / 4) * CG * 4 + i % 4, m[i]);
    }
  };
  int done = 0, replayed = 0;
  bool first = true;
  while (first || replayed < sg.commit || done < sg.len) {
    const auto chunk = replay_chunk(sg.commit, sg.len, replayed, done, GDN_T);
    const int nrep = chunk.replay_rows, nnew = chunk.new_rows;
    const int T = nrep + nnew;
    __syncthreads();
    for (int i = tid; i < T * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      float4 v4; bool have = true;
      if (t < nrep) {
        const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN;
        if (c < 32) have = false;
        else v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      } else {
        const float* co = a.conv_out + (size_t)(sg.start + done + t - nrep) * CONV_CH;
        v4 = *(const float4*)(c < 32 ? co + kh * 128 + c * 4 : c < 64 ? co + KDIM + kh * 128 + (c - 32) * 4 : co + 2 * KDIM + h * 128 + JR * half + (c - 64) * 4);
      }
      if (have) {
        float* dst = c < 32 ? &sq[t][c * 4] : c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4];
        *(float4*)dst = v4;
      }
    }
    if (tid < T) {
      const int t = tid;
      if (t < nrep) { const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN; sg_[t] = st_ld(pr + KDIM + VDIM + h); sb_[t] = st_ld(pr + KDIM + VDIM + HV + h); }
      else { const float* gr = a.gates + (size_t)(sg.start + done + t - nrep) * 96; sg_[t] = gr[h]; sb_[t] = gr[HV + h]; }
    }
    __syncthreads();
    if (sg.keep) {   // new speculative rows -> pending[par] (disjoint from the replay buffer)
      for (int i = tid; i < nnew * 128; i += 256) {
        const int t = nrep + (i >> 7), d = i & 127;
        ST* pw = pwr + (size_t)(done + t - nrep) * PEND_GDN;
        if (h < HK && half == 0) st_st(pw + kh * 128 + d, sk[t][d]);
        if (d < JR && !a.replay_delta) st_st(pw + KDIM + h * 128 + JR * half + d, sv[t][d]);
        if (d == 0 && half == 0) { st_st(pw + KDIM + VDIM + h, sg_[t]); st_st(pw + KDIM + VDIM + HV + h, sb_[t]); }
      }
    }
    if (first && sg.keep && nrep == 0) save();
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
#ifndef EXP_GDN_NOMEM
  if (!sg.keep) save();
#endif
}
template <typename ST, int CG = 4>
__global__ void __launch_bounds__(256, 2) k_gdn(GdnArgs a) {
  pdl_wait(); pdl_launch();
  __shared__ __align__(16) uint8_t sm[GDN_SMEM];
  gdn_body<ST, CG>(a, a.segs[blockIdx.y], blockIdx.x, sm);
}
template <typename ST>
void gdn_dispatch(const GdnArgs& a, int nseg, cudaStream_t st) {
  static int cg = getenv("H5_GDN_CG") ? atoi(getenv("H5_GDN_CG")) : 4;
  if (cg == 8) pdl(k_gdn<ST,8>, dim3(4 * HV, nseg), dim3(256), 0, st, a);
  else if (cg == 16) pdl(k_gdn<ST,16>, dim3(8 * HV, nseg), dim3(256), 0, st, a);
  else pdl(k_gdn<ST,4>, dim3(2 * HV, nseg), dim3(256), 0, st, a);
}
void launch_gdn(const GdnArgs& a, int nseg, StateFormat format, cudaStream_t st) {
  if (format == StateFormat::F16) gdn_dispatch<__half>(a, nseg, st);
  else if (format == StateFormat::BF16) gdn_dispatch<__nv_bfloat16>(a, nseg, st);
  else gdn_dispatch<float>(a, nseg, st);
}

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
__global__ void __launch_bounds__(256) k_attn_pre(AttnArgs a) {
  pdl_wait(); pdl_launch();
  attn_pre_body(a, blockIdx.x, blockIdx.y);
}
void launch_attn_pre(const AttnArgs& a, int rows, cudaStream_t st) { pdl(k_attn_pre, dim3(dim3(NH + NKV, rows)), dim3(256), 0, st, a); }

// Argmax over each logits row. Grid rows, 1024 threads.
template<bool RELAX = false>
__global__ void __launch_bounds__(1024) k_argmax(const float* __restrict__ logits, int ld, int* __restrict__ out,
    const int* tokens = nullptr, const Seg* segments = nullptr, int nseg = 0, int row0 = 0, float margin = 0.f) {
  pdl_wait(); pdl_launch();
  __shared__ float bv[32]; __shared__ int bi[32];
  const float* l = logits + (size_t)blockIdx.x * ld;
  float best = -INFINITY; int idx = 0;
  for (int i = threadIdx.x; i < VOCAB; i += 1024) { float v = l[i]; if (v > best) { best = v; idx = i; } }
  for (int o = 16; o; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffff, best, o); int oi = __shfl_xor_sync(0xffffffff, idx, o);
    if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
  }
  if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = best; bi[threadIdx.x >> 5] = idx; }
  __syncthreads();
  if (threadIdx.x < 32) {
    best = bv[threadIdx.x]; idx = bi[threadIdx.x];
    for (int o = 16; o; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, best, o); int oi = __shfl_xor_sync(0xffffffff, idx, o);
      if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (threadIdx.x == 0) {
      if constexpr (RELAX) {
        const int row = row0 + blockIdx.x;
        for (int s = 0; s < nseg; ++s) {
          const Seg seg = segments[s];
          if (!seg.keep || row < seg.start || row + 1 >= seg.start + seg.len) continue;
          const int proposed = tokens[row + 1];
          // Never relax EOS or a non-finite target; the final bonus row remains greedy.
          if (proposed >= 0 && proposed < VOCAB && relaxed_draft_allowed(idx, proposed, best, l[proposed], margin, VOCAB)) idx = proposed;
          break;
        }
      }
      out[blockIdx.x] = idx;
    }
  }
}
void launch_argmax(const float* logits, int ld, int* out, int rows, cudaStream_t st) { pdl(k_argmax<false>, dim3(rows), dim3(1024), 0, st, logits, ld, out, (const int*)nullptr, (const Seg*)nullptr, 0, 0, 0.f); }
void launch_relaxed_argmax(const float* logits, int ld, int* out, int rows, cudaStream_t st,
    const int* tokens, const Seg* segments, int nseg, int row0, float margin) {
  pdl(k_argmax<true>, dim3(rows), dim3(1024), 0, st, logits, ld, out, tokens, segments, nseg, row0, margin);
}

}  // namespace h5

// One translation unit: attention bodies and the megakernel use the device bodies above.
#include "attn.cu"
#include "mega.cu"
