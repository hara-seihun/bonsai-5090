// Kernel/helpers copied verbatim from src/kernels.cu at
// 524b2ab11af15e587cf0dc0042ad4e0e5502fb5a; see REFERENCE_INT8.md.
#include "reference_int8.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace int8_reference {
namespace {

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
  const bool live = g < N / 32;
  uint4 wa = live ? wchunk[0] : make_uint4(0, 0, 0, 0), wb = live ? wchunk[1] : make_uint4(0, 0, 0, 0);

  for (int kb = 0; kb < nkb; kb++) {
    const int st = kb % GSTAGES;
    cp_wait<1>();
    __syncthreads();
    if (kb + 2 < nkb) load_stage(kb + 2, (kb + 2) % GSTAGES);
    cp_commit();
    uint4 na = wa, nb = wb;
    if (live && kb + 1 < nkb) { na = wchunk[(kb + 1) * 64]; nb = wchunk[(kb + 1) * 64 + 1]; }
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
    for (int nt = 0; nt < 4; nt++) ws[nt] = live ? *(const float2*)(sk + nt * 8) : make_float2(0.f, 0.f);
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
        if (r < R && col < N) {
          float2* o = (float2*)(out + (size_t)r * ldo + col);
          float2 v = make_float2(accf[mt][nt][2 * h], accf[mt][nt][2 * h + 1]);
          if (MODE == 1) { float2 p = *o; v.x += p.x; v.y += p.y; }
          *o = v;
        }
      }
    }
  }
}

cudaError_t configure() {
  static thread_local int configured_device = -1;
  int device;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return e;
  if (configured_device == device) return cudaSuccess;
  constexpr int smem = GSTAGES * GM * 128 + GSTAGES * GM * 4;
  e = cudaFuncSetAttribute(k_gemm_t2<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e == cudaSuccess) configured_device = device;
  return e;
}

cudaError_t abandon(cudaError_t primary, uint4** W, float** S) {
  cudaError_t cleanup = cudaSuccess;
  if (*S) {
    const cudaError_t e = cudaFree(*S);
    if (e == cudaSuccess) *S = nullptr;
    else cleanup = e;
  }
  if (*W) {
    const cudaError_t e = cudaFree(*W);
    if (e == cudaSuccess) *W = nullptr;
    else if (cleanup == cudaSuccess) cleanup = e;
  }
  return cleanup == cudaSuccess ? primary : cleanup;
}

bool aligned(const void* p, std::uintptr_t alignment) {
  return p && (reinterpret_cast<std::uintptr_t>(p) % alignment == 0);
}

}  // namespace

cudaError_t sizes(int N, int K, Requirements* result) {
  if (!result || N <= 0 || N % 32 || K <= 0 || K % 128)
    return cudaErrorInvalidValue;
  const std::size_t n = static_cast<std::size_t>(N);
  const std::size_t kb = static_cast<std::size_t>(K / 128);
  const std::size_t limit = std::numeric_limits<std::size_t>::max();
  if (n > limit / kb) return cudaErrorInvalidValue;
  const std::size_t blocks = n * kb;
  if (blocks > limit / 32) return cudaErrorInvalidValue;
  *result = {blocks * 18, blocks * 32, blocks * sizeof(float)};
  return cudaSuccess;
}

cudaError_t prepare(const std::uint8_t* raw_host, std::size_t raw_bytes,
                    int N, int K, uint4** packed_weights, float** scales) {
  if (!packed_weights || !scales ||
      static_cast<void*>(packed_weights) == static_cast<void*>(scales) ||
      *packed_weights || *scales)
    return cudaErrorInvalidValue;
  Requirements required;
  cudaError_t e = sizes(N, K, &required);
  if (e != cudaSuccess) return e;
  if (!raw_host || raw_bytes != required.raw_q1_bytes)
    return cudaErrorInvalidValue;
  if (required.packed_weight_bytes >
      std::numeric_limits<std::size_t>::max() - required.scale_bytes)
    return cudaErrorInvalidValue;
  auto* host_W = static_cast<std::uint8_t*>(
      std::malloc(required.packed_weight_bytes + required.scale_bytes));
  if (!host_W) return cudaErrorMemoryAllocation;
  auto* host_S = reinterpret_cast<float*>(host_W + required.packed_weight_bytes);
  const int nkb = K / 128;
  for (int g = 0; g < N / 32; ++g) {
    for (int kb = 0; kb < nkb; ++kb) {
      int8_t trit[32][128];
      float sc[32];
      for (int r = 0; r < 32; ++r) {
        const std::uint8_t* blk = raw_host +
            (static_cast<std::size_t>(g * 32 + r) * nkb + kb) * 18;
        const std::uint16_t hs = static_cast<std::uint16_t>(blk[0]) |
            (static_cast<std::uint16_t>(blk[1]) << 8);
        __half_raw half_bits;
        half_bits.x = hs;
        sc[r] = __half2float(__half(half_bits));
        if (!std::isfinite(sc[r])) {
          std::free(host_W);
          return cudaErrorInvalidValue;
        }
        for (int k = 0; k < 128; ++k)
          trit[r][k] = ((blk[2 + k / 8] >> (k % 8)) & 1) ? 1 : -1;
      }
      // Original repack_rows fragment packing, unchanged except W/S ownership.
      std::uint8_t* chunk = host_W +
          (static_cast<std::size_t>(g) * nkb + kb) * 1024;
      std::memset(chunk, 0, 1024);
      for (int l = 0; l < 32; l++) {
        const int gid = l >> 2, tq = l & 3;
        for (int ri = 0; ri < 32; ri++) {
          const int h = ri & 1, j = (ri >> 1) & 3, s = ri >> 3, wi = ri >> 2, f = ri & 3;
          for (int b = 0; b < 4; b++) {
            const int n = 8 * j + gid, k = 32 * s + 16 * h + 4 * tq + b;
            chunk[l * 32 + wi * 4 + b] |= (uint8_t)((trit[n][k] & 3) << (2 * f));
          }
        }
      }
      for (int r = 0; r < 32; r++)
        host_S[static_cast<std::size_t>(kb) * N + g * 32 + r] = sc[r] / 64.f;
    }
  }
  e = configure();
  if (e == cudaSuccess)
    e = cudaMalloc(reinterpret_cast<void**>(packed_weights), required.packed_weight_bytes);
  if (e == cudaSuccess)
    e = cudaMalloc(reinterpret_cast<void**>(scales), required.scale_bytes);
  if (e == cudaSuccess)
    e = cudaMemcpy(*packed_weights, host_W, required.packed_weight_bytes, cudaMemcpyHostToDevice);
  if (e == cudaSuccess)
    e = cudaMemcpy(*scales, host_S, required.scale_bytes, cudaMemcpyHostToDevice);
  std::free(host_W);
  return e == cudaSuccess ? e : abandon(e, packed_weights, scales);
}

cudaError_t project(const std::int8_t* q, const float* sx,
                    const uint4* packed_weights, const float* scales,
                    float* out, int N, int K, int R,
                    float alpha, float beta, cudaStream_t stream) {
  Requirements required;
  cudaError_t e = sizes(N, K, &required);
  if (e != cudaSuccess) return e;
  if (R < 1 || R > 16 || alpha != 1.f || beta != 0.f ||
      !aligned(q, 16) || !aligned(sx, 4) || !aligned(packed_weights, 16) ||
      !aligned(scales, 8) || !aligned(out, 8))
    return cudaErrorInvalidValue;
  e = configure();
  if (e != cudaSuccess) return e;
  constexpr int smem = GSTAGES * GM * 128 + GSTAGES * GM * 4;
  const dim3 grid((N - 1) / GN + 1, (R - 1) / GM + 1);
  k_gemm_t2<0><<<grid, 256, smem, stream>>>(q, sx, R, K, packed_weights, scales, N, out, N);
  return cudaGetLastError();
}

}  // namespace int8_reference
