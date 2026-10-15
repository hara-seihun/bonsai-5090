#include "pair.cuh"
#include "../../src/kernels.cuh"
#include "../../src/fp4_ops.cuh"

namespace binary_fp4_pair {
using namespace h5;
__device__ __forceinline__ unsigned expand(unsigned x) {
  x &= 255; x = (x | (x << 12)) & 0x000f000fu;
  x = (x | (x << 6)) & 0x03030303u;
  x = (x | (x << 3)) & 0x11111111u;
  return 0x22222222u | (x << 3);
}
template<int FW, int WG, int TW> struct Geometry {
  static constexpr int NG = FW * WG, SW = NG * 512, SS = NG * 64;
  static constexpr int AQ = TW * 64, AS = TW * 8;
  static constexpr int STAGE = SW + SS + 2 * (AQ + AS), STAGES = 8;
  static constexpr int SMEM = STAGE * STAGES + STAGES * 16;
};
template<int FW, int WG, int TW>
__global__ void __launch_bounds__((FW + 1) * 32) joint(
    Plane p, Plane e, const uint8_t* W, const uint8_t* S,
    int N, int K, int R, float* out, int ld, bool add) {
  using G = Geometry<FW, WG, TW>;
  constexpr int NT = TW / 8, MT = WG * 2;
  extern __shared__ __align__(128) uint8_t sm[];
  auto* full = reinterpret_cast<uint64_t*>(sm + G::STAGE * G::STAGES);
  auto* empty = full + G::STAGES;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int ngall = N / 32, g0 = blockIdx.y * G::NG, t0 = blockIdx.x * TW;
  int all = K / 128, kb0 = all * blockIdx.z / gridDim.z;
  int nkb = all * (blockIdx.z + 1) / gridDim.z - kb0;
  if (tid == 0) {
    for (int s = 0; s < G::STAGES; s++) { mbar_init(full + s, 1); mbar_init(empty + s, FW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == FW) {
    if (lane == 0) {
      int ng = min(G::NG, ngall - g0), npre = min(nkb, G::STAGES);
      unsigned tx = ng * 576 + 2 * (G::AQ + G::AS);
      uint64_t policy = weight_tma_policy();
      if (l2pf_enabled()) for (int kb = G::STAGES; kb < nkb; kb++) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(W + (size_t(kb0 + kb) * ngall + g0) * 512), "r"(ng * 512) : "memory");
      }
      for (int kb = 0; kb < npre; kb++) {
        auto* s = sm + kb * G::STAGE;
        size_t c = size_t(kb0 + kb) * ngall + g0;
        mbar_expect(full + kb, tx);
        bulk_weight_g2s(s, W + c * 512, ng * 512, full + kb, policy);
        bulk_g2s(s + G::SW, S + c * 64, ng * 64, full + kb);
      }
      pdl_wait();
      for (int kb = 0; kb < nkb; kb++) {
        int slot = kb % G::STAGES;
        auto* s = sm + slot * G::STAGE;
        size_t c = size_t(kb0 + kb) * ngall + g0;
        if (kb >= npre) {
          mbar_wait(empty + slot, ((kb / G::STAGES) & 1) ^ 1);
          mbar_expect(full + slot, tx);
          bulk_weight_g2s(s, W + c * 512, ng * 512, full + slot, policy);
          bulk_g2s(s + G::SW, S + c * 64, ng * 64, full + slot);
        }
        size_t rb = (size_t(t0 / 64) * all + kb0 + kb) * 64 + t0 % 64;
        auto* a = s + G::SW + G::SS;
        bulk_g2s(a, p.q + rb * 64, G::AQ, full + slot);
        bulk_g2s(a + G::AQ, p.sf + rb * 8, G::AS, full + slot);
        bulk_g2s(a + G::AQ + G::AS, e.q + rb * 64, G::AQ, full + slot);
        bulk_g2s(a + 2 * G::AQ + G::AS, e.sf + rb * 8, G::AS, full + slot);
      }
    }
    return;
  }
  int gid = lane >> 2, tq = lane & 3, gw = warp * WG;
  float acc[MT][NT][4] = {}, gp[NT][2], ge[NT][2];
  const float zero[4] = {};
  pdl_wait();
#pragma unroll
  for (int n = 0; n < NT; n++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      int t = t0 + n * 8 + tq * 2 + j;
      gp[n][j] = t < R ? activation_scale(p.g + t) : 0.f;
      ge[n][j] = t < R ? activation_scale(e.g + t) : 0.f;
    }
  }
  for (int kb = 0; kb < nkb; kb++) {
    int slot = kb % G::STAGES;
    mbar_wait(full + slot, (kb / G::STAGES) & 1);
    auto* s = sm + slot * G::STAGE;
    unsigned a[MT][2][4]; float ws[MT][2];
#pragma unroll
    for (int q = 0; q < WG; q++) {
      const uint4 packed = *(const uint4*)(s + (gw + q) * 512 + lane * 16);
      unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (int m = 0; m < 2; m++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          unsigned word = words[m * 2 + h]; unsigned* dst = a[q * 2 + m][h];
          dst[0] = expand(word); dst[2] = expand(word >> 8);
          dst[1] = expand(word >> 16); dst[3] = expand(word >> 24);
        }
      }
      uint2 scales = *(const uint2*)(s + G::SW + (gw + q) * 64 + gid * 8);
      const __half2* sh = (const __half2*)&scales;
      float2 x = __half22float2(sh[0]), y = __half22float2(sh[1]);
      ws[q * 2][0] = x.x; ws[q * 2][1] = x.y;
      ws[q * 2 + 1][0] = y.x; ws[q * 2 + 1][1] = y.y;
    }
#pragma unroll
    for (int n = 0; n < NT; n++) {
      int row = n * 8 + (lane & 7), c = lane >> 3;
      const auto* ap = s + G::SW + G::SS;
      const auto* ae = ap + G::AQ + G::AS;
      unsigned b[4], z[4];
      int off = row * 64 + ((c ^ ((row >> 1) & 3)) << 4);
      ldsm4(b[0], b[1], b[2], b[3], ap + off);
      ldsm4(z[0], z[1], z[2], z[3], ae + off);
      uint2 sb = *(const uint2*)(ap + G::AQ + (n * 8 + gid) * 8);
      uint2 sz = *(const uint2*)(ae + G::AQ + (n * 8 + gid) * 8);
#pragma unroll
      for (int m = 0; m < MT; m++) {
        float dp[4], de[4];
        mma_fp4(dp, a[m][0], b[0], b[1], sb.x, zero);
        mma_fp4(de, a[m][0], z[0], z[1], sz.x, zero);
        mma_fp4(dp, a[m][1], b[2], b[3], sb.y, dp);
        mma_fp4(de, a[m][1], z[2], z[3], sz.y, de);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float pair = __fmaf_rn(de[i], ge[n][i & 1], __fmul_rn(dp[i], gp[n][i & 1]));
          acc[m][n][i] = __fmaf_rn(pair, ws[m][i >> 1], acc[m][n][i]);
        }
      }
    }
    ring_release(empty + slot, lane == 0);
  }
  pdl_wait(); pdl_launch();
  if (g0 + gw >= ngall) return;
  out += size_t(blockIdx.z) * R * ld;
#pragma unroll
  for (int n = 0; n < NT; n++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      int t = t0 + n * 8 + tq * 2 + j;
      if (t >= R) continue;
#pragma unroll
      for (int m = 0; m < MT; m++) {
        if (g0 + gw + m / 2 >= ngall) continue;
#pragma unroll
        for (int r = 0; r < 2; r++) {
          size_t at = size_t(t) * ld + (g0 + gw) * 32 + gid + m * 16 + r * 8;
          float v = acc[m][n][r * 2 + j];
          out[at] = add ? __fadd_rn(out[at], v) : v;
        }
      }
    }
  }
}
__global__ void fold(const float* p, float* y, int N, int R, int ld, int splits, bool add) {
  pdl_wait(); pdl_launch();
  size_t plane = size_t(R) * N;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= plane) return;
  float v = p[i];
  for (int s = 1; s < splits; s++) v = __fadd_rn(v, p[size_t(s) * plane + i]);
  size_t at = (i / N) * ld + i % N;
  y[at] = add ? __fadd_rn(y[at], v) : v;
}
size_t partial_floats(const Plan& p) { return p.splits > 1 && p.n > 0 && p.r > 0 ? size_t(p.splits) * p.n * p.r : 0; }
template<int FW, int WG, int TW>
cudaError_t launch(const Plan& p, Plane a, Plane e, const uint8_t* w, const uint8_t* s, float* y, int ld, cudaStream_t st, bool add) {
  using G = Geometry<FW, WG, TW>;
  auto f = joint<FW, WG, TW>;
  cudaError_t error = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
  if (error != cudaSuccess) return error;
  pdl(f, dim3((p.r + TW - 1) / TW, (p.n / 32 + G::NG - 1) / G::NG, p.splits), dim3((FW + 1) * 32), G::SMEM, st, a, e, w, s, p.n, p.k, p.r, y, ld, add);
  return cudaGetLastError();
}
cudaError_t project(const Plan& p, Plane a, Plane e, const void* w, const void* s, float* y, int ld,
                    float* partials, size_t capacity, cudaStream_t st, bool add) {
  if (!a.q || !a.sf || !a.g || !e.q || !e.sf || !e.g || !w || !s || !y ||
      p.r < 1 || p.r > 16 || p.n <= 0 || p.n % 32 || p.k <= 0 || p.k % 128 || ld < p.n ||
      p.splits < 1 || p.splits > p.k / 128) return cudaErrorInvalidValue;
  if ((uintptr_t(a.q) | uintptr_t(a.sf) | uintptr_t(e.q) | uintptr_t(e.sf) |
       uintptr_t(w) | uintptr_t(s)) & 15u) return cudaErrorInvalidValue;
  if ((uintptr_t(a.g) | uintptr_t(e.g) | uintptr_t(y)) & 3u) return cudaErrorInvalidValue;
  size_t need = partial_floats(p);
  if (need && (!partials || capacity < need || (uintptr_t(partials) & 3u))) return cudaErrorInvalidValue;
  if (need) {
    size_t bytes = need * sizeof(float), output_bytes = size_t(p.r) * ld * sizeof(float);
    uintptr_t pp = uintptr_t(partials), py = uintptr_t(y);
    if (pp <= py ? py - pp < bytes : pp - py < output_bytes) return cudaErrorInvalidValue;
  }
  float* target = need ? partials : y; int stride = need ? p.n : ld;
  cudaError_t error;
  bool add_direct = need ? false : add;
  switch (p.schedule) {
    case Schedule::F2G2T8: error = launch<2,2,8>(p,a,e,(const uint8_t*)w,(const uint8_t*)s,target,stride,st,add_direct); break;
    case Schedule::F4G2T8: error = launch<4,2,8>(p,a,e,(const uint8_t*)w,(const uint8_t*)s,target,stride,st,add_direct); break;
    case Schedule::F8G1T8: error = launch<8,1,8>(p,a,e,(const uint8_t*)w,(const uint8_t*)s,target,stride,st,add_direct); break;
    case Schedule::F4G2T16: error = launch<4,2,16>(p,a,e,(const uint8_t*)w,(const uint8_t*)s,target,stride,st,add_direct); break;
    case Schedule::F8G1T32: error = launch<8,1,32>(p,a,e,(const uint8_t*)w,(const uint8_t*)s,target,stride,st,add_direct); break;
    case Schedule::F4G2T32: error = launch<4,2,32>(p,a,e,(const uint8_t*)w,(const uint8_t*)s,target,stride,st,add_direct); break;
    case Schedule::F8G2T16: error = launch<8,2,16>(p,a,e,(const uint8_t*)w,(const uint8_t*)s,target,stride,st,add_direct); break;
    default: return cudaErrorInvalidValue;
  }
  if (error != cudaSuccess || !need) return error;
  pdl(fold, dim3((size_t(p.r) * p.n + 255) / 256), dim3(256), 0, st, (const float*)partials, y, p.n, p.r, ld, p.splits, add);
  return cudaGetLastError();
}
}
