#pragma once
#include "f07_contract.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace h5 {
// Measured F07 EP2, with half2 rounding boundaries and the original FP32 tail.
__device__ __forceinline__ float2 f07_degree8_product(float g0, float g1, float u0, float u1) {
  const __half2 g = __floats2half2_rn(g0, g1), u = __floats2half2_rn(u0, u1);
  const __half2 z = __hmul2(g, g);
  __half2 v = __hfma2(z, __float2half2_rn(-0.000026285648345947266f), __float2half2_rn(0.0010166168212890625f));
  v = __hfma2(z, v, __float2half2_rn(-0.0176849365234375f));
  v = __hfma2(z, v, __float2half2_rn(0.246826171875f));
  const __half2 s = __hfma2(z, v, __hmul2(g, __float2half2_rn(0.5f)));
  const __half2 h = __hmul2(s, u);
  float h0 = __low2float(h), h1 = __high2float(h);
  if (fabsf(g0) > 3.5f || fabsf(u0) > 60000.f || !isfinite(h0)) h0 = g0 / (1.f + __expf(-g0)) * u0;
  if (fabsf(g1) > 3.5f || fabsf(u1) > 60000.f || !isfinite(h1)) h1 = g1 / (1.f + __expf(-g1)) * u1;
  return make_float2(h0, h1);
}
cudaError_t launch_f07_degree8(const uint8_t* A, const uint8_t* AS, const float* AG,
                              int R, int K, const void* W, const void* S, int N,
                              float* out, int ldo, cudaStream_t st);
} // namespace h5
