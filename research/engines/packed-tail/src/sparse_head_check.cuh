#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace h5 {
// Diagnostic only: full-head logits are the oracle at the same prepared row/config.
static __global__ void sparse_head_check_kernel(const float* compact, const float* full,
                                               const int* ids, int count, unsigned* result) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float a = compact[i], b = full[ids[i]];
  atomicAdd(result, 1u);
  if (__float_as_uint(a) != __float_as_uint(b)) atomicAdd(result + 1, 1u);
  if (!isfinite(a) || !isfinite(b)) atomicAdd(result + 2, 1u);
  else atomicMax(result + 3, __float_as_uint(fabsf(a - b)));
}
inline cudaError_t launch_sparse_head_check(const float* compact, const float* full,
    const int* ids, int count, unsigned* result, cudaStream_t stream) {
  if (!compact || !full || !ids || !result || count < 2) return cudaErrorInvalidValue;
  sparse_head_check_kernel<<<(count + 255) / 256, 256, 0, stream>>>(compact, full, ids, count, result);
  return cudaPeekAtLastError();
}
}
