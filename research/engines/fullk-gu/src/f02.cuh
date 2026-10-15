#pragma once
#include "kernels.cuh"

namespace f02 {
void launch_finish(const h5::PrepArgs&, int, cudaStream_t);
void launch_gemm_fp4_silu(int, const uint8_t*, const uint8_t*, const float*, int, int,
                          const void*, const void*, int, float*, int, cudaStream_t,
                          const float*, int, const float*);

inline cudaError_t launch_h32(const uint8_t* A, const uint8_t* AS, const float* AG,
                             int rows, int K, const void* W, const void* S, int N,
                             float* out, int ldo, cudaStream_t st, const float* RS,
                             const float* signs, const h5::PrepArgs& prepared) {
  if (rows != 512 || K != h5::D || N != h5::GU || ldo != h5::FF ||
      !A || !AS || !AG || !W || !S || !out || !RS || !signs ||
      prepared.x != out || prepared.n != h5::FF || prepared.ldx != ldo ||
      !prepared.q4 || !prepared.s4 || !prepared.g4)
    return cudaErrorInvalidValue;
  // Producer config 2 is FW4/WG2/TOK64, not ordinary fp4.cu config 2.
  launch_gemm_fp4_silu(2, A, AS, AG, rows, K, W, S, N, out, ldo, st, RS, 2, signs);
  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) return result;
  launch_finish(prepared, rows, st);
  return cudaGetLastError();
}
}
