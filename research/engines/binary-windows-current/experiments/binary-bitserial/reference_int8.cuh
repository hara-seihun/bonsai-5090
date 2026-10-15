#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace int8_reference {

struct Requirements {
  std::size_t raw_q1_bytes;
  std::size_t packed_weight_bytes;
  std::size_t scale_bytes;
};

// N > 0, N divisible by 32; K > 0, K divisible by 128.
// The raw host input is row-major Q1_0: FP16 little-endian scale, 16
// LSB-first sign bytes per 128 weights; a set sign bit means +1.
cudaError_t sizes(int N, int K, Requirements* result);

// Blocking, one-time preparation. Output-pointer slots must be distinct and
// initially null. Success transfers two cudaMalloc allocations to the caller;
// cudaFree each after all using streams finish. On failure, successfully freed
// slots are null. A nonnull failure slot is retained if cudaFree itself failed,
// so the caller still has its allocation handle; the cleanup error is returned.
cudaError_t prepare(const std::uint8_t* raw_host, std::size_t raw_bytes,
                    int N, int K, uint4** packed_weights, float** scales);

// Device buffers: q[R,K], sx[R,K/128], W and S from prepare(), out[R,N].
// R is 1..16. Only alpha=1,beta=0 is supported; all other values are rejected.
// Successful launch is asynchronous: the caller checks stream completion.
// Buffers must belong to the current device, not alias, and have cudaMalloc
// alignment (q:16, sx:4, W:16, S/out:8). SX is supplied by the caller's producer.
cudaError_t project(const std::int8_t* q, const float* sx,
                    const uint4* packed_weights, const float* scales,
                    float* out, int N, int K, int R,
                    float alpha, float beta, cudaStream_t stream);

}  // namespace int8_reference
