#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace packed1 {
enum class Route { Lut4, Sign, WarpSign };
struct Plan { int n, k, r, groups_per_split, threads; Route route; };
// Q1_0: row-major K128 blocks, FP16 scale then 16 LSB-first sign bytes; 1 => +scale.
// Prepared signs [K/128][4][N], scales [K/128][N], without padding or expansion.
cudaError_t prepare(const uint8_t* raw, uint32_t* signs, __half* scales,
                    int n, int k, cudaStream_t stream);
size_t sign_bytes(int n, int k);
size_t scale_bytes(int n, int k);
size_t table_bytes(const Plan& p);
size_t partial_bytes(const Plan& p);
cudaError_t validate(const Plan& p);
// Measured binary27B dimension families, R1..8. Sets *out only on success;
// unsupported dimensions/row count return cudaErrorInvalidValue.
cudaError_t select(int n, int k, int r, Plan* out);
// X FP32 [R,K], Y FP32 [R,N]; Y <- alpha W X + beta Y.
// Caller-owned device scratch: tables (Lut4 only), partials (when split count > 1).
// All recurring input table production and continuation are included. Capture-safe, no allocations.
cudaError_t project(const Plan& p, const uint32_t* signs, const __half* scales,
                    const float* x, float* y, float* tables, float* partials,
                    float alpha, float beta, cudaStream_t stream);
}
