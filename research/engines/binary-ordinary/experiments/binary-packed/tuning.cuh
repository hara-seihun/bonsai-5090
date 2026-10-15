#pragma once
#include "packed1.cuh"
namespace packed1::tuning {
enum class Route { Lut8, Lut4Ilp };
struct Plan { int n, k, r, groups_per_split, threads; Route route; };
// Explicit opt-in experiments; packed1::select/project and default source are unchanged.
cudaError_t validate(const Plan& p);
size_t table_bytes(const Plan& p);
size_t partial_bytes(const Plan& p);
// Call once before capture to opt R7/R8 LUT8 kernels into 64KiB dynamic shared memory.
cudaError_t configure();
cudaError_t project(const Plan& p, const uint32_t* signs, const __half* scales,
                    const float* x, float* y, float* tables, float* partials,
                    float alpha, float beta, cudaStream_t stream);
}
