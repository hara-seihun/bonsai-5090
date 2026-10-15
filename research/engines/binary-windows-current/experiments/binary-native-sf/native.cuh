#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace binary_native_sf {
struct Plane { const uint8_t* q; const uint8_t* sf; const float* g; };
enum class Schedule { F2G2T8, F4G2T8, F8G1T8, F4G2T16, F8G1T32, F4G2T32, F8G2T16 };
struct Plan { Schedule schedule; int n, k, r, splits, weight_planes; };
enum class ScaleStatus : uint32_t { Valid = 0, NonFinite = 1 };
cudaError_t configure(Schedule, int weight_planes);
size_t partial_floats(const Plan&);
size_t scale_bytes(int n, int k, int weight_planes);
cudaError_t prepare(const uint16_t* original_scales, uint8_t* native_scales, float* row_factor, ScaleStatus* row_status, int n, int k, int weight_planes, cudaStream_t);
cudaError_t project(const Plan&, Plane primary, Plane residual, const void* binary_weights,
                    const void* native_scales, const float* row_factor, float* output,
                    int output_stride, float* partials, size_t capacity, cudaStream_t, bool add);
}
