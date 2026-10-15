#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace binary_fp4_pair {
// Both planes use modern engine NVFP4 [tile64][K128][row64] layouts.
// W/S are WF8: negative sign bits, 512B chunks + original FP16 K128 scales.
struct Plane { const uint8_t* q; const uint8_t* sf; const float* g; };
enum class Schedule { F2G2T8, F4G2T8, F8G1T8, F4G2T16, F8G1T32, F4G2T32, F8G2T16 };
struct Plan { Schedule schedule; int n, k, r, splits; };
size_t partial_floats(const Plan&);
cudaError_t project(const Plan&, Plane primary, Plane residual, const void* w,
                    const void* original_scales, float* output, int output_stride,
                    float* partials, size_t capacity, cudaStream_t, bool add);
}
