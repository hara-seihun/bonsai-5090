#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace h5 {
// Compact split-major FP32 planes. One workspace per stream; keep it alive through
// every captured graph that borrows it. No allocation or clearing occurs at launch.
inline size_t deterministic_gemm_partial_floats(int R, int N, int split) {
  if (R <= 0 || N <= 0 || split <= 0) return 0;
  if (split == 1) return 0;
  if (size_t(R) > std::numeric_limits<size_t>::max() / size_t(N) / size_t(split)) return 0;
  return size_t(R) * size_t(N) * size_t(split);
}

// carrier=true: WF0 or WF3 with RS=row means. Otherwise RS is the format's
// row scale (may be null for WF0/WF3); S is required unless carrier=true.
// Split1 needs no partial workspace. Split>1 fully overwrites its compact planes
// then folds them in ascending split order, adding the residual exactly once.
cudaError_t launch_gemm_fp4_deterministic(int ci, int split,
    const uint8_t* A, const uint8_t* AS, const float* AG, int R, int K,
    const void* W, const void* S, int N, float* out, int ldo, bool add,
    cudaStream_t stream, const float* RS, int wf, bool carrier,
    float* partials, size_t partial_floats);

// The caller keeps GPU count stable until the stream/graph finishes. Bounds are
// inclusive; a rejected launch leaves output and workspace untouched and streams
// no weights. deterministic=false uses the ordinary atomic split-K sum, clearing
// output on-device only when the predicate is true and !add. WF/carrier contracts
// and deterministic workspace sizing are the same as the entry above.
cudaError_t launch_gemm_fp4_predicated(int ci, int split,
    const uint8_t* A, const uint8_t* AS, const float* AG, int R, int K,
    const void* W, const void* S, int N, float* out, int ldo, bool add,
    cudaStream_t stream, const float* RS, int wf, bool carrier,
    float* partials, size_t partial_floats, bool deterministic,
    const uint32_t* count, uint32_t min_count, uint32_t max_count);

cudaError_t launch_gemm_fp4_partials(int ci, int split,
    const uint8_t* A, const uint8_t* AS, const float* AG, int R, int K,
    const void* W, const void* S, int N, cudaStream_t stream,
    const float* RS, int wf, bool carrier, float* partials, size_t partial_floats);
}
