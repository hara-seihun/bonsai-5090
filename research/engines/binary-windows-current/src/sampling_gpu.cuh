#pragma once
#include "sampling_contract.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <vector>

namespace h5 {
struct DraftLz;   // kernels.cuh
struct GpuSampler {
  unsigned char* seen = nullptr;
  int *prefix = nullptr, *out = nullptr, *error = nullptr, *offsets = nullptr;
  double *scores = nullptr, *sorted = nullptr;
  int *ids = nullptr, *sorted_ids = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  int capacity = 0, vocab = 0;
  bool trace = false;
  std::vector<double> trace_scores;
  std::vector<int> trace_ids;
  GpuSampler() = default;
  GpuSampler(const GpuSampler&) = delete;
  GpuSampler& operator=(const GpuSampler&) = delete;
  ~GpuSampler();
};
// Result: 0 success, -1 invalid contract, -2 nonfinite logits, -9 CUDA error.
int gpu_sample_rows(GpuSampler&, const float* logits, int vocab,
                    const H5SamplingConfig&, uint64_t seed, uint64_t first_index,
                    const unsigned char* seen, const int* prefix, int rows,
                    int* host_out, cudaStream_t);
int gpu_sampler_seen(GpuSampler&, const unsigned char* seen, int vocab, cudaStream_t);
bool sampling_config_valid(const H5SamplingConfig&);
cudaError_t launch_selector_sampled(const __nv_bfloat16* pred, const __nv_bfloat16* succ,
                                    const float* hp, const float* topv, const int* topi,
                                    const int* anchor, int* drafts, float* conf,
                                    int proposals, const H5SamplingConfig&,
                                    uint64_t seed, uint64_t first_index,
                                    const unsigned char* seen, int* error,
                                    double* transition_scores, const DraftLz&, cudaStream_t);
}
