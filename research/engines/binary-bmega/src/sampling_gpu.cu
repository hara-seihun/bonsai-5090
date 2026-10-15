#include "sampling_gpu.cuh"
#include "sampling_noise.h"
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <climits>
#include <vector>

namespace h5 {
namespace {
constexpr int MaxRows = 128;
void release_rows(GpuSampler& s) {
  cudaFree(s.scores); cudaFree(s.sorted); cudaFree(s.ids); cudaFree(s.sorted_ids);
  cudaFree(s.offsets); cudaFree(s.workspace);
  s.scores = s.sorted = nullptr; s.ids = s.sorted_ids = s.offsets = nullptr;
  s.workspace = nullptr; s.workspace_bytes = 0; s.capacity = 0;
}
#define SC(call) do { const auto e = (call); if (e != cudaSuccess) { fprintf(stderr, "GPU sampling: %s: %s\n", #call, cudaGetErrorString(e)); return -9; } } while (0)
int ensure_rows(GpuSampler& s, int rows, int vocab, cudaStream_t st) {
  if (rows <= s.capacity) return 0;
  release_rows(s);
  const size_t n = size_t(rows) * vocab;
  SC(cudaMalloc(&s.scores, n * sizeof(double))); SC(cudaMalloc(&s.sorted, n * sizeof(double)));
  SC(cudaMalloc(&s.ids, n * sizeof(int))); SC(cudaMalloc(&s.sorted_ids, n * sizeof(int)));
  SC(cudaMalloc(&s.offsets, (rows + 1) * sizeof(int)));
  std::vector<int> offsets(rows + 1);
  for (int r = 0; r <= rows; ++r) offsets[r] = r * vocab;
  SC(cudaMemcpyAsync(s.offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, st));
  SC(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, s.workspace_bytes,
       s.scores, s.sorted, s.ids, s.sorted_ids, int(n), rows, s.offsets, s.offsets + 1, 0, 64, st));
  SC(cudaMalloc(&s.workspace, s.workspace_bytes));
  SC(cudaStreamSynchronize(st));
  s.capacity = rows;
  return 0;
}
__global__ void transform_rows(const float* logits, double* scores, int* ids,
                               int vocab, int rows, H5SamplingConfig c,
                               const unsigned char* seen, const int* prefix, int* error) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (token >= vocab || row >= rows) return;
  double score = logits[size_t(row) * vocab + token];
  if (!isfinite(score)) { atomicExch(error, 1); score = -INFINITY; }
  bool present = seen[token] != 0;
  for (int p = 0; !present && p < row; ++p) present = prefix[p] == token;
  if (present) {
    score = score < 0 ? score * c.repetition_penalty : score / c.repetition_penalty;
    score -= c.presence_penalty;
  }
  if (score == 0) score = 0.0; // Radix keys must equate signed zeros, like score comparisons.
  scores[size_t(row) * vocab + token] = score;
  ids[size_t(row) * vocab + token] = token;
}
__global__ void compact_topk(const double* scores, const int* ids,
                             double* candidates, int* candidate_ids,
                             int vocab, int k, int chunks) {
  using Sort = cub::BlockRadixSort<double, 256, 4, int>;
  __shared__ typename Sort::TempStorage temporary;
  double values[4]; int tokens[4];
  const int row = blockIdx.y, chunk = blockIdx.x;
  for (int i = 0; i < 4; ++i) {
    const int token = chunk * 1024 + threadIdx.x * 4 + i;
    values[i] = token < vocab ? scores[size_t(row) * vocab + token] : -INFINITY;
    tokens[i] = token < vocab ? ids[size_t(row) * vocab + token] : INT_MAX;
  }
  Sort(temporary).SortDescending(values, tokens);
  for (int i = 0; i < 4; ++i) {
    const int rank = threadIdx.x * 4 + i;
    if (rank < k) {
      const size_t dst = size_t(row) * chunks * k + chunk * k + rank;
      candidates[dst] = values[i]; candidate_ids[dst] = tokens[i];
    }
  }
}
__global__ void set_offsets(int* offsets, int rows, int stride) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r <= rows) offsets[r] = r * stride;
}
__device__ bool better(double a, int ai, double b, int bi) {
  return a > b || (a == b && ai < bi);
}
__global__ void select_rows(const double* scores, const int* ids, int vocab,
                            H5SamplingConfig c, uint64_t seed, uint64_t first,
                            int* out) {
  __shared__ int retained;
  __shared__ double maximum[256];
  __shared__ int selected[256];
  const int row = blockIdx.x, t = threadIdx.x;
  const double* s = scores + size_t(row) * vocab;
  const int* id = ids + size_t(row) * vocab;
  if (c.temperature == 0) {
    double best = -INFINITY; int token = INT_MAX;
    for (int k = t; k < vocab; k += blockDim.x)
      if (better(s[k], id[k], best, token)) { best = s[k]; token = id[k]; }
    maximum[t] = best; selected[t] = token;
  } else {
    if (t == 0) {
      const int limit = c.top_k ? min(c.top_k, vocab) : vocab;
      int count = 0; double total = 0;
      for (int k = 0; k < limit; ++k) {
        const double weight = exp((s[k] - s[0]) / c.temperature);
        if (weight < c.min_p) break;
        total += weight; ++count;
      }
      double sum = 0; int n = 0;
      do { sum += exp((s[n++] - s[0]) / c.temperature); }
      while (n < count && sum < total * c.top_p);
      retained = n;
    }
    __syncthreads();
    double best = -INFINITY; int token = INT_MAX;
    for (int k = t; k < retained; k += blockDim.x) {
      const double score = (s[k] - s[0]) / c.temperature +
          sampling::noise_gumbel(seed, first + row, uint64_t(id[k]));
      if (better(score, id[k], best, token)) { best = score; token = id[k]; }
    }
    maximum[t] = best; selected[t] = token;
  }
  __syncthreads();
  for (int n = 128; n; n >>= 1) {
    if (t < n && better(maximum[t + n], selected[t + n], maximum[t], selected[t])) {
      maximum[t] = maximum[t + n]; selected[t] = selected[t + n];
    }
    __syncthreads();
  }
  if (t == 0) out[row] = selected[0];
}
}
GpuSampler::~GpuSampler() {
  release_rows(*this); cudaFree(seen); cudaFree(prefix); cudaFree(out); cudaFree(error);
}
bool sampling_config_valid(const H5SamplingConfig& c) {
  return std::isfinite(c.temperature) && c.temperature >= 0 && c.temperature <= 2 &&
    std::isfinite(c.top_p) && c.top_p > 0 && c.top_p <= 1 &&
    std::isfinite(c.min_p) && c.min_p >= 0 && c.min_p <= 1 &&
    std::isfinite(c.presence_penalty) && c.presence_penalty >= -2 && c.presence_penalty <= 2 &&
    std::isfinite(c.repetition_penalty) && c.repetition_penalty > 0 && c.repetition_penalty <= 2 &&
    c.top_k >= 0 && c.top_k <= 128;
}
int gpu_sampler_seen(GpuSampler& s, const unsigned char* seen, int vocab, cudaStream_t st) {
  if (!seen || vocab < 1 || (s.vocab && s.vocab != vocab)) return -1;
  s.vocab = vocab;
  if (!s.seen) SC(cudaMalloc(&s.seen, vocab));
  if (!s.prefix) SC(cudaMalloc(&s.prefix, MaxRows * sizeof(int)));
  if (!s.out) SC(cudaMalloc(&s.out, MaxRows * sizeof(int)));
  if (!s.error) SC(cudaMalloc(&s.error, sizeof(int)));
  SC(cudaMemcpyAsync(s.seen, seen, vocab, cudaMemcpyHostToDevice, st));
  return 0;
}
int gpu_sample_rows(GpuSampler& s, const float* logits, int vocab,
                    const H5SamplingConfig& c, uint64_t seed, uint64_t first,
                    const unsigned char* seen, const int* prefix, int rows,
                    int* host_out, cudaStream_t st) {
  if (!logits || !host_out || rows < 1 || rows > MaxRows || vocab < 1 ||
      !sampling_config_valid(c) || (rows > 1 && !prefix) ||
      first > std::numeric_limits<uint64_t>::max() - uint64_t(rows - 1)) return -1;
  for (int i = 0; i < rows - 1; ++i) if (prefix[i] < 0 || prefix[i] >= vocab) return -1;
  s.trace_scores.clear(); s.trace_ids.clear();
  int rc = gpu_sampler_seen(s, seen, vocab, st); if (rc) return rc;
  rc = ensure_rows(s, rows, vocab, st); if (rc) return rc;
  if (rows > 1) SC(cudaMemcpyAsync(s.prefix, prefix, (rows - 1) * sizeof(int), cudaMemcpyHostToDevice, st));
  SC(cudaMemsetAsync(s.error, 0, sizeof(int), st));
  transform_rows<<<dim3((vocab + 255) / 256, rows), 256, 0, st>>>(
      logits, s.scores, s.ids, vocab, rows, c, s.seen, s.prefix, s.error);
  const double* selected_scores = s.scores;
  const int* selected_ids = s.ids;
  int selected_stride = vocab;
  size_t bytes = s.workspace_bytes;
  if (c.temperature > 0 && c.top_k > 0 && vocab > 1024) {
    const int chunks = (vocab + 1023) / 1024;
    selected_stride = chunks * c.top_k;
    compact_topk<<<dim3(chunks, rows), 256, 0, st>>>(s.scores, s.ids,
        s.sorted, s.sorted_ids, vocab, c.top_k, chunks);
    set_offsets<<<1, 256, 0, st>>>(s.offsets, rows, selected_stride);
    SC(cub::DeviceSegmentedRadixSort::SortPairsDescending(s.workspace, bytes,
         s.sorted, s.scores, s.sorted_ids, s.ids, rows * selected_stride, rows,
         s.offsets, s.offsets + 1, 0, 64, st));
  } else if (c.temperature > 0) {
    set_offsets<<<1, 256, 0, st>>>(s.offsets, rows, vocab);
    SC(cub::DeviceSegmentedRadixSort::SortPairsDescending(s.workspace, bytes,
         s.scores, s.sorted, s.ids, s.sorted_ids, rows * vocab, rows,
         s.offsets, s.offsets + 1, 0, 64, st));
    selected_scores = s.sorted; selected_ids = s.sorted_ids;
  }
  select_rows<<<rows, 256, 0, st>>>(selected_scores, selected_ids,
       selected_stride, c, seed, first, s.out);
  SC(cudaGetLastError());
  int error = 0;
  if (s.trace && c.temperature > 0 && c.top_k > 0) {
    const int count = std::min(c.top_k, vocab);
    s.trace_scores.resize(size_t(rows) * count); s.trace_ids.resize(size_t(rows) * count);
    SC(cudaMemcpy2DAsync(s.trace_scores.data(), count * sizeof(double), selected_scores,
        selected_stride * sizeof(double), count * sizeof(double), rows, cudaMemcpyDeviceToHost, st));
    SC(cudaMemcpy2DAsync(s.trace_ids.data(), count * sizeof(int), selected_ids,
        selected_stride * sizeof(int), count * sizeof(int), rows, cudaMemcpyDeviceToHost, st));
  }
  SC(cudaMemcpyAsync(host_out, s.out, rows * sizeof(int), cudaMemcpyDeviceToHost, st));
  SC(cudaMemcpyAsync(&error, s.error, sizeof(int), cudaMemcpyDeviceToHost, st));
  SC(cudaStreamSynchronize(st));
  return error ? -2 : 0;
}
#undef SC
}
