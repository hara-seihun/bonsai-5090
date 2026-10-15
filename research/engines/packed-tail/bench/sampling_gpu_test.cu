#include "sampling_gpu.cuh"
#include "sampling.h"
#include <cuda_runtime.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

int main() {
  constexpr int vocab = 248320, rows = 16;
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  float* device; assert(cudaMalloc(&device, size_t(rows) * vocab * sizeof(float)) == cudaSuccess);
  std::mt19937 random(913);
  std::normal_distribution<float> distribution(0, 2);
  std::vector<float> logits(size_t(rows) * vocab);
  for (auto& x : logits) x = distribution(random);
  // Include exact ties, positive/negative penalties, and repeated causal prefix IDs.
  for (int r = 0; r < rows; ++r) { logits[size_t(r) * vocab + 17] = 9; logits[size_t(r) * vocab + 19] = 9; }
  std::vector<unsigned char> seen(vocab);
  for (int k = 0; k < vocab; k += 13) seen[k] = 1;
  std::vector<int> prefix(rows - 1);
  for (int i = 0; i < rows - 1; ++i) prefix[i] = i % 3 ? 17 : 19;
  assert(cudaMemcpyAsync(device, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
  h5::GpuSampler state;
  int checked = 0;
  for (int variant = 0; variant < 8; ++variant) {
    h5::sampling::Config config;
    config.temperature = variant == 0 ? 0 : variant == 1 ? .7 : 1;
    config.top_k = variant == 2 || variant == 0 ? 0 : variant == 3 ? 128 : 20;
    config.top_p = variant == 4 ? .00001 : variant == 5 ? 1 : .95;
    config.min_p = variant == 6 ? 1 : .05;
    config.presence_penalty = variant % 2 ? 1.5 : -1;
    config.repetition_penalty = variant % 2 ? 1.2 : .7;
    const auto native = h5::sampling::native_config(config);
    for (int iteration = 0; iteration < 3; ++iteration) {
      const uint64_t seed = iteration == 2 ? UINT32_MAX : 42 + iteration;
      int output[rows];
      assert(h5::gpu_sample_rows(state, device, vocab, native, seed, 91, seen.data(), prefix.data(), rows, output, stream) == 0);
      for (int r = 0; r < rows; ++r) {
        std::vector<int> causal(prefix.begin(), prefix.begin() + r);
        const auto expected = h5::sampling::select(logits.data() + size_t(r) * vocab, vocab, config, seen, seed, 91 + r, causal);
        if (!expected.error.empty() || expected.token != output[r]) {
          fprintf(stderr, "GPU mismatch: variant=%d seed=%llu row=%d expected=%d actual=%d error=%s\n",
              variant, (unsigned long long)seed, r, expected.token, output[r], expected.error.c_str());
          return 1;
        }
        ++checked;
      }
    }
  }
  h5::sampling::Config config; config.temperature = 1; config.top_k = 20; config.top_p = .95; config.min_p = .05;
  auto native = h5::sampling::native_config(config);
  int output[rows];
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 40; ++i)
    assert(h5::gpu_sample_rows(state, device, vocab, native, i, 0, seen.data(), prefix.data(), rows, output, stream) == 0);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  printf("GPU sampler oracle rows=%d; 16-row full-vocab sampling %.6f ms/block, %.6f us/row\n", checked, seconds * 1000 / 40, seconds * 1e6 / (40 * rows));
  std::fill(logits.begin(), logits.end(), -1.f);
  logits[0] = -0.f; logits[1] = 0.f;
  assert(cudaMemcpyAsync(device, logits.data(), vocab * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
  native.top_k = 1;
  assert(h5::gpu_sample_rows(state, device, vocab, native, 0, 0, seen.data(), nullptr, 1, output, stream) == 0 && output[0] == 0);
  logits[0] = std::numeric_limits<float>::quiet_NaN();
  assert(cudaMemcpyAsync(device, logits.data(), sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
  assert(h5::gpu_sample_rows(state, device, vocab, native, 0, 0, seen.data(), nullptr, 1, output, stream) == -2);
  native.temperature = -1;
  assert(h5::gpu_sample_rows(state, device, vocab, native, 0, 0, seen.data(), nullptr, 1, output, stream) == -1);
  cudaFree(device); cudaStreamDestroy(stream);
  puts("GPU sampling contracts passed");
}
