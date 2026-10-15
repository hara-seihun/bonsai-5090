#include "ternary_operands.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
namespace h5 {
int fp4_ncfg();
bool fp4_cfg_wg2(int);
cudaError_t fp4_profile_resources(int, int, int, cudaFuncAttributes*, int*);
const char* fp4_cfg_name(int, int);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
void launch_gemm_fp4_silu(int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, cudaStream_t, const float*, int);
}
static void ck(cudaError_t e) {
  if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
template <class T> static T* upload(const std::vector<T>& v) {
  T* p; ck(cudaMalloc(&p, v.size() * sizeof(T)));
  ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
int main(int argc, char** argv) {
  if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "wf6") && std::strcmp(argv[1], "wf7"))) {
    std::fprintf(stderr, "usage: ternary-expanded-check [wf6|wf7]\n"); return 2;
  }
  const int candidate = argc == 2 && !std::strcmp(argv[1], "wf7") ? 7 : 6;
  const int N = 576, K = 1152, Rp = 640, chunks = (N / 32) * (K / 128), ldo = N + 16;
  std::mt19937 rng(29);
  std::vector<uint8_t> W2(chunks * 1024), S2(chunks * 64, 0), W6(chunks * 2048), S6(chunks * 128);
  for (auto& w : W2) w = rng();
  for (int c = 0; c < chunks; c++) {
    for (int i = 0; i < 32; i++) S2[c * 64 + i] = 0x28 + rng() % 24;
    h5::expand_ternary_chunk(W2.data() + c * 1024, W6.data() + c * 2048);
    h5::expand_ternary_scales(S2.data() + c * 64, S6.data() + c * 128);
  }
  std::vector<uint8_t> A((size_t)Rp * K / 2), AS((size_t)Rp * K / 16);
  for (auto& v : A) v = rng();
  for (auto& v : AS) v = 0x28 + rng() % 24;
  std::vector<float> AG(Rp), RS(N), init((size_t)Rp * ldo);
  for (auto& v : AG) v = 0.03f * (1 + rng() % 7);
  for (auto& v : RS) v = 0.001f * (1 + rng() % 7);
  for (auto& v : init) v = 0.01f * int(rng() % 21) - 0.1f;
  auto dW2 = upload(W2), dS2 = upload(S2), dW6 = upload(W6), dS6 = upload(S6), dA = upload(A), dAS = upload(AS);
  auto dAG = upload(AG), dRS = upload(RS), o2 = upload(init), o6 = upload(init);
  cudaStream_t st; ck(cudaStreamCreate(&st));
  int failures = 0, checked = 0, skipped = 0, split_unequal = 0;
  for (int R : {1, 8, 64, 65, 512, 513})
    for (int ci = 0; ci < h5::fp4_ncfg(); ci++)
      for (int mode : {0, 1, 3})
        for (int ks : {1, 2, 3}) {
          if (mode == 3 && (ks != 1 || !h5::fp4_cfg_wg2(ci))) continue;
          cudaFuncAttributes attrs{}; int blocks2 = 0, blocks6 = 0;
          ck(h5::fp4_profile_resources(ci, ks > 1 ? 2 : mode, 2, &attrs, &blocks2));
          ck(h5::fp4_profile_resources(ci, ks > 1 ? 2 : mode, candidate, &attrs, &blocks6));
          if (blocks2 == 0 || blocks6 == 0) { skipped++; continue; }
          ck(cudaMemcpy(o2, init.data(), init.size() * 4, cudaMemcpyHostToDevice));
          ck(cudaMemcpy(o6, init.data(), init.size() * 4, cudaMemcpyHostToDevice));
          auto launch = [&](int wf, float* out) {
            const auto W = wf == 6 ? dW6 : dW2, S = wf == 2 ? dS2 : dS6;
            if (mode == 3) h5::launch_gemm_fp4_silu(ci, dA, dAS, dAG, R, K, W, S, N, out, ldo, st, dRS, wf);
            else h5::launch_gemm_fp4_cfg(ci, ks, dA, dAS, dAG, R, K, W, S, N, out, ldo, mode == 1, st, dRS, wf);
            return cudaGetLastError();
          };
          const auto e2 = launch(2, o2), e6 = launch(candidate, o6);
          if (e2 != cudaSuccess || e6 != cudaSuccess) {
            std::fprintf(stderr, "launch R%d %s mode%d: WF2=%s WF%d=%s\n", R, h5::fp4_cfg_name(ci, ks), mode, cudaGetErrorString(e2), candidate, cudaGetErrorString(e6));
            if ((e2 != cudaSuccess && e2 != cudaErrorInvalidConfiguration && e2 != cudaErrorLaunchOutOfResources) ||
                (e6 != cudaSuccess && e6 != cudaErrorInvalidConfiguration && e6 != cudaErrorLaunchOutOfResources)) {
              ck(e2); ck(e6);
            }
            ck(cudaStreamSynchronize(st)); skipped++; continue;
          }
          ck(cudaStreamSynchronize(st));
          std::vector<float> h2(init.size()), h6(init.size());
          ck(cudaMemcpy(h2.data(), o2, init.size() * 4, cudaMemcpyDeviceToHost));
          ck(cudaMemcpy(h6.data(), o6, init.size() * 4, cudaMemcpyDeviceToHost));
          int bad = 0, unequal = 0;
          const int width = mode == 3 ? N / 2 : N;
          for (int r = 0; r < Rp; r++) for (int n = 0; n < ldo; n++) {
            const size_t i = (size_t)r * ldo + n;
            if (r >= R || n >= width) {
              const float expected = r < R && mode == 0 && ks > 1 ? 0.f : init[i];
              bad += h2[i] != expected || h6[i] != expected; continue;
            }
            const bool exact = std::memcmp(&h2[i], &h6[i], 4) == 0;
            unequal += !exact;
            // Split-K atomic arrival order can differ; only unsplit launches require bit equality.
            bad += !std::isfinite(h2[i]) || !std::isfinite(h6[i]) ||
              (ks == 1 ? !exact : std::fabs(h2[i] - h6[i]) > 2e-5f * std::max(1.f, std::fabs(h2[i])));
          }
          if (ks > 1) split_unequal += unequal;
          if (bad) { failures++; std::printf("FAIL R%d %s mode%d: %d values\n", R, h5::fp4_cfg_name(ci, ks), mode, bad); }
          checked++;
        }
  std::printf("TERNARY_EXPANDED_%s checked=%d skipped=%d failures=%d split_unequal=%d wf=%d\n",
              failures || !checked ? "FAIL" : "OK", checked, skipped, failures, split_unequal, candidate);
  return failures || !checked;
}
