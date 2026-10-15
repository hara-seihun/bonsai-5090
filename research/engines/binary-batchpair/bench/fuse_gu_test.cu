// Fused post-norm prep + dense5 carrier GU versus separate k_prep2 + carrier GU.
// Random residual rows, norm weights, signs, dense5 payload and carrier means; outputs compared bitwise.
#include "kernels.cuh"
#include "scale_carrier.cuh"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace h5 { void launch_prep(const PrepArgs&, int, cudaStream_t); }
using namespace h5;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)
int main(int argc, char** argv) {
  const int K = 5120, N = 34816, FF = N / 2, ci = argc > 1 ? atoi(argv[1]) : 19;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> ud(0, 242);
  std::vector<float> x(8 * K), nw(K), sg(K), mean(N);
  for (auto& v : x) v = nd(rng) * 3.f;
  for (int i = 0; i < 37; i++) x[(size_t)(rng() % (8 * K))] *= 40.f;   // outliers
  for (auto& v : nw) v = 0.5f + std::fabs(nd(rng));
  for (auto& v : sg) v = (rng() & 1) ? 1.f : -1.f;
  for (auto& v : mean) v = 0.002f + 0.001f * std::fabs(nd(rng));
  std::vector<uint8_t> w((size_t)(N / 32) * (K / 128) * 832);
  for (auto& b : w) b = (uint8_t)ud(rng);
  float *dx, *dnw, *dsg, *dmean, *o1, *o2, *g4; uint8_t *dw, *q4, *s4;
  CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dnw, K * 4)); CK(cudaMalloc(&dsg, K * 4)); CK(cudaMalloc(&dmean, N * 4));
  CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&o1, 8 * FF * 4)); CK(cudaMalloc(&o2, 8 * FF * 4));
  CK(cudaMalloc(&q4, 64 * K / 2)); CK(cudaMalloc(&s4, 64 * K / 16)); CK(cudaMalloc(&g4, 64 * 4));
  CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dnw, nw.data(), K * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dsg, sg.data(), K * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dmean, mean.data(), N * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  int bad = 0;
  for (int R : {1}) {
    CK(cudaMemset(o1, 0xff, 8 * FF * 4)); CK(cudaMemset(o2, 0xff, 8 * FF * 4)); CK(cudaMemset(q4, 0, 64 * K / 2)); CK(cudaMemset(s4, 0, 64 * K / 16));
    PrepArgs a{}; a.x = dx; a.ldx = K; a.norm_w = dnw; a.signs = dsg; a.n = K; a.flags = P_NORM | P_SIGN | P_HADAMARD | P_QUANT;
    a.q4 = q4; a.s4 = s4; a.g4 = g4;
    launch_prep(a, R, st);
    CK(launch_scale_carrier(ci, 1, 3, q4, s4, g4, R, K, dw, N, o1, FF, dmean, st));
    CK(launch_scale_carrier_fused_gu(ci, R, K, dw, N, o2, FF, dmean, dx, K, dnw, dsg, st));
    CK(cudaStreamSynchronize(st));
    std::vector<uint32_t> h1(R * FF), h2(R * FF);
    CK(cudaMemcpy(h1.data(), o1, R * FF * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2.data(), o2, R * FF * 4, cudaMemcpyDeviceToHost));
    int diff = 0; double maxrel = 0;
    for (size_t i = 0; i < h1.size(); i++) if (h1[i] != h2[i]) {
      diff++; float f1, f2; memcpy(&f1, &h1[i], 4); memcpy(&f2, &h2[i], 4);
      maxrel = std::max(maxrel, (double)std::fabs(f1 - f2) / (std::fabs(f1) + 1e-30));
    }
    printf("cfg %d R=%d: %d / %zu outputs differ (max rel %.3g)\n", ci, R, diff, h1.size(), maxrel);
    bad += diff != 0;
  }
  printf(bad ? "FUSE_GU_MISMATCH\n" : "FUSE_GU_BITWISE_OK\n");
  return bad ? 1 : 0;
}
