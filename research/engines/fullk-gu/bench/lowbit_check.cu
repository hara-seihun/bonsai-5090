// WF 4/5 (3/2-bit E2M1-subset) GEMM against WF 1 (NVFP4) carrying the identical operand values:
// the same E2M1 codes as nibbles, each per-32 UE4M3 scale duplicated into the two per-16 scales, the same
// row scales. Both feed identical MMA operands, so outputs must match exactly for every config / split.
//   lowbit-check
#include "kernels.cuh"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace h5 {
int fp4_ncfg();
const char* fp4_cfg_name(int, int);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
}
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); exit(1); } }
template <class T> static T* up(const std::vector<T>& v) { T* p; ck(cudaMalloc(&p, v.size() * sizeof(T)), "m"); ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "c"); return p; }
int main() {
  const int N = 512, K = 1024, nkb = K / 128, NG = N / 32;
  std::mt19937 rng(11);
  int fails = 0;
  for (int bits : {3, 2}) {
    const int CW = bits == 3 ? 1536 : 1024;
    std::vector<uint8_t> code((size_t)N * K), s32((size_t)N * (K / 32));
    for (auto& c : code) c = (uint8_t)((rng() & 1) << 2 | (rng() % (bits == 3 ? 4 : 2)));
    for (auto& s : s32) s = (uint8_t)(0x28 + rng() % 24);   // UE4M3 2^-2 .. ~2^1
    std::vector<float> rs(N); for (auto& r : rs) r = 0.001f * (1 + rng() % 7);
    std::vector<uint8_t> Wl((size_t)NG * nkb * CW, 0), Sl((size_t)NG * nkb * 128), W1((size_t)NG * nkb * 2048, 0), S1((size_t)NG * nkb * 256);
    for (int g = 0; g < NG; g++) for (int kb = 0; kb < nkb; kb++) {
      const size_t c = (size_t)kb * NG + g;
      for (int r = 0; r < 32; r++) for (int b = 0; b < 4; b++) { const uint8_t v = s32[(size_t)(32 * g + r) * (K / 32) + kb * 4 + b]; Sl[c * 128 + r * 4 + b] = v; S1[c * 256 + r * 8 + 2 * b] = v; S1[c * 256 + r * 8 + 2 * b + 1] = v; }
      for (int l = 0; l < 32; l++) {
        const int gid = l >> 2, tq = l & 3; uint32_t words[12] = {};
        for (int n2 = 0; n2 < 16; n2++) {
          const int mm = n2 >> 3, h = (n2 >> 2) & 1, o = n2 & 3, row = 16 * mm + 8 * (o & 1) + gid, k0 = kb * 128 + 64 * h + 32 * (o >> 1) + 8 * tq;
          uint32_t nib = 0;
          for (int j = 0; j < 8; j++) {
            const uint8_t cc = code[(size_t)(32 * g + row) * K + k0 + j]; const uint32_t sg = cc >> 2, mg = cc & 3;
            nib |= (sg << 3 | (2 * mg + 1)) << (4 * j);
            if (bits == 2) { const int wi = n2 >> 1; words[wi] |= (n2 & 1) ? (sg << (4 * j + 2) | mg << (4 * j)) : (sg << (4 * j + 3) | mg << (4 * j + 1)); }
            else { const int q = n2 >> 2, idx = n2 & 3, m1 = mg >> 1, m0 = mg & 1; uint32_t& XY = words[3 * q + (idx >> 1)];
              XY |= (idx & 1) ? (sg << (4 * j + 1) | (uint32_t)m1 << (4 * j)) : (sg << (4 * j + 3) | (uint32_t)m1 << (4 * j + 2));
              static const int zpos[4] = {1, 0, 3, 2}; words[3 * q + 2] |= (uint32_t)m0 << (4 * j + zpos[idx]); }
          }
          memcpy(W1.data() + c * 2048 + (n2 >> 2) * 512 + l * 16 + (n2 & 3) * 4, &nib, 4);
        }
        for (int wi = 0; wi < (bits == 3 ? 12 : 8); wi++) memcpy(Wl.data() + c * CW + (wi >> 2) * 512 + l * 16 + (wi & 3) * 4, &words[wi], 4);
      }
    }
    std::vector<uint8_t> A((size_t)nkb * 64 * 64), AS((size_t)nkb * 64 * 8);
    for (auto& a : A) a = (uint8_t)rng();
    for (auto& s : AS) s = (uint8_t)(0x30 + rng() % 16);
    std::vector<float> AG(64, 1.f);
    auto dWl = up(Wl), dSl = up(Sl), dW1 = up(W1), dS1 = up(S1), dA = up(A), dAS = up(AS); auto dRS = up(rs), dAG = up(AG);
    float *o1, *o2; ck(cudaMalloc(&o1, 8 * N * 4), "m"); ck(cudaMalloc(&o2, 8 * N * 4), "m");
    cudaStream_t st; ck(cudaStreamCreate(&st), "s");
    for (int ci = 0; ci < h5::fp4_ncfg(); ci++)
      for (int ks : {1, 2}) {
        for (int R : {1, 8}) {
          h5::launch_gemm_fp4_cfg(ci, ks, dA, dAS, dAG, R, K, dW1, dS1, N, o1, N, false, st, dRS, 1);
          h5::launch_gemm_fp4_cfg(ci, ks, dA, dAS, dAG, R, K, dWl, dSl, N, o2, N, false, st, dRS, bits == 3 ? 4 : 5);
          ck(cudaStreamSynchronize(st), h5::fp4_cfg_name(ci, ks));
          std::vector<float> h1(R * N), h2(R * N);
          ck(cudaMemcpy(h1.data(), o1, R * N * 4, cudaMemcpyDeviceToHost), "c"); ck(cudaMemcpy(h2.data(), o2, R * N * 4, cudaMemcpyDeviceToHost), "c");
          int diff = 0; double mx = 0; for (int i = 0; i < R * N; i++) { diff += h1[i] != h2[i]; mx = std::max(mx, (double)fabsf(h1[i])); }
          if (diff && ks == 1) { fails++; printf("bits %d %s R %d: %d mismatches (max |ref| %.3g)\n", bits, h5::fp4_cfg_name(ci, ks), R, diff, mx); }
          if (ci == 0 && ks == 1 && R == 8) printf("bits %d: sample out %.5f %.5f, max |out| %.4g\n", bits, h1[0], h2[0], mx);
        }
      }
  }
  printf(fails ? "LOWBIT_CHECK_FAIL %d\n" : "LOWBIT_CHECK_OK\n", fails);
  return fails != 0;
}
