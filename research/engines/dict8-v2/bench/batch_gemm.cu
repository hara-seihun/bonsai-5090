// Large-row GEMM screen for the batch regime: every production carrier (dense5, WF3) projection shape at
// R rows, every tile configuration x split-K, synthetic weights rotated over enough copies that every launch
// streams its weights from DRAM (as in a 64-layer pass). Prints the best configurations per shape with
// TMAC/s against the measured NVFP4 MMA peak, and the whole-pass sum at the best configurations.
//
//   batch-gemm R [R...]            (build: make batch-gemm, with the selected preset's DEFS)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>

namespace h5 {
cudaError_t launch_scale_carrier(int ci, int split, int mode, const uint8_t* A, const uint8_t* AS, const float* AG,
                                 int R, int K, const void* W, int N, float* out, int ldo, const float* mean, cudaStream_t st);
int fp4_ncfg();
bool fp4_cfg_wg2(int);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
void launch_gemm_fp4_silu(int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, cudaStream_t, const float*, int);
const char* fp4_cfg_name(int, int);
}
using namespace h5;

static void ck(cudaError_t e, const char* what) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(1); } }
__global__ void k_fill(uint8_t* p, size_t n, uint32_t seed, uint32_t mod, uint32_t add, uint32_t mask = 255) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = (uint8_t)(add + (mod ? x % mod : x)) & mask;
  }
}
__global__ void k_setf(float* p, size_t n, float v) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v; }

__global__ void k_scales6(uint32_t* p, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = uint32_t(0x30 + i % 16) * 0x01010101u;
}
struct Shape { const char* name; int N, K, mode, count; };   // mode 0 store, 1 add, 3 silu(gate)*up; count per pass
int main(int argc, char** argv) {
  // WF selects the weight format: 3 = dense5 carrier (single-stream preset), 2 = two-bit trits with E4M3 group
  // scales in the MMA scale operand (batch preset), 6 = pre-expanded identical operands.
  // batch-gemm [wf2|wf6|wf7] R [R...]
  int wf = 3, a0 = 1;
  if (argc > 1 && (std::string(argv[1]) == "wf2" || std::string(argv[1]) == "wf6" || std::string(argv[1]) == "wf7")) {
    wf = argv[1][2] - '0'; a0 = 2;
  }
  std::vector<int> Rs; for (int i = a0; i < argc; i++) Rs.push_back(atoi(argv[i]));
  if (Rs.empty()) Rs = {512};
  const Shape shapes[] = {{"in_gdn", 16384, 5120, 0, 48}, {"in_att", 14336, 5120, 0, 16}, {"out", 5120, 6144, 1, 64},
                          {"gu_silu", 34816, 5120, 3, 64}, {"down", 5120, 17408, 1, 64}, {"head128", 248320, 5120, 0, 4}};
  const int Rmax = *std::max_element(Rs.begin(), Rs.end());
  const size_t Rp = (Rmax + 127) / 128 * 128, Kmax = 17408;
  uint8_t *A, *AS; float *AG, *out, *mean;
  ck(cudaMalloc(&A, Rp * Kmax / 2), "A"); ck(cudaMalloc(&AS, Rp * Kmax / 16), "AS"); ck(cudaMalloc(&AG, Rp * 4), "AG");
  ck(cudaMalloc(&out, std::max((size_t)Rmax * 34816, (size_t)128 * 248320) * 4), "out"); ck(cudaMalloc(&mean, 248320 * 4), "mean");
  k_fill<<<512, 256>>>(A, Rp * Kmax / 2, 1, 0, 0);
  k_fill<<<512, 256>>>(AS, Rp * Kmax / 16, 2, 16, 0x30);   // UE4M3 scales 2^-3 .. ~2^-1
  k_setf<<<64, 256>>>(AG, Rp, 1.f); k_setf<<<512, 256>>>(mean, 248320, 1e-3f);
  // Weight copies: 8 x 36 MB (largest non-head shape) > L2, rotated per launch.
  const int NW = 8; const size_t wbytes = (size_t)(34816 / 32) * (5120 / 128) * 832;
  std::vector<uint8_t*> W(NW);
  const size_t cw = wf == 6 ? 2048 : wf == 3 ? 832 : 1024;
  const size_t wb = (size_t)(34816 / 32) * (5120 / 128) * cw;
  for (int i = 0; i < NW; i++) { ck(cudaMalloc(&W[i], wb), "W"); k_fill<<<512, 256>>>(W[i], wb, 10 + i, wf == 3 ? 243 : 0, 0, wf == 6 ? 0xaa : 255); }
  (void)wbytes;
  uint8_t* Wh; const size_t hbytes = (size_t)(248320 / 32) * 40 * cw; ck(cudaMalloc(&Wh, hbytes), "Wh"); k_fill<<<512, 256>>>(Wh, hbytes, 99, wf == 3 ? 243 : 0, 0, wf == 6 ? 0xaa : 255);
  const size_t cs = wf == 6 || wf == 7 ? 128 : 64;
  uint8_t* S2; const size_t sbytes = (size_t)(248320 / 32) * 136 * cs; ck(cudaMalloc(&S2, sbytes), "S2");
  if (wf == 6 || wf == 7) k_scales6<<<512, 256>>>((uint32_t*)S2, sbytes / 4);
  else k_fill<<<512, 256>>>(S2, sbytes, 7, 16, 0x30);
  ck(cudaDeviceSynchronize(), "fill");
  cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  const double peak = 981e12;   // mma_probe nvfp4 m16n8k64, register-resident, this card class
  for (int R : Rs) {
    double pass_us = 0, pass_mac = 0;
    printf("== WF%d R=%d\n", wf, R);
    for (const Shape& s : shapes) {
      const int Rr = s.N == 248320 ? std::min(R, 128) : R;
      struct Res { float us; int ci, ks; };
      std::vector<Res> res;
      for (int ci = 0; ci < fp4_ncfg(); ci++)
        for (int ks : {1, 2, 3, 4, 6}) {
          if (s.mode == 3 && (ks != 1 || !fp4_cfg_wg2(ci))) continue;
          if (s.mode == 0 && s.N == 248320 && ks != 1) continue;
          if (s.K / 128 / ks < 4) continue;
          const int mode = ks > 1 ? 2 : s.mode;
          const int ldo = s.mode == 3 ? s.N / 2 : s.N;
          auto launch = [&](int it) {
            const uint8_t* w = s.N == 248320 ? Wh : W[it % NW];
#ifdef H5_PROFILE_SCALE_CARRIER
            if (wf == 3) return launch_scale_carrier(ci, ks, mode, A, AS, AG, Rr, s.K, w, s.N, out, ldo, mean, st);
#else
            if (wf == 3) return cudaErrorNotSupported;   // carrier kernels not built (ablation builds)
#endif
            if (mode == 3) launch_gemm_fp4_silu(ci, A, AS, AG, Rr, s.K, w, S2, s.N, out, ldo, st, mean, wf);
            else launch_gemm_fp4_cfg(ci, ks, A, AS, AG, Rr, s.K, w, S2, s.N, out, ldo, s.mode == 1, st, mean, wf);
            return cudaPeekAtLastError();
          };
          if (launch(0) != cudaSuccess) { (void)cudaGetLastError(); continue; }
          for (int it = 1; it < 3; it++) launch(it);
          const int reps = 12;
          cudaEventRecord(e0, st);
          for (int it = 0; it < reps; it++) launch(it);
          cudaEventRecord(e1, st); ck(cudaEventSynchronize(e1), "sync");
          float ms; cudaEventElapsedTime(&ms, e0, e1);
          if (cudaGetLastError() != cudaSuccess) continue;
          res.push_back({ms * 1e3f / reps, ci, ks});
        }
      std::sort(res.begin(), res.end(), [](const Res& a, const Res& b) { return a.us < b.us; });
      const double mac = (double)Rr * s.N * s.K;
      printf("%-8s N=%6d K=%5d R=%4d:", s.name, s.N, s.K, Rr);
      for (size_t i = 0; i < std::min<size_t>(4, res.size()); i++)
        printf("  [%s] %.1f us %.0f TMAC/s (%.0f%%)", fp4_cfg_name(res[i].ci, res[i].ks), res[i].us, mac / res[i].us / 1e6, 100 * mac / res[i].us / 1e6 / (peak / 1e12));
      printf("\n");
      if (!res.empty()) { pass_us += res[0].us * s.count; pass_mac += mac * s.count; }
    }
    printf("pass at best configs: %.2f ms, %.0f TMAC/s (%.0f%% of %.0f), %.2f us/row\n", pass_us / 1e3, pass_mac / pass_us / 1e6,
           100 * pass_mac / pass_us / 1e6 / (peak / 1e12), peak / 1e12, pass_us / R);
    fflush(stdout);
  }
  return 0;
}
