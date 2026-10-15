#include "kernels.cuh"
#include "deterministic_gemm.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace h5 { void launch_prep(const PrepArgs&, int, cudaStream_t); }
static void ck(cudaError_t e) {
  if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); }
}
template<class T> T* alloc(size_t n) {
  T* p; ck(cudaMalloc(&p, n * sizeof(T))); return p;
}
__global__ void bytes(uint8_t* p, size_t n, unsigned mod, unsigned seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    p[i] = (i * 29 + i / 89 + seed * 13) % mod;
}
__global__ void fill(float* p, size_t n, float value) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = value;
}
__global__ void residual(float* p, size_t n, int seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    p[i] = (int((i * 17 + seed * 31) % 127) - 63) * 0.015625f;
}
__global__ void norm_sign(float* norm, float* sign, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    norm[i] = 0.5f + float(i % 31) * 0.03125f; sign[i] = i % 3 ? 1.f : -1.f;
  }
}
static bool equal(const char* name, const void* a, const void* b, size_t n) {
  std::vector<uint8_t> x(n), y(n);
  ck(cudaMemcpy(x.data(), a, n, cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(y.data(), b, n, cudaMemcpyDeviceToHost));
  if (x == y) return true;
  for (size_t i = 0; i < n; i++) if (x[i] != y[i]) {
    fprintf(stderr, "%s differs at byte %zu: %02x != %02x\n", name, i, x[i], y[i]); break;
  }
  return false;
}
int main(int argc, char** argv) {
  const bool bound_only = argc == 2 && !strcmp(argv[1], "--bound640");
  if (argc != 1 && !bound_only) { fprintf(stderr, "usage: ordered-prep-bench [--bound640]\n"); return 2; }
  const std::vector<int> screen = bound_only ? std::vector<int>{0, 1, 4} : std::vector<int>{0, 1, 2, 3, 4};
  setenv("H5_PREP_CHUNK", "2", 1); setenv("H5_PREP_G4", "0.01", 1); setenv("H5_PREP_WIDE", "multi", 1);
  cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  constexpr int arms = 5, copies = 32;
  const char* names[arms] = {"separate", "serial", "load4", "fenced4", "bound640"};
  const char* pipe[arms] = {nullptr, "0", "1", "2", "3"};
  for (int R : {8, 16}) for (int K : {6144, 17408}) {
    const int N = 5120, split = K == 6144 ? 6 : 17, ci = 26;
    const size_t plane = size_t(R) * N, ab = size_t(64) * K / 2, qb = size_t(64) * N / 2;
    auto A = alloc<uint8_t>(ab), AS = alloc<uint8_t>(ab / 8), S = alloc<uint8_t>(size_t(N) * K / 64);
    auto AG = alloc<float>(64), RS = alloc<float>(N), part = alloc<float>(plane * split);
    auto norm = alloc<float>(N), sign = alloc<float>(N);
    const size_t wb = size_t(N / 32) * (K / 128) * 832;
    std::vector<uint8_t*> weights;
    for (int j = 0; j < copies; j++) {
      weights.push_back(alloc<uint8_t>(wb)); bytes<<<512, 256, 0, st>>>(weights.back(), wb, 243, j);
    }
    bytes<<<64, 256, 0, st>>>(A, ab, 256, 0);
    ck(cudaMemsetAsync(AS, 0x38, ab / 8, st)); ck(cudaMemsetAsync(S, 0x38, size_t(N) * K / 64, st));
    fill<<<64, 256, 0, st>>>(AG, 64, 0.125f); fill<<<64, 256, 0, st>>>(RS, N, 0.001f);
    norm_sign<<<64, 256, 0, st>>>(norm, sign, N);
    float* out[arms]; h5::PrepArgs prep[arms]{};
    for (int arm : screen) {
      out[arm] = alloc<float>(plane);
      auto& a = prep[arm]; a.x = out[arm]; a.ldx = N; a.n = N; a.norm_w = norm; a.signs = sign;
      a.flags = h5::P_NORM | h5::P_SIGN | h5::P_HADAMARD | h5::P_QUANT;
      a.q4 = alloc<uint8_t>(qb); a.s4 = alloc<uint8_t>(qb / 8); a.g4 = alloc<float>(64);
      ck(cudaMemsetAsync(a.q4, 0xa5, qb, st)); ck(cudaMemsetAsync(a.s4, 0xa5, qb / 8, st));
    }
    auto run = [&](int arm, int copy) {
      residual<<<64, 256, 0, st>>>(out[arm], plane, copy);
      if (arm == 0) {
        ck(h5::launch_gemm_fp4_deterministic(ci, split, A, AS, AG, R, K, weights[copy], S, N, out[arm], N, true,
                                           st, RS, 3, true, part, plane * split));
        h5::launch_prep(prep[arm], R, st);
      } else {
        setenv("H5_PREP_ORDERED_PIPE", pipe[arm], 1);
        ck(h5::launch_gemm_fp4_partials(ci, split, A, AS, AG, R, K, weights[copy], S, N, st, RS, 3, true, part, plane * split));
        ck(h5::launch_prep_ordered(prep[arm], {part, out[arm], R, split}, st));
      }
    };
    for (int rep = 0; rep < 5; rep++) {
      for (int arm : screen) run(arm, rep);
      ck(cudaStreamSynchronize(st));
      for (int arm : screen) {
        if (arm == 0) continue;
        const bool ok = equal("residual", out[0], out[arm], plane * 4)
                      & equal("q4", prep[0].q4, prep[arm].q4, qb)
                      & equal("s4", prep[0].s4, prep[arm].s4, qb / 8)
                      & equal("g4", prep[0].g4, prep[arm].g4, R * 4);
        if (!ok) { fprintf(stderr, "R=%d K=%d arm=%s rep=%d exactness failed\n", R, K, names[arm], rep); return 1; }
      }
    }
    printf("R=%d K=%d split=%d exact=1 copies=%d weight_ring_bytes=%zu\n", R, K, split, copies, wb * copies);
    cudaGraphExec_t exec[arms];
    for (int arm : screen) {
      cudaGraph_t graph; ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      for (int j = 0; j < copies; j++) run(arm, j);
      ck(cudaStreamEndCapture(st, &graph)); ck(cudaGraphInstantiate(&exec[arm], graph, 0)); ck(cudaGraphDestroy(graph));
      for (int i = 0; i < 8; i++) ck(cudaGraphLaunch(exec[arm], st));
    }
    cudaEvent_t begin, end; ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
    for (int left : screen) for (int right : screen) {
      if (right <= left) continue;
      for (int pass = 0; pass < 12; pass++) {
        const int arm = pass % 4 == 1 || pass % 4 == 2 ? right : left;
        ck(cudaEventRecord(begin, st)); for (int i = 0; i < 40; i++) ck(cudaGraphLaunch(exec[arm], st));
        ck(cudaEventRecord(end, st)); ck(cudaEventSynchronize(end));
        float ms; ck(cudaEventElapsedTime(&ms, begin, end));
        printf("R=%d K=%d split=%d pair=%s:%s pass=%d arm=%s us=%.6f\n",
               R, K, split, names[left], names[right], pass, names[arm], ms * 1000 / 40 / copies);
      }
    }
    ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
    for (int arm : screen) {
      ck(cudaGraphExecDestroy(exec[arm]));
      for (void* p : {(void*)out[arm], (void*)prep[arm].q4, (void*)prep[arm].s4, (void*)prep[arm].g4}) ck(cudaFree(p));
    }
    for (auto* w : weights) ck(cudaFree(w));
    for (void* p : {(void*)A, (void*)AS, (void*)S, (void*)AG, (void*)RS, (void*)part, (void*)norm, (void*)sign}) ck(cudaFree(p));
  }
  ck(cudaStreamDestroy(st)); printf("failures=0\n"); return 0;
}
