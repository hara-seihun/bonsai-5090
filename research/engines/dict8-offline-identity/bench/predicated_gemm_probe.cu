// Contract and invocation: bench/PREDICATED_GEMM.md. Link with the preset's fp4.o.
#include "deterministic_gemm.cuh"
#include "scale_carrier.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#ifndef H5_PROFILE_SCALE_CARRIER
#error "predicated_gemm_probe requires H5_PROFILE_SCALE_CARRIER and a matching fp4.o"
#endif

namespace h5 {
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int,
    const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
}
static void ck(cudaError_t e, const char* label) {
  if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(e)); std::exit(1); }
}
struct Memory {
  std::vector<void*> pointers;
  template<class T> T* make(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "allocate");
    pointers.push_back(p);
    return p;
  }
  ~Memory() { for (void* p : pointers) ck(cudaFree(p), "free"); }
};
template<class T> static void upload(T* p, const std::vector<T>& v, cudaStream_t st) {
  ck(cudaMemcpyAsync(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, st), "upload");
}
static std::vector<float> download(const float* p, size_t n, cudaStream_t st) {
  std::vector<float> v(n);
  ck(cudaMemcpyAsync(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost, st), "download");
  ck(cudaStreamSynchronize(st), "join");
  return v;
}
__global__ void set_count(uint32_t* count, uint32_t value) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x == 0) {
    // Release before publishing: the head must acquire completion, not launch.
    __nanosleep(1000);
    *count = value;
  }
}
static void launch_count(uint32_t* count, uint32_t value, cudaStream_t st) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(1); cfg.blockDim = dim3(32); cfg.stream = st;
  cfg.attrs = &attr; cfg.numAttrs = 1;
  ck(cudaLaunchKernelEx(&cfg, set_count, count, value), "count producer");
}
struct Fixture {
  static constexpr int K = 512, CI = 0;
  static constexpr float OUT_SENTINEL = 0.375f, PART_SENTINEL = -123.25f;
  Memory memory;
  int R, N, ldo;
  size_t outputs, planes;
  uint8_t *A, *AS, *W;
  __half* S;
  float *AG, *means, *out, *partials;
  uint32_t* count;
  std::vector<float> initial_out, initial_partials;
  Fixture(int rows, int columns, cudaStream_t st) : R(rows), N(columns), ldo(columns + 32),
      outputs(size_t(rows) * ldo), planes(size_t(rows) * columns * 4) {
    const size_t ab = size_t((R + 63) / 64) * 64 * K / 2;
    const size_t wb = size_t(N / 32) * (K / 128) * 832;
    const size_t scales = size_t(N / 32) * (K / 128) * 32;
    A = memory.make<uint8_t>(ab); AS = memory.make<uint8_t>(ab / 8);
    W = memory.make<uint8_t>(wb); S = memory.make<__half>(scales);
    AG = memory.make<float>((R + 63) / 64 * 64); means = memory.make<float>(N);
    out = memory.make<float>(outputs); partials = memory.make<float>(planes);
    count = memory.make<uint32_t>(1);
    std::vector<uint8_t> a(ab), as(ab / 8, 0x38), w(wb);
    for (size_t i = 0; i < a.size(); ++i) a[i] = uint8_t((i * 73 + (i >> 7) * 19 + 31) & 255);
    for (size_t i = 0; i < w.size(); ++i) w[i] = uint8_t((i * 37 + (i >> 8) * 17 + 53) % 243);
    std::vector<__half> s(scales);
    for (size_t i = 0; i < s.size(); ++i) s[i] = __float2half(0.002f + float(i % 13) * 0.00025f);
    std::vector<float> ag((R + 63) / 64 * 64);
    for (size_t i = 0; i < ag.size(); ++i) ag[i] = 0.03125f + float(i % 5) * 0.00390625f;
    upload(A, a, st); upload(AS, as, st); upload(W, w, st); upload(S, s, st); upload(AG, ag, st);
    ck(h5::scale_carrier_prepare(S, means, nullptr, N, K, st), "prepare means");
    ck(cudaStreamSynchronize(st), "prepare fixture");
    for (float v : download(means, N, st)) if (!std::isfinite(v) || v <= 0.f) {
      std::fprintf(stderr, "invalid prepared mean\n"); std::exit(1);
    }
    initial_out.assign(outputs, OUT_SENTINEL);
    initial_partials.assign(planes, PART_SENTINEL);
  }
  void reset(cudaStream_t st) {
    upload(out, initial_out, st); upload(partials, initial_partials, st);
  }
  cudaError_t control(bool deterministic, bool carrier, int split, bool add, cudaStream_t st) {
    if (deterministic) return h5::launch_gemm_fp4_deterministic(CI, split, A, AS, AG, R, K,
        W, S, N, out, ldo, add, st, means, 3, carrier, partials, planes);
    if (carrier) {
      if (split > 1 && !add) ck(cudaMemsetAsync(out, 0, outputs * sizeof(float), st), "control clear");
      return h5::launch_scale_carrier(CI, split, split > 1 ? 2 : add ? 1 : 0,
          A, AS, AG, R, K, W, N, out, ldo, means, st);
    }
    h5::launch_gemm_fp4_cfg(CI, split, A, AS, AG, R, K, W, S, N, out, ldo, add, st, means, 3);
    return cudaGetLastError();
  }
  cudaError_t predicated(bool deterministic, bool carrier, int split, bool add, bool active,
      uint32_t lo, uint32_t hi, cudaStream_t st) {
    // Aligned, nonnull and deliberately unmapped. Any operand read on the skipped
    // path fails the CUDA join rather than merely leaving plausible sentinels.
    const uintptr_t inaccessible = 0x10000;
    return h5::launch_gemm_fp4_predicated(CI, split,
        active ? A : reinterpret_cast<const uint8_t*>(inaccessible),
        active ? AS : reinterpret_cast<const uint8_t*>(inaccessible),
        active ? AG : reinterpret_cast<const float*>(inaccessible), R, K,
        active ? W : reinterpret_cast<const void*>(inaccessible),
        active ? S : reinterpret_cast<const __half*>(inaccessible), N,
        out, ldo, add, st, active ? means : reinterpret_cast<const float*>(inaccessible),
        3, carrier, partials, planes, deterministic, count, lo, hi);
  }
};
static bool exact(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
}
int main() {
  // Exercise PDL even if the invoking shell normally disables it. Set before the
  // library's first launch initializes its process-local launch configuration.
  if (setenv("H5_PDL", "1", 1)) { std::perror("setenv"); return 1; }
  cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, 0), "device");
  if (prop.major != 12 || prop.minor != 0) {
    std::fprintf(stderr, "requires sm_120a, got %s cc=%d.%d\n", prop.name, prop.major, prop.minor); return 2;
  }
  cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
  unsigned cases = 0, failures = 0;
  for (int R : {1, 8}) for (bool dense : {false, true}) {
    Fixture f(R, dense ? 1024 : 512, st);
    const uint32_t lo = dense ? 513 : 2;
    const uint32_t hi = dense ? std::numeric_limits<uint32_t>::max() : 512;
    for (bool carrier : {false, true}) for (bool deterministic : {false, true})
      for (int split : {1, 4}) for (bool add : {false, true}) {
        f.reset(st); ck(f.control(deterministic, carrier, split, add, st), "control");
        const auto reference = download(f.out, f.outputs, st);
        const auto reference_partials = download(f.partials, f.planes, st);
        bool useful_reference = false;
        for (int r = 0; r < R; ++r) for (int n = 0; n < f.N; ++n) {
          const float v = reference[size_t(r) * f.ldo + n];
          if (!std::isfinite(v)) { std::fprintf(stderr, "nonfinite control\n"); return 1; }
          useful_reference |= v != 0.f && v != Fixture::OUT_SENTINEL;
        }
        if (!useful_reference) { std::fprintf(stderr, "degenerate control\n"); return 1; }
        for (uint32_t value : {0u, 1u, 2u, 512u, 513u}) {
          const bool active = value >= lo && value <= hi;
          cudaGraph_t graph; cudaGraphExec_t exec;
          ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), "capture");
          launch_count(f.count, value, st);
          ck(f.predicated(deterministic, carrier, split, add, active, lo, hi, st), "predicated launch");
          ck(cudaStreamEndCapture(st, &graph), "end capture");
          ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
          bool passed = true;
          double max_abs = 0;
          for (int replay = 0; replay < 2; ++replay) {
            f.reset(st);
            const uint32_t poison = active ? 0 : lo;
            ck(cudaMemcpyAsync(f.count, &poison, sizeof(poison), cudaMemcpyHostToDevice, st), "poison count");
            ck(cudaGraphLaunch(exec, st), "replay");
            const auto got = download(f.out, f.outputs, st);
            const auto got_partials = download(f.partials, f.planes, st);
            if (!active) {
              passed &= exact(got, f.initial_out) && exact(got_partials, f.initial_partials);
            } else {
              for (int r = 0; r < R; ++r) for (int n = 0; n < f.ldo; ++n) {
                const size_t i = size_t(r) * f.ldo + n;
                if (n >= f.N || deterministic || split == 1) {
                  passed &= !std::memcmp(&got[i], &reference[i], sizeof(float));
                } else {
                  const double err = std::fabs(double(got[i]) - double(reference[i]));
                  max_abs = std::max(max_abs, err);
                  passed &= std::isfinite(got[i]) && err <= 1e-6 + 2e-6 * std::fabs(double(reference[i]));
                }
              }
              passed &= exact(got_partials, reference_partials);
            }
          }
          ++cases; if (!passed) ++failures;
          std::printf("%s N%d R%d carrier%d ordered%d split%d add%d count%u active%d atomic_max_abs=%.9g\n",
              passed ? "PASS" : "FAIL", f.N, R, int(carrier), int(deterministic), split,
              int(add), value, int(active), max_abs);
          ck(cudaGraphExecDestroy(exec), "destroy exec"); ck(cudaGraphDestroy(graph), "destroy graph");
        }
      }
  }
  ck(cudaStreamDestroy(st), "destroy stream");
  std::printf("predicated_gemm_probe cases=%u failures=%u\n", cases, failures);
  return failures ? 1 : 0;
}
