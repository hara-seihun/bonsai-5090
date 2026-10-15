#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace h5 {
void launch_prep(const PrepArgs&, int, cudaStream_t);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int,
                       const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
void launch_gemm_fp4_precleared_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int,
                                  const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
}

static void ck(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Allocations {
  std::vector<void*> pointers;
  template<class T> T* create(size_t n) {
    T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T))); pointers.push_back(p);
    ck(cudaMemset(p, 0, n * sizeof(T))); return p;
  }
  template<class T> T* upload(const std::vector<T>& values) {
    T* p = create<T>(values.size());
    ck(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
  }
  ~Allocations() { for (void* p : pointers) cudaFree(p); }
};
template<class T> static std::vector<T> read(const T* p, size_t n) {
  std::vector<T> values(n); ck(cudaMemcpy(values.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return values;
}
template<class T> static void exact(const std::vector<T>& a, const std::vector<T>& b, const char* name) {
  if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)))
    throw std::runtime_error(std::string("changed preparation: ") + name);
}
static double close(const std::vector<float>& a, const std::vector<float>& b, const char* name) {
  double maximum = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      throw std::runtime_error(std::string("poison/nonfinite output: ") + name + " at " + std::to_string(i));
    double difference = std::abs(double(a[i]) - b[i]);
    maximum = std::max(maximum, difference);
    if (difference > 3e-5 * (1 + std::abs(double(a[i]))))
      throw std::runtime_error(std::string("GEMM mismatch: ") + name + " at " + std::to_string(i));
  }
  return maximum;
}
struct Prepared {
  h5::PrepArgs args{};
  float* input_output;
  float* ab_output;
  int rows;
  static constexpr int K = 5120, N = 16384, AB = 96, GUARD = 32;
  Prepared(Allocations& memory, const float* x, const float* norm, const float* signs, int r) : rows(r) {
    args.x = x; args.norm_w = norm; args.signs = signs; args.n = K; args.ldx = K; args.ldo = K;
    args.flags = h5::P_NORM | h5::P_SIGN | h5::P_HADAMARD | h5::P_QUANT | h5::P_OUT_NORM | h5::P_STORE_F32;
    args.q4 = memory.create<uint8_t>(64 * K / 2); args.s4 = memory.create<uint8_t>(64 * K / 16);
    args.g4 = memory.create<float>(r); args.out_norm = memory.create<float>(r * K);
    args.out_f32 = memory.create<float>(r * K);
    input_output = memory.create<float>(r * N + 2 * GUARD) + GUARD;
    ab_output = memory.create<float>(r * AB + 2 * GUARD) + GUARD;
  }
  void poison(cudaStream_t stream) const {
    ck(cudaMemsetAsync(input_output - GUARD, 0xff, (size_t(rows) * N + 2 * GUARD) * 4, stream));
    ck(cudaMemsetAsync(ab_output - GUARD, 0xff, (size_t(rows) * AB + 2 * GUARD) * 4, stream));
  }
  void guard() const {
    for (auto region : {std::pair<float*, int>{input_output, rows * N}, std::pair<float*, int>{ab_output, rows * AB}}) {
      for (float* p : {region.first - GUARD, region.first + region.second}) {
        auto data = read(reinterpret_cast<uint32_t*>(p), GUARD);
        for (auto v : data) if (v != 0xffffffffu) throw std::runtime_error("clear/GEMM wrote outside declared buffer");
      }
    }
  }
};
struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ~Graph() { if (executable) cudaGraphExecDestroy(executable); if (graph) cudaGraphDestroy(graph); }
};

static constexpr const char* INVALID_CLEAR = "prep output clearing requires row norm mapping0/2 and aligned declared outputs";
static void rejection(const char* name, const h5::PrepArgs& args, int rows, cudaStream_t stream,
                      const char* expected = INVALID_CLEAR) {
  bool rejected = false;
  try { h5::launch_prep(args, rows, stream); }
  catch (const std::runtime_error& error) {
    if (std::strcmp(error.what(), expected))
      throw std::runtime_error(std::string("unexpected host rejection for ") + name + ": " + error.what());
    rejected = true;
    printf("{\"state\":\"rejected\",\"case\":\"%s\",\"error_kind\":\"prep_clear_argument\"}\n", name);
  }
  if (!rejected) {
    ck(cudaStreamSynchronize(stream));
    throw std::runtime_error(std::string("missing native host rejection: ") + name);
  }
}

static void run_case(int rows, int ab_splits, cudaStream_t stream) {
  using P = Prepared;
  Allocations memory;
  std::vector<float> x(rows * P::K), norm(P::K), signs(P::K);
  for (int i = 0; i < P::K; ++i) { norm[i] = .8f + (i % 11) * .03125f; signs[i] = (i % 3) ? 1.f : -1.f; }
  auto input = memory.upload(x); auto nw = memory.upload(norm); auto sg = memory.upload(signs);
  Prepared old(memory, input, nw, sg, rows), fresh(memory, input, nw, sg, rows);
  fresh.args.clear_a = fresh.input_output; fresh.args.clear_a_stride = P::N;
  fresh.args.clear_b = fresh.ab_output; fresh.args.clear_b_stride = P::AB;
  uint32_t rng = 8129;
  auto next = [&]() { rng = 1664525u * rng + 1013904223u; return rng >> 16; };
  const size_t blocks = P::K / 128;
  std::vector<uint8_t> labels(size_t(P::N / 32) * blocks * 832);
  for (auto& v : labels) v = next() % 243;
  std::vector<__half> scales(size_t(P::N / 32) * blocks * 32);
  for (auto& v : scales) v = __float2half(.015625f + (next() % 7) * .00390625f);
  auto weight = memory.upload(labels); auto scale = memory.upload(scales);
  std::vector<uint8_t> ab_weights(size_t(P::AB / 32) * blocks * 2048);
  for (auto& v : ab_weights) v = next() & 255;
  std::vector<uint8_t> ab_scales(size_t(P::AB / 32) * blocks * 256, 0x38);
  std::vector<float> row_scales(P::AB, .015625f);
  auto aw = memory.upload(ab_weights); auto as = memory.upload(ab_scales); auto rs = memory.upload(row_scales);
  auto launch = [&](Prepared& prepared, bool preclear) {
    h5::launch_prep(prepared.args, rows, stream);
    auto gemm = preclear ? h5::launch_gemm_fp4_precleared_cfg : h5::launch_gemm_fp4_cfg;
    gemm(26, 4, prepared.args.q4, prepared.args.s4, prepared.args.g4, rows, P::K,
         weight, scale, P::N, prepared.input_output, P::N, false, stream, nullptr, 3);
    gemm(2, ab_splits, prepared.args.q4, prepared.args.s4, prepared.args.g4, rows, P::K,
         aw, as, P::AB, prepared.ab_output, P::AB, false, stream, rs, 1);
    ck(cudaGetLastError());
  };
  // Populate the GEMM function table and its CUDA attributes before capture.
  launch(old, false); launch(fresh, true); ck(cudaStreamSynchronize(stream));
  Graph graphs[2];
  for (int route = 0; route < 2; ++route) {
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch(route ? fresh : old, route != 0);
    ck(cudaStreamEndCapture(stream, &graphs[route].graph));
    ck(cudaGraphInstantiate(&graphs[route].executable, graphs[route].graph, 0));
  }
  std::vector<float> prior;
  double input_max = 0, ab_max = 0;
  for (int visit = 0; visit < 6; ++visit) {
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = .7f * sinf(float(i % 8191) * .071f + visit * .43f) + .13f * cosf(float(i % 71) + visit);
    ck(cudaMemcpyAsync(input, x.data(), x.size() * 4, cudaMemcpyHostToDevice, stream));
    old.poison(stream); fresh.poison(stream);
    if (visit % 2) {
      ck(cudaGraphLaunch(graphs[1].executable, stream)); ck(cudaGraphLaunch(graphs[0].executable, stream));
    } else {
      ck(cudaGraphLaunch(graphs[0].executable, stream)); ck(cudaGraphLaunch(graphs[1].executable, stream));
    }
    ck(cudaStreamSynchronize(stream)); old.guard(); fresh.guard();
    exact(read(old.args.q4, 64 * P::K / 2), read(fresh.args.q4, 64 * P::K / 2), "q4");
    exact(read(old.args.s4, 64 * P::K / 16), read(fresh.args.s4, 64 * P::K / 16), "s4");
    exact(read(old.args.g4, rows), read(fresh.args.g4, rows), "g4");
    auto old_norm = read(old.args.out_norm, rows * P::K);
    exact(old_norm, read(fresh.args.out_norm, rows * P::K), "norm");
    exact(read(old.args.out_f32, rows * P::K), read(fresh.args.out_f32, rows * P::K), "Hadamard floats");
    for (float v : old_norm) if (!std::isfinite(v)) throw std::runtime_error("nonfinite norm");
    if (visit && !std::memcmp(prior.data(), old_norm.data(), prior.size() * 4))
      throw std::runtime_error("changed input did not change captured preparation");
    prior = old_norm;
    input_max = std::max(input_max, close(read(old.input_output, rows * P::N), read(fresh.input_output, rows * P::N), "input"));
    ab_max = std::max(ab_max, close(read(old.ab_output, rows * P::AB), read(fresh.ab_output, rows * P::AB), "AB"));
  }
  // Negative controls must observe poison when the owned split-K zero is absent.
  old.poison(stream);
  h5::launch_prep(old.args, rows, stream);
  h5::launch_gemm_fp4_precleared_cfg(26, 4, old.args.q4, old.args.s4, old.args.g4,
    rows, P::K, weight, scale, P::N, old.input_output, P::N, false, stream, nullptr, 3);
  ck(cudaStreamSynchronize(stream));
  auto missing_input_clear = read(old.input_output, rows * P::N);
  if (std::all_of(missing_input_clear.begin(), missing_input_clear.end(), [](float x) { return std::isfinite(x); }))
    throw std::runtime_error("missing-input-clear negative control did not retain poison");
  if (ab_splits > 1) {
    h5::launch_gemm_fp4_precleared_cfg(2, ab_splits, old.args.q4, old.args.s4, old.args.g4,
      rows, P::K, aw, as, P::AB, old.ab_output, P::AB, false, stream, rs, 1);
    ck(cudaStreamSynchronize(stream));
    auto output = read(old.ab_output, rows * P::AB);
    if (std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("missing-clear negative control did not retain poison");
  }
  if (rows == 1 && ab_splits == 1) {
    auto invalid = fresh.args; invalid.clear_a_stride = P::N - 1; rejection("unaligned-stride", invalid, rows, stream);
    invalid = fresh.args; invalid.clear_a = fresh.input_output + 1; rejection("unaligned-pointer", invalid, rows, stream);
    invalid = fresh.args; invalid.clear_b = fresh.ab_output + 1; rejection("unaligned-second-pointer", invalid, rows, stream);
    invalid = fresh.args; invalid.clear_a_stride = 0; rejection("zero-stride", invalid, rows, stream);
    invalid = fresh.args; invalid.clear_a = nullptr; rejection("stride-without-primary-pointer", invalid, rows, stream);
    invalid = fresh.args; invalid.clear_b = nullptr; rejection("stride-without-secondary-pointer", invalid, rows, stream);
    invalid = fresh.args; invalid.flags &= ~h5::P_NORM; rejection("not-norm", invalid, rows, stream);
    rejection("zero-rows", fresh.args, 0, stream);
    invalid = fresh.args; invalid.n = 5136; rejection("unaligned-width", invalid, rows, stream);
    invalid = fresh.args; invalid.n = 1024 * 18; rejection("oversize-width", invalid, rows, stream);
    invalid = fresh.args; invalid.ldx = P::K - 1; rejection("short-input-stride", invalid, rows, stream);
    invalid = fresh.args; invalid.rowmap = memory.create<int>(rows); rejection("mapped-input", invalid, rows, stream);
    invalid = fresh.args; invalid.clear_b = invalid.clear_a;
    rejection("overlapping-outputs", invalid, rows, stream, "prep clear outputs overlap");
    invalid = fresh.args; invalid.clear_a = const_cast<float*>(invalid.x);
    rejection("aliases-input", invalid, rows, stream, "prep clear output aliases a live prep operand");
    invalid = fresh.args; invalid.clear_a = invalid.out_norm;
    rejection("aliases-normalized-output", invalid, rows, stream, "prep clear output aliases a live prep operand");
    invalid = fresh.args; invalid.clear_a = nullptr; invalid.clear_b = nullptr;
    rejection("strides-without-pointers", invalid, rows, stream, "prep output clear stride requires its pointer");
    auto reject_gemm = [&](const char* name, int ci, int splits, int r, int k, int stride, bool add) {
      bool rejected = false;
      try {
        h5::launch_gemm_fp4_precleared_cfg(ci, splits, fresh.args.q4, fresh.args.s4, fresh.args.g4,
          r, k, weight, scale, P::N, fresh.input_output, stride, add, stream, nullptr, 3);
      } catch (const std::runtime_error& e) {
        if (std::strcmp(e.what(), "precleared GEMM requires valid nonadd full-row operands")) throw;
        rejected = true;
        printf("{\"state\":\"rejected\",\"case\":\"%s\",\"error_kind\":\"precleared_gemm_argument\"}\n", name);
      }
      if (!rejected) throw std::runtime_error(std::string("missing precleared GEMM rejection: ") + name);
    };
    reject_gemm("gemm-add", 26, 4, rows, P::K, P::N, true);
    reject_gemm("gemm-zero-rows", 26, 4, 0, P::K, P::N, false);
    reject_gemm("gemm-short-output-stride", 26, 4, rows, P::K, P::N - 1, false);
    reject_gemm("gemm-invalid-K", 26, 4, rows, P::K - 1, P::N, false);
    reject_gemm("gemm-zero-splits", 26, 0, rows, P::K, P::N, false);
    reject_gemm("gemm-invalid-config", -1, 4, rows, P::K, P::N, false);
  }
  printf("{\"state\":\"passed\",\"rows\":%d,\"input_splits\":4,\"ab_splits\":%d,\"graph_visits\":6,\"prep\":\"bitwise_equal\",\"input_max_abs\":%.9g,\"ab_max_abs\":%.9g}\n",
         rows, ab_splits, input_max, ab_max);
}

int main(int argc, char** argv) {
  try {
    if (argc != 3 || (strcmp(argv[1], "0") && strcmp(argv[1], "2")) ||
        (strcmp(argv[2], "0") && strcmp(argv[2], "1")))
      throw std::runtime_error("usage: prep-clear-test MAPPING(0|2) PDL(0|1)");
    if (setenv("H5_PREP_CHUNK", argv[1], 1) || setenv("H5_PDL", argv[2], 1))
      throw std::runtime_error("environment setup failed");
    const int scale_status = !strcmp(argv[1], "0") ? unsetenv("H5_PREP_G4") : setenv("H5_PREP_G4", ".015625", 1);
    if (scale_status) throw std::runtime_error("scale environment setup failed");
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int rows : {1, 8}) for (int splits : {1, 2, 4, 8}) run_case(rows, splits, stream);
    ck(cudaStreamDestroy(stream));
    printf("{\"state\":\"complete\",\"mapping\":%s,\"pdl\":%s,\"cases\":8}\n", argv[1], argv[2]);
    return 0;
  } catch (const std::exception& error) {
    fprintf(stderr, "prep-clear-test error: %s\n", error.what()); return 1;
  }
}
