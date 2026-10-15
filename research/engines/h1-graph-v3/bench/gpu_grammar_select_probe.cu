// Standalone: nvcc -O2 -std=c++17 -arch=sm_120 -Isrc -I<json-vendor> bench/gpu_grammar_select_probe.cu -o gpu-grammar-select-probe
#include "gpu_grammar_select.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace h5::gpu_grammar;
constexpr int VOCABULARY = 248320;
constexpr float GUARD_VALUE = 12345.5f;
constexpr int GUARD_ID = -12345;

static void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
    std::exit(2);
  }
}
static void require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}
struct Memory {
  std::vector<void*> allocations;
  template<class T> T* make(size_t n) {
    T* p = nullptr;
    cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&p), n * sizeof(T)));
    allocations.push_back(p);
    return p;
  }
  ~Memory() { for (void* p : allocations) cuda_check(cudaFree(p)); }
};
struct Adjustment { int id; float value; };
struct Case {
  std::string name;
  std::vector<int> legal;
  std::vector<float> scores;
  std::vector<Adjustment> adjustments;
  RuntimeError incoming_error = RuntimeError::None;
};
static std::vector<int> legal_ids(int n) {
  std::vector<int> ids(n);
  // Coprime stride distributes legal IDs across all partitions and vocabulary edges.
  for (int i = 0; i < n; ++i) ids[i] = (int64_t(i) * 8191 + VOCABULARY - 1) % VOCABULARY;
  std::sort(ids.begin(), ids.end());
  require(std::adjacent_find(ids.begin(), ids.end()) == ids.end(), "legal IDs unique");
  return ids;
}
static Case make_case(const std::string& name, int n) {
  Case c{name, legal_ids(n), {}, {}, RuntimeError::None};
  c.scores.resize(n);
  for (int i = 0; i < n; ++i) c.scores[i] = float((int64_t(i) * 37) % 113 - 56) * 0.25f;
  return c;
}
struct Candidate { float value; int id; };
static std::vector<Candidate> reference(const Case& c) {
  std::vector<Candidate> candidates;
  for (size_t i = 0; i < c.legal.size(); ++i) {
    float score = c.scores[i];
    for (const auto& adjustment : c.adjustments) if (adjustment.id == c.legal[i]) {
      // Force one binary32 rounding per adjustment, independent of device helper code.
      volatile float rounded = score + adjustment.value;
      score = rounded;
    }
    if (std::isfinite(score)) candidates.push_back({score, c.legal[i]});
  }
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    return a.value != b.value ? a.value > b.value : a.id < b.id;
  });
  return candidates;
}
struct Fixture {
  Memory memory;
  float* dense = memory.make<float>(2 * VOCABULARY);
  float* sparse = memory.make<float>(COMPACT_CAPACITY);
  uint32_t* source = memory.make<uint32_t>((VOCABULARY + 31) / 32);
  uint32_t* mask = memory.make<uint32_t>((VOCABULARY + 31) / 32);
  int* ids = memory.make<int>(COMPACT_CAPACITY);
  uint32_t* count = memory.make<uint32_t>(1);
  RowResult* result = memory.make<RowResult>(1);
  float* out_values = memory.make<float>(SELECT_TOP + 2);
  int* out_ids = memory.make<int>(SELECT_TOP + 2);
  int2* adjustments = memory.make<int2>(16);
  SelectWorkspace workspace{memory.make<float>(SELECT_PARTS * SELECT_TOP),
                            memory.make<int>(SELECT_PARTS * SELECT_TOP)};
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  explicit Fixture() { cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Fixture() {
    if (executable) cuda_check(cudaGraphExecDestroy(executable));
    if (graph) cuda_check(cudaGraphDestroy(graph));
    cuda_check(cudaStreamDestroy(stream));
  }
  void prepare(const Case& c) {
    require(c.adjustments.size() <= 16, "adjustment capacity");
    require(c.legal.size() == c.scores.size(), "score cardinality");
    std::fill(source, source + (VOCABULARY + 31) / 32, 0u);
    std::fill(mask, mask + (VOCABULARY + 31) / 32, 0xdeadbeefu);
    std::fill(ids, ids + COMPACT_CAPACITY, -99);
    std::fill(dense, dense + VOCABULARY, 9999.f);
    std::fill(sparse, sparse + COMPACT_CAPACITY, 9999.f);
    const bool compact = c.legal.size() <= COMPACT_CAPACITY;
    for (size_t i = 0; i < c.legal.size(); ++i) {
      const int id = c.legal[i];
      source[id / 32] |= 1u << (id % 32);
      dense[id] = compact ? std::numeric_limits<float>::quiet_NaN() : c.scores[i];
      if (compact) sparse[i] = c.scores[i];
    }
    if (!compact) std::fill(sparse, sparse + COMPACT_CAPACITY, std::numeric_limits<float>::quiet_NaN());
    for (int i = 0; i < SELECT_TOP + 2; ++i) {
      out_values[i] = GUARD_VALUE;
      out_ids[i] = GUARD_ID;
    }
    std::fill(workspace.values, workspace.values + SELECT_PARTS * SELECT_TOP, 99999.f);
    std::fill(workspace.ids, workspace.ids + SELECT_PARTS * SELECT_TOP, 0);
    // Graphs have fixed adjustment_count; unused entries cannot match any token ID.
    for (int i = 0; i < 16; ++i) {
      adjustments[i].x = i < int(c.adjustments.size()) ? c.adjustments[i].id : -1;
      const float delta = i < int(c.adjustments.size()) ? c.adjustments[i].value : 0.f;
      std::memcpy(&adjustments[i].y, &delta, sizeof(delta));
    }
    *count = 1234567;
    *result = RowResult{State{17, 0}, State{23, 1}, 0, 77, uint32_t(c.incoming_error)};
  }
  template<int TOP> void enqueue(const int* dense_row = nullptr) {
    cuda_check(launch_external_mask(source, (VOCABULARY + 31) / 32, mask, ids, count, result, stream));
    cuda_check(launch_select<TOP>(dense, sparse, VOCABULARY, mask, ids, count,
        out_values + 1, out_ids + 1, result, adjustments, 16, workspace, stream, dense_row));
  }
  template<int TOP> void capture(const int* dense_row = nullptr) {
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    enqueue<TOP>(dense_row);
    cuda_check(cudaStreamEndCapture(stream, &graph));
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  }
  template<int TOP> void check(const Case& c, const char* route) {
    const std::string label = c.name + "/top" + std::to_string(TOP) + "/" + route;
    auto expected = reference(c);
    RuntimeError error = c.incoming_error;
    if (c.legal.empty()) error = RuntimeError::NoLegalTokens;
    else if (error == RuntimeError::None && c.legal.size() > 1 && expected.empty())
      error = RuntimeError::NonfiniteLogits;
    require(result->error == uint32_t(error), label + " typed error");
    require(*count == c.legal.size() && result->allowed == c.legal.size(), label + " allowed count");
    require(result->before.id == 23 && result->before.ended == 1 && result->after.id == 23 &&
        result->after.ended == 1 && result->complete == 77, label + " row state preserved");
    require(out_values[0] == GUARD_VALUE && out_ids[0] == GUARD_ID, label + " leading guard");
    for (int k = TOP + 1; k < SELECT_TOP + 2; ++k)
      require(out_values[k] == GUARD_VALUE && out_ids[k] == GUARD_ID, label + " trailing guard");
    for (int k = 0; k < TOP; ++k) {
      int id = -1;
      float value = -std::numeric_limits<float>::infinity();
      if (error == RuntimeError::None) {
        if (c.legal.size() == 1) { id = c.legal[0]; if (!k) value = 0.f; }
        else { id = k < int(expected.size()) ? expected[k].id : expected[0].id;
          if (k < int(expected.size())) value = expected[k].value; }
      }
      require(out_ids[k + 1] == id, label + " rank " + std::to_string(k) + " ID expected " +
          std::to_string(id) + " got " + std::to_string(out_ids[k + 1]));
      require(out_values[k + 1] == value, label + " rank " + std::to_string(k) + " value");
      if (error == RuntimeError::None) require(std::binary_search(c.legal.begin(), c.legal.end(), out_ids[k + 1]),
          label + " output legal including padding");
    }
  }
};
static std::vector<Case> cases() {
  std::vector<Case> out;
  for (int n : {0, 1, 2, 7, 15, 16, 17, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1025, VOCABULARY})
    out.push_back(make_case("cardinality-" + std::to_string(n), n));
  for (int n : {1, 2, 512, 513, VOCABULARY}) {
    auto c = make_case("all-ties-" + std::to_string(n), n);
    std::fill(c.scores.begin(), c.scores.end(), 0.f);
    out.push_back(c);
  }
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (int n : {2, 7, 512, 513, VOCABULARY}) {
    auto c = make_case("adjustments-" + std::to_string(n), n);
    c.adjustments = {{c.legal.back(), 64.f}, {c.legal.back(), 32.f},
                     {c.legal.front(), -128.f}, {-1, 99999.f}};
    out.push_back(c);
    c = make_case("filtered-nan-inf-" + std::to_string(n), n);
    for (int i = 0; i < n; ++i) c.scores[i] = i % 4 == 0 ? nan : i % 4 == 1 ? inf : i % 4 == 2 ? -inf : 3.f;
    c.scores[n - 1] = -7.f;
    out.push_back(c);
    c = make_case("all-nonfinite-" + std::to_string(n), n);
    for (int i = 0; i < n; ++i) c.scores[i] = i % 3 == 0 ? nan : i % 3 == 1 ? inf : -inf;
    out.push_back(c);
    c = make_case("adjustment-nonfinite-" + std::to_string(n), n);
    std::fill(c.scores.begin(), c.scores.end(), nan);
    c.scores[0] = 1.f;
    c.adjustments = {{c.legal[0], inf}};
    out.push_back(c);
  }
  for (int n : {1, 512, 513}) {
    auto c = make_case("sticky-error-" + std::to_string(n), n);
    c.incoming_error = RuntimeError::InvalidState;
    out.push_back(c);
  }
  auto singleton = make_case("singleton-nan-ignores-adjustment", 1);
  singleton.scores[0] = nan;
  singleton.adjustments = {{singleton.legal[0], inf}};
  out.push_back(singleton);
  auto padding = make_case("one-finite-top16-padding-wide", 513);
  std::fill(padding.scores.begin(), padding.scores.end(), nan);
  padding.scores[400] = -123.f;
  out.push_back(padding);
  return out;
}
template<int TOP> static void run(const std::vector<Case>& inputs, int& passed) {
  Fixture f;
  f.capture<TOP>();
  for (const auto& c : inputs) {
    f.prepare(c);
    f.enqueue<TOP>();
    cuda_check(cudaStreamSynchronize(f.stream));
    f.check<TOP>(c, "direct");
    ++passed;
    for (int replay = 0; replay < 2; ++replay) {
      f.prepare(c);
      cuda_check(cudaGraphLaunch(f.executable, f.stream));
      cuda_check(cudaStreamSynchronize(f.stream));
      f.check<TOP>(c, "graph-replay");
      ++passed;
    }
  }
  auto singleton = make_case("singleton-null-logits", 1);
  f.prepare(singleton);
  cuda_check(launch_external_mask(f.source, (VOCABULARY + 31) / 32, f.mask, f.ids, f.count, f.result, f.stream));
  cuda_check(launch_select<TOP>(nullptr, nullptr, VOCABULARY, f.mask, f.ids, f.count,
      f.out_values + 1, f.out_ids + 1, f.result, nullptr, 0, f.workspace, f.stream));
  cuda_check(cudaStreamSynchronize(f.stream));
  f.check<TOP>(singleton, "null-logits");
  ++passed;
  require(launch_select<TOP>(f.dense, f.sparse, VOCABULARY, f.mask, f.ids, f.count,
      f.out_values + 1, f.out_ids + 1, f.result, nullptr, 0, {nullptr, f.workspace.ids}, f.stream) ==
      cudaErrorInvalidValue, "missing workspace typed CUDA error");
  require(launch_select<TOP>(f.dense, f.sparse, 0, f.mask, f.ids, f.count,
      f.out_values + 1, f.out_ids + 1, f.result, nullptr, 0, f.workspace, f.stream) ==
      cudaErrorInvalidValue, "invalid vocabulary typed CUDA error");
}
template<int TOP> static void run_mapped(int& passed) {
  Fixture f;
  int* dense_row = f.memory.make<int>(1);
  f.capture<TOP>(dense_row);
  for (int n : {513, 512, VOCABULARY, 1, 513, 0, 7, 1025}) {
    auto c = make_case("mapped-row-" + std::to_string(n), n);
    for (int replay = 0; replay < 2; ++replay) {
      f.prepare(c);
      *dense_row = n > int(COMPACT_CAPACITY) ? 1 : -1;
      std::copy(f.dense, f.dense + VOCABULARY, f.dense + VOCABULARY);
      std::fill(f.dense, f.dense + VOCABULARY, 99999.f);
      if (!replay) f.enqueue<TOP>(dense_row);
      else cuda_check(cudaGraphLaunch(f.executable, f.stream));
      cuda_check(cudaStreamSynchronize(f.stream));
      f.check<TOP>(c, replay ? "mapped-graph-replay" : "mapped-direct");
      ++passed;
    }
  }
  for (int n : {1, 7, 512}) {
    auto c = make_case("compact-negative-mapping-null-dense-" + std::to_string(n), n);
    f.prepare(c);
    *dense_row = -1;
    cuda_check(launch_external_mask(f.source, (VOCABULARY + 31) / 32, f.mask, f.ids, f.count, f.result, f.stream));
    cuda_check(launch_select<TOP>(nullptr, n == 1 ? nullptr : f.sparse, VOCABULARY, f.mask, f.ids,
        f.count, f.out_values + 1, f.out_ids + 1, f.result, f.adjustments, 16, f.workspace, f.stream, dense_row));
    cuda_check(cudaStreamSynchronize(f.stream));
    f.check<TOP>(c, "null-dense");
    ++passed;
  }
}
int main() {
  int passed = 0;
  const auto inputs = cases();
  run<1>(inputs, passed);
  run<SELECT_TOP>(inputs, passed);
  run_mapped<1>(passed);
  run_mapped<SELECT_TOP>(passed);
  std::printf("PASS: GPU grammar selector %d executions; TOP1/16 singleton, compact<=512, wide>=513/full248320, original-ID ties, adjustments, NaN/Inf filtering, safe padding, typed errors, dense row mapping, graph replay\n", passed);
  return 0;
}
