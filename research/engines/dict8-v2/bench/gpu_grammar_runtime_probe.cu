// Build separately: nvcc -std=c++17 -Isrc -I/workspace/h5-llama/vendor bench/gpu_grammar_runtime_probe.cu -o /tmp/gpu-grammar-runtime-probe
#include "gpu_grammar_token_compile.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace h5::gpu_grammar;

static void cuda_check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}
struct Allocations {
  std::vector<void*> pointers;
  template<class T> T* make(size_t count) {
    T* pointer = nullptr;
    cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&pointer), sizeof(T) * count));
    pointers.push_back(pointer);
    return pointer;
  }
  ~Allocations() { for (void* pointer : pointers) cudaFree(pointer); }
};
static DevicePlan compile_tokens(Allocations& memory, BytePlan source,
                                  DeviceVocabulary vocabulary) {
  auto* counts = memory.make<uint32_t>(source.states);
  auto* offsets = memory.make<uint32_t>(source.states);
  cuda_check(cudaMemsetAsync(counts, 0, source.states * sizeof(uint32_t), nullptr));
  cuda_check(launch_token_counts(source, vocabulary, counts, nullptr));
  cuda_check(cudaDeviceSynchronize());
  uint32_t sparse_size = 0, wide_size = 0;
  for (uint32_t state = 0; state < source.states; ++state) {
    if (counts[state] <= COMPACT_CAPACITY) {
      offsets[state] = sparse_size;
      sparse_size += counts[state];
    } else offsets[state] = wide_size++;
  }
  auto* sparse_ids = memory.make<int>(std::max(1u, sparse_size));
  auto* sparse_next = memory.make<uint32_t>(std::max(1u, sparse_size));
  auto* wide_next = memory.make<uint32_t>(std::max(size_t(1), size_t(wide_size) * vocabulary.size));
  auto* wide_mask = memory.make<uint32_t>(std::max(size_t(1), size_t(wide_size) * ((vocabulary.size + 31u) / 32u)));
  auto* wide_ids = memory.make<int>(std::max(size_t(1), size_t(wide_size) * COMPACT_CAPACITY));
  auto* dense_next = memory.make<uint32_t>(size_t(TOKEN_COMPILE_CHUNK) * vocabulary.size);
  for (uint32_t first = 0; first < source.states; first += TOKEN_COMPILE_CHUNK)
    cuda_check(launch_token_chunk(source, vocabulary, first,
        std::min(TOKEN_COMPILE_CHUNK, source.states - first), counts, offsets,
        dense_next, sparse_ids, sparse_next, wide_next, wide_mask, wide_ids, nullptr));
  cuda_check(cudaDeviceSynchronize());
  return DevicePlan{source.accepting, source.states, counts, offsets, sparse_ids,
                    sparse_next, wide_next, wide_mask, wide_ids};
}
#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)

static int compact_boundary_probe() {
  constexpr uint32_t size = 514, words = (size + 31) / 32;
  Allocations memory;
  auto* transitions = memory.make<uint32_t>(3 * 256);
  auto* accepting = memory.make<uint8_t>(3);
  std::fill(transitions, transitions + 3 * 256, INVALID);
  accepting[0] = accepting[1] = 0; accepting[2] = 1;
  transitions['a'] = transitions[256 + 'a'] = transitions[256 + 'b'] = 2;
  auto* offsets = memory.make<uint32_t>(size + 1);
  auto* bytes = memory.make<uint8_t>(size - 1);
  auto* eos = memory.make<uint8_t>(size);
  auto* eos_ids = memory.make<int>(1);
  for (uint32_t token = 0; token < size; ++token) {
    offsets[token] = token;
    eos[token] = token == size - 1;
    if (token < size - 1) bytes[token] = token < COMPACT_CAPACITY ? 'a' : 'b';
  }
  offsets[size] = size - 1;
  eos_ids[0] = size - 1;
  const DeviceVocabulary vocabulary{offsets, bytes, eos, size, eos_ids, 1};
  auto* plan = memory.make<DevicePlan>(1);
  *plan = compile_tokens(memory, BytePlan{transitions, accepting, 3}, vocabulary);
  CHECK(plan->counts[0] == 512 && plan->counts[1] == 513);
  CHECK(plan->sparse_ids[plan->offsets[0] + 511] == 511);
  CHECK(plan->sparse_next[plan->offsets[0] + 511] == 2);
  CHECK(plan->wide_next[size_t(plan->offsets[1]) * size + 512] == 2);
  CHECK(plan->wide_next[size_t(plan->offsets[1]) * size + 513] == INVALID);
  auto* input = memory.make<DeviceInput>(2);
  auto* rows = memory.make<DeviceRow>(2);
  auto* results = memory.make<RowResult>(2);
  input[0] = DeviceInput{plan, State{0, 0}, 0, 0};
  input[1] = DeviceInput{plan, State{1, 0}, 0, 0};
  auto* mask = memory.make<uint32_t>(words);
  auto* ids = memory.make<int>(COMPACT_CAPACITY);
  auto* count = memory.make<uint32_t>(1);
  auto* selected = memory.make<int>(1);
  cuda_check(launch_init(2, input, 0, nullptr, vocabulary, rows, results, nullptr));
  for (int row = 0; row < 2; ++row) {
    cuda_check(cudaMemsetAsync(mask, 0xff, words * sizeof(uint32_t), nullptr));
    cuda_check(launch_legal(row, rows, vocabulary, results, mask, ids, count, nullptr));
    cuda_check(cudaDeviceSynchronize());
    CHECK(results[row].error == 0 && *count == uint32_t(512 + row));
    CHECK(results[row].allowed == *count);
    for (uint32_t rank = 0; rank < COMPACT_CAPACITY; ++rank) {
      CHECK(ids[rank] == int(rank));
      if (!row) {
        CHECK(plan->sparse_ids[plan->offsets[0] + rank] == int(rank));
        CHECK(plan->sparse_next[plan->offsets[0] + rank] == 2);
      } else CHECK(plan->wide_ids[size_t(plan->offsets[1]) * COMPACT_CAPACITY + rank] == int(rank));
    }
    for (uint32_t word = 0; word < words; ++word) {
      CHECK(mask[word] == (word < 16 ? 0xffffffffu : uint32_t(row)));
      if (row) CHECK(mask[word] == plan->wide_mask[size_t(plan->offsets[1]) * words + word]);
    }
  }
  for (int row = 0; row < 2; ++row) {
    *selected = 511 + row;
    cuda_check(launch_advance(row, rows, vocabulary, selected, results, nullptr));
    cuda_check(cudaDeviceSynchronize());
    CHECK(results[row].error == 0 && results[row].after.id == 2);
    CHECK(results[row].complete == 1 && results[row].after.ended == 0);
  }
  cuda_check(launch_init(2, input, 0, nullptr, vocabulary, rows, results, nullptr));
  cuda_check(cudaDeviceSynchronize());
  for (int row = 0; row < 2; ++row) {
    *selected = 512 + row;
    cuda_check(launch_advance(row, rows, vocabulary, selected, results, nullptr));
    cuda_check(cudaDeviceSynchronize());
    CHECK(results[row].error == uint32_t(RuntimeError::RejectedToken));
    CHECK(results[row].after.id == uint32_t(row) && results[row].after.ended == 0);
  }
  return 0;
}

int main() {
  CHECK(compact_boundary_probe() == 0);
  constexpr uint32_t size = 777, eos_id = 600, states = 19;
  Allocations memory;
  auto* transitions = memory.make<uint32_t>(states * 256);
  auto* accepting = memory.make<uint8_t>(states);
  for (uint32_t i = 0; i < states * 256; ++i) transitions[i] = INVALID;
  for (uint32_t i = 0; i < states; ++i) accepting[i] = i == 2;
  transitions['a'] = 1;
  transitions[256 + 'b'] = 2;
  transitions[0xc3] = 3;
  transitions[3 * 256 + 0xa9] = 2;
  transitions[2 * 256 + ' '] = 2;
  transitions[17 * 256 + 'a'] = 18;
  transitions[18 * 256 + 'b'] = 2;
  const BytePlan byte_plan{transitions, accepting, states};
  auto* plan = memory.make<DevicePlan>(1);
  auto* offsets = memory.make<uint32_t>(size + 1);
  auto* bytes = memory.make<uint8_t>(size + 1);
  auto* eos = memory.make<uint8_t>(size);
  uint32_t used = 0;
  for (uint32_t token = 0; token < size; ++token) {
    offsets[token] = used;
    eos[token] = token == eos_id;
    if (token == 2 || token == eos_id) continue;
    if (token == 3) { bytes[used++] = 'a'; bytes[used++] = 'b'; }
    else bytes[used++] = token == 1 ? 'b' : token == 4 ? ' ' :
                         token == 5 ? 0xc3 : token == 6 ? 0xa9 : 'a';
  }
  offsets[size] = used;
  auto* eos_ids = memory.make<int>(1);
  eos_ids[0] = eos_id;
  const DeviceVocabulary vocabulary{offsets, bytes, eos, size, eos_ids, 1};
  *plan = compile_tokens(memory, byte_plan, vocabulary);
  CHECK(plan->counts[0] == 772 && plan->counts[1] == 1);
  CHECK(plan->counts[2] == 2 && plan->counts[3] == 1);
  CHECK(plan->counts[16] == 0 && plan->counts[17] == 771 && plan->counts[18] == 1);
  CHECK(plan->wide_ids[size_t(plan->offsets[17]) * COMPACT_CAPACITY] == 0);
  CHECK(plan->wide_next[size_t(plan->offsets[17]) * size + 3] == 2);
  CHECK(plan->wide_next[size_t(plan->offsets[17]) * size + 5] == INVALID);
  auto* input = memory.make<DeviceInput>(4);
  auto* rows = memory.make<DeviceRow>(4);
  auto* results = memory.make<RowResult>(4);
  auto* prefix = memory.make<int>(2);
  prefix[0] = 0; prefix[1] = 4;
  input[0] = DeviceInput{plan, State{0, 0}, 0, 0};
  input[1] = DeviceInput{plan, State{0, 0}, 0, 1};
  input[2] = DeviceInput{nullptr, State{0, 0}, 0, 0};
  input[3] = DeviceInput{plan, State{0, 0}, 0, 2};
  auto* mask = memory.make<uint32_t>((size + 31) / 32);
  auto* ids = memory.make<int>(COMPACT_CAPACITY);
  auto* count = memory.make<uint32_t>(1);
  auto* selected = memory.make<int>(1);
  cuda_check(launch_init(4, input, 2, prefix, vocabulary, rows, results, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(results[1].before.id == 1 && results[1].after.id == 1);
  CHECK(results[3].error == uint32_t(RuntimeError::InvalidPrefix));
  CHECK(results[3].after.id == 0);

  cuda_check(launch_legal(0, rows, vocabulary, results, mask, ids, count, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(*count == 772 && results[0].allowed == 772);
  CHECK(ids[0] == 0 && ids[1] == 3 && ids[2] == 5 && ids[3] == 7);
  for (uint32_t i = 1; i < COMPACT_CAPACITY; ++i) CHECK(ids[i] > ids[i - 1]);
  CHECK(!(mask[eos_id / 32] & (1u << (eos_id % 32))));
  CHECK(mask[(size - 1) / 32] & (1u << ((size - 1) % 32)));

  cuda_check(launch_legal(1, rows, vocabulary, results, mask, ids, count, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(*count == 1 && ids[0] == 1 && ids[1] == -1);
  *selected = 1;
  cuda_check(launch_advance(1, rows, vocabulary, selected, results, nullptr));
  cuda_check(launch_legal(1, rows, vocabulary, results, mask, ids, count, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(results[1].after.id == 2 && results[1].complete == 1);
  CHECK(*count == 1 && ids[0] == int(eos_id));
  CHECK(!(mask[4 / 32] & (1u << (4 % 32))));
  *selected = eos_id;
  cuda_check(launch_advance(1, rows, vocabulary, selected, results, nullptr));
  cuda_check(launch_legal(1, rows, vocabulary, results, mask, ids, count, nullptr));
  cuda_check(launch_advance(1, rows, vocabulary, selected, results, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(results[1].error == 0 && results[1].after.ended == 1 && *count == 1);
  *selected = 0;
  cuda_check(launch_advance(1, rows, vocabulary, selected, results, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(results[1].error == uint32_t(RuntimeError::RejectedToken));
  CHECK(results[1].after.ended == 1);

  cuda_check(launch_legal(2, rows, vocabulary, results, mask, ids, count, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(*count == size && ids[511] == 511 && mask[24] == 0x1ffu);
  *selected = 2;
  cuda_check(launch_advance(2, rows, vocabulary, selected, results, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(results[2].error == 0 && results[2].after.id == 0 && !results[2].complete);
  *selected = size;
  cuda_check(launch_advance(2, rows, vocabulary, selected, results, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(results[2].error == uint32_t(RuntimeError::InvalidToken));

  input[0].state = State{3, 0};
  cuda_check(launch_init(1, input, 2, prefix, vocabulary, rows, results, nullptr));
  cuda_check(launch_legal(0, rows, vocabulary, results, mask, ids, count, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(*count == 1 && ids[0] == 6);
  *selected = 6;
  cuda_check(launch_advance(0, rows, vocabulary, selected, results, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(results[0].after.id == 2 && results[0].complete);

  input[0].state = State{2, 0};
  input[0].prefix_offset = 1;
  input[0].prefix_length = 1;
  cuda_check(launch_init(1, input, 2, prefix, vocabulary, rows, results, nullptr));
  cuda_check(launch_legal(0, rows, vocabulary, results, mask, ids, count, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(results[0].error == 0 && results[0].after.id == 2 && !results[0].after.ended);
  CHECK(*count == 1 && ids[0] == int(eos_id));
  CHECK(!(mask[4 / 32] & (1u << (4 % 32))));
  input[0].prefix_offset = input[0].prefix_length = 0;

  input[0].state = State{1, 0};
  transitions[256 + 'b'] = INVALID;
  *plan = compile_tokens(memory, byte_plan, vocabulary);
  CHECK(plan->counts[1] == 0);
  cuda_check(launch_init(1, input, 2, prefix, vocabulary, rows, results, nullptr));
  cuda_check(launch_legal(0, rows, vocabulary, results, mask, ids, count, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(*count == 0 && ids[0] == -1);
  CHECK(results[0].error == uint32_t(RuntimeError::NoLegalTokens));
  for (uint32_t i = 0; i < (size + 31) / 32; ++i) CHECK(mask[i] == 0);
  input[0].state = State{99, 0};
  cuda_check(launch_init(1, input, 2, prefix, vocabulary, rows, results, nullptr));
  cuda_check(cudaDeviceSynchronize());
  CHECK(results[0].error == uint32_t(RuntimeError::InvalidState));
  std::puts("GPU grammar runtime contract passed");
}
