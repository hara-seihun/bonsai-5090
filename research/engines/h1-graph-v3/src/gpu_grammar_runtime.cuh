#pragma once
#include "gpu_grammar.h"
#include "gpu_grammar_cache.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace h5::gpu_grammar {
constexpr uint32_t COMPACT_CAPACITY = 512;
enum class RuntimeError : uint32_t {
  None = 0, InvalidPlan = 1, InvalidState = 2, InvalidVocabulary = 3,
  InvalidPrefix = 4, InvalidToken = 5, RejectedToken = 6, NoLegalTokens = 7,
  NonfiniteLogits = 8
};
struct DevicePlan {
  const uint8_t* accepting;
  uint32_t states;
  const uint32_t* counts;
  const uint32_t* offsets;
  const int* sparse_ids;
  const uint32_t* sparse_next;
  const uint32_t* wide_next;
  const uint32_t* wide_mask;
  const int* wide_ids;
};
struct DeviceVocabulary {
  const uint32_t* offsets;
  const uint8_t* bytes;
  const uint8_t* eos;
  uint32_t size;
  const int* eos_ids = nullptr;
  uint32_t eos_count = 0;
};
struct DeviceInput { const DevicePlan* plan; State state; uint32_t prefix_offset, prefix_length; };
struct DeviceRow { const DevicePlan* plan; };
namespace runtime_detail {
__device__ inline RuntimeError validate(const DevicePlan* plan, State state, DeviceVocabulary vocab) {
  if (!vocab.size || vocab.size > 0x7fffffffu || !vocab.eos || !vocab.eos_ids || !vocab.eos_count)
    return RuntimeError::InvalidVocabulary;
  if (!plan) return RuntimeError::None;
  if (!plan->states || !plan->accepting || !plan->counts || !plan->offsets) return RuntimeError::InvalidPlan;
  if (state.id >= plan->states || state.ended > 1 || (state.ended && !plan->accepting[state.id]))
    return RuntimeError::InvalidState;
  return RuntimeError::None;
}
__device__ inline RuntimeError consume(const DevicePlan* plan, State before, DeviceVocabulary vocab,
    int token, State* after) {
  *after = before;
  if (token < 0 || uint32_t(token) >= vocab.size) return RuntimeError::InvalidToken;
  if (!plan) return RuntimeError::None;
  if (vocab.eos[token]) {
    if (!plan->accepting[before.id]) return RuntimeError::RejectedToken;
    after->ended = 1;
    return RuntimeError::None;
  }
  if (before.ended) return RuntimeError::RejectedToken;
  const uint32_t count = plan->counts[before.id], offset = plan->offsets[before.id];
  uint32_t next = INVALID;
  if (count <= COMPACT_CAPACITY) {
    if (count && (!plan->sparse_ids || !plan->sparse_next)) return RuntimeError::InvalidPlan;
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (plan->sparse_ids[offset + mid] < token) lo = mid + 1;
      else hi = mid;
    }
    if (lo < count && plan->sparse_ids[offset + lo] == token) next = plan->sparse_next[offset + lo];
  } else {
    if (!plan->wide_next) return RuntimeError::InvalidPlan;
    next = plan->wide_next[size_t(offset) * vocab.size + token];
  }
  if (next == INVALID) return RuntimeError::RejectedToken;
  if (next >= plan->states) return RuntimeError::InvalidState;
  after->id = next;
  return RuntimeError::None;
}
__device__ inline uint32_t complete(const DevicePlan* plan, State state) {
  return plan && plan->accepting && state.id < plan->states ? uint32_t(plan->accepting[state.id] != 0) : 0;
}
static __global__ void initialize_kernel(int count, const DeviceInput* input, int prefix_count,
    const int* prefix, DeviceVocabulary vocab, DeviceRow* rows, RowResult* results) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const DeviceInput source = input[row];
  rows[row].plan = source.plan;
  State state = source.state;
  RuntimeError error = validate(source.plan, state, vocab);
  if (error == RuntimeError::None && (source.prefix_offset > uint32_t(prefix_count) ||
      source.prefix_length > uint32_t(prefix_count) - source.prefix_offset || (source.prefix_length && !prefix)))
    error = RuntimeError::InvalidPrefix;
  for (uint32_t p = 0; error == RuntimeError::None && p < source.prefix_length; ++p) {
    State next;
    error = consume(source.plan, state, vocab, prefix[source.prefix_offset + p], &next);
    if (error == RuntimeError::RejectedToken || error == RuntimeError::InvalidToken) error = RuntimeError::InvalidPrefix;
    if (error == RuntimeError::None) state = next;
  }
  if (error != RuntimeError::None) state = source.state;
  results[row] = RowResult{state, state, 0, error == RuntimeError::None ? complete(source.plan, state) : 0, uint32_t(error)};
}
static __global__ void legal_cached_kernel(int row, const DeviceRow* rows, DeviceVocabulary vocab,
    RowResult* results, uint32_t* mask, int* ids, uint32_t* count) {
  __shared__ uint32_t error, total, offset;
  __shared__ bool eos_only, wide;
  const DevicePlan* plan = rows[row].plan;
  const State state = results[row].after;
  const size_t words = (size_t(vocab.size) + 31) / 32;
  if (!threadIdx.x) {
    error = results[row].error;
    if (!error) error = uint32_t(validate(plan, state, vocab));
    eos_only = !error && plan && plan->accepting[state.id];
    total = error ? 0 : !plan ? vocab.size : eos_only ? vocab.eos_count : plan->counts[state.id];
    offset = !error && plan ? plan->offsets[state.id] : 0;
    wide = plan && !eos_only && total > COMPACT_CAPACITY;
    if (!error && !total) error = uint32_t(RuntimeError::NoLegalTokens);
    if (!error && plan && !eos_only && (wide ? (!plan->wide_mask || !plan->wide_ids) : !plan->sparse_ids))
      error = uint32_t(RuntimeError::InvalidPlan);
  }
  __syncthreads();
  for (size_t w = threadIdx.x; w < words; w += blockDim.x) {
    uint32_t bits = 0;
    if (!error && !plan) {
      bits = 0xffffffffu;
      if (w + 1 == words && (vocab.size & 31)) bits = (1u << (vocab.size & 31)) - 1;
    } else if (!error && wide) bits = plan->wide_mask[size_t(offset) * words + w];
    mask[w] = bits;
  }
  for (uint32_t i = threadIdx.x; i < COMPACT_CAPACITY; i += blockDim.x) {
    int id = -1;
    if (!error && i < total) id = !plan ? int(i) : eos_only ? vocab.eos_ids[i] : wide ?
        plan->wide_ids[size_t(offset) * COMPACT_CAPACITY + i] : plan->sparse_ids[offset + i];
    ids[i] = id;
  }
  __syncthreads();
  if (!error && plan && !wide) {
    const int* legal = eos_only ? vocab.eos_ids : plan->sparse_ids + offset;
    for (uint32_t i = threadIdx.x; i < total; i += blockDim.x) {
      const uint32_t id = uint32_t(legal[i]);
      atomicOr(mask + id / 32, 1u << (id & 31));
    }
  }
  if (!threadIdx.x) {
    *count = results[row].allowed = error ? 0 : total;
    results[row].before = state;
    results[row].complete = error ? 0 : complete(plan, state);
    results[row].error = error;
  }
}
static __global__ void advance_kernel(int row, const DeviceRow* rows, DeviceVocabulary vocab,
    const int* selected, RowResult* results) {
  RowResult& result = results[row];
  result.before = result.after;
  if (result.error) return;
  const DevicePlan* plan = rows[row].plan;
  RuntimeError error = validate(plan, result.before, vocab);
  State after = result.before;
  if (error == RuntimeError::None) error = consume(plan, result.before, vocab, *selected, &after);
  result.error = uint32_t(error);
  if (error == RuntimeError::None) result.after = after;
  result.complete = (error == RuntimeError::None || error == RuntimeError::RejectedToken || error == RuntimeError::InvalidToken)
      ? complete(plan, result.after) : 0;
}
}
inline cudaError_t launch_init(int rows, const DeviceInput* input, int prefix_count,
    const int* prefix, DeviceVocabulary vocab, DeviceRow* device_rows, RowResult* results, cudaStream_t stream) {
  if (rows < 0 || prefix_count < 0 || (rows && (!input || !device_rows || !results))) return cudaErrorInvalidValue;
  if (!rows) return cudaSuccess;
  runtime_detail::initialize_kernel<<<(uint32_t(rows) + 127u) / 128u, 128, 0, stream>>>(
      rows, input, prefix_count, prefix, vocab, device_rows, results);
  return cudaGetLastError();
}
inline cudaError_t launch_initialize(int rows, const DeviceInput* input, int prefix_count,
    const int* prefix, DeviceVocabulary vocab, DeviceRow* device_rows, RowResult* results, cudaStream_t stream) {
  return launch_init(rows, input, prefix_count, prefix, vocab, device_rows, results, stream);
}
inline cudaError_t launch_legal(int row, const DeviceRow* rows, DeviceVocabulary vocab,
    RowResult* results, uint32_t* mask, int* ids, uint32_t* count, cudaStream_t stream) {
  if (row < 0 || !rows || !results || !mask || !ids || !count || vocab.size > 0x7fffffffu) return cudaErrorInvalidValue;
  runtime_detail::legal_cached_kernel<<<1, 256, 0, stream>>>(row, rows, vocab, results, mask, ids, count);
  return cudaGetLastError();
}
inline cudaError_t launch_advance(int row, const DeviceRow* rows, DeviceVocabulary vocab,
    const int* selected, RowResult* results, cudaStream_t stream) {
  if (row < 0 || !rows || !results || !selected || vocab.size > 0x7fffffffu) return cudaErrorInvalidValue;
  runtime_detail::advance_kernel<<<1, 1, 0, stream>>>(row, rows, vocab, selected, results);
  return cudaGetLastError();
}
}
