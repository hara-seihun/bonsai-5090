#pragma once
#include "gpu_grammar_runtime.cuh"

namespace h5::gpu_grammar {

struct BytePlan {
  const uint32_t* transitions;
  const uint8_t* accepting;
  uint32_t states;
};
namespace token_compile_detail {
__device__ inline uint32_t token_next(BytePlan source, DeviceVocabulary vocabulary,
                                      uint32_t state, uint32_t token) {
  if (vocabulary.eos[token]) return source.accepting[state] ? state : INVALID;
  const uint32_t begin = vocabulary.offsets[token], end = vocabulary.offsets[token + 1];
  if (begin == end) return INVALID;
  for (uint32_t byte = begin; byte < end; ++byte) {
    state = source.transitions[size_t(state) * 256 + vocabulary.bytes[byte]];
    if (state == INVALID) return INVALID;
  }
  return state;
}

static __global__ void counts_kernel(BytePlan source, DeviceVocabulary vocabulary,
                                      uint32_t* counts) {
  __shared__ uint32_t warp_counts[8];
  const uint32_t token = blockIdx.x * 256u + threadIdx.x;
  const bool legal = token < vocabulary.size &&
      token_next(source, vocabulary, blockIdx.y, token) != INVALID;
  const uint32_t bits = __ballot_sync(0xffffffffu, legal);
  if (!(threadIdx.x & 31)) warp_counts[threadIdx.x >> 5] = __popc(bits);
  __syncthreads();
  if (!threadIdx.x) {
    uint32_t total = 0;
    for (uint32_t warp = 0; warp < 8; ++warp) total += warp_counts[warp];
    if (total) atomicAdd(counts + blockIdx.y, total);
  }
}

static __global__ void dense_kernel(BytePlan source, DeviceVocabulary vocabulary,
    uint32_t first_state, uint32_t* dense_next) {
  const uint32_t token = blockIdx.x * 256u + threadIdx.x;
  if (token < vocabulary.size)
    dense_next[size_t(blockIdx.y) * vocabulary.size + token] =
        token_next(source, vocabulary, first_state + blockIdx.y, token);
}

static __global__ void pack_kernel(DeviceVocabulary vocabulary, uint32_t first_state,
    const uint32_t* counts, const uint32_t* offsets, const uint32_t* dense_next,
    int* sparse_ids, uint32_t* sparse_next, uint32_t* wide_next,
    uint32_t* wide_mask, int* wide_ids) {
  __shared__ uint32_t warp_counts[8];
  const uint32_t local_state = blockIdx.x, state = first_state + local_state;
  const uint32_t destination = offsets[state];
  const bool wide = counts[state] > COMPACT_CAPACITY;
  const uint32_t* dense = dense_next + size_t(local_state) * vocabulary.size;
  if (wide) {
    const size_t words = (size_t(vocabulary.size) + 31) / 32;
    for (size_t base = 0; base < vocabulary.size; base += 256) {
      const size_t token = base + threadIdx.x;
      const uint32_t next = token < vocabulary.size ? dense[token] : INVALID;
      if (token < vocabulary.size)
        wide_next[size_t(destination) * vocabulary.size + token] = next;
      const uint32_t bits = __ballot_sync(0xffffffffu, next != INVALID);
      if (!(threadIdx.x & 31) && token / 32 < words)
        wide_mask[size_t(destination) * words + token / 32] = bits;
    }
    for (uint32_t i = threadIdx.x; i < COMPACT_CAPACITY; i += blockDim.x)
      wide_ids[size_t(destination) * COMPACT_CAPACITY + i] = -1;
  }

  // Each thread owns a contiguous token slice, so prefix ranks are sorted IDs.
  const size_t slice = (size_t(vocabulary.size) + 255) / 256;
  const size_t begin = size_t(threadIdx.x) * slice;
  const size_t candidate_end = begin + slice;
  const size_t end = candidate_end < vocabulary.size ? candidate_end : vocabulary.size;
  uint32_t local_count = 0;
  for (size_t token = begin; token < end; ++token) local_count += dense[token] != INVALID;
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t inclusive = local_count;
  for (uint32_t distance = 1; distance < 32; distance <<= 1) {
    const uint32_t preceding = __shfl_up_sync(0xffffffffu, inclusive, distance);
    if (lane >= distance) inclusive += preceding;
  }
  if (lane == 31) warp_counts[warp] = inclusive;
  __syncthreads();
  uint32_t rank = inclusive - local_count;
  for (uint32_t w = 0; w < warp; ++w) rank += warp_counts[w];
  for (size_t token = begin; token < end; ++token) {
    const uint32_t next = dense[token];
    if (next == INVALID) continue;
    if (wide) {
      if (rank < COMPACT_CAPACITY)
        wide_ids[size_t(destination) * COMPACT_CAPACITY + rank] = int(token);
    } else {
      sparse_ids[size_t(destination) + rank] = int(token);
      sparse_next[size_t(destination) + rank] = next;
    }
    ++rank;
  }
}

inline bool valid_source(BytePlan source, DeviceVocabulary vocabulary) {
  return source.states && source.transitions && source.accepting && vocabulary.size &&
      vocabulary.size <= 0x7fffffffu && vocabulary.offsets && vocabulary.eos;
}
} // namespace token_compile_detail

// Boundary-only: validated byte DFA/vocabulary, caller-zeroed counts, no sync/allocation.
inline cudaError_t launch_token_counts(BytePlan source, DeviceVocabulary vocabulary,
                                       uint32_t* counts, cudaStream_t stream) {
  if (!token_compile_detail::valid_source(source, vocabulary) ||
      source.states > 65535u || !counts)
    return cudaErrorInvalidValue;
  const dim3 grid((vocabulary.size + 255u) / 256u, source.states);
  token_compile_detail::counts_kernel<<<grid, 256, 0, stream>>>(source, vocabulary, counts);
  return cudaGetLastError();
}

// offsets is sparse entry offset when count<=512, otherwise wide row index.
// dense_next holds state_count*V entries and may be reused on the same stream.
inline cudaError_t launch_token_chunk(BytePlan source, DeviceVocabulary vocabulary,
    uint32_t first_state, uint32_t state_count, const uint32_t* counts,
    const uint32_t* offsets, uint32_t* dense_next, int* sparse_ids,
    uint32_t* sparse_next, uint32_t* wide_next, uint32_t* wide_mask,
    int* wide_ids, cudaStream_t stream) {
  if (!token_compile_detail::valid_source(source, vocabulary) || !counts || !offsets ||
      !dense_next || !state_count || state_count > TOKEN_COMPILE_CHUNK ||
      first_state >= source.states || state_count > source.states - first_state)
    return cudaErrorInvalidValue;
  const dim3 grid((vocabulary.size + 255u) / 256u, state_count);
  token_compile_detail::dense_kernel<<<grid, 256, 0, stream>>>(
      source, vocabulary, first_state, dense_next);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  token_compile_detail::pack_kernel<<<state_count, 256, 0, stream>>>(
      vocabulary, first_state, counts, offsets, dense_next,
      sparse_ids, sparse_next, wide_next, wide_mask, wide_ids);
  return cudaGetLastError();
}
} // namespace h5::gpu_grammar
