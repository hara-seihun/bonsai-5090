#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>
#include <cstdint>
#include <climits>

namespace h5 {

// These are prepared images, not GGUF rows. INT8 and FP4 WF0 have different
// fragment wiring and different chunk order, despite both using 1024 bytes.
enum class SparseHeadFormat : int { Int8Trits = 0, Fp4Trits = 1, Nvfp4 = 2, Dense5 = 3 };
enum class SparseHeadError : int {
  None, UnsupportedFormat, InvalidShape, EmptyRows, InvalidCount,
  MissingBuffer, InsufficientCapacity, AliasedBuffer, InvalidAdjustments, Cuda
};
struct SparseHeadResult {
  SparseHeadError error;
  cudaError_t cuda_error;
  explicit operator bool() const { return error == SparseHeadError::None; }
};
struct SparseHeadLayout {
  // row_alignment=128 for INT8; FP4 permits 32 or 128 (caller chooses).
  int rows, groups, kblocks;
  size_t weight_bytes, scale_bytes, row_scale_bytes, status_bytes;
};

inline SparseHeadResult sparse_head_layout(SparseHeadFormat format, int count,
                                            int K, int row_alignment, SparseHeadLayout* out) {
  if (!out) return {SparseHeadError::MissingBuffer, cudaSuccess};
  int cw, cs;
  switch (format) {
    case SparseHeadFormat::Int8Trits: cw = 1024; cs = 128; break;
    case SparseHeadFormat::Fp4Trits: cw = 1024; cs = 64; break;
    case SparseHeadFormat::Nvfp4: cw = 2048; cs = 256; break;
    case SparseHeadFormat::Dense5: cw = 832; cs = 64; break;
    default: return {SparseHeadError::UnsupportedFormat, cudaSuccess};
  }
  if ((row_alignment != 32 && row_alignment != 128) ||
      (format == SparseHeadFormat::Int8Trits && row_alignment != 128))
    return {SparseHeadError::InvalidShape, cudaSuccess};
  const int alignment = row_alignment;
  if (K <= 0 || K % 128 || K / 128 > 65535)
    return {SparseHeadError::InvalidShape, cudaSuccess};
  if (count == 0) return {SparseHeadError::EmptyRows, cudaSuccess};
  if (count < 0 || count > INT_MAX - alignment + 1)
    return {SparseHeadError::InvalidCount, cudaSuccess};
  const int rows = ((count + alignment - 1) / alignment) * alignment;
  const size_t chunks = size_t(rows / 32) * (K / 128);
  *out = {rows, rows / 32, K / 128, chunks * cw, chunks * cs,
          size_t(rows) * sizeof(float), size_t(rows / 32) * sizeof(unsigned)};
  return {SparseHeadError::None, cudaSuccess};
}

struct SparseHeadSource {
  const void* weights;
  const void* scales;
  const float* row_scales; // null for ordinary WF0; carrier means may use this
  int rows, K;
  SparseHeadFormat format;
};
struct SparseHeadDestination {
  void* weights;
  void* scales;
  float* row_scales;
  // One status per padded 32-row group. Bit 0 = out-of-range ID. Every
  // invocation overwrites all statuses; no memset or atomic reduction needed.
  unsigned* status;
  size_t weight_bytes, scale_bytes, row_scale_bytes, status_bytes;
};

namespace sparse_head_detail {
__host__ __device__ inline size_t fp_word(int lane, int word) {
  return size_t(word >> 2) * 128 + lane * 4 + (word & 3);
}
__host__ __device__ inline int half_scale_index(int row) {
  return (row & 7) * 4 + (row >> 4) * 2 + ((row >> 3) & 1);
}
__host__ __device__ inline size_t dense5_byte_index(int lane, int byte) {
  return byte < 24 ? size_t(byte / 4) * 128 + lane * 4 + byte % 4
                   : 768 + lane * 2 + byte - 24;
}
// Dense5 labels straddle output-row segments, so copy their base-3 digits,
// not whole labels. The neutral zero trit is digit 1, including tail filler.
__host__ __device__ inline unsigned dense5_digit(unsigned label, int digit) {
  switch (digit) {
    case 0: return label % 3;
    case 1: return (label / 3) % 3;
    case 2: return (label / 9) % 3;
    case 3: return (label / 27) % 3;
    case 4: return (label / 81) % 3;
    default: return 1;
  }
}
__host__ __device__ inline uint8_t gather_dense5_byte(
    const uint8_t* src, const int* ids, int source_groups, int kb, int lane, int byte) {
  unsigned label = 0, power = 1;
#pragma unroll
  for (int digit = 0; digit < 5; ++digit, power *= 3) {
    const int index = byte * 5 + digit;
    unsigned value = 1;
    if (index < 128) {
      const int word = index / 16;
      const int row = 16 * (word / 4) + 8 * (word & 1) + lane / 4;
      const int id = ids[row];
      if (id >= 0) {
        const int sr = id & 31, source_lane = (sr & 7) * 4 + (lane & 3);
        const int source_word = (sr / 16) * 4 + (word & 2) + ((sr / 8) & 1);
        const int source_index = source_word * 16 + index % 16;
        unsigned packed = src[(size_t(kb) * source_groups + id / 32) * 832 +
                              dense5_byte_index(source_lane, source_index / 5)];
        value = dense5_digit(packed, source_index % 5);
      }
    }
    label += value * power;
  }
  return uint8_t(label);
}
// Copy whole FP4/NVFP4 operand words. Only INT8 packs fields from two
// independent selected rows into one word; each field spans four K bytes.
__host__ __device__ inline uint32_t gather_word(
    SparseHeadFormat format, const uint32_t* src, const int* ids,
    int source_groups, int kblocks, int kb, int index) {
  if (format == SparseHeadFormat::Dense5) {
    const int lane = index < 192 ? (index % 32) : (index - 192) / 2;
    const int byte = index < 192 ? (index / 32) * 4 : 24;
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
      const int l = index < 192 ? lane : (index - 192) * 2 + i / 2;
      const int b = index < 192 ? byte + i : 24 + i % 2;
      result |= uint32_t(gather_dense5_byte(reinterpret_cast<const uint8_t*>(src),
                        ids, source_groups, kb, l, b)) << (8 * i);
    }
    return result;
  }
  if (format == SparseHeadFormat::Int8Trits) {
    const int lane = index / 8, word = index & 7;
    const int gid = lane >> 2, tq = lane & 3;
    uint32_t value = 0;
    for (int pair = 0; pair < 2; ++pair) {
      const int row = 8 * (2 * (word & 1) + pair) + gid;
      const int id = ids[row];
      if (id < 0) continue;
      const int sr = id & 31;
      const size_t chunk = size_t(id / 32) * kblocks + kb;
      const int source_lane = (sr & 7) * 4 + tq;
      const int source_word = (word >> 1) * 2 + (sr >> 4);
      const uint32_t v = src[chunk * 256 + source_lane * 8 + source_word];
      value |= ((v >> (4 * ((sr >> 3) & 1))) & 0x0F0F0F0Fu) << (4 * pair);
    }
    return value;
  }
  const int plane = index / 128, lane = (index % 128) / 4;
  const int word = plane * 4 + (index & 3), tq = lane & 3, gid = lane >> 2;
  const bool nv = format == SparseHeadFormat::Nvfp4;
  const int row = nv ? 16 * (word >> 3) + 8 * (word & 1) + gid
                     : 16 * (word >> 2) + 8 * (word & 1) + gid;
  const int id = ids[row];
  if (id < 0) return 0;
  const int sr = id & 31;
  const int source_word = nv ? (sr >> 4) * 8 + (word & 6) + ((sr >> 3) & 1)
                             : (sr >> 4) * 4 + (word & 2) + ((sr >> 3) & 1);
  const size_t chunk = size_t(kb) * source_groups + id / 32;
  return src[chunk * (nv ? 512 : 256) + fp_word((sr & 7) * 4 + tq, source_word)];
}

static __global__ void gather(SparseHeadSource src, SparseHeadDestination dst,
                             const int* row_ids, int count, SparseHeadLayout layout,
                             const uint32_t* device_count = nullptr) {
  if (device_count) {
    const uint32_t n = *device_count;
    if (n < 2 || n > uint32_t(count)) return;
    count = int(n);
  }
  const int g = blockIdx.x, kb = blockIdx.y, t = threadIdx.x;
  __shared__ int ids[32];
  __shared__ unsigned invalid;
  if (t == 0) invalid = 0;
  __syncthreads();
  if (t < 32) {
    const int row = g * 32 + t;
    const int id = row < count ? row_ids[row] : -1;
    const bool bad = row < count && (id < 0 || id >= src.rows);
    ids[t] = bad ? -1 : id;
    if (bad) atomicOr(&invalid, 1u);
  }
  __syncthreads();
  if (t == 0 && kb == 0) dst.status[g] = invalid;
  const bool nv = src.format == SparseHeadFormat::Nvfp4;
  const bool int8 = src.format == SparseHeadFormat::Int8Trits;
  const int words = nv ? 512 : src.format == SparseHeadFormat::Dense5 ? 208 : 256;
  const size_t chunk = int8 ? size_t(g) * layout.kblocks + kb
                           : size_t(kb) * layout.groups + g;
  auto* w = static_cast<uint32_t*>(dst.weights) + chunk * words;
  if (src.format == SparseHeadFormat::Dense5) {
    auto* bytes = reinterpret_cast<uint8_t*>(w);
    for (int i = t; i < 832; i += blockDim.x) {
      const int lane = i < 768 ? (i % 128) / 4 : (i - 768) / 2;
      const int byte = i < 768 ? (i / 128) * 4 + i % 4 : 24 + i % 2;
      bytes[i] = gather_dense5_byte(static_cast<const uint8_t*>(src.weights),
                                   ids, src.rows / 32, kb, lane, byte);
    }
  } else {
    for (int i = t; i < words; i += blockDim.x)
      w[i] = gather_word(src.format, static_cast<const uint32_t*>(src.weights),
                         ids, src.rows / 32, layout.kblocks, kb, i);
  }
  if (int8 && t < 32) {
    const int id = ids[t];
    static_cast<float*>(dst.scales)[size_t(kb) * layout.rows + g * 32 + t] =
        id < 0 ? 0.f : static_cast<const float*>(src.scales)[size_t(kb) * src.rows + id];
  } else if (!int8 && !nv && t < 32) {
    const int id = ids[t];
    auto* s = static_cast<uint16_t*>(dst.scales);
    const auto* ss = static_cast<const uint16_t*>(src.scales);
    s[chunk * 32 + half_scale_index(t)] = id < 0 ? 0 :
        ss[(size_t(kb) * (src.rows / 32) + id / 32) * 32 + half_scale_index(id & 31)];
  } else if (nv) {
    const int row = t / 8, b = t & 7, id = ids[row];
    static_cast<uint8_t*>(dst.scales)[chunk * 256 + t] = id < 0 ? 0 :
        static_cast<const uint8_t*>(src.scales)[
          (size_t(kb) * (src.rows / 32) + id / 32) * 256 + (id & 31) * 8 + b];
  }
  if (kb == 0 && t < 32 && src.row_scales)
    dst.row_scales[g * 32 + t] = ids[t] < 0 ? 0.f : src.row_scales[ids[t]];
}
} // namespace sparse_head_detail

// IDs are device int[count], in compact-logit order; duplicates are preserved.
// No allocations, synchronization, activation preparation or arithmetic changes.
// Recurring cost: one CTA per (32 output rows,128 K), one launch. Logical
// source weight reads are count*K/2 bytes INT8 (two row-pairs per packed word),
// count*K/4 WF0, count*K/2 WF1; destination writes are layout.weight_bytes.
// Dense5 recombines five base3 digits per output label, with up to 5 byte reads
// per label (source cache-line reuse amortizes these); output is 26B/128 trits.
// Add selected scale reads, layout.scale_bytes writes, and optional count*4
// row-scale reads / layout.rows*4 writes. Each CTA loads its 32 IDs; physical
// traffic depends on scattered-row cache-line reuse. Consumer GEMM reads the
// prepared output again. Cache the prepared matrix when the legal list repeats.
// Source/destination allocations must not overlap. Use on the consuming stream;
// this is intentionally an ordinary launch, NOT PDL: existing GEMMs prefetch
// weights before their griddepcontrol.wait, so gather must finish first.
inline SparseHeadResult launch_sparse_head_gather(
    const SparseHeadSource& src, const SparseHeadDestination& dst,
    const int* row_ids, int count, int row_alignment, cudaStream_t stream) {
  SparseHeadLayout layout;
  auto result = sparse_head_layout(src.format, count, src.K, row_alignment, &layout);
  if (!result) return result;
  if (src.rows <= 0 || src.rows % 32)
    return {SparseHeadError::InvalidShape, cudaSuccess};
  if (!src.weights || !src.scales || !row_ids || !dst.weights || !dst.scales || !dst.status ||
      (src.format == SparseHeadFormat::Nvfp4 && !src.row_scales) ||
      (src.row_scales && !dst.row_scales))
    return {SparseHeadError::MissingBuffer, cudaSuccess};
  if (dst.weight_bytes < layout.weight_bytes || dst.scale_bytes < layout.scale_bytes ||
      dst.status_bytes < layout.status_bytes ||
      (src.row_scales && dst.row_scale_bytes < layout.row_scale_bytes))
    return {SparseHeadError::InsufficientCapacity, cudaSuccess};
  if (src.weights == dst.weights || src.scales == dst.scales ||
      (src.row_scales && src.row_scales == dst.row_scales))
    return {SparseHeadError::AliasedBuffer, cudaSuccess};
  sparse_head_detail::gather<<<dim3(layout.groups, layout.kblocks), 256, 0, stream>>>(
      src, dst, row_ids, count, layout);
  const auto err = cudaGetLastError();
  return err == cudaSuccess ? SparseHeadResult{SparseHeadError::None, cudaSuccess}
                           : SparseHeadResult{SparseHeadError::Cuda, err};
}

inline SparseHeadResult launch_sparse_head_gather_device_count(
    const SparseHeadSource& src, const SparseHeadDestination& dst,
    const int* row_ids, const uint32_t* count, int capacity, cudaStream_t stream) {
  SparseHeadLayout layout{};
  auto result = sparse_head_layout(src.format, capacity, src.K, 128, &layout);
  if (!result) return result;
  if (!count || !row_ids || !src.weights || !src.scales || !dst.weights || !dst.scales || !dst.status ||
      src.rows <= 0 || src.rows % 32 || (src.row_scales && !dst.row_scales))
    return {SparseHeadError::MissingBuffer, cudaSuccess};
  if (dst.weight_bytes < layout.weight_bytes || dst.scale_bytes < layout.scale_bytes ||
      dst.status_bytes < layout.status_bytes ||
      (src.row_scales && dst.row_scale_bytes < layout.row_scale_bytes))
    return {SparseHeadError::InsufficientCapacity, cudaSuccess};
  sparse_head_detail::gather<<<dim3(layout.groups, layout.kblocks), 256, 0, stream>>>(
      src, dst, row_ids, capacity, layout, count);
  const auto error = cudaGetLastError();
  return {error == cudaSuccess ? SparseHeadError::None : SparseHeadError::Cuda, error};
}

namespace sparse_head_detail {
__device__ __forceinline__ bool better(float value, int id, float other, int other_id) {
  return value > other || (value == other && id < other_id);
}

template<int TOP> static __global__ void select(
    const float* logits, int ld, const int* ids, int count,
    float* topv, int* topi, const int2* adj, int adj_k, int row0, const int* rowmap) {
  const int t = threadIdx.x, output_row = blockIdx.x;
  const int compact_row = row0 + output_row;
  const int original_row = rowmap ? rowmap[compact_row] : compact_row;
  float values[TOP]; int tokens[TOP];
#pragma unroll
  for (int i = 0; i < TOP; ++i) { values[i] = -CUDART_INF_F; tokens[i] = INT_MAX; }
  for (int i = t; i < count; i += blockDim.x) {
    const int id = ids[i];
    float value = logits[size_t(output_row) * ld + i];
    for (int j = 0; j < adj_k; ++j) {
      const int2 delta = adj[size_t(original_row) * adj_k + j];
      if (delta.x == id) value = __fadd_rn(value, __int_as_float(delta.y));
    }
    if constexpr (TOP > 1) { if (!isfinite(value)) continue; }
    bool duplicate = false;
#pragma unroll
    for (int j = 0; j < TOP; ++j) duplicate |= tokens[j] == id;
    if (duplicate) continue;
#pragma unroll
    for (int j = 0; j < TOP; ++j) if (better(value, id, values[j], tokens[j])) {
#pragma unroll
      for (int k = TOP - 1; k > j; --k) { values[k] = values[k - 1]; tokens[k] = tokens[k - 1]; }
      values[j] = value; tokens[j] = id; break;
    }
  }
  __shared__ float scores[8];
  __shared__ int indices[8], selected;
  for (int k = 0; k < TOP; ++k) {
    float value = values[0]; int id = tokens[0];
    for (int offset = 16; offset; offset >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu, value, offset);
      const int other_id = __shfl_down_sync(0xffffffffu, id, offset);
      if ((t & 31) + offset < 32 && better(other, other_id, value, id)) { value = other; id = other_id; }
    }
    if ((t & 31) == 0) { scores[t / 32] = value; indices[t / 32] = id; }
    __syncthreads();
    if (t == 0) {
      value = -CUDART_INF_F; id = INT_MAX;
      for (int w = 0; w < 8; ++w) if (better(scores[w], indices[w], value, id)) {
        value = scores[w]; id = indices[w];
      }
      selected = id;
      topi[size_t(output_row) * TOP + k] = id == INT_MAX ? ids[0] : id;
      if (topv) topv[size_t(output_row) * TOP + k] = value;
    }
    __syncthreads();
    if (tokens[0] == selected) {
#pragma unroll
      for (int j = 0; j + 1 < TOP; ++j) { values[j] = values[j + 1]; tokens[j] = tokens[j + 1]; }
      values[TOP - 1] = -CUDART_INF_F; tokens[TOP - 1] = INT_MAX;
    }
    __syncthreads();
  }
}

inline SparseHeadResult selection_args(const float* logits, int ld, const int* ids,
                                       int count, const int* out, int rows) {
  if (count == 0) return {SparseHeadError::EmptyRows, cudaSuccess};
  if (count < 0) return {SparseHeadError::InvalidCount, cudaSuccess};
  if (ld < count || rows <= 0) return {SparseHeadError::InvalidShape, cudaSuccess};
  if (!logits || !ids || !out) return {SparseHeadError::MissingBuffer, cudaSuccess};
  return {SparseHeadError::None, cudaSuccess};
}
inline SparseHeadResult launch_result() {
  const auto e = cudaGetLastError();
  return e == cudaSuccess ? SparseHeadResult{SparseHeadError::None, cudaSuccess}
                         : SparseHeadResult{SparseHeadError::Cuda, e};
}
} // namespace sparse_head_detail

// Input IDs must be valid original vocabulary IDs. Padding logits are ignored.
// row0/rowmap select the original adjustment row as in launch_logit_adj; unused
// adjustment entries have token=-1. Adjustment tokens within each row are unique.
inline SparseHeadResult launch_sparse_argmax(
    const float* logits, int ld, const int* ids, int count, int* out, int rows,
    const int2* adj, int adj_k, int row0, const int* rowmap, cudaStream_t stream) {
  auto r = sparse_head_detail::selection_args(logits, ld, ids, count, out, rows);
  if (!r) return r;
  if (row0 < 0 || row0 > INT_MAX - rows || adj_k < 0 || (adj_k > 0) != (adj != nullptr))
    return {SparseHeadError::InvalidAdjustments, cudaSuccess};
  sparse_head_detail::select<1><<<rows, 256, 0, stream>>>(
      logits, ld, ids, count, nullptr, out, adj, adj_k, row0, rowmap);
  return sparse_head_detail::launch_result();
}

// Exactly 16 outputs per row, value descending / original ID ascending. As in
// grammar top-k, non-finite values are excluded; exhausted entries are -inf and
// ids[0] (a legal token), never token zero by accident. No scratch allocation.
inline SparseHeadResult launch_sparse_topk(
    const float* logits, int ld, const int* ids, int count,
    float* topv, int* topi, int rows, cudaStream_t stream) {
  auto r = sparse_head_detail::selection_args(logits, ld, ids, count, topi, rows);
  if (!r) return r;
  if (!topv) return {SparseHeadError::MissingBuffer, cudaSuccess};
  sparse_head_detail::select<16><<<rows, 256, 0, stream>>>(
      logits, ld, ids, count, topv, topi, nullptr, 0, 0, nullptr);
  return sparse_head_detail::launch_result();
}

} // namespace h5
