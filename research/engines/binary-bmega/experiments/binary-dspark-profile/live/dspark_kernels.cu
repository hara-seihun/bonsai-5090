#include "dspark_kernels.cuh"
#include <cmath>
#include <climits>
#include <limits>

namespace h5::dspark {
namespace {

bool positive_matrix(int rows, int width) {
    return rows > 0 && width > 0 &&
           size_t(rows) <= std::numeric_limits<size_t>::max() / sizeof(float) / size_t(width);
}
bool finite_positive(float x) { return std::isfinite(x) && x > 0.0f; }
bool valid_geometry(KvGeometry g) {
    if (g.batch_slots <= 0 || g.layers <= 0 || g.ring <= 0 || g.q_heads <= 0 ||
        g.kv_heads <= 0 || g.head_dim != 128 || g.q_heads % g.kv_heads != 0 ||
        g.q_heads > 65535 || g.kv_heads > 65535) return false;
    size_t n = sizeof(__half);
    const int dims[] = {g.batch_slots, g.layers, g.ring, g.kv_heads, g.head_dim};
    for (int d : dims) {
        if (n > std::numeric_limits<size_t>::max() / size_t(d)) return false;
        n *= size_t(d);
    }
    return true;
}
__device__ void fail(int* status, DeviceError error) {
    atomicCAS(status, 0, static_cast<int>(error));
}
template<int Threads>
__device__ float block_sum(float v) {
    __shared__ float warps[Threads / 32];
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < Threads / 32; ++w) total += warps[w];
    __syncthreads();
    return total;
}
__device__ size_t tag_index(KvGeometry g, int slot, int layer, int cache) {
    return (size_t(slot) * g.layers + layer) * g.ring + cache;
}
__device__ size_t kv_index(KvGeometry g, int slot, int layer, int cache, int head, int d) {
    return (tag_index(g, slot, layer, cache) * g.kv_heads + head) * g.head_dim + d;
}

__global__ void rms_kernel(const float* x, const float* w, float* y,
                           int width, float epsilon, bool normalize) {
    const size_t base = size_t(blockIdx.x) * width;
    float sum = 0.0f;
    if (normalize) {
        for (size_t i = threadIdx.x; i < size_t(width); i += blockDim.x) {
            const float v = x[base + i];
            sum = fmaf(v, v, sum);
        }
        sum = rsqrtf(block_sum<256>(sum) / float(width) + epsilon);
    }
    for (size_t i = threadIdx.x; i < size_t(width); i += blockDim.x) {
        const float weight = w ? w[i] : 1.0f;
        y[base + i] = normalize ? x[base + i] * sum * weight : x[base + i];
    }
}
__global__ void convert_kernel(const float* x, __nv_bfloat16* y, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) y[i] = __float2bfloat16_rn(x[i]);
}
template<bool Silu>
__global__ void binary_kernel(const float* x, const float* b, float* y, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        const float v = x[i], u = b[i];
        y[i] = Silu ? (v / (1.0f + expf(-v))) * u : v + u;
    }
}
unsigned linear_grid(size_t n) { return unsigned((n + 255) / 256 < 65535 ? (n + 255) / 256 : 65535); }

__global__ void embedding_kernel(EmbeddingArgs a) {
    const int row = blockIdx.x;
    const int id = a.ids[row];
    const int offset = a.snr_two_rows ? a.block_offsets[row] : 0;
    const bool good_id = id >= 0 && id < a.vocab;
    const bool good_offset = !a.snr_two_rows || (offset >= 0 && offset < a.block_rows);
    if (threadIdx.x == 0) {
        if (!good_id) fail(a.device_error, DeviceError::token_out_of_range);
        if (!good_offset) fail(a.device_error, DeviceError::invalid_block);
    }
    for (size_t d = threadIdx.x; d < size_t(a.width); d += blockDim.x) {
        const size_t out = size_t(row) * a.width + d;
        if (!good_id || !good_offset) { a.output[out] = nanf(""); continue; }
        const size_t in = size_t(id) * a.width + d;
        float v = a.table_bf16 ? __bfloat162float(a.table_bf16[in]) : a.table_f32[in];
        if (a.snr_two_rows) v += a.snr_two_rows[size_t(offset == 0 ? 0 : 1) * a.width + d];
        if (a.mode) v += a.mode[d];
        if (!isfinite(v)) fail(a.device_error, DeviceError::nonfinite_value);
        a.output[out] = v;
    }
}

__global__ void qkv_kernel(QkvArgs a) {
    __shared__ float normed[128];
    const int row = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int nq = a.emit_q ? a.geometry.q_heads : 0;
    const bool is_q = h < nq;
    const int head = is_q ? h : h - nq;
    const int pos = a.positions[row], slot = a.batch_slots[row], cache = a.cache_indices[row];
    const bool valid = pos >= 0 && slot >= 0 && slot < a.geometry.batch_slots &&
                       cache >= 0 && cache < a.geometry.ring;
    if (!valid) {
        if (d == 0) fail(a.device_error, DeviceError::row_out_of_range);
        if (is_q) a.q_rotated[(size_t(row) * a.geometry.q_heads + head) * 128 + d] = nanf("");
        return;
    }
    const size_t base = size_t(row) * a.qkv_stride;
    const float v = a.qkv[base + (is_q ? a.q_offset : a.k_offset) + head * 128 + d];
    const float inv = rsqrtf(block_sum<128>(v * v) / 128.0f + a.epsilon);
    normed[d] = v * inv * (is_q ? a.q_norm[d] : a.k_norm[d]);
    __syncthreads();
    float out = normed[d];
    if (d < a.n_rot) {
        const bool split = a.pairing == RopePairing::split_half;
        const int pair = split ? d % (a.n_rot / 2) : d / 2;
        const int lo = split ? pair : 2 * pair;
        const int hi = split ? pair + a.n_rot / 2 : 2 * pair + 1;
        const float theta = float(pos) * a.rope_scale * powf(a.rope_base, -2.0f * float(pair) / float(a.n_rot));
        float sn, cs;
        sincosf(theta, &sn, &cs);
        out = d == lo ? normed[lo] * cs - normed[hi] * sn : normed[lo] * sn + normed[hi] * cs;
    }
    if (!isfinite(out)) fail(a.device_error, DeviceError::nonfinite_value);
    if (is_q) {
        a.q_rotated[(size_t(row) * a.geometry.q_heads + head) * 128 + d] = out;
    } else {
        const size_t dst = kv_index(a.geometry, slot, a.layer, cache, head, d);
        const float value = a.qkv[base + a.v_offset + head * 128 + d];
        const __half key_h = __float2half_rn(out), value_h = __float2half_rn(value);
        if (!isfinite(value) || !isfinite(__half2float(key_h)) || !isfinite(__half2float(value_h)))
            fail(a.device_error, DeviceError::nonfinite_value);
        a.keys[dst] = key_h;
        a.values[dst] = value_h;
        if (head == 0 && d == 0) a.cache_positions[tag_index(a.geometry, slot, a.layer, cache)] = pos;
    }
}

__device__ void attention_key(const AttentionArgs& a, int slot, int cache, int kh,
                             float q, float& max_score, float& denominator, float& accum) {
    const size_t index = kv_index(a.geometry, slot, a.layer, cache, kh, threadIdx.x);
    const float k = __half2float(a.keys[index]), value = __half2float(a.values[index]);
    const float score = block_sum<128>(q * k) * a.qk_scale;
    if (!isfinite(score) || !isfinite(value)) fail(a.device_error, DeviceError::nonfinite_value);
    const float next_max = fmaxf(max_score, score);
    const float alpha = expf(max_score - next_max), p = expf(score - next_max);
    accum = accum * alpha + p * value;
    denominator = denominator * alpha + p;
    max_score = next_max;
}
__global__ void attention_kernel(AttentionArgs a) {
    const int row = blockIdx.x, qh = blockIdx.y, d = threadIdx.x;
    const size_t out_index = (size_t(row) * a.geometry.q_heads + qh) * 128 + d;
    const int bid = a.block_ids[row];
    if (bid < 0 || bid >= a.block_count) {
        if (d == 0) fail(a.device_error, DeviceError::invalid_block);
        a.output[out_index] = nanf(""); return;
    }
    const AttentionBlock b = a.blocks[bid];
    bool valid = b.batch_slot >= 0 && b.batch_slot < a.geometry.batch_slots &&
                 b.anchor_position >= 0 && b.context_begin >= 0 && b.context_begin <= b.anchor_position &&
                 b.query_begin >= 0 && b.query_count > 0 && b.query_count <= 16 &&
                 b.query_begin <= a.rows - b.query_count &&
                 b.anchor_position <= INT_MAX - b.query_count &&
                 a.geometry.ring >= a.context_window + b.query_count &&
                 row >= b.query_begin && row < b.query_begin + b.query_count;
    if (valid) {
        for (int i = 0; i < b.query_count; ++i) {
            const int r = b.query_begin + i, cache = a.cache_indices[r];
            valid = valid && a.block_ids[r] == bid && a.batch_slots[r] == b.batch_slot &&
                    a.positions[r] == b.anchor_position + i && cache >= 0 && cache < a.geometry.ring;
            if (valid) valid = a.cache_positions[tag_index(a.geometry, b.batch_slot, a.layer, cache)] == a.positions[r];
            for (int j = 0; j < i; ++j) valid = valid && cache != a.cache_indices[b.query_begin + j];
        }
    }
    if (!valid) {
        if (d == 0) fail(a.device_error, DeviceError::invalid_block);
        a.output[out_index] = nanf(""); return;
    }
    const int begin = max(b.context_begin, max(0, b.anchor_position - a.context_window));
    const int kh = qh / (a.geometry.q_heads / a.geometry.kv_heads);
    const float q = a.q_rotated[out_index];
    float max_score = -INFINITY, denominator = 0.0f, accum = 0.0f;
    int count = 0;
    for (int pos = begin; pos < b.anchor_position; ++pos) {
        const int cache = pos % a.geometry.ring;
        if (a.cache_positions[tag_index(a.geometry, b.batch_slot, a.layer, cache)] != pos)
            continue;
        attention_key(a, b.batch_slot, cache, kh, q, max_score, denominator, accum);
        ++count;
    }
    for (int i = 0; i < b.query_count; ++i)
        attention_key(a, b.batch_slot, a.cache_indices[b.query_begin + i], kh, q, max_score, denominator, accum);
    if (count != b.anchor_position - begin) {
        if (d == 0) fail(a.device_error, DeviceError::missing_kv);
        a.output[out_index] = nanf("");
    } else {
        const float out = accum / denominator;
        if (!isfinite(out)) fail(a.device_error, DeviceError::nonfinite_value);
        a.output[out_index] = out;
    }
}

__global__ void markov_kernel(MarkovArgs a) {
    const int row = blockIdx.y, lane = threadIdx.x & 31;
    const size_t token = size_t(blockIdx.x) * 8 + threadIdx.x / 32;
    if (token >= size_t(a.output_count)) return;
    const int source = a.source_ids[row];
    const int output_id = a.output_ids ? a.output_ids[token] : int(token);
    const size_t index = size_t(row) * a.output_count + token;
    const bool valid_output = output_id >= 0 && output_id < a.source_vocab &&
                             (!a.output_ids || token == 0 || a.output_ids[token-1] < output_id);
    if (!valid_output) {
        if (lane == 0) { fail(a.device_error, DeviceError::invalid_output_domain); a.logits[index] = nanf(""); }
        return;
    }
    if (source < 0 || source >= a.source_vocab) {
        if (lane == 0) { fail(a.device_error, DeviceError::token_out_of_range); a.logits[index] = nanf(""); }
        if (token == 0 && a.source_latent)
            for (size_t r = lane; r < size_t(a.rank); r += 32) a.source_latent[size_t(row) * a.rank + r] = nanf("");
        return;
    }
    if (token == 0 && a.source_latent)
        for (size_t r = lane; r < size_t(a.rank); r += 32)
            a.source_latent[size_t(row) * a.rank + r] = __bfloat162float(a.codebook_a[size_t(source) * a.rank + r]);
    if (output_id == a.mask_id) {
        if (lane == 0) a.logits[index] = -INFINITY;
        return;
    }
    float dot = 0.0f;
    for (size_t r = lane; r < size_t(a.rank); r += 32)
        dot = fmaf(__bfloat162float(a.codebook_a[size_t(source) * a.rank + r]),
                   __bfloat162float(a.codebook_b[token * a.rank + r]), dot);
    for (int d = 16; d; d >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, d);
    if (lane == 0) {
        const float out = a.base_logits[index] + dot;
        if (!isfinite(out)) fail(a.device_error, DeviceError::nonfinite_value);
        a.logits[index] = out;
    }
}

__global__ void argmax_kernel(const float* logits, int32_t* winner, float* scores,
                              float* probability, int* status, int vocab,
                              const int32_t* output_ids, int source_vocab) {
    __shared__ float maxima[256];
    __shared__ int indices[256], invalid[256];
    const int t = threadIdx.x, row = blockIdx.x;
    const size_t base = size_t(row) * vocab;
    float best = -INFINITY;
    int id = INT_MAX, bad = 0;
    for (size_t i = t; i < size_t(vocab); i += 256) {
        const float x = logits[base + i];
        if (isnan(x) || x == INFINITY) bad |= 1;
        if (output_ids && (output_ids[i] < 0 || output_ids[i] >= source_vocab ||
                           (i > 0 && output_ids[i-1] >= output_ids[i]))) bad |= 2;
        if (x > best || (x == best && int(i) < id)) { best = x; id = int(i); }
    }
    maxima[t] = best; indices[t] = id; invalid[t] = bad;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (t < stride) {
            const float other = maxima[t + stride];
            const int other_id = indices[t + stride];
            if (other > maxima[t] || (other == maxima[t] && other_id < indices[t])) {
                maxima[t] = other; indices[t] = other_id;
            }
            invalid[t] |= invalid[t + stride];
        }
        __syncthreads();
    }
    best = maxima[0]; id = indices[0];
    const bool valid = !invalid[0] && isfinite(best) && id < vocab;
    float denominator = 0.0f;
    if (valid && probability) {
        for (size_t i = t; i < size_t(vocab); i += 256) denominator += expf(logits[base + i] - best);
        denominator = block_sum<256>(denominator);
    }
    if (t == 0) {
        if (!valid) fail(status, (invalid[0] & 2) ? DeviceError::invalid_output_domain :
                               invalid[0] ? DeviceError::nonfinite_value : DeviceError::no_candidate);
        winner[row] = valid ? (output_ids ? output_ids[id] : id) : -1;
        if (scores) scores[row] = valid ? best : nanf("");
        if (probability) probability[row] = valid ? 1.0f / denominator : nanf("");
    }
}

} // namespace

cudaError_t launch_rms_norm(const float* x, const float* weight, float* y,
                           int rows, int width, float epsilon, cudaStream_t stream) {
    if (!x || !y || !positive_matrix(rows, width) || !finite_positive(epsilon)) return cudaErrorInvalidValue;
    rms_kernel<<<rows, 256, 0, stream>>>(x, weight, y, width, epsilon, true);
    return cudaPeekAtLastError();
}
cudaError_t launch_f32_to_bf16(const float* x, __nv_bfloat16* y,
                             int rows, int width, cudaStream_t stream) {
    if (!x || !y || !positive_matrix(rows, width)) return cudaErrorInvalidValue;
    const size_t n = size_t(rows) * width;
    convert_kernel<<<linear_grid(n), 256, 0, stream>>>(x, y, n);
    return cudaPeekAtLastError();
}
cudaError_t launch_add(const float* x, const float* residual, float* y,
                      int rows, int width, cudaStream_t stream) {
    if (!x || !residual || !y || !positive_matrix(rows, width)) return cudaErrorInvalidValue;
    const size_t n = size_t(rows) * width;
    binary_kernel<false><<<linear_grid(n), 256, 0, stream>>>(x, residual, y, n);
    return cudaPeekAtLastError();
}
cudaError_t launch_silu_mul(const float* gate, const float* up, float* y,
                           int rows, int width, cudaStream_t stream) {
    if (!gate || !up || !y || !positive_matrix(rows, width)) return cudaErrorInvalidValue;
    const size_t n = size_t(rows) * width;
    binary_kernel<true><<<linear_grid(n), 256, 0, stream>>>(gate, up, y, n);
    return cudaPeekAtLastError();
}
cudaError_t launch_target_taps(const float* x, float* y, int rows, int taps,
                              int width, bool normalize, float epsilon, cudaStream_t stream) {
    if (!x || !y || taps <= 0 || rows <= 0 || rows > INT_MAX / taps ||
        !positive_matrix(rows * taps, width) || (normalize && !finite_positive(epsilon))) return cudaErrorInvalidValue;
    rms_kernel<<<rows * taps, 256, 0, stream>>>(x, nullptr, y, width, epsilon, normalize);
    return cudaPeekAtLastError();
}
cudaError_t launch_embedding(const EmbeddingArgs& a, cudaStream_t stream) {
    if ((a.table_bf16 == nullptr) == (a.table_f32 == nullptr) || !a.ids || !a.output || !a.device_error ||
        !positive_matrix(a.rows, a.width) || !positive_matrix(a.vocab, a.width) ||
        (a.snr_two_rows && (!a.block_offsets || a.block_rows <= 0))) return cudaErrorInvalidValue;
    embedding_kernel<<<a.rows, 256, 0, stream>>>(a);
    return cudaPeekAtLastError();
}
cudaError_t launch_qkv_rope_cache(const QkvArgs& a, cudaStream_t stream) {
    const auto g = a.geometry;
    if (!valid_geometry(g) || !a.qkv || !a.k_norm || !a.positions || !a.batch_slots || !a.cache_indices ||
        !a.keys || !a.values || !a.cache_positions || !a.device_error || a.rows <= 0 ||
        a.layer < 0 || a.layer >= g.layers || !finite_positive(a.epsilon) ||
        !finite_positive(a.rope_base) || !finite_positive(a.rope_scale) ||
        a.n_rot < 0 || a.n_rot > 128 || a.n_rot % 2 ||
        (a.pairing != RopePairing::split_half && a.pairing != RopePairing::interleaved) ||
        a.k_offset < 0 || a.v_offset < 0 || a.qkv_stride <= 0 ||
        int64_t(a.k_offset) + int64_t(g.kv_heads) * 128 > a.qkv_stride ||
        int64_t(a.v_offset) + int64_t(g.kv_heads) * 128 > a.qkv_stride ||
        !positive_matrix(a.rows, a.qkv_stride) ||
        (a.emit_q && (!a.q_norm || !a.q_rotated || a.q_offset < 0 ||
                      int64_t(a.q_offset) + int64_t(g.q_heads) * 128 > a.qkv_stride ||
                      g.q_heads + g.kv_heads > 65535))) return cudaErrorInvalidValue;
    qkv_kernel<<<dim3(a.rows, (a.emit_q ? g.q_heads : 0) + g.kv_heads), 128, 0, stream>>>(a);
    return cudaPeekAtLastError();
}
cudaError_t launch_attention(const AttentionArgs& a, cudaStream_t stream) {
    const auto g = a.geometry;
    if (!valid_geometry(g) || !a.q_rotated || !a.keys || !a.values || !a.cache_positions ||
        !a.positions || !a.batch_slots || !a.cache_indices || !a.block_ids || !a.blocks ||
        !a.output || !a.device_error || a.rows <= 0 || a.layer < 0 || a.layer >= g.layers ||
        a.block_count <= 0 || a.context_window <= 0 || a.context_window > INT_MAX - 16 ||
        g.ring <= a.context_window || !finite_positive(a.qk_scale) ||
        !positive_matrix(a.rows, g.q_heads * 128)) return cudaErrorInvalidValue;
    attention_kernel<<<dim3(a.rows, g.q_heads), 128, 0, stream>>>(a);
    return cudaPeekAtLastError();
}
cudaError_t launch_markov(const MarkovArgs& a, cudaStream_t stream) {
    if (!a.base_logits || !a.codebook_a || !a.codebook_b || !a.source_ids || !a.logits || !a.device_error ||
        !positive_matrix(a.rows, a.output_count) || !positive_matrix(a.source_vocab, a.rank) ||
        !positive_matrix(a.output_count, a.rank) || a.output_count > a.source_vocab ||
        (!a.output_ids && a.output_count != a.source_vocab) ||
        a.rows > 65535 || a.mask_id < 0 || a.mask_id >= a.source_vocab || a.source_vocab < 2) return cudaErrorInvalidValue;
    markov_kernel<<<dim3(unsigned((size_t(a.output_count) + 7) / 8), a.rows), 256, 0, stream>>>(a);
    return cudaPeekAtLastError();
}
cudaError_t launch_argmax(const float* logits, int32_t* winner_ids,
                         float* winner_logits, float* winner_probability,
                         int* device_error, int rows, int vocab, cudaStream_t stream) {
    if (!logits || !winner_ids || !device_error || !positive_matrix(rows, vocab)) return cudaErrorInvalidValue;
    argmax_kernel<<<rows, 256, 0, stream>>>(logits, winner_ids, winner_logits, winner_probability, device_error, vocab, nullptr, vocab);
    return cudaPeekAtLastError();
}
cudaError_t launch_candidate_argmax(const float* logits, const int32_t* output_ids,
                                   int32_t* winner_ids, int* device_error, int rows,
                                   int output_count, int source_vocab, cudaStream_t stream) {
    if (!logits || !winner_ids || !device_error || !positive_matrix(rows, output_count) ||
        source_vocab <= 0 || output_count > source_vocab ||
        (!output_ids && output_count != source_vocab)) return cudaErrorInvalidValue;
    argmax_kernel<<<rows, 256, 0, stream>>>(logits, winner_ids, nullptr, nullptr, device_error,
                                         output_count, output_ids, source_vocab);
    return cudaPeekAtLastError();
}

} // namespace h5::dspark
