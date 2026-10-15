#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace h5::dspark {

// Host geometry/pointer errors return cudaErrorInvalidValue. Device-index errors
// set the caller-owned sticky status; zero it before a sequence, inspect after
// stream completion. Launch success alone does not attest device input validity.
// All arrays are tight device buffers of their declared sizes. Basic elementwise
// operations and RMS allow exact input/output aliasing; all other outputs must
// be disjoint from their inputs. Partial overlaps are not supported.
enum class DeviceError : int {
    success = 0, token_out_of_range = 1, row_out_of_range = 2,
    missing_kv = 3, invalid_block = 4, nonfinite_value = 5,
    no_candidate = 6, invalid_output_domain = 7
};

cudaError_t launch_rms_norm(const float* x, const float* weight, float* y,
                           int rows, int width, float epsilon, cudaStream_t stream);
cudaError_t launch_f32_to_bf16(const float* x, __nv_bfloat16* y,
                             int rows, int width, cudaStream_t stream);
cudaError_t launch_add(const float* x, const float* residual, float* y,
                      int rows, int width, cudaStream_t stream);
cudaError_t launch_silu_mul(const float* gate, const float* up, float* y,
                           int rows, int width, cudaStream_t stream);
// false copies unchanged; true normalizes each width-wide tap independently.
cudaError_t launch_target_taps(const float* x, float* y, int rows, int taps,
                              int width, bool normalize, float epsilon,
                              cudaStream_t stream);

struct EmbeddingArgs {
    const __nv_bfloat16* table_bf16;
    const float* table_f32; // exactly one table pointer must be set
    const int32_t* ids;
    const int32_t* block_offsets; // required with snr_two_rows; 0 means anchor
    const float* snr_two_rows; // optional [2,width], max-SNR then min-SNR
    const float* mode; // optional [width], added to every draft row
    float* output;
    int* device_error;
    int rows, width, vocab, block_rows;
};
cudaError_t launch_embedding(const EmbeddingArgs& args, cudaStream_t stream);

// Dense cache: [batch_slots,layers,ring,kv_heads,head_dim]. Tags are
// [batch_slots,layers,ring], initialized to -1. cache_indices explicitly select
// writes. Historical position p resides at p%ring with tag p; current block
// entries may use any explicit cache indices. Writers have unique
// (slot,layer,cache) entries in a launch.
struct KvGeometry {
    int batch_slots, layers, ring, q_heads, kv_heads, head_dim;
};
enum class RopePairing : int { split_half = 0, interleaved = 1 };
struct QkvArgs {
    const float* qkv;
    const float* q_norm; // [head_dim], required when emit_q
    const float* k_norm; // [head_dim], required
    const int32_t* positions;
    const int32_t* batch_slots;
    const int32_t* cache_indices;
    float* q_rotated; // [rows,q_heads,head_dim], unscaled; null for KV only
    __half* keys;
    __half* values;
    int32_t* cache_positions;
    int* device_error;
    KvGeometry geometry;
    int rows, layer, qkv_stride, q_offset, k_offset, v_offset, n_rot;
    bool emit_q;
    RopePairing pairing;
    float epsilon, rope_base, rope_scale;
};
cudaError_t launch_qkv_rope_cache(const QkvArgs& args, cudaStream_t stream);

struct AttentionBlock {
    int batch_slot, anchor_position, context_begin, query_begin, query_count;
};
struct AttentionArgs {
    const float* q_rotated; // [rows,q_heads,head_dim], unscaled
    const __half* keys;
    const __half* values;
    const int32_t* cache_positions;
    const int32_t* positions; // [rows], positions anchor+i within each block
    const int32_t* batch_slots; // [rows]
    const int32_t* cache_indices; // [rows], current block's physical KV entries
    const int32_t* block_ids; // [rows], explicit query-to-block map
    const AttentionBlock* blocks; // [block_count], device memory
    float* output; // [rows,q_heads,head_dim]
    int* device_error;
    KvGeometry geometry;
    int rows, layer, block_count, context_window;
    float qk_scale;
};
// Full bidirectional block attention. Each historical position p in
// [max(context_begin,anchor-context_window,0),anchor) requires tag p at p%ring;
// missing/stale historical entries raise missing_kv. Only explicitly listed
// current block entries are visible at/after anchor. Stale pending rows are not.
cudaError_t launch_attention(const AttentionArgs& args, cudaStream_t stream);

struct MarkovArgs {
    const float* base_logits; // [rows,output_count], may alias logits
    const __nv_bfloat16* codebook_a; // [source_vocab,rank], actual previous-token lookup
    const __nv_bfloat16* codebook_b; // [output_count,rank], selected output rows
    const int32_t* source_ids; // [rows], actual IDs, including anchors outside selection
    const int32_t* output_ids; // [output_count], increasing actual IDs; null means full identity domain
    float* logits; // [rows,output_count], actual mask_id suppressed to -infinity
    float* source_latent; // optional [rows,rank] F32 lookup for learned confidence
    int* device_error;
    int rows, source_vocab, output_count, rank, mask_id;
};
// One sequential step per row. No hidden correction or confidence head here.
cudaError_t launch_markov(const MarkovArgs& args, cudaStream_t stream);
// Lowest column wins exact ties. Optional winner_probability is domain softmax
// probability, not the learned confidence-head output.
cudaError_t launch_argmax(const float* logits, int32_t* winner_ids,
                         float* winner_logits, float* winner_probability,
                         int* device_error,
                         int rows, int vocab, cudaStream_t stream);
// Output IDs must be strictly increasing actual source-domain IDs. Winners are
// mapped before export; probability is normalized over selected output columns.
cudaError_t launch_candidate_argmax(const float* logits, const int32_t* output_ids,
                                   int32_t* winner_ids, int* device_error, int rows,
                                   int output_count, int source_vocab, cudaStream_t stream);

} // namespace h5::dspark
