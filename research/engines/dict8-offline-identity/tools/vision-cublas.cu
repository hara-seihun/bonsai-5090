// Encoder-only CUDA backend: all supported matmuls execute cuBLAS in ggml-cuda.cu.
// Excluded dispatch families are explicit failures, never dummy successful calls.
#include "common.cuh"
#include "mmvq.cuh"
#include "mmvf.cuh"
#include "fattn.cuh"

bool ggml_cuda_should_use_mmq(ggml_type, int, int64_t, int64_t) { return false; }
bool ggml_cuda_should_use_mmvq(ggml_type, int, int64_t) { return false; }
bool ggml_cuda_should_use_mmvf(ggml_type, int, const int64_t*, const size_t*, int64_t) { return false; }
bool ggml_cuda_should_use_mmf(ggml_type, int, int, const int64_t*, const size_t*, int, bool) { return false; }
int get_mmvq_mmid_max_batch(ggml_type, int) { return 0; }

void ggml_cuda_mul_mat_q(ggml_backend_cuda_context&, const ggml_tensor*, const ggml_tensor*, const ggml_tensor*,
                       ggml_tensor*, const ggml_tensor*, const ggml_tensor*, const ggml_tensor*, void*, bool, const float*) {
    GGML_ABORT("encoder-only CUDA backend: MMQ is not compiled");
}
size_t ggml_cuda_mul_mat_q_q8_size(const ggml_tensor*, const ggml_tensor*) {
    GGML_ABORT("encoder-only CUDA backend: MMQ fusion is not compiled");
}
void ggml_cuda_mul_mat_q_fused_two(ggml_backend_cuda_context&, const ggml_tensor*, const ggml_tensor*, const ggml_tensor*,
                                 ggml_tensor*, ggml_tensor*, const ggml_tensor*, const float*) {
    GGML_ABORT("encoder-only CUDA backend: MMQ fusion is not compiled");
}
void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context&, const ggml_tensor*, const ggml_tensor*, const ggml_tensor*,
                           ggml_tensor*, const ggml_cuda_mm_fusion_args_host*) {
    GGML_ABORT("encoder-only CUDA backend: MMVQ is not compiled");
}
void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context&, const ggml_tensor*, const ggml_tensor*, const ggml_tensor*,
                           ggml_tensor*, const ggml_cuda_mm_fusion_args_host*) {
    GGML_ABORT("encoder-only CUDA backend: MMVF is not compiled");
}
void ggml_cuda_mul_mat_f(ggml_backend_cuda_context&, const ggml_tensor*, const ggml_tensor*, const ggml_tensor*, ggml_tensor*) {
    GGML_ABORT("encoder-only CUDA backend: MMF is not compiled");
}
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context&, ggml_tensor*) {
    GGML_ABORT("encoder-only CUDA backend: flash attention is not compiled");
}
bool ggml_cuda_flash_attn_ext_supported(int, const ggml_tensor*) { return false; }
size_t ggml_cuda_flash_attn_ext_get_alloc_size(int, const ggml_tensor*) { return 0; }
