#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <variant>

namespace h5 {
// Borrowed immutable row means. The engine owns allocation, load-time preparation and lifetime.
class ScaleCarrierView {
  const float* const means_;
  const int rows_, contraction_;
  ScaleCarrierView(const float* p, int n, int k) : means_(p), rows_(n), contraction_(k) {}
public:
  static std::variant<ScaleCarrierView, cudaError_t> make(const float* p, int n, int k) {
    if (!p || n <= 0 || n % 32 || k <= 0 || k % 128) return cudaErrorInvalidValue;
    return ScaleCarrierView(p, n, k);
  }
  const float* means() const { return means_; }
  int rows() const { return rows_; }
  int contraction() const { return contraction_; }
};

// stats may be null; when present it is float4[N] = mean,min,max,variance.
// Invalid source scale data writes NaN means/stats; loading must check finite prepared values once.
cudaError_t scale_carrier_prepare(const void* packed_scales, float* mean, float4* stats,
                                 int N, int K, cudaStream_t stream);
cudaError_t scale_carrier_resources(int ci, int mode, cudaFuncAttributes* attrs, int* resident_blocks);
// Explicit native two-bit coordinate. Same means and output policy; W must be
// WF0 1024B/group/K-block, not dense5. No preparation/repacking during launch.
cudaError_t scale_carrier_resources_wf0(int ci, int mode, cudaFuncAttributes* attrs, int* resident_blocks);
cudaError_t launch_scale_carrier_wf0(int ci, int split, int mode, const uint8_t* A, const uint8_t* AS,
                                    const float* AG, int R, int K, const void* W, int N,
                                    float* out, int ldo, const float* means, cudaStream_t stream);
// mode0 store, mode1 residual add, mode2 atomic partial, mode3 fused interleaved GU.
// Never zeros output. For mode2 nonadd caller memsets/preclears; add caller retains its residual.
bool scale_carrier_fuse_supported(int ci);
struct TailArgs;
// Carrier dense5 projection whose producer tail prepares the next GEMM's activations (tail_prep.cuh):
// kind 1 chunk prep (SiLU gate/up, mode 3), kind 2 row-norm prep (add projections, mode 1/2). R <= 8.
cudaError_t launch_scale_carrier_tail(int ci, int split, int mode, int kind, const uint8_t* A, const uint8_t* AS, const float* AG,
                                      int R, int K, const void* W, int N, float* out, int ldo, const float* mean,
                                      const TailArgs& ta, cudaStream_t stream);
cudaError_t launch_scale_carrier_fused_gu(int ci, int R, int K, const void* W, int N, float* out, int ldo, const float* mean,
                                          const float* x, int ldx, const float* norm_w, const float* signs, cudaStream_t st);
cudaError_t launch_scale_carrier(int ci, int split, int mode, const uint8_t* A, const uint8_t* AS,
                                const float* AG, int R, int K, const void* W, int N,
                                float* out, int ldo, const float* means, cudaStream_t stream);
inline cudaError_t launch_scale_carrier(int ci, int split, int mode, const uint8_t* A,
                                       const uint8_t* AS, const float* AG, int R, const void* W,
                                       float* out, int ldo, const ScaleCarrierView& view,
                                       cudaStream_t stream) {
  return launch_scale_carrier(ci, split, mode, A, AS, AG, R, view.contraction(), W,
                             view.rows(), out, ldo, view.means(), stream);
}
inline cudaError_t launch_scale_carrier_wf0(int ci, int split, int mode, const uint8_t* A,
                                           const uint8_t* AS, const float* AG, int R, const void* W,
                                           float* out, int ldo, const ScaleCarrierView& view,
                                           cudaStream_t stream) {
  return launch_scale_carrier_wf0(ci, split, mode, A, AS, AG, R, view.contraction(), W,
                                 view.rows(), out, ldo, view.means(), stream);
}
}
