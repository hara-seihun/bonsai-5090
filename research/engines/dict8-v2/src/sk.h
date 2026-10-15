// Stream-K weight GEMM (src/sk.cu) for steps of at most eight rows.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
namespace h5 {
enum SkMode : int {
  SK_ATOMIC = 0,   // out[t][n] += value (output precleared, or a residual sum)
  SK_STORE = 1,    // out[t][n] = value; split tiles meet in the workspace
  SK_SILU = 2,     // interleaved gate/up image: out[t][n/2] = silu(gate) * up; split tiles meet in the workspace
};
struct SkArgs {
  const uint8_t* A; const uint8_t* AS; const float* AG;   // NVFP4 activations, rows 0..R-1 of tile 0
  int R, K, N;
  const uint8_t* W; const uint8_t* S;                     // weights; S = NVFP4 block scales (WF 1 only)
  float* out; int ldo;
  const float* RS;                                        // per-output-row scale
  float* ws; int* cnt;                                    // all-zero between launches (STORE, SILU)
  int prewait;                                            // weight stages issued before the predecessor wait
  int* next;                                              // work counter of this launch site; zero between launches
  int chunk;                                              // K blocks per work item
  // Optional second segment (ATOMIC, NVFP4 weights, same activations), e.g. the alpha/beta projection
  // riding inside the in-projection launch. N2 = 0: none.
  const uint8_t* W2 = nullptr; const uint8_t* S2 = nullptr; const float* RS2 = nullptr; int N2 = 0; float* out2 = nullptr; int ldo2 = 0;
};
struct SkShape { int fw, wg, stages; };
int sk_nshapes();
SkShape sk_shape(int i);
// variant: 0 normal; measurement variants 1 copy only, 2 compute only, 3 lane-private codebooks, 4 both 2+3 (dense5 only)
cudaError_t sk_launch(int shape, int wf, int mode, int variant, const SkArgs& a, int grid, cudaStream_t st);
int sk_occupancy(int shape, int wf, int mode, int variant);
// K blocks per work item so that a launch of `grid` CTAs has about items_per_cta items per CTA.
int sk_chunk(int shape, int wf, int N, int K, int N2, int grid, int items_per_cta);
constexpr int SK_SITES = 4096;   // work counters per engine; capture assigns one per launch site
inline size_t sk_workspace_floats(int N) { return ((size_t)N / 32 + 16) * 256; }
inline size_t sk_counter_ints(int N) { return (size_t)N / 32 + 16; }
}
