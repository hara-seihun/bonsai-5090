#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace h5 {
// Unset flag is the established serving route. Unknown values are rejected at load.
enum class FullKGuMode { Serving, Mixed, Plain, RingMixed, RingPlain };
cudaError_t launch_fullk_gu(FullKGuMode mode, const uint8_t* weights,
  const uint8_t* q4, const uint8_t* s4, const float* ag,
  const float* interleaved_means, const uint32_t* book,
  int rows, int features, int contraction, float* output, cudaStream_t stream);
cudaError_t launch_fullk_ring(int cfg, FullKGuMode mode, const uint8_t* q4,const uint8_t* s4,const float* ag,int rows,int contraction,const uint8_t* weights,int outputs,float* out,int stride,const float* means,const uint32_t* book,cudaStream_t stream);
cudaError_t compare_fullk_gu(const float* a,const float* b,int count,uint32_t* stats,cudaStream_t stream);
void fullk_ring_set_prewait_cap(int cap);
void fullk_ring_set_l2pf_stages(int count);
}
