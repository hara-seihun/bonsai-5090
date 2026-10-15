#pragma once
#include "kernels.cuh"
namespace f1 {
// fixed=0 selects row-amax. Input x is already RMS-normalized, as in saved fixture.
void prep(const float* x, const float* signs, int K, int R, float fixed,
          uint8_t* q, uint8_t* scales, float* global, cudaStream_t st);
// gu_fused emits NVFP4 directly, using permuted gate/up weights and natural-K hidden layout.
void gu_fused(int tokw, const uint8_t* A,const uint8_t* AS,const float* AG,int R,int K,
              const void* W,const void* S,int N,const float* signs,float fixed,
              uint8_t* Q,uint8_t* QS,float* QG,cudaStream_t st);
// transformed down weights: compact=false => expanded E2M1 nibbles, same fp16 scale groups.
void down(int tokw,bool compact,const uint8_t* A,const uint8_t* AS,const float* AG,
          int R,int K,const void* W,const void* S,int N,float* Y,cudaStream_t st);
}
