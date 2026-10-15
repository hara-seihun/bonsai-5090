#pragma once
#include "../binary-bitserial/bitserial.cuh"
namespace signed_fp4 {
enum class Schedule { F4G2T8, F4G2T16, F4G2T32, F8G1T32, F8G2T32 };
enum class WeightLayout { WF8, Packed1 };
struct Plan { int n,k,rows,splits; Schedule schedule; WeightLayout layout; };
// Caller-owned [3 signed-magnitude radix4 planes][tile64][K128][row64][64B].
struct Digits { uint8_t* data; size_t capacity; };
size_t digit_bytes(int k,int rows);
size_t partial_floats(const Plan&);
cudaError_t validate(const Plan&);
// Set CUDA kernel shared-memory limits once before capturing any project calls.
cudaError_t configure(Schedule);
// sx=max(abs(x))/63; q=RNE(x*63/max). Same K128 status/DC contract.
cudaError_t produce(const float*,int k,int rows,bitserial::Carrier,cudaStream_t);
// q outside [-63,63] sets asynchronous ScaleRange status/error.
cudaError_t encode(bitserial::Carrier,int k,int rows,Digits,cudaStream_t);
// Y <- alpha * projection + beta * Y; beta=0 never reads Y.
// Plan/scratch are explicit. Invalid carrier status propagates NaN, not an A8 fallback.
cudaError_t project(const Plan&,const void* wf8,const void* fp16_scales,
                    bitserial::Carrier,Digits,float* y,int stride,float* partials,
                    size_t partial_capacity,float alpha,float beta,cudaStream_t);
}
