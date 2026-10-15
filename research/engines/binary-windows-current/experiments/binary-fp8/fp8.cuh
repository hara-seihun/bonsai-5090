#pragma once
#include "../binary-bitserial/bitserial.cuh"
#include <cuda_fp8.h>
namespace binary_fp8 {
struct Plan {int n,k,r,groups_per_split,n_tile;};
struct Carrier {uint8_t* q; float* scales; bitserial::InputStatus* status; bitserial::InputStatus* error=nullptr;};
cudaError_t validate(const Plan& p);
size_t partial_bytes(const Plan& p);
// Original prepared Q1 signs/scales, same as bitserial. No persistent W8.
cudaError_t produce(const float* x,int k,int r,Carrier out,cudaStream_t stream);
cudaError_t project_carrier(const Plan& p,const uint32_t* signs,const __half* wscale,Carrier in,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t stream);
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* wscale,const float* x,Carrier scratch,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t stream);
}
