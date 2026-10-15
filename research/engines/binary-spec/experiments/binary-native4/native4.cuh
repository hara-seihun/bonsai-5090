#pragma once
#include "../binary-bitserial/bitserial.cuh"
namespace native4 {
struct Plan {int n,k,r,groups_per_split,n_tile;};
struct Weights {uint32_t* nibbles; int8_t* dc; const __half* scales;};
struct Carrier {uint32_t* low; uint32_t* high; bitserial::Carrier a8;};
cudaError_t validate(const Plan& p);
size_t weight_bytes(int n,int k); // signed W4 [K/128,16,N]
size_t dc_bytes(int n,int k); // half of exact signed weight sum per K128, in [-64,64]
size_t plane_bytes(int k,int r); // each unsigned activation nibble plane [R,K/128,16]
size_t partial_bytes(const Plan& p);
cudaError_t prepare(const uint32_t* signs,Weights out,int n,int k,cudaStream_t stream);
cudaError_t repack(int k,int r,Carrier out,cudaStream_t stream); // consumes already-produced original A8
cudaError_t produce(const float* x,int k,int r,Carrier out,cudaStream_t stream);
cudaError_t project_carrier(const Plan& p,Weights w,Carrier in,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t stream);
cudaError_t project(const Plan& p,Weights w,const float* x,Carrier scratch,float* y,int ldo,float* partial,float alpha,float beta,cudaStream_t stream);
}
