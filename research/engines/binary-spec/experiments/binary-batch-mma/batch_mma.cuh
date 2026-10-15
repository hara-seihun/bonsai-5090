#pragma once
#include "../binary-bitserial/bitserial.cuh"
namespace batch_mma {
struct Plan { int n,k,r,groups_per_split,n_tile; };
// Original packed signs [K/128,4,N], original FP16 scales [K/128,N].
// N32, K128; R1..32 (masked M16 token tails). N tiles32/64/128.
// No persistent expanded weight representation.
cudaError_t validate(const Plan& p);
// Best cold candidate from the negative bounded round: GU/down/head65536/head248320,
// R5/8/16/32. Other valid geometry returns cudaErrorNotSupported; caller may use explicit Plan.
// Selection does not recommend replacing stable DP4A.
cudaError_t select(int n,int k,int r,Plan* out);
// Size helpers require a validated plan. Output row stride is explicit and >=N.
size_t partial_bytes(const Plan& p);
// Carrier token-row views are supported by pointer offsets with original K/group strides.
// Scratch is caller-owned [R,splits,N] FP32 and may differ from the DP4A plan's scratch size.
// Capture-safe, no allocation/host synchronization/configuration in recurring calls.
cudaError_t project_carrier(const Plan& p,const uint32_t* signs,const __half* weight_scales,
                            bitserial::Carrier in,float* y,int ldo,float* partials,
                            float alpha,float beta,cudaStream_t stream);
// Inclusive producer+MMA+K128 scale/DC continuation. X tightly packed [R,K].
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* weight_scales,const float* x,
                    bitserial::Carrier scratch,float* y,int ldo,float* partials,
                    float alpha,float beta,cudaStream_t stream);
}
