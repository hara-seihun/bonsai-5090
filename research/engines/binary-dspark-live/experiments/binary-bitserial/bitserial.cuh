#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
namespace bitserial {
enum class Route { Popcount, Dp4a01 };
struct Plan { int n,k,r,groups_per_split,threads; Route route; };
enum class InputStatus : uint32_t { Valid=0, NonFinite=1, ScaleRange=2 };
struct Carrier {
  int8_t* q;                 // Dp4a01: [R,K], absent for Popcount
  uint32_t* planes;           // Popcount: [R,K/128,8,4], absent for Dp4a01
  float* scales;              // [R,K/128]
  int32_t* dc;                // [R,K/128], sum of signed q bytes
  InputStatus* status;        // [R,K/128], asynchronous typed producer outcome
  InputStatus* error=nullptr; // Optional persistent aggregate; null explicitly disables aggregation.
                             // Caller clears once/body, then checks after its normal stream synchronization.
};
// Supports every R1..16; other row counts are explicit errors.
cudaError_t validate(const Plan& p);
// Explicit opt-in schedule seeded by measured geometries; unchanged default engine routes.
// Sets *out only on success. Unmeasured positive dimensions use the declared scheduling formula.
cudaError_t select(int n,int k,int r,Route route,Plan* out);
// Recommended opt-in DP4A map; does not change the FP32 engine's own selection.
cudaError_t select(int n,int k,int r,Plan* out);
size_t sign_bytes(int n,int k);
size_t scale_bytes(int n,int k);
size_t partial_bytes(const Plan& p);
cudaError_t prepare(const uint8_t* raw,uint32_t* signs,__half* scales,int n,int k,cudaStream_t stream);
// sx=max(abs(x))/127; q=round-to-nearest-even(x*(127/max)), all-zero group -> q=sx=0.
// Producer supports R1..65535 independently of projection's R1..16 tile limit.
// Exact maps are conditional on status==Valid, not claims of FP32 activation exactness.
// Each group writes status: nonfinite input or nonzero max outside [1e-30,1e30] is an explicit error.
cudaError_t produce(Route route,const float* x,int k,int r,Carrier out,cudaStream_t stream);
// Given a valid A8 carrier, applies original FP16 K128 weight scales in FP32 continuation.
// X and output tightly packed; Y <- alpha W dequant(q) + beta Y.
// Invalid carrier status produces NaN output and retains the typed producer error for the caller.
cudaError_t project_carrier(const Plan& p,const uint32_t* signs,const __half* weight_scales,
                            Carrier in,float* y,float* partials,float alpha,float beta,cudaStream_t stream);
// Complete recurring path: producer, packed map, K128 scaling, split-K continuation.
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* weight_scales,const float* x,
                    Carrier scratch,float* y,float* partials,float alpha,float beta,cudaStream_t stream);
}
