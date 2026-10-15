#pragma once
#include "../binary-bitserial/bitserial.cuh"
namespace composition {
struct Plan {int n,k,r,tile,chunk;};
cudaError_t validate(Plan);
size_t partial_bytes(Plan);
cudaError_t fused(Plan,const uint32_t*,const __half*,bitserial::Carrier,bitserial::Carrier,float*,cudaStream_t);
cudaError_t unfused_tail(const float*,int,int,bitserial::Carrier,float*,cudaStream_t);
}
