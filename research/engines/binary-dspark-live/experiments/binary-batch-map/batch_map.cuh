#pragma once
#include "../binary-bitserial/bitserial.cuh"
namespace batch_map {
struct Carrier { uint32_t* packed; bitserial::Carrier metadata; int k, rows; };
struct Plan { int n,k,rows,first_token,groups_per_split,threads; };
cudaError_t prepare_counts(const uint32_t* signs,uint8_t* counts,int n,int k,cudaStream_t st);
cudaError_t produce(const float* x,Carrier out,cudaStream_t st);
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* scales,const uint8_t* counts,
                    Carrier in,float* y,float* partial,float alpha,float beta,cudaStream_t st);
}
