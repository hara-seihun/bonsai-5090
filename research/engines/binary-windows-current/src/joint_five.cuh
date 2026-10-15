#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "../experiments/binary-bitserial/bitserial.cuh"
namespace p12_joint_five {
size_t bytes(int k,int rows);
cudaError_t initialize();
cudaError_t encode(bitserial::Carrier,int k,int rows,uint8_t*,size_t,cudaStream_t);
cudaError_t project(const uint8_t* packed1,const void* fp16_scales,bitserial::Carrier,const uint8_t* digits,int n,int k,int rows,float* out,int stride,float alpha,float beta,cudaStream_t);
}
