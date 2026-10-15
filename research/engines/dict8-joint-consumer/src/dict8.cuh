#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#ifdef _WIN32
#define DICT8_EXPORT __declspec(dllexport)
#else
#define DICT8_EXPORT __attribute__((visibility("default")))
#endif
namespace h5 {
cudaError_t launch_gu_dict8(const uint8_t*,const uint8_t*,const float*,int,int,const uint8_t*,int,float*,int,const float*,const uint32_t*,cudaStream_t,float* pairs=nullptr);
cudaError_t launch_gu_dict8_reference(const uint8_t*,const uint8_t*,const float*,int,int,const uint8_t*,int,float*,int,const float*,cudaStream_t,float* pairs);
cudaError_t compare_dict8_pairs(const float*,const float*,const float*,const float*,uint32_t*,cudaStream_t);
cudaError_t compare_dict8_quant(const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,const float*,uint32_t*,cudaStream_t);
}
