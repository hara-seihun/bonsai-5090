#pragma once
#include "signed_digits.cuh"
namespace sparse_fp4 {
using Plan=signed_fp4::Plan;
using Schedule=signed_fp4::Schedule;
using WeightLayout=signed_fp4::WeightLayout;
constexpr int ListCapacity=32;
struct Digits { uint8_t* data; size_t capacity; uint8_t* counts; size_t count_capacity; uint16_t* entries; size_t entry_capacity; };
size_t digit_bytes(int k,int rows);
size_t count_bytes(int k,int rows);
size_t entry_words(int k,int rows);
size_t partial_floats(const Plan&);
cudaError_t validate(const Plan&);
cudaError_t configure(Schedule);
cudaError_t encode(bitserial::Carrier,int k,int rows,Digits,cudaStream_t);
cudaError_t project(const Plan&,const void* packed_signs,const void* fp16_scales,
                    bitserial::Carrier,Digits,float* y,int stride,float* partials,
                    size_t partial_capacity,float alpha,float beta,cudaStream_t);
}
