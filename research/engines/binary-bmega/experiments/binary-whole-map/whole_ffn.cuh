#pragma once
#include "signed_digits.cuh"
namespace whole_ffn {
struct Plan { int hidden,k,rows; signed_fp4::Schedule schedule; };
// Static relabel: each GU CTA has 128 gate rows then the corresponding 128 up rows.
cudaError_t prepare(const uint8_t* joined_q1,int hidden,int k,uint32_t* signs,__half* scales,cudaStream_t);
cudaError_t configure(signed_fp4::Schedule);
cudaError_t validate(const Plan&);
// Given ExactA8 input digits. Complete GU/SFU/K128 A8/direct ExactA8-digit continuation.
// next.q may be null: no GU, hidden, or A8-byte global tensor is required.
// next scales/DC/status/error remain caller owned; error is sticky until caller clears it.
cudaError_t gate_up(const Plan&,const uint32_t* signs,const __half* scales,
                    bitserial::Carrier input,signed_fp4::Digits input_digits,
                    bitserial::Carrier next,signed_fp4::Digits next_digits,cudaStream_t);
struct Weights { const uint32_t* gate_up; const __half* gate_up_scales; const uint32_t* down; const __half* down_scales; };
struct Scratch {
  signed_fp4::Digits input_digits;
  bitserial::Carrier next;
  signed_fp4::Digits next_digits;
  float* down_partials;
  size_t down_partial_capacity;
};
// Complete State1→State2 path; charges input recoding and applies optional residual via alpha/beta.
cudaError_t project(const Plan&,const signed_fp4::Plan& down_plan,Weights,
                    bitserial::Carrier input,Scratch,float* output,int stride,
                    float alpha,float beta,cudaStream_t);
}
