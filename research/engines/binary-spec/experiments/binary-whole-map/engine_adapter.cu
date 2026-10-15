#include "signed_digits.cuh"
#include "../binary-a8-fp4/digits.cuh"

// Linker wrappers retain the immutable engine's actual plan, projection policy,
// scales, A8 producer and complete consuming suffix. Only the carrier changes.
static signed_fp4::Schedule schedule(a8_fp4::Schedule s) {
  switch(s) {
    case a8_fp4::Schedule::F4G2T8: return signed_fp4::Schedule::F4G2T8;
    case a8_fp4::Schedule::F4G2T16: return signed_fp4::Schedule::F4G2T16;
    case a8_fp4::Schedule::F4G2T32: return signed_fp4::Schedule::F4G2T32;
    case a8_fp4::Schedule::F8G1T32: return signed_fp4::Schedule::F8G1T32;
    case a8_fp4::Schedule::F8G2T32: return signed_fp4::Schedule::F8G2T32;
  }
  return static_cast<signed_fp4::Schedule>(-1);
}
static signed_fp4::WeightLayout layout(a8_fp4::WeightLayout l) {
  switch(l) {
    case a8_fp4::WeightLayout::WF8: return signed_fp4::WeightLayout::WF8;
    case a8_fp4::WeightLayout::Packed1: return signed_fp4::WeightLayout::Packed1;
  }
  return static_cast<signed_fp4::WeightLayout>(-1);
}
extern "C" size_t wrap_bytes(int,int) asm("__wrap__ZN6a8_fp411digit_bytesEii");
extern "C" size_t wrap_bytes(int k,int r) { return signed_fp4::digit_bytes(k,r); }
extern "C" cudaError_t wrap_config(a8_fp4::Schedule) asm("__wrap__ZN6a8_fp49configureENS_8ScheduleE");
extern "C" cudaError_t wrap_config(a8_fp4::Schedule s) { return signed_fp4::configure(schedule(s)); }
extern "C" cudaError_t wrap_encode(bitserial::Carrier,int,int,a8_fp4::Digits,cudaStream_t) asm("__wrap__ZN6a8_fp46encodeEN9bitserial7CarrierEiiNS_6DigitsEP11CUstream_st");
extern "C" cudaError_t wrap_encode(bitserial::Carrier c,int k,int r,a8_fp4::Digits d,cudaStream_t st) {
  return signed_fp4::encode(c,k,r,{d.data,d.capacity,signed_fp4::Encoding::ExactA8},st);
}
extern "C" cudaError_t wrap_project(const a8_fp4::Plan&,const void*,const void*,bitserial::Carrier,a8_fp4::Digits,float*,int,float*,size_t,float,float,cudaStream_t) asm("__wrap__ZN6a8_fp47projectERKNS_4PlanEPKvS4_N9bitserial7CarrierENS_6DigitsEPfiS8_mffP11CUstream_st");
extern "C" cudaError_t wrap_project(const a8_fp4::Plan& p,const void* w,const void* scales,bitserial::Carrier c,a8_fp4::Digits d,float* y,int ld,float* partials,size_t capacity,float alpha,float beta,cudaStream_t st) {
  signed_fp4::Plan q{p.n,p.k,p.rows,p.splits,schedule(p.schedule),layout(p.layout)};
  return signed_fp4::project(q,w,scales,c,{d.data,d.capacity,signed_fp4::Encoding::ExactA8},y,ld,partials,capacity,alpha,beta,st);
}
