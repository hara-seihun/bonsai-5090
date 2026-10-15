#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define F1_HD __host__ __device__ __forceinline__
#else
#define F1_HD inline
#endif
namespace f1 {
// Eight independent biased4-bit channels; each intermediate nibble stays in[0,15],
// so native32-bit add/sub cannot carry/borrow across channels.
F1_HD uint32_t magnitude_label(uint32_t z) {
  uint32_t neg=(~z>>3)&0x11111111u;
  uint32_t mag=((z&0x77777777u)^(neg*7u))+neg;
  return mag|(neg<<3);
}
F1_HD void h4_swar(uint32_t x,uint32_t y,uint32_t* r) {
  constexpr uint32_t M=0x11111111u,B=0x22222222u,E=0x88888888u;
  uint32_t a=B+((x>>1)&M)-(((x>>3)&M)<<1);
  uint32_t b=B+(x&M)-(((x>>2)&M)<<1);
  uint32_t c=B+((y>>1)&M)-(((y>>3)&M)<<1);
  uint32_t d=B+(y&M)-(((y>>2)&M)<<1);
  r[0]=magnitude_label(a+b+c+d);
  r[1]=magnitude_label(a+c+(E-b-d));
  r[2]=magnitude_label(a+b+(E-c-d));
  r[3]=magnitude_label(a+d+(E-b-c));
}
}
#undef F1_HD
