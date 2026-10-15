#include <cuda_runtime.h>

// Compile-only copies of the WF2 scale and WF5 operand transforms. No model or launch harness.
__global__ void scale_extract_current(const unsigned* x, unsigned* y) {
  const unsigned sb4=x[threadIdx.x]; const int tq=threadIdx.x&3;
#pragma unroll
  for (int mm=0;mm<2;mm++) {
    const unsigned b=(sb4>>(8*(2*mm+(tq&1))))&0xff;
    y[2*threadIdx.x+mm]=b*0x01010101u;
  }
}
__global__ void scale_extract_prmt(const unsigned* x, unsigned* y) {
  const unsigned sb4=x[threadIdx.x]; const int tq=threadIdx.x&3;
#pragma unroll
  for (int mm=0;mm<2;mm++) y[2*threadIdx.x+mm]=__byte_perm(sb4,0,(tq&1)*0x1111+mm*0x2222);
}
__global__ void decode_current(const unsigned* x, unsigned* y) {
  const unsigned r=x[threadIdx.x];
  y[2*threadIdx.x]=r&0xaaaaaaaau; y[2*threadIdx.x+1]=(r<<1)&0xaaaaaaaau;
}
__global__ void decode_lowbit2_current(const unsigned* x, unsigned* y) {
  const unsigned r=x[threadIdx.x];
  y[2*threadIdx.x]=(r&0xaaaaaaaau)|0x11111111u;
  y[2*threadIdx.x+1]=((r<<1)&0xaaaaaaaau)|0x11111111u;
}
__global__ void decode_lowbit2_lop3(const unsigned* x, unsigned* y) {
  const unsigned r=x[threadIdx.x], shifted=r<<1; unsigned a,b;
  asm volatile("lop3.b32 %0,%1,%2,%3,0xea;" : "=r"(a) : "r"(r),"r"(0xaaaaaaaau),"r"(0x11111111u));
  asm volatile("lop3.b32 %0,%1,%2,%3,0xea;" : "=r"(b) : "r"(shifted),"r"(0xaaaaaaaau),"r"(0x11111111u));
  y[2*threadIdx.x]=a; y[2*threadIdx.x+1]=b;
}
