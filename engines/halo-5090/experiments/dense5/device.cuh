#include "../../src/compact.cuh"
namespace c5 {
using h5::dense5_registers;
__global__ void dense5_certificate(unsigned* out) {
  __shared__ uint32_t book[256];
  __shared__ uint8_t chunk[832];
  int lane=threadIdx.x;
  for(int code=lane;code<256;code+=32) {
    unsigned x=code,v=0;
#pragma unroll
    for(int j=0;j<5;j++,x/=3) {unsigned t=x%3; v|=(t==0?10u:t==2?2u:0u)<<(4*j);}
    book[code]=v;
  }
  int q=blockIdx.x*32+lane;
  for(int b=0;b<26;b++) {
    int pos=b<24?(b/4)*128+lane*4+b%4:768+lane*2+b-24;
    chunk[pos]=(q+b*17)%243;
  }
  __syncwarp();
  unsigned regs[16];dense5_registers(chunk,lane,book,regs);
  if(q<243)for(int w=0;w<16;w++)out[q*16+w]=regs[w];
}
void decoder_test(unsigned* out,cudaStream_t st) {dense5_certificate<<<8,32,0,st>>>(out);}
}
