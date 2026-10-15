#pragma once
#include "compact.hpp"
namespace h5 {
__device__ __forceinline__ void dense5_labels(const uint8_t* chunk, int lane, uint32_t* labels) {
#pragma unroll
  for (int p = 0; p < 6; p++) labels[p] = *(const uint32_t*)(chunk + p * 128 + lane * 4);
  labels[6] = *(const uint16_t*)(chunk + 768 + lane * 2);
}
__device__ __forceinline__ unsigned dense5_word(const uint32_t* labels, const uint32_t* book, int w) {
  const int b = (8 * w) / 5, sh = ((8 * w) % 5) * 4;
  const unsigned lo = book[(labels[b / 4] >> (8 * (b % 4))) & 255];
  const unsigned hi = book[(labels[(b + 1) / 4] >> (8 * ((b + 1) % 4))) & 255];
  unsigned v = (lo >> sh) | (hi << (20 - sh));
  if (sh >= 12) v |= book[(labels[(b + 2) / 4] >> (8 * ((b + 2) % 4))) & 255] << (40 - sh);
  return v;
}
__device__ __forceinline__ void dense5_registers(const uint8_t* chunk, int lane, const uint32_t* book, unsigned* regs) {
  uint32_t labels[7];
#pragma unroll
  for(int p=0;p<6;p++) labels[p]=*(const uint32_t*)(chunk+p*128+lane*4);
  labels[6]=*(const uint16_t*)(chunk+768+lane*2);
  uint32_t expanded[26];
#pragma unroll
  for(int b=0;b<26;b++) expanded[b]=book[(labels[b/4]>>(8*(b%4)))&255];
#pragma unroll
  for(int w=0;w<16;w++) {
    const int b=(8*w)/5, sh=((8*w)%5)*4;
    uint32_t v=(expanded[b]>>sh)|(expanded[b+1]<<(20-sh));
    if(sh>=12) v|=expanded[b+2]<<(40-sh);
    regs[w]=v;
  }
}
}
