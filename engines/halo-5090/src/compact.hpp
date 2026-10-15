#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
namespace dense5 {
inline uint32_t fp4(int t) { return t == 0 ? 0u : t < 0 ? 10u : 2u; }
inline uint32_t book(unsigned q) {
  uint32_t out = 0;
  for (unsigned j = 0; j < 5; j++, q /= 3) out |= fp4(int(q % 3) - 1) << (4 * j);
  return out;
}
inline void decode_ptq(const uint8_t* b, int8_t* t, uint16_t& s) {
  int at=0;
  for(int stage=0;stage<3;stage++) {
    int off=stage==0?0:stage==1?16:24, count=stage==0?16:stage==1?8:2, digits=stage==2?4:5;
    for(int j=0;j<digits;j++) for(int k=0;k<count;k++) {
      unsigned q=b[off+k]; for(int a=0;a<j;a++)q=(q*3)&255;
      t[at++]=int((q*3)>>8)-1;
    }
  }
  memcpy(&s,b+26,2);
}
// A lane's sixteen native 8-nibble E2M1 operand registers become one 128-trit stream.
// Six uint32 planes plus one uint16 plane keep every load warp-coalesced.
inline void pack_lane(uint8_t* chunk,int lane,const int8_t* t) {
  for(int byte=0;byte<26;byte++) {
    unsigned q=0,p=1;
    for(int j=0;j<5;j++,p*=3) q+=unsigned(byte*5+j<128?t[byte*5+j]+1:1)*p;
    size_t pos=byte<24?(byte/4)*128+lane*4+byte%4:768+lane*2+byte-24;
    chunk[pos]=uint8_t(q);
  }
}
}
