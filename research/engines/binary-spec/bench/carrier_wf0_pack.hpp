#pragma once
#include "compact.hpp"
#include <stdexcept>
#include <vector>
// Physical groups/K blocks and lanes are unchanged. Each native two-bit word
// holds two eight-nibble MMA registers interleaved in bits1,3 and bits0,2.
inline std::vector<uint8_t> carrier_pack_wf0(const std::vector<uint8_t>& dense) {
 if(dense.empty()||dense.size()%832)throw std::runtime_error("invalid dense5 packed extent");
 std::vector<uint8_t> native(dense.size()/832*1024);
 for(size_t chunk=0;chunk<dense.size()/832;chunk++)for(int lane=0;lane<32;lane++){
  const uint8_t*src=dense.data()+chunk*832;uint8_t*dst=native.data()+chunk*1024;
  uint32_t labels[26];
  for(int b=0;b<26;b++){size_t pos=b<24?(b/4)*128+lane*4+b%4:768+lane*2+b-24;labels[b]=dense5::book(src[pos]);}
  auto nibble=[&](int i){return (labels[i/5]>>(4*(i%5)))&15u;};
  for(int word=0;word<8;word++){
   uint32_t value=0;
   for(int j=0;j<8;j++)value|=(nibble(word*16+j)|(nibble(word*16+8+j)>>1))<<(4*j);
   // Independent full-word assembly must match the original tensor operands.
   uint32_t lo=0,hi=0;for(int j=0;j<8;j++){lo|=nibble(word*16+j)<<(4*j);hi|=nibble(word*16+8+j)<<(4*j);}
   if((value&0xAAAAAAAAu)!=lo||((value<<1)&0xAAAAAAAAu)!=hi)throw std::runtime_error("native carrier operand mismatch");
   size_t pos=(word>>2)*512+lane*16+(word&3)*4;memcpy(dst+pos,&value,4);
  }
 }
 return native;
}
