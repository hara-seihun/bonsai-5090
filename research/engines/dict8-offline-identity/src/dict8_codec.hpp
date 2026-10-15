#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <variant>
#include <vector>
namespace h5::dict8 {
constexpr size_t entries=6561, pair_bytes=1664, native_pair_bytes=2048;
enum class Error { Shape, InvalidNativeNibble };
struct Key { uint16_t value; };
inline std::variant<Key,Error> key(uint32_t word) {
  unsigned k=0,p=1;
  for(unsigned j=0;j<8;j++,p*=3) {
    const unsigned n=(word>>(4*j))&15;
    unsigned d;
    switch(n) { case 10:d=0;break;case 0:d=1;break;case 2:d=2;break;default:return Error::InvalidNativeNibble; }
    k+=d*p;
  }
  return Key{static_cast<uint16_t>(k)};
}
constexpr uint32_t word(unsigned code) {
  uint32_t w=0;
  for(unsigned j=0;j<8;j++,code/=3)w|=(code%3==0?10u:code%3==1?0u:2u)<<(4*j);
  return w;
}
inline std::array<uint32_t,entries> dictionary() {
  std::array<uint32_t,entries> d{};
  for(unsigned k=0;k<entries;k++)d[k]=word(k);
  return d;
}
inline uint32_t native_word(const uint8_t* chunk,unsigned lane,unsigned w) {
  const unsigned mm=w/8,h=(w/4)%2,i=w%4;
  const unsigned v=mm*4+h*2+(i&1);
  uint32_t x;std::memcpy(&x,chunk+(v/4)*512+lane*16+(v%4)*4,4);
  return (x<<(i/2))&0xaaaaaaaau;
}
inline std::variant<std::array<uint8_t,pair_bytes>,Error> pack_pair(const uint8_t* gate,const uint8_t* up) {
  if(!gate||!up)return Error::Shape;
  std::array<uint8_t,pair_bytes> out{};
  for(unsigned lane=0;lane<32;lane++) {
    uint32_t labels[13]{};
    for(unsigned w=0;w<16;w++)for(unsigned arm=0;arm<2;arm++) {
      const auto result=key(native_word(arm?up:gate,lane,w));
      if(auto e=std::get_if<Error>(&result))return *e;
      const unsigned k=std::get<Key>(result).value,bit=26*w+13*arm,b=bit/32,s=bit%32;
      labels[b]|=k<<s;
      if(s>19)labels[b+1]|=k>>(32-s);
    }
    for(unsigned i=0;i<13;i++)std::memcpy(out.data()+i*128+lane*4,&labels[i],4);
  }
  return out;
}
inline uint16_t unpack_key(const uint8_t* pair,unsigned lane,unsigned w,unsigned arm) {
  const unsigned bit=26*w+13*arm,b=bit/32,s=bit%32;
  uint32_t lo,hi=0;std::memcpy(&lo,pair+b*128+lane*4,4);
  if(s>19)std::memcpy(&hi,pair+(b+1)*128+lane*4,4);
  return static_cast<uint16_t>(((uint64_t(hi)<<32|lo)>>s)&8191);
}
}
