#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <variant>
#include "dict8_codec.hpp"
namespace joint53 {
enum class Error { Shape, NativeNibble, InvalidLabel };
class Label {
  uint16_t bits_;
  explicit Label(unsigned bits):bits_(uint16_t(bits)){}
 public:
  unsigned bits()const{return bits_;}
  static std::variant<Label,Error> admit(unsigned bits){
    if(bits>=8192 || (bits&255)>=243 || (bits>>8)>=27)return Error::InvalidLabel;
    return Label(bits);
  }
  static std::variant<Label,Error> encode(uint32_t native){
    auto k=h5::dict8::key(native);
    if(std::holds_alternative<h5::dict8::Error>(k))return Error::NativeNibble;
    auto code=std::get<h5::dict8::Key>(k).value;
    return Label((code%243)|((code/243)<<8));
  }
};
constexpr uint32_t low(unsigned q,unsigned n){
  uint32_t r=0;for(unsigned j=0;j<n;j++,q/=3)r|=(q%3==0?10u:q%3==1?0u:2u)<<(4*j);return r;
}
inline uint32_t decode(Label label){return low(label.bits()&255,5)|(low(label.bits()>>8,3)<<20);}
using Pair=std::array<uint8_t,1664>;
inline std::variant<Pair,Error> pack(const uint8_t* g,const uint8_t* u){
 if(!g||!u)return Error::Shape;
 Pair out{};
 for(unsigned lane=0;lane<32;lane++){
  uint32_t plane[13]{};
  for(unsigned w=0;w<16;w++)for(unsigned arm=0;arm<2;arm++){
   auto enc=Label::encode(h5::dict8::native_word(arm?u:g,lane,w));
   if(auto e=std::get_if<Error>(&enc))return *e;
   unsigned bits=std::get<Label>(enc).bits(),i=2*w+arm,b=8+(5*i)/32,s=(5*i)%32;
   plane[i/4]|=(bits&255)<<(8*(i%4));
   plane[b]|=(bits>>8)<<s;
   if(s>27)plane[b+1]|=(bits>>8)>>(32-s);
  }
  for(unsigned p=0;p<13;p++)std::memcpy(out.data()+128*p+4*lane,plane+p,4);
 }
 return out;
}
inline std::variant<Label,Error> extract(const uint8_t* pair,unsigned lane,unsigned w,unsigned arm){
 if(!pair||lane>=32||w>=16||arm>=2)return Error::Shape;
 unsigned i=2*w+arm,b=8+5*i/32,s=5*i%32;uint32_t p[13];
 for(unsigned j=0;j<13;j++)std::memcpy(p+j,pair+128*j+4*lane,4);
 uint32_t hi=p[b]>>s;if(s>27)hi|=p[b+1]<<(32-s);
 return Label::admit(((p[i/4]>>(8*(i%4)))&255)|((hi&31)<<8));
}
}
