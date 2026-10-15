#pragma once
#include "model.hpp"
namespace source14 {
struct Packed {unsigned K,F;std::vector<uint8_t>raw,w;std::vector<uint16_t>s;};
inline Packed prepare(Weights weights){Packed r{weights.K,weights.F,std::move(weights.raw),{}, {}};unsigned nkb=r.K/128,ng=r.F/32;r.w.resize(uint64_t(nkb)*ng*512);r.s.resize(uint64_t(nkb)*r.F);
 for(unsigned kb=0;kb<nkb;kb++)for(unsigned g=0;g<ng;g++){
  for(unsigned tid=0;tid<32;tid++){unsigned row=g*32+16*((tid&3)>>1)+8*(tid&1)+tid/4;auto*p=r.raw.data()+(uint64_t(row)*nkb+kb)*18;r.s[(uint64_t(kb)*ng+g)*32+tid]=p[0]|uint16_t(p[1])<<8;}
  for(unsigned tid=0;tid<128;tid++){unsigned lane=tid/4,w=2*(tid&3),gid=lane>>2,tq=lane&3,row=g*32+16*(w>>2)+gid,h=(w>>1)&1;auto signs=[&](unsigned rr){auto*p=r.raw.data()+(uint64_t(rr)*nkb+kb)*18;unsigned b=2+8*h+tq;return unsigned(uint8_t(~p[b]))|(unsigned(uint8_t(~p[b+4]))<<8);};uint32_t word=signs(row)|(signs(row+8)<<16);std::memcpy(r.w.data()+(uint64_t(kb)*ng+g)*512+tid*4,&word,4);}
 }return r;
}
}
