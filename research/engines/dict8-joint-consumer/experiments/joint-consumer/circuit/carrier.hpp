#pragma once
#include <array>
#include <cstdint>
#include <variant>

namespace ternary5 {
using U = std::uint32_t;
using Triple = std::array<std::uint8_t,3>;
enum class Error { invalid_nibble, code_out_of_range, reserved_code, CircuitInvariant };
template<class T> using Result = std::variant<T,Error>;

// NVIDIA LOP3 indexing: bit 4*a+2*b+c of the immediate.
// This CPU implementation evaluates Boolean minterms, never a data lookup.
template<unsigned I> inline U lop(U a,U b,U c) {
    U r=0;
    if constexpr(I&1) r|=~a&~b&~c;
    if constexpr(I&2) r|=~a&~b& c;
    if constexpr(I&4) r|=~a& b&~c;
    if constexpr(I&8) r|=~a& b& c;
    if constexpr(I&16)r|= a&~b&~c;
    if constexpr(I&32)r|= a&~b& c;
    if constexpr(I&64)r|= a& b&~c;
    if constexpr(I&128)r|=a& b& c;
    return r;
}
struct Planes { U m0,s0,m1,s1,m2,s2; };
inline Planes circuit(U a,U b,U c,U d,U e) {
    U m0=lop<0xd5>(a,d,e);
    U s0=lop<0x82>(d,c,m0);
    U m1=lop<0x97>(b,d,e);
    U s1=lop<0xc0>(c,m1,0);
    U m2=lop<0x3c>(b,m1,0);
    U s2=lop<0x24>(a,b,m1);
    return {m0,s0,m1,s1,m2,s2};
}
inline Triple decode_raw(unsigned x) {
    auto z=circuit((x>>0)&1,(x>>1)&1,(x>>2)&1,(x>>3)&1,(x>>4)&1);
    return {std::uint8_t(((z.m0&1)<<1)|((z.s0&1)<<3)),
            std::uint8_t(((z.m1&1)<<1)|((z.s1&1)<<3)),
            std::uint8_t(((z.m2&1)<<1)|((z.s2&1)<<3))};
}
inline bool native(unsigned x) { return x==0 || x==2 || x==10; }
inline bool reserved(unsigned x) { return x==3 || x==7 || x==22 || x==23 || x==28; }
class Code5 {
    std::uint8_t value_;
    explicit Code5(std::uint8_t v):value_(v){}
public:
    unsigned value() const {return value_;}
    static Result<Code5> parse(unsigned x) {
        if(x>=32)return Error::code_out_of_range;
        if(reserved(x))return Error::reserved_code;
        return Code5(std::uint8_t(x));
    }
};
inline Triple decode(Code5 x) { return decode_raw(x.value()); }
inline Result<Code5> encode(Triple t) {
    for(auto v:t)if(!native(v))return Error::invalid_nibble;
    // Fixed preparation-only inversion of the six gates; no dictionary.
    for(unsigned x=0;x<32;x++)if(!reserved(x) && decode_raw(x)==t)return Code5::parse(x);
    return Error::CircuitInvariant;
}
struct Payload8 { U low4; std::uint8_t high1; };
// Serialized payload is low4 little-endian (4 bytes), then high1 (1 byte).
// sizeof(Payload8) is deliberately NOT the serialized representation.
inline Result<Payload8> encode24(const std::array<std::uint8_t,24>& source) {
    U lo=0; unsigned hi=0;
    for(unsigned j=0;j<8;j++) {
        auto r=encode({source[j],source[8+j],source[16+j]});
        if(auto e=std::get_if<Error>(&r))return *e;
        unsigned x=std::get<Code5>(r).value();
        lo|=(x&15)<<(4*j); hi|=(x>>4)<<j;
    }
    return Payload8{lo,std::uint8_t(hi)};
}
inline Result<Payload8> validate(Payload8 p) {
    for(unsigned j=0;j<8;j++) {
        unsigned x=((p.low4>>(4*j))&15)|(((p.high1>>j)&1)<<4);
        auto r=Code5::parse(x);
        if(auto e=std::get_if<Error>(&r))return *e;
    }
    return p;
}
inline U spread8(U x) {
    x=lop<0xa8>(x,x<<12,0x000f000f);
    x=lop<0xa8>(x,x<<6,0x03030303);
    x=lop<0xa8>(x,x<<3,0x11111111);
    return x;
}
inline U assemble(U m,U s) {
    U mag=m&0x22222222;
    U sign=s<<2;
    return lop<0xea>(sign,0x88888888,mag);
}
// 25 straight-line data operations: 3 aligns + 7 spread/align + 6 LUT + 9 pack.
// Precondition: validate(p) succeeded in the preparation/host boundary.
inline std::array<U,3> decode8_prevalidated(Payload8 p) {
    U a=p.low4<<1, b=p.low4, c=p.low4>>1, d=p.low4>>2;
    U e=spread8(p.high1)<<1;
    auto z=circuit(a,b,c,d,e);
    return {assemble(z.m0,z.s0),assemble(z.m1,z.s1),assemble(z.m2,z.s2)};
}
inline Result<U> encode_raw_pair(U gate,U up) {
    for(unsigned j=0;j<8;j++)if(!native((gate>>(4*j))&15)||!native((up>>(4*j))&15))return Error::invalid_nibble;
    return gate|(up>>1);
}
inline Result<std::array<U,2>> decode_raw_pair(U raw) {
    U gate=raw&0xaaaaaaaau;
    U up=(raw<<1)&0xaaaaaaaau;
    for(unsigned j=0;j<8;j++)if(!native((gate>>(4*j))&15)||!native((up>>(4*j))&15))return Error::reserved_code;
    return std::array<U,2>{gate,up};
}
inline U decode_tail8_prevalidated(U x) {
    x=lop<0xa8>(x,x<<8,0x00ff00ffu);
    x=lop<0xa8>(x,x<<4,0x0f0f0f0fu);
    x=lop<0xa8>(x,x<<2,0x33333333u);
    U mag=lop<0xc0>(x<<1,0x22222222u,0u);
    return lop<0xea>(x<<2,0x88888888u,mag);
}
inline Result<U> decode_tail8(unsigned x) {
    if(x>=65536)return Error::code_out_of_range;
    for(unsigned j=0;j<8;j++)if(((x>>(2*j))&3)==2)return Error::reserved_code;
    return decode_tail8_prevalidated(x);
}
inline Result<std::array<U,3>> decode8(Payload8 p) {
    auto r=validate(p);
    if(auto e=std::get_if<Error>(&r))return *e;
    return decode8_prevalidated(std::get<Payload8>(r));
}
}
