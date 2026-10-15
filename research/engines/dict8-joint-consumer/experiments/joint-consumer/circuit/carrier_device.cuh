#pragma once
#include <stdint.h>
// Host validates canonical codes {0..31}\{3,7,22,23,28} before admission.
// No runtime dictionary, shared memory, or data-dependent indexing.
namespace ternary5_device {
struct Words3 { uint32_t w0,w1,w2; };
template<unsigned IMM> __device__ __forceinline__ uint32_t lop(uint32_t a,uint32_t b,uint32_t c) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(r) : "r"(a),"r"(b),"r"(c),"n"(IMM));
    return r;
}
__device__ __forceinline__ uint32_t assemble(uint32_t m,uint32_t s) {
    uint32_t mag=lop<0xc0>(m,0x22222222u,0u);
    return lop<0xea>(s<<2,0x88888888u,mag);
}
// L has low four code bits in each nibble; H low byte has the fifth bits.
// Outputs reproduce consecutive source nibbles [0..7],[8..15],[16..23].
// Source triple j=(source[j],source[8+j],source[16+j]).
// 25 specified PTX-level data ops, excluding materialization/moves/loads/addressing:
// 10 shifts,15 LOP3; no PRMT. Compiler scheduling/register allocation unmeasured.
__device__ __forceinline__ Words3 decode8(uint32_t L,uint32_t H) {
    uint32_t a=L<<1, b=L, c=L>>1, d=L>>2;
    uint32_t e=lop<0xa8>(H,H<<12,0x000f000fu);
    e=lop<0xa8>(e,e<<6,0x03030303u);
    e=lop<0xa8>(e,e<<3,0x11111111u);
    e<<=1;
    uint32_t m0=lop<0xd5>(a,d,e);
    uint32_t s0=lop<0x82>(d,c,m0);
    uint32_t m1=lop<0x97>(b,d,e);
    uint32_t s1=lop<0xc0>(c,m1,0u);
    uint32_t m2=lop<0x3c>(b,m1,0u);
    uint32_t s2=lop<0x24>(a,b,m1);
    return {assemble(m0,s0),assemble(m1,s1),assemble(m2,s2)};
}
struct Words2 { uint32_t gate,up; };
// Selected pair-level tail: raw=gateWord|(upWord>>1), 3 data operations.
__device__ __forceinline__ Words2 decode_raw_pair(uint32_t raw) {
    return {lop<0xc0>(raw,0xaaaaaaaau,0u),
            lop<0xc0>(raw<<1,0xaaaaaaaau,0u)};
}
// Alternative compact per-operand tail, NOT the selected 1728B pair layout.
// Tail code has eight canonical m|(s<<1) fields (00,01,11), low 16 bits only.
// Ten specified data ops: five shifts and five LOP3.
__device__ __forceinline__ uint32_t decode_tail8(uint32_t x) {
    x=lop<0xa8>(x,x<<8,0x00ff00ffu);
    x=lop<0xa8>(x,x<<4,0x0f0f0f0fu);
    x=lop<0xa8>(x,x<<2,0x33333333u);
    uint32_t mag=lop<0xc0>(x<<1,0x22222222u,0u);
    return lop<0xea>(x<<2,0x88888888u,mag);
}
}
