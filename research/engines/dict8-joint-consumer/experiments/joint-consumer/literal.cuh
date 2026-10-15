#pragma once
// Six-LUT coordinate synthesized by joint-circuit-worker; exact nibble output.
// Host canonical inverse and exhaustive boundary witness live in circuit/carrier.hpp.
template<unsigned I> __device__ __forceinline__ unsigned lc_lop(unsigned a,unsigned b,unsigned c){
 unsigned v;asm("lop3.b32 %0,%1,%2,%3,%4;":"=r"(v):"r"(a),"r"(b),"r"(c),"n"(I));return v;
}
__device__ __forceinline__ unsigned lc_assemble(unsigned m,unsigned s){return lc_lop<0xea>(s<<2,0x88888888,m&0x22222222);}
__device__ __forceinline__ void lc_decode(unsigned L,unsigned H,unsigned (&out)[3]){
 unsigned a=L<<1,b=L,c=L>>1,d=L>>2;
 unsigned e=lc_lop<0xa8>(H,H<<12,0x000f000f);
 e=lc_lop<0xa8>(e,e<<6,0x03030303);
 e=lc_lop<0xa8>(e,e<<3,0x11111111)<<1;
 unsigned m0=lc_lop<0xd5>(a,d,e),s0=lc_lop<0x82>(d,c,m0);
 unsigned m1=lc_lop<0x97>(b,d,e),s1=lc_lop<0xc0>(c,m1,0);
 unsigned m2=lc_lop<0x3c>(b,m1,0),s2=lc_lop<0x24>(a,b,m1);
 out[0]=lc_assemble(m0,s0);out[1]=lc_assemble(m1,s1);out[2]=lc_assemble(m2,s2);
}
