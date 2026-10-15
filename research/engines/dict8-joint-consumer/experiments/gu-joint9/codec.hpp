#pragma once
#include <cstdint>
namespace joint9 {
// Five GU pairs = ten base-three digits, two independent five-digit bytes.
// Low/high digit order g0,u0,g1,u1,g2 / u2,g3,u3,g4,u4.
// These are joint coordinates; multiplication still uses separate gate/up words.
inline uint32_t table(unsigned c,bool odd){
 uint32_t v=0;unsigned g=0,u=0;
 for(unsigned j=0;j<5;j++,c/=3){unsigned d=c%3,q=d==0?10:d==2?2:0;
  if(((j+unsigned(odd))&1)==0){v|=q<<(4*g++);}else{v|=q<<(16+4*u++);}}
 // odd table stores g2 low/u3 high; even stores g3 low/u2 high.
 return v;
}
inline void decode5(unsigned lo,unsigned hi,uint32_t&g,uint32_t&u){
 unsigned a=table(lo,false),b=table(hi,true);
 g=(a&0xfff)|((b&0xff)<<12);u=(a>>16)|((b>>16)<<8);
}
inline unsigned encode5(const int8_t*g,const int8_t*u){
 unsigned lo=0,hi=0,p=1;
 for(int j=0;j<5;j++){int pair=j/2;int t=j%2?u[pair]:g[pair];lo+=(t+1)*p;p*=3;}
 p=1;
 for(int j=0;j<5;j++){int t=j%2?g[3+j/2]:u[2+j/2];hi+=(t+1)*p;p*=3;}
 return lo|(hi<<8);
}
}
