#include "codec.hpp"
#include <cstdio>
#include <array>
#include <cstdint>
int main(){
 unsigned tested=0;
 for(unsigned x=0;x<59049;x++){
  unsigned q=x;int8_t g[5],u[5];uint32_t eg=0,eu=0;
  for(int i=0;i<5;i++){g[i]=int(q%3)-1;q/=3;u[i]=int(q%3)-1;q/=3;
   eg|=(g[i]<0?10:g[i]>0?2:0)<<(4*i);eu|=(u[i]<0?10:u[i]>0?2:0)<<(4*i);}
  unsigned c=joint9::encode5(g,u);uint32_t dg,du;joint9::decode5(c&255,c>>8,dg,du);
  if(dg!=eg||du!=eu)return 2;tested++;
 }
 std::array<unsigned,4096> visits{};
 for(int lane=0;lane<32;lane++)for(int w=0;w<16;w++)for(int j=0;j<8;j++){
  int mm=w/8,h=w/4%2,i=w%4;
  int row=16*mm+8*(i&1)+(lane>>2),k=64*h+32*(i>>1)+8*(lane&3)+j;
  if(row<0||row>=32||k<0||k>=128)return 3;
  visits[row*128+k]++;
 }
 for(unsigned n:visits)if(n!=1)return 4;
 std::printf("{\"joint_patterns\":%u,\"covered_pairs\":4096,\"coordinate_visits\":1,\"gpu_execution\":false}\n",tested);
}
