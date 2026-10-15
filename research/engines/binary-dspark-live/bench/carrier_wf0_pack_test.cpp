#include "carrier_wf0_pack.hpp"
#include <cstdio>
int main(){
 std::vector<uint8_t>dense(256*832);
 for(size_t i=0;i<dense.size();i++)dense[i]=i%256;
 auto native=carrier_pack_wf0(dense);
 size_t checked=0;
 for(int chunk=0;chunk<256;chunk++)for(int lane=0;lane<32;lane++)for(int word=0;word<8;word++){
  uint32_t value;memcpy(&value,native.data()+chunk*1024+(word>>2)*512+lane*16+(word&3)*4,4);
  for(int half=0;half<2;half++)for(int j=0;j<8;j++){
   int i=word*16+half*8+j,b=i/5,sh=i%5;size_t pos=b<24?(b/4)*128+lane*4+b%4:768+lane*2+b-24;
   unsigned q=dense[chunk*832+pos];for(int k=0;k<sh;k++)q/=3;int t=int(q%3)-1;
   unsigned got=((value<<(half?1:0))>>(4*j))&10u;
   if(got!=dense5::fp4(t))throw std::runtime_error("direct trit mismatch");checked++;
  }
 }
 bool rejected=false;try{carrier_pack_wf0(std::vector<uint8_t>(833));}catch(const std::runtime_error&){rejected=true;}if(!rejected)return 1;
 printf("native operands exact: %zu trits including all256 labels; invalid extent rejected\n",checked);
}
