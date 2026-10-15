#define main inherited_main
#include "check.cpp"
#undef main
#include "circuit/carrier.hpp"
using LitPair=std::array<uint8_t,1728>;
LitPair make_literal(const uint8_t* g,const uint8_t* u){
 LitPair pair{};
 for(unsigned lane=0;lane<32;lane++){
  unsigned words[32];for(unsigned q=0;q<32;q++)words[q]=h5::dict8::native_word(q%2?u:g,lane,q/2);
  for(unsigned group=0;group<10;group++){
   std::array<uint8_t,24> trits{};
   for(unsigned j=0;j<3;j++)for(unsigned t=0;t<8;t++)trits[8*j+t]=(words[3*group+j]>>(4*t))&15;
   auto enc=ternary5::encode24(trits);require(std::holds_alternative<ternary5::Payload8>(enc));
   auto code=std::get<ternary5::Payload8>(enc);
   std::memcpy(pair.data()+128*group+4*lane,&code.low4,4);pair[1280+32*group+lane]=code.high1;
  }
  unsigned tail=words[30]|(words[31]>>1);
  std::memcpy(pair.data()+1600+4*lane,&tail,4);
 }
 return pair;
}
int main(int argc,char**argv){try{
 require(argc==3);auto g=read(argv[1]),u=read(argv[2]);require(g.size()==35840&&u.size()==35840);unsigned count=0;
 for(unsigned kb=0;kb<40;kb++){
  std::array<std::array<int8_t,128>,32> gt,ut;std::array<uint8_t,2048> source{};
  for(unsigned r=0;r<32;r++){gt[r]=ptq(g.data()+(r*40+kb)*28);ut[r]=ptq(u.data()+(r*40+kb)*28);}
  original(gt,source.data());original(ut,source.data()+1024);
  auto pair=make_literal(source.data(),source.data()+1024);
  for(unsigned lane=0;lane<32;lane++){
   unsigned decoded[32];
   for(unsigned group=0;group<10;group++){
    unsigned L;std::memcpy(&L,pair.data()+128*group+4*lane,4);unsigned H=pair[1280+32*group+lane];
    auto words=std::get<std::array<unsigned,3>>(ternary5::decode8({L,uint8_t(H)}));
    for(unsigned j=0;j<3;j++)decoded[3*group+j]=words[j];
   }
   unsigned tail;std::memcpy(&tail,pair.data()+1600+4*lane,4);decoded[30]=tail&0xaaaaaaaa;decoded[31]=(tail<<1)&0xaaaaaaaa;
   for(unsigned q=0;q<32;q++){require(decoded[q]==h5::dict8::native_word(source.data()+1024*(q%2),lane,q/2));count++;}
  }
 }
 std::cout<<"{\"literal_layout_pair_bytes\":1728,\"real_native_words\":"<<count<<",\"mismatches\":0,\"gpu_execution\":false}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
