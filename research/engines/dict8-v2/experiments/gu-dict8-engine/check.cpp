#include "dict8_codec.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
using namespace h5;
std::vector<uint8_t> read(const char* path){std::ifstream f(path,std::ios::binary);assert(f);return {std::istreambuf_iterator<char>(f),{}};}
std::array<int8_t,128> decode(const uint8_t* b){
 std::array<int8_t,128> t{};int e=0;const int stages[2][2]={{0,16},{16,8}};
 for(auto& s:stages)for(int n=0;n<5;n++)for(int m=0;m<s[1];m++){unsigned q=b[s[0]+m];for(int k=0;k<n;k++)q=(q*3)&255;t[e++]=int((q*3)>>8)-1;}
 for(int n=0;n<4;n++)for(int h=0;h<2;h++){unsigned q=b[24+h];for(int k=0;k<n;k++)q=(q*3)&255;t[e++]=int((q*3)>>8)-1;}assert(e==128);return t;
}
void original(const std::array<std::array<int8_t,128>,32>& t,uint8_t* dst){
 for(int l=0;l<32;l++)for(int w=0;w<8;w++) {
  const int mm=w/4,h=(w/2)%2,r=w%2,row=16*mm+8*r+l/4;uint32_t v=0;
  for(int j=0;j<8;j++) {
   const int a=t[row][64*h+8*(l%4)+j],b=t[row][64*h+32+8*(l%4)+j];
   v|=uint32_t((a!=0)*2+(a<0)*8+(b!=0)+(b<0)*4)<<(4*j);
  }
  std::memcpy(dst+(w/4)*512+l*16+(w%4)*4,&v,4);
 }
}
int main(int argc,char** argv){
 assert(argc==4);auto book=dict8::dictionary();
 for(unsigned k=0;k<6561;k++){auto v=dict8::key(book[k]);assert(std::holds_alternative<dict8::Key>(v));assert(std::get<dict8::Key>(v).value==k);}
 for(unsigned n=0;n<16;n++)if(n!=0&&n!=2&&n!=10)assert(std::holds_alternative<dict8::Error>(dict8::key(n)));
 auto g=read(argv[1]),u=read(argv[2]),expected=read(argv[3]);assert(g.size()==32*40*28&&u.size()==g.size()&&expected.size()==40*1664);
 unsigned words=0;
 for(int kb=0;kb<40;kb++) {
  std::array<std::array<int8_t,128>,32> gt,ut;std::array<uint8_t,2048> source{};
  for(int r=0;r<32;r++){gt[r]=decode(g.data()+(r*40+kb)*28);ut[r]=decode(u.data()+(r*40+kb)*28);}
  original(gt,source.data());original(ut,source.data()+1024);
  auto result=dict8::pack_pair(source.data(),source.data()+1024);assert(!std::holds_alternative<dict8::Error>(result));
  const auto& packed=std::get<std::array<uint8_t,dict8::pair_bytes>>(result);
  assert(std::memcmp(packed.data(),expected.data()+kb*1664,1664)==0);
  for(unsigned lane=0;lane<32;lane++)for(unsigned w=0;w<16;w++)for(unsigned arm=0;arm<2;arm++) {
   unsigned k=dict8::unpack_key(packed.data(),lane,w,arm);assert(k<6561);assert(book[k]==dict8::native_word(source.data()+1024*arm,lane,w));words++;
  }
 }
 assert(std::holds_alternative<dict8::Error>(dict8::pack_pair(nullptr,nullptr)));
 std::cout<<"{\"dictionary_codes\":6561,\"original_native_words\":"<<words<<",\"fixture_bytes\":66560,\"mismatches\":0,\"gpu_work\":false}\n";
}
