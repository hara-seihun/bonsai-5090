#include "split53.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
void require(bool v){if(!v)throw std::runtime_error("witness mismatch");}
std::vector<uint8_t> read(const char* p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("fixture unavailable");return {std::istreambuf_iterator<char>(f),{}};}
// Independent source PTQ decode and WF0 placement inherited from existing CPU witness.
std::array<int8_t,128> ptq(const uint8_t* b){
 std::array<int8_t,128> t{};int e=0;const int stages[2][2]={{0,16},{16,8}};
 for(auto& s:stages)for(int n=0;n<5;n++)for(int m=0;m<s[1];m++){unsigned q=b[s[0]+m];for(int k=0;k<n;k++)q=(q*3)&255;t[e++]=int((q*3)>>8)-1;}
 for(int n=0;n<4;n++)for(int h=0;h<2;h++){unsigned q=b[24+h];for(int k=0;k<n;k++)q=(q*3)&255;t[e++]=int((q*3)>>8)-1;}require(e==128);return t;
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
int main(int argc,char**argv){try{
 require(argc==3);unsigned admitted=0,rejected=0,words=0;
 for(unsigned b=0;b<8192;b++){
  auto x=joint53::Label::admit(b);
  if(std::holds_alternative<joint53::Error>(x)){rejected++;continue;}
  auto l=std::get<joint53::Label>(x);auto w=joint53::decode(l);auto back=joint53::Label::encode(w);
  require(std::get<joint53::Label>(back).bits()==b);admitted++;
 }
 require(admitted==6561 && rejected==1631);
 for(unsigned k=0;k<6561;k++){
  unsigned w=h5::dict8::word(k);auto label=std::get<joint53::Label>(joint53::Label::encode(w));require(joint53::decode(label)==w);
  // Old global dictionary and both tiny tables return exactly the same U32.
  require((joint53::low(label.bits()&255,5)|(joint53::low(label.bits()>>8,3)<<20))==w);
 }
 for(unsigned n=0;n<16;n++)if(n!=0&&n!=2&&n!=10)for(unsigned pos=0;pos<8;pos++)require(std::holds_alternative<joint53::Error>(joint53::Label::encode(n<<(4*pos))));
 auto g=read(argv[1]),u=read(argv[2]);require(g.size()==32*40*28&&u.size()==g.size());
 for(unsigned kb=0;kb<40;kb++){
  std::array<std::array<int8_t,128>,32> gt,ut;std::array<uint8_t,2048> source{};
  for(unsigned r=0;r<32;r++){gt[r]=ptq(g.data()+(r*40+kb)*28);ut[r]=ptq(u.data()+(r*40+kb)*28);}
  original(gt,source.data());original(ut,source.data()+1024);
  auto pair=std::get<joint53::Pair>(joint53::pack(source.data(),source.data()+1024));
  for(unsigned lane=0;lane<32;lane++)for(unsigned w=0;w<16;w++)for(unsigned arm=0;arm<2;arm++){
   unsigned exp=h5::dict8::native_word(source.data()+1024*arm,lane,w);
   auto label=std::get<joint53::Label>(joint53::extract(pair.data(),lane,w,arm));
   require(joint53::decode(label)==exp);words++;
  }
 }
 require(std::holds_alternative<joint53::Error>(joint53::Label::admit(8192)));
 require(std::holds_alternative<joint53::Error>(joint53::pack(nullptr,nullptr)));
 require(std::holds_alternative<joint53::Error>(joint53::extract(g.data(),32,0,0)));
 std::cout<<"{\"legal_labels\":"<<admitted<<",\"invalid_13bit_labels\":"<<rejected<<",\"real_native_words\":"<<words<<",\"mismatches\":0,\"gpu_work\":false}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
