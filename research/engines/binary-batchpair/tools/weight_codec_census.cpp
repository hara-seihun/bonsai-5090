// Exact source-block census for a lossless nonzero-mask/sign representation.
// This measures payload opportunity; no decoding-time or GPU speed claim.
#include "gguf.h"
#include "compact.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <cstdint>
using Hist = std::array<uint64_t,129>;
struct Counts { Hist nnz{}; std::array<uint64_t,3> trits{}; uint64_t blocks=0; };
static std::array<unsigned,256> count5{}, count4{};
static std::array<std::array<unsigned,3>,256> hist5{}, hist4{};
static std::array<unsigned,129> mask_bits{};
static void init() {
 for(unsigned q=0;q<256;q++) {
  unsigned v=q;
  for(unsigned j=0;j<5;j++) { unsigned d=(v*3)>>8; hist5[q][d]++; if(j<4)hist4[q][d]++; v=(v*3)&255; }
  count5[q]=5-hist5[q][1]; count4[q]=4-hist4[q][1];
 }
}
static void exact_mask_bits() {
 // Each Pascal row is positive and bounded by row128's maximum, <2^125.
 std::array<unsigned __int128,129> p{};p[0]=1;
 for(unsigned n=1;n<=128;n++)for(unsigned k=n;k>0;k--)p[k]+=p[k-1];
 for(unsigned m=0;m<=128;m++){auto z=p[m]-1;unsigned b=0;while(z){b++;z>>=1;}mask_bits[m]=b;}
}
static void add(Counts& c,const uint8_t* p) {
 unsigned nz=0;
 for(unsigned j=0;j<26;j++) {
  auto& h=j<24?hist5[p[j]]:hist4[p[j]];
  nz+=j<24?count5[p[j]]:count4[p[j]];
  for(unsigned d=0;d<3;d++)c.trits[d]+=h[d];
 }
 if(nz>128)throw std::runtime_error("invalid nonzero count");
 c.nnz[nz]++;c.blocks++;
}
static void emit(const char* name,const Counts& c) {
 uint64_t ranked_bits=0,mask_sign_bits=0;double entropy=0;
 for(unsigned m=0;m<=128;m++) {
  ranked_bits+=c.nnz[m]*(8+mask_bits[m]+m);
  mask_sign_bits+=c.nnz[m]*(128+m);
 }
 double total=double(c.blocks)*128;
 for(auto n:c.trits)if(n){double p=n/total;entropy-=p*std::log2(p);}
 printf("{\"tensor\":\"%s\",\"blocks\":%llu,\"trits\":[%llu,%llu,%llu],\"dense5_bits\":%llu,\"rank_mask_sign_bits\":%llu,\"raw_mask_sign_bits\":%llu,\"marginal_trit_entropy\":%.12g,\"nnz_hist\":[",name,(unsigned long long)c.blocks,(unsigned long long)c.trits[0],(unsigned long long)c.trits[1],(unsigned long long)c.trits[2],(unsigned long long)(c.blocks*208),(unsigned long long)ranked_bits,(unsigned long long)mask_sign_bits,entropy);
 for(unsigned m=0;m<=128;m++)printf("%s%llu",m?",":"",(unsigned long long)c.nnz[m]);puts("]}");
}
int main(int argc,char**argv){try{
 if(argc!=2){fprintf(stderr,"usage: weight-codec-census MODEL.gguf\n");return 2;}
 init();exact_mask_bits();
 for(unsigned q=0;q<256;q++) {
  uint8_t b[28]{}; for(unsigned j=0;j<26;j++)b[j]=q;
  int8_t t[128];uint16_t s;dense5::decode_ptq(b,t,s);std::array<unsigned,3> h{};
  for(auto v:t)h[unsigned(v+1)]++;
  for(unsigned d=0;d<3;d++)if(h[d]!=24*hist5[q][d]+2*hist4[q][d])throw std::runtime_error("table/source decoder mismatch");
 }
 h5::Gguf gg;gg.open(argv[1]);Counts total;
 for(auto& t:gg.tensors)if(t.type==143&&t.ne.size()==2){
  if(t.ne[0]%128||t.bytes!=t.ne[0]*t.ne[1]/128*28)throw std::runtime_error("invalid PTQ block extent");
  Counts c;for(size_t off=0;off<t.bytes;off+=28)add(c,t.data+off);
  emit(t.name.c_str(),c);total.blocks+=c.blocks;
  for(unsigned m=0;m<=128;m++)total.nnz[m]+=c.nnz[m];for(unsigned d=0;d<3;d++)total.trits[d]+=c.trits[d];
 }
 if(!total.blocks)throw std::runtime_error("no PTQ matrices");emit("TOTAL",total);
 if(ferror(stdout))throw std::runtime_error("output write failed");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
