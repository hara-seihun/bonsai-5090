#include <array>
#include <bit>
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
struct Candidate {std::string name;int bits;int kind;int argument;};
uint32_t label(int d,const Candidate& c){
  switch(c.kind){
    case 0:return unsigned(d)&((1u<<c.argument)-1);
    case 1:return unsigned((d+c.argument)%c.bits);
    case 2:return uint32_t(3*d+c.argument);
    case 3:return std::bit_cast<uint32_t>(float(d));
    case 4:{float rounded=std::nearbyint(float(d)/float(1<<c.argument));return std::bit_cast<uint32_t>(rounded);}
  }
  __builtin_unreachable();
}
int main(){
  if(std::fesetround(FE_TONEAREST)!=0)return 2;
  constexpr int bound=128*127;
  std::vector<Candidate> candidates;
  for(int bits=8;bits<=16;bits++)candidates.push_back({"mask",bits,0,bits});
  for(int modulus:{255,256,2047,2048,8191,8192,16381,16384,32513,32768})candidates.push_back({"residue",modulus,1,2*bound});
  for(int bias:{-16384,0,16384})candidates.push_back({"affine_3x",32,2,bias});
  candidates.push_back({"native_f32_integer",32,3,0});
  for(int shift:{1,2,3,4})candidates.push_back({"rounded_shift",32,4,shift});
  int matches=0;
  for(auto c:candidates){std::map<uint32_t,int> seen;bool match=true;int left=0,right=0;
    for(int d=-bound;d<=bound;d++){auto [i,inserted]=seen.emplace(label(d,c),d);if(!inserted&&i->second!=d){match=false;left=i->second;right=d;break;}}
    if(match)matches++;
    printf("{\"grammar\":\"%s\",\"parameter\":%d,\"bias_or_shift\":%d,\"match\":%s,\"left\":%d,\"right\":%d,\"checked_domain\":[%d,%d],\"target\":\"integer_K128_dot_up_to_relabelling\"}\n",c.name.c_str(),c.bits,c.argument,match?"true":"false",left,right,-bound,bound);
  }
  // Whole finite suffix: one variable gate dot, up=1, remaining127 hidden lanes=1.
  // Original half scale and dynamic activation scale enter through the effective scale.
  for(float scale:{1.f/1024,1.f/128,1.f/16,1.f}){
    std::map<uint64_t,int> classes;
    for(int d=-bound;d<=bound;d++){
      float gate=float(d)*scale,hidden=gate/(1.f+std::exp(-gate));
      float maximum=std::fmax(1.f,std::fabs(hidden));
      int q=int(std::nearbyint(hidden*(127.f/maximum)));
      uint64_t key=(uint64_t(std::bit_cast<uint32_t>(maximum/127.f))<<32)|uint32_t(q);
      classes.emplace(key,d);
    }
    printf("{\"whole_suffix_fibres\":%zu,\"domain_size\":%d,\"effective_scale\":%.9g,\"observer\":\"A8_q_and_dynamic_block_scale\"}\n",classes.size(),2*bound+1,scale);
  }
  for(auto c:candidates){
    bool match=true;int left=0,right=0;float distinguishing_scale=0;
    for(float scale:{1.f/1024,1.f/128,1.f/16,1.f}){
      std::map<uint32_t,std::pair<int,uint64_t>> seen;
      for(int d=-bound;d<=bound;d++){
        float gate=float(d)*scale,hidden=gate/(1.f+std::exp(-gate));
        float maximum=std::fmax(1.f,std::fabs(hidden));int q=int(std::nearbyint(hidden*(127.f/maximum)));
        uint64_t key=(uint64_t(std::bit_cast<uint32_t>(maximum/127.f))<<32)|uint32_t(q);
        auto [i,inserted]=seen.emplace(label(d,c),std::pair{d,key});
        if(!inserted&&i->second.second!=key){match=false;left=i->second.first;right=d;distinguishing_scale=scale;break;}
      }
      if(!match)break;
    }
    printf("{\"whole_suffix_candidate\":\"%s\",\"parameter\":%d,\"bias_or_shift\":%d,\"sufficient\":%s,\"left\":%d,\"right\":%d,\"distinguishing_effective_scale\":%.9g}\n",c.name.c_str(),c.bits,c.argument,match?"true":"false",left,right,distinguishing_scale);
  }
  // Two K128 groups: raw sum labels collide before original row-dependent scales.
  const int first0=1,first1=0,second0=0,second1=1;
  const float scale0=.5f,scale1=1.f;
  float a=std::fma(float(first1),scale1,std::fma(float(first0),scale0,0.f));
  float b=std::fma(float(second1),scale1,std::fma(float(second0),scale0,0.f));
  float ha=(a/(1.f+std::exp(-a))),hb=(b/(1.f+std::exp(-b)));
  // The remaining127 output lanes are1, so requant preserves different first bytes too.
  int qa=int(std::nearbyint(ha*127.f)),qb=int(std::nearbyint(hb*127.f));
  if(first0+first1!=second0+second1||a==b||qa==qb)return 3;
  printf("{\"scale_blind_sum_rejection\":true,\"dots_A\":[1,0],\"dots_B\":[0,1],\"effective_scales\":[0.5,1],\"up\":1,\"scaled_gate_A\":%.9g,\"scaled_gate_B\":%.9g,\"SiLU_A\":%.9g,\"SiLU_B\":%.9g,\"observer_q_A\":%d,\"observer_q_B\":%d}\n",a,b,ha,hb,qa,qb);
  // Exhaust the ISA nibble multiply/AND composition over four arbitrary signed bytes.
  for(unsigned n=0;n<16;n++){unsigned expanded=(n*0x00204081u)&0x01010101u;
    for(int lane=0;lane<4;lane++)for(int q=-128;q<=127;q++){
      int selected=((expanded>>(8*lane))&255)*q,dc=q,dot=2*selected-dc;
      if(dot!=((n>>lane)&1?q:-q))return 4;
    }
  }
  fprintf(stderr,"PASS %zu candidate labels, %d exact fibre matches, 16384 nibble-byte cases; scale-blind endpoint counterexample retained\n",candidates.size(),matches);
}
