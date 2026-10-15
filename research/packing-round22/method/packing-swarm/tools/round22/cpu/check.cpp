#include "../../round17/native/packed.hpp"
#include "../../round18/grammar.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <iostream>
#include <omp.h>
constexpr unsigned K=5120,KB=40,SW=4096,SWS=512,SA=2048,SAS=256,STAGE=9216;
namespace witness22 {
enum class Error {arguments=2,input,model,weights,coordinates,words,fold,owner};
static int fail(Error e){std::cerr<<"{\"state\":\"CPU_refused\",\"typed_error\":"<<unsigned(e)<<"}\n";return unsigned(e);}
static float mul(float a,float b){volatile float c=a*b;return c;}
static float add(float a,float b){volatile float c=a+b;return c;}
static float e2(unsigned x){constexpr float v[8]={0,.5,1,1.5,2,3,4,6};return x&8?-v[x&7]:v[x&7];}
static float e4(unsigned x){unsigned e=(x>>3)&15,m=x&7;float f=e?std::ldexp(float(8+m),int(e)-10):std::ldexp(float(m),-9);return x&128?-f:f;}
static uint64_t fnv(const void*p,size_t n){auto*b=static_cast<const uint8_t*>(p);uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;i++){h^=b[i];h*=1099511628211ull;}return h;}
template<class T>static uint64_t fnv(const std::vector<T>&v){return fnv(v.data(),v.size()*sizeof(T));}
static void feed(uint64_t&h,uint32_t w){for(unsigned i=0;i<4;i++){h^=(w>>(8*i))&255;h*=1099511628211ull;}}
template<class T>static bool input(std::vector<T>&v,size_t n){v.resize(n);std::cin.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));return bool(std::cin);}
struct Tape {std::vector<uint8_t>q,s;std::vector<float>g;};
struct LUT {std::vector<int32_t>raw,native;};
static uint64_t qoff(unsigned row,unsigned kb,unsigned k){return (uint64_t(kb)*64+row)*64+(((k/32)^((row>>1)&3))*16)+k%32/2;}
static LUT tables(const Tape&t,unsigned B){LUT l{std::vector<int32_t>(uint64_t(B)*KB*16*256),std::vector<int32_t>(uint64_t(B)*KB*32*16)};
 for(unsigned row=0;row<B;row++)for(unsigned kb=0;kb<KB;kb++){
  int32_t x[128];for(unsigned k=0;k<128;k++){float v=mul(e2((t.q[qoff(row,kb,k)]>>(4*(k&1)))&15),e4(t.s[(uint64_t(kb)*64+row)*8+k/16]));float grid=std::ldexp(v,10);if(!std::isfinite(grid)||grid!=float(int32_t(grid)))std::abort();x[k]=int32_t(grid);}
  size_t a=(size_t(row)*KB+kb)*16*256,b=(size_t(row)*KB+kb)*32*16;
  for(unsigned block=0;block<16;block++)for(unsigned code=0;code<256;code++){int32_t d=0;for(unsigned k=0;k<8;k++)d+=code&(1u<<k)?x[block*8+k]:-x[block*8+k];l.raw[a+block*256+code]=d;}
  for(unsigned block=0;block<32;block++)for(unsigned code=0;code<16;code++){int32_t d=0;for(unsigned k=0;k<4;k++)d+=code&(1u<<k)?-x[block*4+k]:x[block*4+k];l.native[b+block*16+code]=d;}
 }return l;}
struct Output {unsigned f,t;};
static std::vector<Output> candidate(unsigned g0,unsigned ng,unsigned B,std::vector<unsigned>&seen,uint64_t&inactive){std::vector<Output> v;for(unsigned warp=0;warp<4;warp++)for(unsigned q=0;q<2;q++)for(unsigned lane=0;lane<32;lane++)for(unsigned n=0;n<4;n++)for(unsigned j=0;j<2;j++)for(unsigned mm=0;mm<2;mm++)for(unsigned r=0;r<2;r++){
 unsigned g=g0+warp*2+q,t=n*8+(lane&3)*2+j,f=g*32+mm*16+r*8+(lane>>2);if(g>=ng||t>=B){++inactive;continue;}++seen[size_t(t)*ng*32+f];v.push_back({f,t});}return v;}
static std::vector<Output> strongest(unsigned g0,unsigned ng,unsigned B,std::vector<unsigned>&seen,uint64_t&inactive){std::vector<Output> v;for(unsigned warp=0;warp<8;warp++)for(unsigned n=0;n<4;n++)for(unsigned lane=0;lane<32;lane++)for(unsigned m=0;m<2;m++)for(unsigned r=0;r<2;r++)for(unsigned j=0;j<2;j++){
 unsigned group=g0+warp,feature=group*32+m*16+r*8+lane/4,token=n*8+(lane%4)*2+j;if(group>=ng||token>=B){++inactive;continue;}++seen[size_t(token)*ng*32+feature];v.push_back({feature,token});}return v;}
struct Counts {uint64_t stage_q=0,stage_s=0,stage_w=0,stage_ws=0,global_scale_words=0,folds=0,outputs=0,owners=0,inactive=0,half_dots=0;};
static bool project(const std::array<Tape,2>&tapes,const std::array<LUT,2>&lut,const source14::Packed&p,unsigned B,bool cand,const std::array<std::vector<uint32_t>,3>&saved,const std::array<std::vector<uint64_t>,3>&traces,Counts&counts){
 unsigned ng=p.F/32,ncta=(ng+7)/8;std::vector<std::vector<Output>> dictionary(ncta);std::vector<unsigned> seen(size_t(B)*p.F);
 for(unsigned cta=0;cta<ncta;cta++)dictionary[cta]=cand?candidate(cta*8,ng,B,seen,counts.inactive):strongest(cta*8,ng,B,seen,counts.inactive);
 if(!std::all_of(seen.begin(),seen.end(),[](unsigned n){return n==1;}))return false;
 std::atomic<bool>good{true};
 #pragma omp parallel for num_threads(8) schedule(static)
 for(unsigned cta=0;cta<ncta;cta++){
  unsigned g0=cta*8,live=std::min(8u,ng-g0);const auto&dict=dictionary[cta];size_t extent=dict.size();
  std::array<std::vector<float>,2>acc{std::vector<float>(extent),std::vector<float>(extent)},rawacc=acc;
  std::array<std::vector<uint64_t>,2>digest{std::vector<uint64_t>(extent,14695981039346656037ull),std::vector<uint64_t>(extent,14695981039346656037ull)};
  std::array<std::array<uint8_t,STAGE>,8>ring{};
  std::array<std::array<float,32>,2>shared_globals{};
  for(unsigned pass=0;pass<2;pass++)for(unsigned token=0;token<B;token++)shared_globals[pass][token]=tapes[pass].g[token];
  std::array<std::vector<unsigned>,2>owners{std::vector<unsigned>(extent,~0u),std::vector<unsigned>(extent,~0u)};
  for(unsigned kb=0;kb<KB;kb++){
   auto&st=ring[kb%8];std::fill(st.begin(),st.end(),0xcd);
   std::memcpy(st.data(),p.w.data()+(size_t(kb)*ng+g0)*512,live*512);
   std::memcpy(st.data()+SW,p.s.data()+(size_t(kb)*ng+g0)*32,live*64);
   for(unsigned pass=0;pass<2;pass++){
    unsigned qo=SW+SWS+pass*(SA+SAS),so=qo+SA;
    std::memcpy(st.data()+qo,tapes[pass].q.data()+size_t(kb)*64*64,SA);
    std::memcpy(st.data()+so,tapes[pass].s.data()+size_t(kb)*64*8,SAS);
    for(unsigned token=0;token<32;token++)for(unsigned k=0;k<128;k+=2)if(st[qo+token*64+(((k/32)^((token>>1)&3))*16)+k%32/2]!=tapes[pass].q[qoff(token,kb,k)])good=false;
    for(unsigned token=0;token<32;token++)for(unsigned k=0;k<8;k++)if(st[so+token*8+k]!=tapes[pass].s[(size_t(kb)*64+token)*8+k])good=false;
   }
   for(unsigned pass=0;pass<2;pass++)for(size_t i=0;i<extent;i++){
    auto[f,row]=dict[i];unsigned g=f/32-g0,rr=f%32,mm=rr/16,r=(rr%16)/8,gid=rr%8;auto*raw=p.raw.data()+(size_t(f)*KB+kb)*18;
    size_t a=(size_t(row)*KB+kb)*16*256,b=(size_t(row)*KB+kb)*32*16;int32_t rd=0,halves[2]={};
    for(unsigned block=0;block<16;block++){
     rd+=lut[pass].raw[a+block*256+raw[2+block]];unsigned h=block/8,tq=block%4,upper=(block%8)/4,lane=gid*4+tq,w=mm*4+h*2+r;uint16_t word;std::memcpy(&word,st.data()+g*512+lane*16+w*2,2);unsigned code=(word>>(upper*8))&255;
     halves[h]+=lut[pass].native[b+block*2*16+(code&15)]+lut[pass].native[b+(block*2+1)*16+(code>>4)];
    }
    int32_t nd=halves[0]+halves[1];uint16_t ns;std::memcpy(&ns,st.data()+SW+g*64+(gid*4+mm*2+r)*2,2);uint16_t rs=raw[0]|uint16_t(raw[1])<<8;
    if(rd!=nd||ns!=rs)good=false;
    rawacc[pass][i]=std::fma(std::ldexp(float(rd),-10),source14::full(rs),rawacc[pass][i]);
    acc[pass][i]=std::fma(std::ldexp(float(nd),-10),source14::full(ns),acc[pass][i]);
    if(std::bit_cast<uint32_t>(acc[pass][i])!=std::bit_cast<uint32_t>(rawacc[pass][i]))good=false;
    feed(digest[pass][i],uint32_t(nd));feed(digest[pass][i],ns);feed(digest[pass][i],std::bit_cast<uint32_t>(acc[pass][i]));
   }
  }
  for(size_t i=0;i<extent;i++){
   auto[f,row]=dict[i];float primary=add(mul(acc[0][i],shared_globals[0][row]),0.f),final=add(mul(acc[1][i],shared_globals[1][row]),primary);
   float rawprimary=add(mul(rawacc[0][i],tapes[0].g[row]),0.f),rawfinal=add(mul(rawacc[1][i],tapes[1].g[row]),rawprimary);
   if(std::bit_cast<uint32_t>(primary)!=std::bit_cast<uint32_t>(rawprimary)||std::bit_cast<uint32_t>(final)!=std::bit_cast<uint32_t>(rawfinal))good=false;
   for(unsigned pass=0;pass<2;pass++){size_t at=(size_t(pass)*B+row)*p.F+f;uint32_t word=std::bit_cast<uint32_t>(pass?final:primary);for(unsigned arm=0;arm<3;arm++)if(word!=saved[arm][at]||digest[pass][i]!=traces[arm][at])good=false;if(owners[pass][i]!=~0u)good=false;owners[pass][i]=0;}
  }
  for(auto&pass:owners)if(!std::all_of(pass.begin(),pass.end(),[](unsigned status){return status==0;}))good=false;
 }
 counts.stage_q=uint64_t(ncta)*KB*2*SA;counts.stage_s=uint64_t(ncta)*KB*2*SAS;counts.stage_w=uint64_t(ng)*KB*512;counts.stage_ws=uint64_t(ng)*KB*64;counts.folds=uint64_t(2)*B*p.F*KB;counts.outputs=uint64_t(2)*B*p.F;counts.owners=counts.outputs;counts.global_scale_words=uint64_t(ncta)*2*B;counts.half_dots=counts.folds*2;
 return good.load();
}
static void emit(const Counts&c){std::cout<<"{\"stage_q_bytes\":"<<c.stage_q<<",\"stage_scale_bytes\":"<<c.stage_s<<",\"stage_weight_bytes\":"<<c.stage_w<<",\"stage_weight_scale_bytes\":"<<c.stage_ws<<",\"once_CTA_global_scale_words\":"<<c.global_scale_words<<",\"complete_pass_owner_words\":"<<c.owners<<",\"exact_K128_dot_scale_fold_checks\":"<<c.folds<<",\"exact_K64_half_dots\":"<<c.half_dots<<",\"complete_pass_output_words\":"<<c.outputs<<",\"inactive_output_coordinates\":"<<c.inactive<<"}";}
}
int main(int argc,char**argv){using namespace witness22;if(argc!=3)return fail(Error::arguments);unsigned B;auto text=std::string(argv[2]);auto parsed=std::from_chars(text.data(),text.data()+text.size(),B);if(parsed.ec!=std::errc{}||parsed.ptr!=text.data()+text.size()||(B!=1&&B!=8&&B!=32))return fail(Error::arguments);std::string name=argv[1];if(name!="input0"&&name!="input3"&&name!="ffn0")return fail(Error::arguments);auto family=readers18::family(name=="ffn0"?"ffn":"input",name=="input3"?3:0,B,name=="ffn0"?2:3);auto*f=std::get_if<readers18::Family>(&family);if(!f)return fail(Error::arguments);unsigned F=0;for(auto&r:f->readers)F+=r.F;
 std::array<Tape,2>t;for(auto&v:t)if(!input(v.q,64*K/2)||!input(v.s,64*K/16)||!input(v.g,B))return fail(Error::input);
 std::array<std::vector<uint32_t>,3>saved;std::array<std::vector<uint64_t>,3>traces;for(auto&v:saved)if(!input(v,uint64_t(2)*B*F))return fail(Error::input);for(auto&v:traces)if(!input(v,uint64_t(2)*B*F))return fail(Error::input);char extra;if(std::cin.get(extra))return fail(Error::input);
 source14::Model model;if(!model.open("~/work/packing-mma-round13/archive/packing-swarm/round13/model-index.txt"))return fail(Error::model);
 source14::Weights raw{K,0,{}};for(auto&r:f->readers)for(auto&[tensor,width]:readers18::tensors(r)){auto result=model.bytes(tensor,41,K,width);auto*v=std::get_if<std::vector<uint8_t>>(&result);if(!v||v->size()!=uint64_t(width)*KB*18)return fail(Error::weights);raw.F+=width;raw.raw.insert(raw.raw.end(),v->begin(),v->end());}
 auto p=source14::prepare(std::move(raw));auto before_raw=fnv(p.raw),before_sign=fnv(p.w),before_scale=fnv(p.s);auto before_t=t;
 std::array<LUT,2>lut{tables(t[0],B),tables(t[1],B)};Counts candidate_counts,control_counts;
 if(!project(t,lut,p,B,true,saved,traces,candidate_counts)||!project(t,lut,p,B,false,saved,traces,control_counts))return fail(Error::fold);
 if(fnv(p.raw)!=before_raw||fnv(p.w)!=before_sign||fnv(p.s)!=before_scale)return fail(Error::weights);
 for(unsigned i=0;i<2;i++)if(t[i].q!=before_t[i].q||t[i].s!=before_t[i].s||t[i].g!=before_t[i].g)return fail(Error::coordinates);
 std::cout<<"{\"state\":\"paired_saved_tape_CPU_pass\",\"family\":\""<<name<<"\",\"B\":"<<B<<",\"F\":"<<F<<",\"source_and_weights_preserved\":true,\"source_weight_fnv1a64\":\""<<std::hex<<before_raw<<"\",\"native_sign_fnv1a64\":\""<<before_sign<<"\",\"native_scale_fnv1a64\":\""<<before_scale<<"\",\"primary_q_fnv1a64\":\""<<fnv(t[0].q)<<"\",\"innovation_q_fnv1a64\":\""<<fnv(t[1].q)<<"\",\"primary_s_fnv1a64\":\""<<fnv(t[0].s)<<"\",\"innovation_s_fnv1a64\":\""<<fnv(t[1].s)<<std::dec<<"\",\"candidate\":";emit(candidate_counts);std::cout<<",\"equally_paired_FW8\":";emit(control_counts);std::cout<<",\"CUDA_primitives_executed\":false}\n";
}
