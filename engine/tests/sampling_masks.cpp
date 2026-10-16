#include "sampling.h"
#include <cassert>
#include <cstdio>
#include <cstddef>
using namespace h5::sampling;
extern "C" int qc_sample_cpu(const float*l,int vocab,const H5SamplingConfig*c,uint64_t seed,uint64_t index,const unsigned char*seen){Config x;x.temperature=c->temperature;x.top_p=c->top_p;x.min_p=c->min_p;x.presence_penalty=c->presence_penalty;x.repetition_penalty=c->repetition_penalty;x.top_k=c->top_k;std::vector<unsigned char>v(seen,seen+vocab);return select(l,vocab,x,v,seed,index).token;}
int main(){
 const float inf=std::numeric_limits<float>::infinity();
 float masked[]={-inf,2,1};unsigned char seen[]={1,1,0};
 H5SamplingConfig abi{0,1,0,0,1,0};
 static_assert(sizeof(H5SamplingConfig)==48);
 static_assert(offsetof(H5SamplingConfig,temperature)==0 && offsetof(H5SamplingConfig,top_p)==8 && offsetof(H5SamplingConfig,min_p)==16 && offsetof(H5SamplingConfig,presence_penalty)==24 && offsetof(H5SamplingConfig,repetition_penalty)==32 && offsetof(H5SamplingConfig,top_k)==40);
 int token=qc_sample_cpu(masked,3,&abi,7,0,seen);

 assert(token==1);Config c;std::vector<unsigned char> v{1,1,0};
 float none[]={-inf,-inf,-inf};auto r=select(none,3,c,v,0,0);assert(r.token==-1 && r.error=="no permitted target logits");
 float nan[]={0,std::numeric_limits<float>::quiet_NaN(),1};assert(!select(nan,3,c,v,0,0).error.empty());
 float pos[]={0,inf,1};assert(!select(pos,3,c,v,0,0).error.empty());
 float tie[]={-inf,2,2};assert(select(tie,3,c,v,0,0).token==1);
 int checks=0;
 for(double temp:{0.,0.2,0.7,2.}) for(double p:{0.2,0.9,1.}) for(double min:{0.,0.1,1.}) for(int k:{0,1,2,128}) for(double rep:{0.1,1.,2.}) for(double pres:{-2.,0.,2.}){
   c.temperature=temp;c.top_p=p;c.min_p=min;c.top_k=k;c.repetition_penalty=rep;c.presence_penalty=pres;
   for(int seed=0;seed<8;++seed){r=select(masked,3,c,v,seed,seed*3);assert(r.error.empty() && (r.token==1 || r.token==2));++checks;}
 }
 float sole[]={-inf,-inf,-3};for(int k:{0,1,128}){c.top_k=k;r=select(sole,3,c,v,7,2);assert(r.token==2 && r.error.empty());}
 printf("{\"fixed\":true,\"masked_controls\":%d,\"negative_inf_excluded\":true,\"allmasked_error\":true,\"nan_positive_inf_error\":true,\"tie_lower_id\":true,\"config_bytes\":48}\n",checks);

}
