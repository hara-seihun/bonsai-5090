#include "attention.cuh"
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
using namespace halo6000;
namespace cg=cooperative_groups;
#define CUDA(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)

template<bool Coarse>
__global__ void attention_control(LayerWeights w,StateView s,AttentionBuffers b,const float *rawk,ActivationBuffer a,int layer,int slot,int pos,int repeats){
    auto grid=cg::this_grid();
    for(int it=0;it<repeats;it++){
        for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<KVHeads*HeadDim;i+=gridDim.x*blockDim.x)b.k[i]=rawk[i];grid.sync();
        if constexpr(Coarse){attention_norm_rope_append_phase(w,s,layer,slot,pos,b);grid.sync();}
        else{attention_norm_rope_phase(w,pos,b);grid.sync();attention_append_phase(s,layer,slot,pos,b);grid.sync();}
        attention_scores_phase(s,layer,slot,pos,b);grid.sync();
        attention_softmax_phase(s,pos,b);grid.sync();
        if constexpr(Coarse){attention_value_gate_q8_phase(s,layer,slot,pos,b,a);grid.sync();}
        else{attention_value_gate_phase(s,layer,slot,pos,b);grid.sync();quantize_phase(b.attended,a);grid.sync();}
    }
}
struct Device {
    std::vector<void*>p;
    template<class T>T*make(size_t n){T*x;CUDA(cudaMalloc(&x,n*sizeof(T)));p.push_back(x);return x;}
    template<class T>T*copy(const std::vector<T>&v){auto*x=make<T>(v.size());CUDA(cudaMemcpy(x,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return x;}
    ~Device(){for(auto*x:p)cudaFree(x);}
};
template<class T>std::vector<T>get(const T*p,size_t n){std::vector<T>v(n);CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
uint16_t hbits(float x){half h=__float2half_rn(x);uint16_t v;memcpy(&v,&h,2);return v;}
float hfloat(uint16_t x){half h;memcpy(&h,&x,2);return __half2float(h);}
struct FixtureHeader {char magic[8];uint32_t context,position,layer,slot,slots;};
struct Input {
    int context,pos,layer=5,slot=1,slots=2;
    std::vector<float>qgate,k,v,qgamma,kgamma;
    std::vector<uint16_t>prefixk,prefixv;
    Input(int c=64,int p=0):context(c),pos(p),qgate(QueryHeads*2*HeadDim),k(KVHeads*HeadDim),v(KVHeads*HeadDim),qgamma(HeadDim),kgamma(HeadDim),prefixk((size_t)p*KVHeads*HeadDim),prefixv(prefixk.size()){}
    void randomize(int scenario){
        std::mt19937 gen(128+scenario+pos);std::normal_distribution<float>normal(0,1);
        for(auto*a:{&qgate,&k,&v})for(auto&x:*a)x=normal(gen)*.5f;
        for(auto*a:{&qgamma,&kgamma})for(auto&x:*a)x=1.f+normal(gen)*.05f;
        for(auto*a:{&prefixk,&prefixv})for(auto&x:*a)x=hbits(normal(gen)*.5f);
        if(scenario==1){for(int h=0;h<QueryHeads;h++)for(int d=0;d<HeadDim;d++)qgate[h*2*HeadDim+d]=0;std::fill(k.begin(),k.end(),0);}
        if(scenario==2){for(int h=0;h<QueryHeads;h++)for(int d=0;d<HeadDim;d++)qgate[h*2*HeadDim+HeadDim+d]=(d%2?80:-80);}
        if(scenario==3){std::fill(prefixk.begin(),prefixk.end(),0);std::fill(prefixv.begin(),prefixv.end(),0);std::fill(qgate.begin(),qgate.end(),0);std::fill(k.begin(),k.end(),0);}
    }
    bool read(const std::string&path){
        std::ifstream f(path,std::ios::binary);FixtureHeader h{};f.read((char*)&h,sizeof(h));if(memcmp(h.magic,"H6ATTN1",8)||h.context<1||h.position>=h.context||h.layer>=AttentionLayers||h.slots<1||h.slot>=h.slots)return false;
        *this=Input(h.context,h.position);layer=h.layer;slot=h.slot;slots=h.slots;
        for(auto*a:{&qgate,&k,&v,&qgamma,&kgamma})f.read((char*)a->data(),a->size()*4);
        for(auto*a:{&prefixk,&prefixv})f.read((char*)a->data(),a->size()*2);
        return f&&f.peek()==EOF;
    }
};
struct Reference {
    std::vector<double>q,k,prob,out;std::vector<uint16_t>key,value;
    Reference(const Input&i):q(QueryHeads*HeadDim),k(KVHeads*HeadDim),prob((size_t)QueryHeads*(i.pos+1)),out(QueryHeads*HeadDim),key(i.prefixk),value(i.prefixv){
        auto norm=[&](const float*src,const std::vector<float>&gamma,double*dst){
            double ss=0;for(int d=0;d<HeadDim;d++)ss+=(double)src[d]*src[d];double inv=1/std::sqrt(ss/HeadDim+(double)NormEpsilon);
            for(int d=0;d<HeadDim;d++)dst[d]=src[d]*inv*gamma[d];
            for(int d=0;d<RotaryDim/2;d++){float angle=(float)i.pos*powf(AttentionRopeThetaScale,(float)d);double sn=std::sin((double)angle),cs=std::cos((double)angle),x=dst[d],y=dst[d+RotaryDim/2];dst[d]=x*cs-y*sn;dst[d+RotaryDim/2]=x*sn+y*cs;}
        };
        for(int h=0;h<QueryHeads;h++)norm(i.qgate.data()+h*2*HeadDim,i.qgamma,q.data()+h*HeadDim);
        for(int h=0;h<KVHeads;h++)norm(i.k.data()+h*HeadDim,i.kgamma,k.data()+h*HeadDim);
        key.resize((size_t)(i.pos+1)*KVHeads*HeadDim);value.resize(key.size());
        for(int d=0;d<KVHeads*HeadDim;d++){key[i.prefixk.size()+d]=hbits((float)k[d]);value[i.prefixv.size()+d]=hbits(i.v[d]);}
        for(int h=0;h<QueryHeads;h++){
            int kh=h/(QueryHeads/KVHeads);double mx=-INFINITY;int len=i.pos+1;
            for(int t=0;t<len;t++){double sum=0;for(int d=0;d<HeadDim;d++)sum+=q[h*HeadDim+d]*hfloat(key[((size_t)t*KVHeads+kh)*HeadDim+d]);prob[h*len+t]=sum*AttentionScale;mx=std::max(mx,prob[h*len+t]);}
            double sum=0;for(int t=0;t<len;t++){prob[h*len+t]=std::exp(prob[h*len+t]-mx);sum+=prob[h*len+t];}for(int t=0;t<len;t++)prob[h*len+t]/=sum;
            for(int d=0;d<HeadDim;d++){double x=0;for(int t=0;t<len;t++)x+=prob[h*len+t]*hfloat(value[((size_t)t*KVHeads+kh)*HeadDim+d]);out[h*HeadDim+d]=x/(1+std::exp(-(double)i.qgate[h*2*HeadDim+HeadDim+d]));}
        }
    }
};
struct Diff{double absolute=0,relative=0;};
Diff diff(const std::vector<float>&a,const std::vector<double>&b){double err=0,norm=0,ab=0;for(size_t i=0;i<a.size();i++){if(!std::isfinite(a[i])){fprintf(stderr,"nonfinite\n");exit(3);}double d=a[i]-b[i];ab=std::max(ab,std::abs(d));err+=d*d;norm+=b[i]*b[i];}return{ab,std::sqrt(err/std::max(norm,1e-30))};}
struct Output {std::vector<float>q,k,prob,out,scales;std::vector<int8_t>qs;std::vector<uint16_t>key,value;bool state_isolated=true;float us=0;};
Output run(const Input&i,int block,int requested_grid,int repeats=1,bool coarse=false){
    Device m;LayerWeights w{};w.q_norm=m.copy(i.qgamma);w.k_norm=m.copy(i.kgamma);AttentionBuffers b{m.copy(i.qgate),m.copy(i.k),m.copy(i.v),m.make<float>(QueryHeads*HeadDim),m.make<float>((size_t)QueryHeads*i.context),m.make<float>(QueryHeads*HeadDim)};
    ActivationBuffer a{};a.qs=m.make<int8_t>(QueryHeads*HeadDim);a.scale=m.make<float>(QueryHeads*HeadDim/32);a.k=QueryHeads*HeadDim;a.rows=1;a.group=32;a.q_stride=a.k;a.scale_stride=a.k/32;
    auto*rawk=m.copy(i.k);size_t total=(size_t)AttentionLayers*i.slots*i.context*KVHeads*HeadDim;
    StateView s{};s.slots=i.slots;s.context=i.context;s.key=m.make<uint16_t>(total);s.value=m.make<uint16_t>(total);std::vector<int32_t>positions(i.slots,123456);s.positions=m.copy(positions);
    CUDA(cudaMemset(s.key,0x55,total*2));CUDA(cudaMemset(s.value,0x33,total*2));CUDA(cudaMemset(b.scores,0x66,(size_t)QueryHeads*i.context*4));
    size_t base=attention_cache_offset(s,i.layer,i.slot,0);if(!i.prefixk.empty()){CUDA(cudaMemcpy(s.key+base,i.prefixk.data(),i.prefixk.size()*2,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(s.value+base,i.prefixv.data(),i.prefixv.size()*2,cudaMemcpyHostToDevice));}
    auto kernel=coarse?attention_control<true>:attention_control<false>;
    int active=0,sm=0;CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,block,0));CUDA(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0));int grid=std::min(requested_grid,active*sm);if(grid<1)exit(4);
    int layer=i.layer,slot=i.slot,pos=i.pos;void*args[]={&w,&s,&b,&rawk,&a,&layer,&slot,&pos,&repeats};cudaEvent_t st,en;CUDA(cudaEventCreate(&st));CUDA(cudaEventCreate(&en));CUDA(cudaEventRecord(st));CUDA(cudaLaunchCooperativeKernel((void*)kernel,dim3(grid),dim3(block),args,0,nullptr));CUDA(cudaEventRecord(en));CUDA(cudaEventSynchronize(en));float ms;CUDA(cudaEventElapsedTime(&ms,st,en));CUDA(cudaEventDestroy(st));CUDA(cudaEventDestroy(en));
    Output o;o.us=ms*1000/repeats;o.qs=get(a.qs,a.k);o.scales=get(a.scale,a.k/32);o.q=get(b.q,QueryHeads*HeadDim);o.k=get(b.k,KVHeads*HeadDim);o.out=get(b.attended,QueryHeads*HeadDim);auto scores=get(b.scores,(size_t)QueryHeads*i.context);
    for(int h=0;h<QueryHeads;h++)for(int t=0;t<=i.pos;t++)o.prob.push_back(scores[(size_t)h*i.context+t]);
    auto keys=get(s.key,total),values=get(s.value,total);size_t current=base+i.prefixk.size();
    for(size_t x=0;x<total;x++){
        if(x>=base&&x<current){o.state_isolated&=keys[x]==i.prefixk[x-base]&&values[x]==i.prefixv[x-base];}
        else if(x>=current&&x<current+KVHeads*HeadDim){o.state_isolated&=keys[x]==hbits(o.k[x-current])&&values[x]==hbits(i.v[x-current]);}
        else o.state_isolated&=keys[x]==0x5555&&values[x]==0x3333;
    }
    o.state_isolated&=get(s.positions,i.slots)==positions;
    for(int h=0;h<QueryHeads;h++)for(int t=i.pos+1;t<i.context;t++){uint32_t bits;float f=scores[(size_t)h*i.context+t];memcpy(&bits,&f,4);o.state_isolated&=bits==0x66666666;}
    o.key.assign(keys.begin()+current,keys.begin()+current+KVHeads*HeadDim);o.value.assign(values.begin()+current,values.begin()+current+KVHeads*HeadDim);return o;
}
void save(const std::string&p,const Output&o){std::ofstream f(p,std::ios::binary);for(auto*a:{&o.q,&o.k})f.write((const char*)a->data(),a->size()*4);for(auto*a:{&o.key,&o.value})f.write((const char*)a->data(),a->size()*2);for(auto*a:{&o.prob,&o.out})f.write((const char*)a->data(),a->size()*4);if(!f)exit(5);}
int main(int argc,char**argv){
    if(argc>=4&&!strcmp(argv[1],"--fixture")){
        Input i;if(!i.read(argv[2])){fprintf(stderr,"invalid H6ATTN1 fixture\n");return 6;}auto o=run(i,256,128);save(argv[3],o);
        if(argc>=5){std::ifstream f(argv[4],std::ios::binary);std::vector<float>expected(QueryHeads*HeadDim);f.read((char*)expected.data(),expected.size()*4);if(!f||f.peek()!=EOF)return 6;std::vector<double>ref(expected.begin(),expected.end());auto e=diff(o.out,ref);printf("{\"fork_callback_max_abs\":%.9g,\"fork_callback_relative_rms\":%.9g,\"state_isolated\":%s}\n",e.absolute,e.relative,o.state_isolated?"true":"false");return o.state_isolated&&(e.relative<3e-5||e.absolute<3e-6)?0:7;}
        printf("{\"fixture_processed\":true,\"state_isolated\":%s}\n",o.state_isolated?"true":"false");return o.state_isolated?0:7;
    }
    bool pass=true;int cases=0;
    const int cs[]={64,64,64,64,1024,1024,8192,8192};const int ps[]={0,1,31,63,511,1023,4095,8191};
    for(int j=0;j<8;j++){
        int scenario=j==7?0:j%4;Input i(cs[j],ps[j]);i.randomize(scenario);Reference r(i);auto a=run(i,128,64),b=run(i,256,128),c=run(i,256,128,1,true);
        auto equal=[](const Output&a,const Output&b){return a.q==b.q&&a.k==b.k&&a.prob==b.prob&&a.out==b.out&&a.key==b.key&&a.value==b.value&&a.qs==b.qs&&a.scales==b.scales;};
        bool same=equal(a,b),coarsened=equal(b,c)&&c.state_isolated;
        auto eq=diff(a.q,r.q),ek=diff(a.k,r.k),ep=diff(a.prob,r.prob),eo=diff(a.out,r.out);
        bool cpu=eq.absolute<3e-3&&ek.absolute<3e-3&&(ep.absolute<3e-5||ep.relative<3e-4)&&(eo.absolute<3e-5||eo.relative<3e-4);
        bool ok=same&&coarsened&&cpu&&a.state_isolated&&b.state_isolated;pass&=ok;cases++;
        printf("{\"context\":%d,\"position\":%d,\"scenario\":%d,\"cpu_guard\":%s,\"schedule_bit_equal\":%s,\"coarsened_bit_equal\":%s,\"cache_position_isolated\":%s,\"q_max_abs\":%.9g,\"k_max_abs\":%.9g,\"prob_max_abs\":%.9g,\"output_max_abs\":%.9g,\"output_relative_rms\":%.9g}\n",i.context,i.pos,scenario,cpu?"true":"false",same?"true":"false",coarsened?"true":"false",a.state_isolated&&b.state_isolated?"true":"false",eq.absolute,ek.absolute,ep.absolute,eo.absolute,eo.relative);fflush(stdout);
    }
    for(int pos:{127,1023,8191})for(bool coarse:{false,true}){
        Input i(pos+1,pos);i.randomize(0);run(i,256,128,4,coarse);auto o=run(i,256,128,32,coarse);
        printf("{\"position\":%d,\"coarsened\":%s,\"cooperative_attention_us\":%.9g,\"includes_rawK_restore\":true,\"grid_barriers\":%d,\"includes_Q8_output\":true,\"excludes_projection\":true}\n",pos,coarse?"true":"false",o.us,coarse?5:7);
    }
    printf("{\"passed\":%s,\"cases\":%d,\"fork_callback_fixture_run\":false}\n",pass?"true":"false",cases);return pass?0:1;
}
