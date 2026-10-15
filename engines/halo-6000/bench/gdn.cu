#include "gdn.cuh"
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cmath>

using namespace halo6000;
namespace cg = cooperative_groups;
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);} } while(0)

__global__ void gdn_control(GdnView v, GdnScratch s, ActivationBuffer a, int mode) {
    auto grid = cg::this_grid();
    if (mode == 2) { gdn_conv_prepare(v,s); grid.sync(); }
    else { gdn_conv(v,s); grid.sync(); gdn_prepare(v,s); grid.sync(); }
    gdn_recur(v,s); grid.sync();
    if (mode) gdn_finish_q8(v,s,a);
    else { gdn_finish(v,s); grid.sync(); quantize_phase(v.out,a); }
}

struct Device {
    std::vector<float*> allocations;
    float *make(size_t n) { float *p; CUDA(cudaMalloc(&p,n*sizeof(float))); allocations.push_back(p); return p; }
    float *copy(const std::vector<float>&x){float*p=make(x.size());CUDA(cudaMemcpy(p,x.data(),x.size()*4,cudaMemcpyHostToDevice));return p;}
    ~Device(){for(auto*p:allocations)cudaFree(p);}
};
std::vector<float> read_device(const float*p,size_t n){std::vector<float>x(n);CUDA(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost));return x;}
struct Input {
    std::vector<float> qkv,z,alpha,beta,conv,a,dt,norm,state,history;
    Input():qkv(ConvChannels),z(Inner),alpha(ValueHeads),beta(ValueHeads),conv(ConvChannels*ConvTaps),
            a(ValueHeads),dt(ValueHeads),norm(ValueDim),state(ValueHeads*ValueDim*StateDim),history(ConvChannels*3){}
    std::vector<std::vector<float>*> arrays(){return{&qkv,&z,&alpha,&beta,&conv,&a,&dt,&norm,&state,&history};}
    void randomize(int scenario) {
        std::mt19937 gen(817+scenario);
        std::normal_distribution<float> normal(0,1);
        for(auto*p:arrays())for(auto&x:*p)x=normal(gen)*.2f;
        for(auto&x:state)x*=.2f;
        for(auto&x:a)x=-std::abs(x)-.1f;
        for(auto&x:norm)x+=1.f;
        if(scenario==1){std::fill(state.begin(),state.end(),0);std::fill(history.begin(),history.end(),0);}
        if(scenario==2){for(int h=0;h<ValueHeads;++h){alpha[h]=(h&1)?80.f:-80.f;beta[h]=(h&2)?80.f:-80.f;}}
        if(scenario==3){for(int c=0;c<2*KeyHeads*StateDim;++c){qkv[c]=0;for(int j=0;j<3;++j)history[c*3+j]=0;}}
        if(scenario==4){for(int c=0;c<2*KeyHeads*StateDim;++c){qkv[c]*=1e-8f;for(int j=0;j<3;++j)history[c*3+j]*=1e-8f;}}
    }
    bool file(const std::string&p){std::ifstream f(p,std::ios::binary);for(auto*a:arrays())f.read((char*)a->data(),a->size()*4);return bool(f)&&f.peek()==EOF;}
};
struct Reference {
    std::vector<double> state,history,qkv,decay,beta,core,out,log_decay,softplus;
    Reference(const Input&i):state(i.state.begin(),i.state.end()),history(i.history.begin(),i.history.end()),
        qkv(ConvChannels),decay(ValueHeads),beta(ValueHeads),core(Inner),out(Inner),log_decay(ValueHeads),softplus(ValueHeads){}
    void step(const Input&i) {
        for(int c=0;c<ConvChannels;++c){
            double sum=0;for(int t=0;t<3;++t)sum+=history[c*3+t]*i.conv[c*4+t];sum+=double(i.qkv[c])*i.conv[c*4+3];
            qkv[c]=sum/(1+std::exp(-sum));
            history[c*3]=history[c*3+1];history[c*3+1]=history[c*3+2];history[c*3+2]=i.qkv[c];
        }
        for(int h=0;h<2*KeyHeads;++h){double sum=0;for(int k=0;k<StateDim;++k)sum+=qkv[h*StateDim+k]*qkv[h*StateDim+k];
            double scale=1/std::sqrt(std::max(sum,double(NormEpsilon)*NormEpsilon));for(int k=0;k<StateDim;++k)qkv[h*StateDim+k]*=scale;}
        for(int h=0;h<ValueHeads;++h){
            double x=double(i.alpha[h])+i.dt[h];double sp=x>20?x:std::log1p(std::exp(x));
            softplus[h]=sp;log_decay[h]=double(i.a[h])*sp;
            decay[h]=std::exp(log_decay[h]);beta[h]=1/(1+std::exp(-double(i.beta[h])));
            int kh=h%KeyHeads;
            for(int col=0;col<ValueDim;++col){
                double kv=0;for(int k=0;k<StateDim;++k)kv+=state[(h*ValueDim+col)*StateDim+k]*qkv[KeyHeads*StateDim+kh*StateDim+k];
                double delta=(qkv[2*KeyHeads*StateDim+h*ValueDim+col]-decay[h]*kv)*beta[h];
                double result=0;for(int k=0;k<StateDim;++k){auto&st=state[(h*ValueDim+col)*StateDim+k];
                    st=decay[h]*st+qkv[KeyHeads*StateDim+kh*StateDim+k]*delta;result+=st*qkv[kh*StateDim+k];}
                core[h*ValueDim+col]=result/std::sqrt(double(StateDim));
            }
            double sum=0;for(int col=0;col<ValueDim;++col)sum+=core[h*ValueDim+col]*core[h*ValueDim+col];
            double rms=1/std::sqrt(sum/ValueDim+double(NormEpsilon));
            for(int col=0;col<ValueDim;++col){int o=h*ValueDim+col;double z=i.z[o];out[o]=core[o]*rms*i.norm[col]*z/(1+std::exp(-z));}
        }
    }
};
struct Error {double relative=0,absolute=0;};
Error difference(const std::vector<float>&got,const std::vector<double>&ref){
    double sq=0,norm=0,ab=0;
    for(size_t i=0;i<got.size();++i){if(!std::isfinite(got[i])){std::fprintf(stderr,"nonfinite output\n");std::exit(3);}double d=double(got[i])-ref[i];sq+=d*d;norm+=ref[i]*ref[i];ab=std::max(ab,std::abs(d));}
    return{std::sqrt(sq/std::max(norm,1e-30)),ab};
}
struct Output {std::vector<float>qkv,decay,beta,core,out,state,history,scales;std::vector<int8_t>qs;std::vector<float>log_decay,softplus;};
template<class T>bool bit_equal(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0;}
bool exact_equal(const Output&a,const Output&b){return bit_equal(a.qkv,b.qkv)&&bit_equal(a.decay,b.decay)&&bit_equal(a.beta,b.beta)&&bit_equal(a.core,b.core)&&bit_equal(a.out,b.out)&&bit_equal(a.state,b.state)&&bit_equal(a.history,b.history)&&bit_equal(a.scales,b.scales)&&bit_equal(a.qs,b.qs)&&bit_equal(a.log_decay,b.log_decay)&&bit_equal(a.softplus,b.softplus);}
Output run(Input&i,int steps,int block,int grid,int mode=0){
    Device mem;
    GdnView v{mem.copy(i.qkv),mem.copy(i.z),mem.copy(i.alpha),mem.copy(i.beta),mem.copy(i.conv),mem.copy(i.a),mem.copy(i.dt),mem.copy(i.norm),mem.copy(i.state),mem.copy(i.history),mem.make(Inner)};
    GdnScratch s{mem.make(ConvChannels),mem.make(ValueHeads),mem.make(ValueHeads),mem.make(Inner),mem.make(ValueHeads),mem.make(ValueHeads)};
    ActivationBuffer activation;
    activation.qs=reinterpret_cast<int8_t*>(mem.make(Inner/4));activation.scale=mem.make(Inner/32);
    activation.k=Inner;activation.rows=1;activation.group=32;activation.q_stride=Inner;activation.scale_stride=Inner/32;
    int active=0;CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,gdn_control,block,0));
    int sm=0;CUDA(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0));grid=std::min(grid,active*sm);
    if(grid<1){std::fprintf(stderr,"cooperative kernel cannot reside\n");std::exit(4);}
    void*args[]={&v,&s,&activation,&mode};
    for(int t=0;t<steps;++t)CUDA(cudaLaunchCooperativeKernel((void*)gdn_control,dim3(grid),dim3(block),args,0,nullptr));
    CUDA(cudaDeviceSynchronize());
    Output out{read_device(s.qkv,ConvChannels),read_device(s.decay,ValueHeads),read_device(s.beta,ValueHeads),read_device(s.core,Inner),read_device(v.out,Inner),read_device(v.state,ValueHeads*ValueDim*StateDim),read_device(v.history,ConvChannels*3),read_device(activation.scale,Inner/32),std::vector<int8_t>(Inner),read_device(s.log_decay,ValueHeads),read_device(s.softplus,ValueHeads)};
    CUDA(cudaMemcpy(out.qs.data(),activation.qs,Inner,cudaMemcpyDeviceToHost));return out;
}
void save(const std::string&p,const Output&o){std::ofstream f(p,std::ios::binary);for(auto*v:{&o.qkv,&o.decay,&o.beta,&o.core,&o.out,&o.state,&o.history})f.write((const char*)v->data(),v->size()*4);if(!f)std::exit(5);}
int main(int argc,char**argv){
    if(argc==3){Input i;if(!i.file(argv[1])){std::fprintf(stderr,"invalid fixture\n");return 6;}auto o=run(i,1,256,128);save(argv[2],o);return 0;}
    bool pass=true;double max_rel=0,max_abs=0;
    for(int scenario=0;scenario<5;++scenario)for(int steps:{1,4,16}){
        Input i;i.randomize(scenario);Reference ref(i);for(int t=0;t<steps;++t)ref.step(i);
        Output a=run(i,steps,128,64),b=run(i,steps,256,128),c=run(i,steps,256,128,1),d=run(i,steps,256,128,2);
        bool exact=exact_equal(a,b),producer=exact_equal(b,c)&&exact_equal(b,d);
        double rel=0,absolute=0;
        auto check=[&](const std::vector<float>&v,const std::vector<double>&w){auto e=difference(v,w);rel=std::max(rel,e.relative);absolute=std::max(absolute,e.absolute);return e.relative<3e-5||e.absolute<1e-7;};
        bool cpu=check(a.qkv,ref.qkv)&&check(a.decay,ref.decay)&&check(a.beta,ref.beta)&&check(a.core,ref.core)&&check(a.out,ref.out)&&check(a.state,ref.state)&&check(a.history,ref.history)&&check(a.log_decay,ref.log_decay)&&check(a.softplus,ref.softplus);
        pass&=exact&&cpu&&producer;max_rel=std::max(max_rel,rel);max_abs=std::max(max_abs,absolute);
        std::printf("{\"scenario\":%d,\"steps\":%d,\"cpu_equal\":%s,\"schedule_bit_equal\":%s,\"producer_bit_equal\":%s,\"max_relative\":%.9g,\"max_absolute\":%.9g}\n",scenario,steps,cpu?"true":"false",exact?"true":"false",producer?"true":"false",rel,absolute);
    }
    std::printf("{\"passed\":%s,\"cases\":15,\"comparison\":\"memcmp\",\"max_relative\":%.9g,\"max_absolute\":%.9g}\n",pass?"true":"false",max_rel,max_abs);
    return pass?0:1;
}
