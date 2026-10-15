#include "../../src/dspark_quant.cuh"
#include "../../src/kernels.cuh"
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace h5 { void launch_prep(const PrepArgs&, int, cudaStream_t); }
using h5::dspark::ActivationQuantError;
static void ck(cudaError_t e) { if(e != cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(1); } }
static float e4(uint8_t b) { return __half2float(__half(__nv_cvt_fp8_to_halfraw(b,__NV_E4M3))); }
static float f4(unsigned c) { unsigned m=c&7; float v=m<4?.5f*m:m==4?2.f:m==5?3.f:m==6?4.f:6.f; return c&8?-v:v; }
struct Host {
    int R,K;
    std::vector<uint8_t> q,s;
    std::vector<float> g;
    Host(int r,int k):R(r),K(k),q(((r+63)/64)*64*k/2),s(((r+63)/64)*64*k/16),g(r){}
    size_t base(int r,int k) const { return (size_t(r/64)*(K/128)+k/128)*64+r%64; }
    size_t qo(int r,int k) const { int o=k%128; return base(r,k)*64+(((o/32)^((r%64>>1)&3))*16)+(o%32)/2; }
    float decode(int r,int k) const { unsigned c=(q[qo(r,k)]>>(4*(k&1)))&15; return f4(c)*(g[r]*e4(s[base(r,k)*8+(k%128)/16])); }
};
static Host scalar(const std::vector<float>& x,int R,int K,const Host* ref) {
    Host out(R,K);
    std::vector<float> v(K);
    for(int r=0;r<R;++r) {
        float am=0;
        for(int k=0;k<K;++k) {
            v[k]=x[r*K+k];
            if(ref) {
                unsigned c=(ref->q[ref->qo(r,k)]>>(4*(k&1)))&15;
                float scale=ref->g[r]*e4(ref->s[ref->base(r,k)*8+(k%128)/16]);
                v[k]=std::fma(-f4(c),scale,v[k]);
            }
            am=std::max(am,std::fabs(v[k]));
        }
        float g=std::max(am/2688.f,1e-30f), ig=1.f/g;
        out.g[r]=g;
        for(int k=0;k<K;k+=16) {
            float bm=0;
            for(int i=0;i<16;++i) bm=std::max(bm,std::fabs(v[k+i]));
            float target=bm/6.f*ig;
            uint8_t sb=__nv_cvt_float_to_fp8(target,__NV_SATFINITE,__NV_E4M3);
            float sv=e4(sb);
            if(sv<target && sb<0x7e) sv=e4(++sb);
            float rs=sv>0?ig/sv:0;
            out.s[out.base(r,k)*8+(k%128)/16]=sb;
            for(int i=0;i<16;i+=2) out.q[out.qo(r,k+i)]=__nv_cvt_float2_to_fp4x2(make_float2(v[k+i]*rs,v[k+i+1]*rs),__NV_E2M1,cudaRoundNearest);
        }
    }
    return out;
}
struct Device {
    uint8_t *q,*s;
    float* g;
    Device(const Host& h) { ck(cudaMalloc(&q,h.q.size())); ck(cudaMalloc(&s,h.s.size())); ck(cudaMalloc(&g,h.g.size()*4)); ck(cudaMemset(q,0,h.q.size())); ck(cudaMemset(s,0,h.s.size())); }
    ~Device(){ cudaFree(q);cudaFree(s);cudaFree(g); }
    void read(Host& h) { ck(cudaMemcpy(h.q.data(),q,h.q.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(h.s.data(),s,h.s.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(h.g.data(),g,h.g.size()*4,cudaMemcpyDeviceToHost)); }
};
static void same(const Host& a,const Host& b,const char* name) {
    if(a.q==b.q && a.s==b.s && !std::memcmp(a.g.data(),b.g.data(),a.g.size()*4))return;
    for(size_t i=0;i<a.q.size();++i)if(a.q[i]!=b.q[i]) { std::fprintf(stderr,"%s q[%zu]=%u vs %u\n",name,i,a.q[i],b.q[i]);break; }
    for(size_t i=0;i<a.s.size();++i)if(a.s[i]!=b.s[i]) { std::fprintf(stderr,"%s s[%zu]=%u vs %u\n",name,i,a.s[i],b.s[i]);break; }
    for(size_t i=0;i<a.g.size();++i)if(std::memcmp(&a.g[i],&b.g[i],4))std::fprintf(stderr,"%s g[%zu]=%.9g vs %.9g\n",name,i,a.g[i],b.g[i]);
    std::exit(1);
}
int main() {
    setenv("H5_PREP_CHUNK","0",1);unsetenv("H5_PREP_G4");setenv("H5_PREP_WIDE","0",1);
    constexpr int R=4;
    ActivationQuantError* err;
    ck(cudaMalloc(&err,4));
    for(int K:{5120,25600}) {
        std::vector<float>x(R*K);
        uint32_t rng=19;
        for(int i=0;i<R*K;++i){rng=rng*1664525u+1013904223u;x[i]=float(int(rng%4097)-2048)/1536.f;}
        std::fill(x.begin()+K,x.begin()+2*K,0.f);
        for(int k=0;k<K;++k)x[2*K+k]=-std::fabs(x[2*K+k]);
        float* input;ck(cudaMalloc(&input,x.size()*4));ck(cudaMemcpy(input,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        Host got0(R,K),got1(R,K),shared0(R,K),shared1(R,K);
        Device d0(got0),d1(got1);
        ck(cudaMemset(err,0,4));
        ck(h5::dspark::launch_prepare_activations(input,R,K,d0.q,d0.s,d0.g,err,nullptr,nullptr,nullptr,0));
        ck(h5::dspark::launch_prepare_activations(input,R,K,d1.q,d1.s,d1.g,err,d0.q,d0.s,d0.g,0));
        ck(cudaDeviceSynchronize());d0.read(got0);d1.read(got1);
        ActivationQuantError error;ck(cudaMemcpy(&error,err,4,cudaMemcpyDeviceToHost));
        if(error!=ActivationQuantError::success){std::fprintf(stderr,"unexpected device status %u\n",unsigned(error));return 1;}
        Host ref0=scalar(x,R,K,nullptr),ref1=scalar(x,R,K,&ref0);
        same(got0,ref0,"scalar first");same(got1,ref1,"scalar residual");
        if(K==5120) {
            Device s0(shared0),s1(shared1);
            h5::PrepArgs a{};a.x=input;a.ldx=K;a.n=K;a.flags=h5::P_QUANT;a.q4=s0.q;a.s4=s0.s;a.g4=s0.g;
            h5::launch_prep(a,R,0);ck(cudaGetLastError());
            a.flags|=h5::P_FP4_RESIDUAL;a.q4=s1.q;a.s4=s1.s;a.g4=s1.g;a.ref_q4=s0.q;a.ref_s4=s0.s;a.ref_g4=s0.g;
            h5::launch_prep(a,R,0);ck(cudaGetLastError());ck(cudaDeviceSynchronize());s0.read(shared0);s1.read(shared1);
            same(got0,shared0,"shared first");same(got1,shared1,"shared residual");
        }
        // An error survives subsequent valid preparation until the caller clears it.
        float nan=std::nanf("");ck(cudaMemcpy(input,&nan,4,cudaMemcpyHostToDevice));
        ck(h5::dspark::launch_prepare_activations(input,R,K,d0.q,d0.s,d0.g,err,nullptr,nullptr,nullptr,0));
        ck(cudaMemcpy(input,x.data(),4,cudaMemcpyHostToDevice));
        ck(h5::dspark::launch_prepare_activations(input,R,K,d0.q,d0.s,d0.g,err,nullptr,nullptr,nullptr,0));
        ck(cudaDeviceSynchronize());ck(cudaMemcpy(&error,err,4,cudaMemcpyDeviceToHost));
        if(error!=ActivationQuantError::non_finite_input){std::fprintf(stderr,"sticky finite error lost\n");return 1;}
        ck(cudaFree(input));std::printf("activation parity K%d R4 first/residual/padding/sticky PASS\n",K);
    }
    ck(cudaFree(err));return 0;
}
