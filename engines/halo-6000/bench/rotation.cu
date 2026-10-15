#include "rotation.cuh"
#include <cooperative_groups.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
using namespace halo6000;
namespace cg=cooperative_groups;
#define CUDA(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)

template<bool Fused>__global__ void rotation_control(const float*x,float*y,int rows,int K,int B,const float*signs,int64_t sign_stride,ActivationBuffer a,int repeats){
    __shared__ float scratch[4096];auto grid=cg::this_grid();
    for(int it=0;it<repeats;it++){
        if constexpr(Fused){rotation_q8_rows_phase(x,y,rows,K,B,signs,sign_stride,a,scratch);grid.sync();}
        else{rotation_rows_phase(x,y,rows,K,B,signs,sign_stride,scratch);grid.sync();
            for(int r=0;r<rows;r++){ActivationBuffer row=a;row.qs+=r*a.q_stride;row.scale+=r*a.scale_stride;quantize_phase(y+r*K,row);}grid.sync();}
    }
}
struct Device{std::vector<void*>p;template<class T>T*make(size_t n){T*x;CUDA(cudaMalloc(&x,n*sizeof(T)));p.push_back(x);return x;}template<class T>T*copy(const std::vector<T>&v){T*x=make<T>(v.size());CUDA(cudaMemcpy(x,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return x;}~Device(){for(auto*x:p)cudaFree(x);}};
template<class T>std::vector<T>get(T*p,size_t n){std::vector<T>v(n);CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
std::vector<double>transform(std::vector<double>x,int B,const float*signs){if(!B)return x;for(int base=0;base<(int)x.size();base+=B){for(int s=1;s<B;s*=2)for(int p=0;p<B/2;p++){int lo=base+(p/s)*(2*s)+p%s,hi=lo+s;double l=x[lo],r=x[hi];x[lo]=l+r;x[hi]=l-r;}double inv=1/std::sqrt((double)B);for(int i=0;i<B;i++)x[base+i]*=inv*(signs?signs[base+i]:1.f);}return x;}
struct Input{int rows,K,B;int64_t sign_stride;std::vector<float>x,signs;Input(int r,int k,int b,int mode,int scenario):rows(r),K(k),B(b),sign_stride(mode==2?k:0),x(r*k),signs(mode?(mode==2?r*k:k):0){std::mt19937 gen(4096+13*b+r+scenario);std::normal_distribution<float>n(0,1);for(auto&v:x)v=n(gen)*2;for(auto&v:signs)v=(gen()&1)?-1:1;if(scenario==1){std::fill(x.begin(),x.end(),0);for(int row=0;row<rows;row++)x[row*K+(row*17)%K]=1;}if(scenario==2){for(int i=0;i<r*k;i++)x[i]=(i%2?-1.f:1.f)*(i%7==0?1e-8f:1.f);}if(scenario==3){for(int i=0;i<r*k;i++)x[i]=(i%2?-0.f:0.f);}}};
struct Output{std::vector<float>y,scale;std::vector<int8_t>q;bool padding=true;float us=0;};
Output run(const Input&i,bool fused,int requested_grid=64,int repeats=1,bool inplace=false){
    Device m;float*x=m.copy(i.x),*y=inplace?x:m.make<float>(i.x.size()),*signs=i.signs.empty()?nullptr:m.copy(i.signs);ActivationBuffer a{};a.k=i.K;a.rows=i.rows;a.group=32;a.q_stride=i.K+32;a.scale_stride=i.K/32+3;a.qs=m.make<int8_t>(i.rows*a.q_stride);a.scale=m.make<float>(i.rows*a.scale_stride);CUDA(cudaMemset(a.qs,0x55,i.rows*a.q_stride));CUDA(cudaMemset(a.scale,0x55,i.rows*a.scale_stride*4));
    int active=0,sm=0;auto kernel=fused?rotation_control<true>:rotation_control<false>;CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,256,0));CUDA(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0));int grid=std::min(requested_grid,active*sm),rows=i.rows,K=i.K,B=i.B;int64_t ss=i.sign_stride;if(grid<1)exit(3);void*args[]={&x,&y,&rows,&K,&B,&signs,&ss,&a,&repeats};
    cudaEvent_t st,en;CUDA(cudaEventCreate(&st));CUDA(cudaEventCreate(&en));CUDA(cudaEventRecord(st));CUDA(cudaLaunchCooperativeKernel((void*)kernel,dim3(grid),dim3(256),args,0,nullptr));CUDA(cudaEventRecord(en));CUDA(cudaEventSynchronize(en));float ms;CUDA(cudaEventElapsedTime(&ms,st,en));CUDA(cudaEventDestroy(st));CUDA(cudaEventDestroy(en));Output o;o.us=ms*1000/repeats;o.y=get(y,i.x.size());auto q=get(a.qs,i.rows*a.q_stride);auto scale=get(a.scale,i.rows*a.scale_stride);
    for(int r=0;r<i.rows;r++){for(int k=0;k<i.K;k++)o.q.push_back(q[r*a.q_stride+k]);for(int g=0;g<i.K/32;g++)o.scale.push_back(scale[r*a.scale_stride+g]);for(int k=i.K;k<a.q_stride;k++)o.padding&=q[r*a.q_stride+k]==0x55;for(int g=i.K/32;g<a.scale_stride;g++){uint32_t b;memcpy(&b,&scale[r*a.scale_stride+g],4);o.padding&=b==0x55555555;}}
    return o;
}
bool bytes_equal(const Output&a,const Output&b){return a.y.size()==b.y.size()&&!memcmp(a.y.data(),b.y.data(),a.y.size()*4)&&a.q==b.q&&a.scale.size()==b.scale.size()&&!memcmp(a.scale.data(),b.scale.data(),a.scale.size()*4);}
int main(){bool pass=true;int cases=0;
    for(int K:{512,4096})for(int B:{0,128,256,512,4096})if(B==0||(B<=K&&K%B==0))for(int mode:{0,1,2})for(int scenario:{0,1,2,3}){
        int rows=K==512?8:1;Input i(rows,K,B,mode,scenario);auto a=run(i,false),b=run(i,true),c=run(i,true,1,1,true);bool exact=bytes_equal(a,b)&&bytes_equal(b,c),zero=B!=0||!memcmp(a.y.data(),i.x.data(),i.x.size()*4);double ab=0,sq=0,norm=0,inverse=0,equiv=0,native_equiv=0;
        for(int r=0;r<rows;r++){
            std::vector<double>x(i.x.begin()+r*K,i.x.begin()+(r+1)*K);const float*signs=i.signs.empty()?nullptr:i.signs.data()+r*i.sign_stride;auto expected=transform(x,B,signs);std::vector<double>back(a.y.begin()+r*K,a.y.begin()+(r+1)*K);if(B&&signs)for(int k=0;k<K;k++)back[k]*=signs[k];back=transform(back,B,nullptr);
            std::vector<double>w(K);for(int k=0;k<K;k++)w[k]=std::sin(k*.37+r*.21);auto rw=transform(w,B,signs);double raw=0,ideal=0,native=0;
            for(int k=0;k<K;k++){double d=a.y[r*K+k]-expected[k];ab=std::max(ab,std::abs(d));sq+=d*d;norm+=expected[k]*expected[k];inverse=std::max(inverse,std::abs(back[k]-x[k]));raw+=w[k]*x[k];ideal+=rw[k]*expected[k];native+=rw[k]*a.y[r*K+k];}
            equiv=std::max(equiv,std::abs(raw-ideal));native_equiv=std::max(native_equiv,std::abs(raw-native));
        }
        double rel=std::sqrt(sq/std::max(norm,1e-30));bool ok=exact&&zero&&a.padding&&b.padding&&c.padding&&(rel<8e-7||ab<1e-7)&&inverse<4e-6&&equiv<1e-9&&native_equiv<1e-3;pass&=ok;cases++;
        printf("{\"rows\":%d,\"K\":%d,\"B\":%d,\"sign_mode\":%d,\"scenario\":%d,\"cpu_guard\":%s,\"fused_split_inplace_bit_equal\":%s,\"padding_unchanged\":%s,\"disabled_identity\":%s,\"cpu_max_abs\":%.9g,\"cpu_relative_rms\":%.9g,\"inverse_max_abs\":%.9g,\"double_Wx_equivalence_abs\":%.9g,\"native_Wx_equivalence_abs\":%.9g}\n",rows,K,B,mode,scenario,ok?"true":"false",exact?"true":"false",a.padding&&b.padding&&c.padding?"true":"false",zero?"true":"false",ab,rel,inverse,equiv,native_equiv);fflush(stdout);
    }
    for(auto geometry:{std::pair<int,int>{1,4096},std::pair<int,int>{8,512}})for(int B:{0,128,256,512,4096})if(B==0||(B<=geometry.second&&geometry.second%B==0))for(bool fused:{false,true}){Input i(geometry.first,geometry.second,B,1,0);run(i,fused,188,4);auto o=run(i,fused,188,128);printf("{\"rows\":%d,\"K\":%d,\"B\":%d,\"fused\":%s,\"rotation_Q8_us\":%.9g,\"grid_barriers\":%d,\"shared_scratch_bytes\":16384}\n",i.rows,i.K,B,fused?"true":"false",o.us,fused?1:2);}
    printf("{\"passed\":%s,\"cases\":%d}\n",pass?"true":"false",cases);return pass?0:1;
}
