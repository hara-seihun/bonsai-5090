#include "signed_digits.cuh"
#include "../binary-a8-fp4/digits.cuh"
#include "../../src/kernels.cuh"
#include "../../src/gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define CUDA(x) do{auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(err));exit(2);}}while(0)
namespace h5 {void launch_argmax(const float*,int,int*,int,cudaStream_t);}
unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
struct Memory{std::vector<void*> all;template<class T>T* alloc(size_t n){T* p;CUDA(cudaMalloc(&p,std::max(size_t(1),n)*sizeof(T)));all.push_back(p);CUDA(cudaMemset(p,0,std::max(size_t(1),n)*sizeof(T)));return p;}~Memory(){for(auto p:all)cudaFree(p);}};
bitserial::Carrier carrier(Memory& m,int k,int r){return {m.alloc<int8_t>(size_t(r)*k),nullptr,m.alloc<float>(size_t(r)*k/128),m.alloc<int32_t>(size_t(r)*k/128),m.alloc<bitserial::InputStatus>(size_t(r)*k/128),m.alloc<bitserial::InputStatus>(1)};}
bitserial::Carrier view(bitserial::Carrier c,int k,int t){c.q+=size_t(t)*k;c.scales+=size_t(t)*k/128;c.dc+=size_t(t)*k/128;c.status+=size_t(t)*k/128;return c;}
struct Dp{int k,n,r;std::vector<bitserial::Plan> p;std::vector<float*> partial;Dp(Memory& m,int N,int K,int R):k(K),n(N),r(R){for(int t=0;t<r;t+=16){bitserial::Plan plan;CUDA(bitserial::select(n,k,std::min(16,r-t),&plan));p.push_back(plan);partial.push_back(m.alloc<float>(bitserial::partial_bytes(plan)/4));}}void run(const uint32_t* w,const __half* s,bitserial::Carrier c,float* y,cudaStream_t st){int t=0;for(size_t i=0;i<p.size();i++){CUDA(bitserial::project_carrier(p[i],w,s,view(c,k,t),y+size_t(t)*n,partial[i],1,0,st));t+=p[i].r;}}};
__global__ void raw_reference(const uint8_t* raw,bitserial::Carrier c,float* y,const float* initial,int n,int k,int r,int stride,float alpha,float beta){int col=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,t=blockIdx.y;if(col>=n||t>=r)return;float sum=0;for(int g=0;g<k/128;g++){const uint8_t* b=raw+(size_t(col)*k/128+g)*18;int dot=0;for(int j=lane;j<128;j+=32){int q=c.q[(size_t(t)*k/128+g)*128+j];dot+=((b[2+j/8]>>(j%8))&1)?q:-q;}for(int d=16;d;d/=2)dot+=__shfl_down_sync(0xffffffff,dot,d);float scale=__half2float(*(const __half*)b)*c.scales[size_t(t)*k/128+g];sum=fmaf(float(dot),scale,sum);}if(lane==0)y[size_t(t)*stride+col]=beta==0?alpha*sum:fmaf(alpha,sum,beta*initial[size_t(t)*stride+col]);}
__global__ void next(const float* y,const float* residual,float* h,int n,int r,bool gu){h5::pdl_wait();h5::pdl_launch();int cn=gu?n/2:n;size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(r)*cn)return;int t=i/cn,j=i%cn;if(gu){float g=y[size_t(t)*n+j];h[i]=g/(1.f+__expf(-g))*y[size_t(t)*n+cn+j];}else h[i]=y[i]+residual[i];}
__global__ void evict(unsigned* p,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]++;}
struct Graph{cudaGraph_t g;cudaGraphExec_t e;};
template<class F>Graph capture(cudaStream_t st,F f){Graph g;CUDA(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));f();CUDA(cudaStreamEndCapture(st,&g.g));CUDA(cudaGraphInstantiate(&g.e,g.g,0));return g;}
float timed(Graph g,cudaStream_t st,unsigned* trash,size_t n,bool cold){cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));std::vector<float> ts;for(int i=0;i<9;i++){if(cold)evict<<<(n+255)/256,256,0,st>>>(trash,n);CUDA(cudaEventRecord(a,st));CUDA(cudaGraphLaunch(g.e,st));CUDA(cudaEventRecord(b,st));CUDA(cudaEventSynchronize(b));float ms;CUDA(cudaEventElapsedTime(&ms,a,b));if(i>=2)ts.push_back(ms*1000);}std::sort(ts.begin(),ts.end());CUDA(cudaEventDestroy(a));CUDA(cudaEventDestroy(b));return ts[3];}
void destroy(Graph g){CUDA(cudaGraphExecDestroy(g.e));CUDA(cudaGraphDestroy(g.g));}
void check(const std::vector<float>& x,const std::vector<float>& y,double* maximum,double* relative){double sq=0,ss=0;*maximum=0;for(size_t i=0;i<x.size();i++){if(!std::isfinite(x[i])||!std::isfinite(y[i]))exit(3);double d=double(x[i])-y[i];*maximum=std::max(*maximum,std::abs(d));sq+=d*d;ss+=double(y[i])*y[i];}*relative=sqrt(sq/std::max(ss,1e-30));if(*relative>1e-5){fprintf(stderr,"rms %.9g max %.9g\n",*relative,*maximum);exit(3);}}
void witness(const uint8_t* host,int k,cudaStream_t st,FILE* file,signed_fp4::WeightLayout layout){
  constexpr int n=96,ld=128,prefix=32;Memory m;auto* raw=m.alloc<uint8_t>(size_t(n)*k/128*18);auto* w=m.alloc<uint8_t>(size_t(n)*k/8);auto* s=m.alloc<uint16_t>(size_t(n)*k/128);auto* pw=m.alloc<uint32_t>(size_t(n)*k/128*4);auto* ps=m.alloc<__half>(size_t(n)*k/128);
  CUDA(cudaMemcpy(raw,host,size_t(n)*k/128*18,cudaMemcpyHostToDevice));CUDA(h5::launch_repack_q1_fp4(raw,k,n,w,s,n/32,1,0,true,st));CUDA(bitserial::prepare(raw,pw,ps,n,k,st));
  const void* weights=layout==signed_fp4::WeightLayout::WF8?(const void*)w:(const void*)pw;
  const void* scales=layout==signed_fp4::WeightLayout::WF8?(const void*)s:(const void*)ps;
  for(int r:{5,16,32,65}){
    auto c=carrier(m,k,r);size_t sz=size_t(r)*k;std::vector<int8_t> q(sz);for(size_t i=0;i<sz;i++)q[i]=int8_t(i%256-128);std::vector<float> sx(sz/128);for(size_t i=0;i<sx.size();i++)sx[i]=float(i%17+1)/1024.f;
    CUDA(cudaMemcpy(c.q,q.data(),sz,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(c.scales,sx.data(),sx.size()*4,cudaMemcpyHostToDevice));
    signed_fp4::Digits d{m.alloc<uint8_t>(signed_fp4::digit_bytes(k,r)),signed_fp4::digit_bytes(k,r),signed_fp4::Encoding::ExactA8};CUDA(signed_fp4::encode(c,k,r,d,st));
    size_t count=size_t(r+2)*ld;auto* y=m.alloc<float>(count);auto* ref=m.alloc<float>(count);auto* initial=m.alloc<float>(count);auto* partial=m.alloc<float>(size_t(r)*n*8);
    std::vector<float> seed(count,-9876.25f),href(count),hy(count);for(int t=0;t<r;t++)for(int j=0;j<n;j++)seed[prefix+size_t(t)*ld+j]=float(int(mix(t*n+j)%1024)-512)/1024;
    CUDA(cudaMemcpy(initial,seed.data(),count*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(ref,seed.data(),count*4,cudaMemcpyHostToDevice));raw_reference<<<dim3((n+3)/4,r),128,0,st>>>(raw,c,ref+prefix,initial+prefix,n,k,r,ld,.75f,.25f);CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(href.data(),ref,count*4,cudaMemcpyDeviceToHost));
    for(int cfg=0;cfg<5;cfg++)for(int split:{1,4,8}){
      signed_fp4::Plan p{n,k,r,split,signed_fp4::Schedule(cfg),layout};CUDA(cudaMemcpyAsync(y,initial,count*4,cudaMemcpyDeviceToDevice,st));CUDA(signed_fp4::project(p,weights,scales,c,d,y+prefix,ld,partial,size_t(r)*n*8,.75f,.25f,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hy.data(),y,count*4,cudaMemcpyDeviceToHost));double max,rel;
      std::vector<float> active,expected;for(int t=0;t<r;t++)for(int j=0;j<n;j++){active.push_back(hy[prefix+size_t(t)*ld+j]);expected.push_back(href[prefix+size_t(t)*ld+j]);}check(active,expected,&max,&rel);
      for(size_t i=0;i<count;i++){bool live=i>=prefix&&i<prefix+size_t(r)*ld&&(i-prefix)%ld<size_t(n);if(!live&&hy[i]!=seed[i]){fprintf(stderr,"guard overwritten\n");exit(3);}}
      fprintf(file,"{\"kind\":\"witness\",\"layout\":\"%s\",\"r\":%d,\"cfg\":%d,\"splits\":%d,\"max_error\":%.9g,\"relative_rms\":%.9g,\"q_range\":[-128,127],\"alpha\":0.75,\"beta\":0.25,\"stride\":128,\"guards\":true}\n",layout==signed_fp4::WeightLayout::WF8?"wf8":"packed1",r,cfg,split,max,rel);
    }
    auto status=bitserial::InputStatus::NonFinite;CUDA(cudaMemcpy(c.status,&status,sizeof(status),cudaMemcpyHostToDevice));signed_fp4::Plan p{n,k,r,4,signed_fp4::Schedule::F4G2T32,layout};CUDA(signed_fp4::project(p,weights,scales,c,d,y+prefix,ld,partial,size_t(r)*n*8,1,0,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hy.data(),y,count*4,cudaMemcpyDeviceToHost));for(int j=0;j<n;j++)if(!std::isnan(hy[prefix+j])){fprintf(stderr,"invalidstatus not NaN\n");exit(3);}
    auto* x=m.alloc<float>(sz);std::vector<float> hx(sz,1.f);hx[0]=NAN;hx[128]=1e31f;CUDA(cudaMemcpy(x,hx.data(),sz*4,cudaMemcpyHostToDevice));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(signed_fp4::encode(c,k,r,d,st));CUDA(signed_fp4::project(p,weights,scales,c,d,y+prefix,ld,partial,size_t(r)*n*8,1,0,st));CUDA(cudaStreamSynchronize(st));bitserial::InputStatus error;CUDA(cudaMemcpy(&error,c.error,sizeof(error),cudaMemcpyDeviceToHost));if(error!=bitserial::InputStatus::ScaleRange)exit(3);CUDA(cudaMemcpy(hy.data(),y,count*4,cudaMemcpyDeviceToHost));for(int j=0;j<n;j++)if(!std::isnan(hy[prefix+j]))exit(3);
    if(signed_fp4::encode(c,k,r,{d.data,d.capacity-1,d.encoding},st)!=cudaErrorInvalidValue)exit(3);
    auto invalid=p;invalid.layout=signed_fp4::WeightLayout(99);if(signed_fp4::validate(invalid)!=cudaErrorInvalidValue)exit(3);
    CUDA(cudaMemset(x,0,sz*4));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(signed_fp4::encode(c,k,r,d,st));CUDA(signed_fp4::project(p,weights,scales,c,d,y+prefix,ld,partial,size_t(r)*n*8,1,0,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hy.data(),y,count*4,cudaMemcpyDeviceToHost));for(int t=0;t<r;t++)for(int j=0;j<n;j++)if(hy[prefix+size_t(t)*ld+j]!=0)exit(3);CUDA(cudaMemcpy(&error,c.error,sizeof(error),cudaMemcpyDeviceToHost));if(error!=bitserial::InputStatus::ScaleRange)exit(3);
  }
  fflush(file);puts("witness passed q[-128,127], R5/16/32/65, alpha/beta,stride,tails,status,error,zero");
}
int run(int argc,char** argv){
  const int twice_e2m1[8]={0,1,2,3,4,6,8,12};
  for(int q=-128;q<128;q++){
    unsigned a=unsigned(q)&3,b=(unsigned(q)>>2)&3;int c=q>>4,mag=abs(c);unsigned bc=b;
    if(mag==5||mag==7){c+=(b&1)?1:-1;bc=(0x95b4u>>(4*b))&15;}
    mag=abs(c);unsigned cc=(mag<=4?mag:mag/2+2)|(c<0?8u:0u);
    auto decode=[&](unsigned x){return twice_e2m1[x&7]*(x&8?-1:1);};
    if(decode(a)+4*decode(bc)+16*decode(cc)!=q){fprintf(stderr,"arithmetic E2M1 recode failed q%d\n",q);return 3;}
  }
  if(argc!=5){fprintf(stderr,"signed_digits_bench MODEL witness|gu|down|head R OUTPUT\n");return 1;}
  std::string kind=argv[2];const bool wit=kind=="witness";int r=atoi(argv[3]);
  if((!wit&&kind!="down"&&kind!="gu"&&kind!="head")||(!wit&&r!=16&&r!=32))return 1;
  for(int cfg=0;cfg<5;cfg++){CUDA(signed_fp4::configure(signed_fp4::Schedule(cfg)));CUDA(a8_fp4::configure(a8_fp4::Schedule(cfg)));}
  h5::Gguf model;model.open(argv[1]);auto tensor=model.t(kind=="gu"?"blk.0.ffn_gate.weight":kind=="head"?"output.weight":"blk.0.ffn_down.weight");
  std::vector<uint8_t> joined;if(kind=="gu"){auto up=model.t("blk.0.ffn_up.weight");joined.assign(tensor.data,tensor.data+tensor.bytes);joined.insert(joined.end(),up.data,up.data+up.bytes);tensor.data=joined.data();tensor.bytes=joined.size();tensor.ne[1]+=up.ne[1];}
  int n=tensor.ne[1],k=tensor.ne[0];
  cudaStream_t st;CUDA(cudaStreamCreate(&st));FILE* file=fopen(argv[4],"w");if(!file)return 2;
  if(wit){witness(tensor.data,k,st,file,signed_fp4::WeightLayout::Packed1);fclose(file);return 0;}
  Memory m;size_t blocks=size_t(n)*k/128,out=size_t(n)*r;
  auto* raw=m.alloc<uint8_t>(blocks*18);auto* signs=m.alloc<uint32_t>(blocks*4);auto* ws=m.alloc<__half>(blocks);
  CUDA(cudaMemcpy(raw,tensor.data,blocks*18,cudaMemcpyHostToDevice));CUDA(bitserial::prepare(raw,signs,ws,n,k,st));
  auto c=carrier(m,k,r);auto c8=carrier(m,k,r);
  signed_fp4::Digits d{m.alloc<uint8_t>(signed_fp4::digit_bytes(k,r)),signed_fp4::digit_bytes(k,r),signed_fp4::Encoding::ExactA8};
  a8_fp4::Digits d4{m.alloc<uint8_t>(a8_fp4::digit_bytes(k,r)),a8_fp4::digit_bytes(k,r)};
  auto* x=m.alloc<float>(size_t(r)*k);auto* y=m.alloc<float>(out);auto* control=m.alloc<float>(out);auto* partial=m.alloc<float>(out*16);
  auto* residual=m.alloc<float>(out);auto* h=m.alloc<float>(out);int cn=kind=="gu"?n/2:n;auto nc=carrier(m,cn,r);auto* tokens=m.alloc<int>(r);
  std::vector<float> hx(size_t(r)*k),hr(out),hy(out),href(out),href8(out),hc(out),hh(out),hhref(out);
  for(size_t i=0;i<hx.size();i++)hx[i]=float(int(mix(i+999)%65536)-32768)/32761;
  for(size_t i=0;i<hr.size();i++)hr[i]=float(int(mix(i+21)%65536)-32768)/32761;
  CUDA(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(residual,hr.data(),out*4,cudaMemcpyHostToDevice));
  Dp dp(m,n,k,r);CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c8,st));
  CUDA(signed_fp4::encode(c,k,r,d,st));CUDA(a8_fp4::encode(c,k,r,d4,st));
  raw_reference<<<dim3((n+3)/4,r),128,0,st>>>(raw,c,control,control,n,k,r,n,1,0);CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(href.data(),control,out*4,cudaMemcpyDeviceToHost));
  raw_reference<<<dim3((n+3)/4,r),128,0,st>>>(raw,c8,control,control,n,k,r,n,1,0);CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(href8.data(),control,out*4,cudaMemcpyDeviceToHost));
  std::vector<int8_t> q(size_t(r)*k);CUDA(cudaMemcpy(q.data(),c.q,q.size(),cudaMemcpyDeviceToHost));
  size_t trashn=128*1024*1024/4;auto* trash=m.alloc<unsigned>(trashn);
  auto continuation=[&](){if(kind=="head"){h5::launch_argmax(y,n,tokens,r,st);return;}h5::pdl(next,dim3((size_t(cn)*r+255)/256),dim3(256),0,st,(const float*)y,(const float*)residual,h,n,r,kind=="gu");CUDA(bitserial::produce(bitserial::Route::Dp4a01,h,cn,r,nc,st));};
  auto record=[&](const char* route,const char* boundary,int cfg,int split,bool full8,bool cont,auto body){
    Graph g=capture(st,body);float warm=timed(g,st,trash,trashn,false),cold=timed(g,st,trash,trashn,true);
    CUDA(cudaMemcpy(hy.data(),y,out*4,cudaMemcpyDeviceToHost));double max,rel;check(hy,full8?href8:href,&max,&rel);
    bool exact=false;
    if(cfg>=0&&!full8){a8_fp4::Plan p4{n,k,r,split,a8_fp4::Schedule(cfg),a8_fp4::WeightLayout::Packed1};CUDA(a8_fp4::encode(c,k,r,d4,st));CUDA(a8_fp4::project(p4,signs,ws,c,d4,control,n,partial,out*16,1,0,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hc.data(),control,out*4,cudaMemcpyDeviceToHost));
      for(size_t i=0;i<out;i++)if(hc[i]!=hy[i]){fprintf(stderr,"same-plan 3/4 mismatch %zu cfg%d split%d\n",i,cfg,split);exit(3);}exact=true;}
    double hmax=0,hrel=0;unsigned invalid=0;
    if(cont&&kind!="head"){hh.resize(size_t(cn)*r);hhref.resize(size_t(cn)*r);CUDA(cudaMemcpy(hh.data(),h,hh.size()*4,cudaMemcpyDeviceToHost));
      for(size_t i=0;i<hh.size();i++){if(kind=="gu"){int t=i/cn,j=i%cn;float gate=href[size_t(t)*n+j];hhref[i]=gate/(1.f+std::exp(-gate))*href[size_t(t)*n+cn+j];}else hhref[i]=href[i]+hr[i];}check(hh,hhref,&hmax,&hrel);
      std::vector<bitserial::InputStatus> status(size_t(r)*cn/128);CUDA(cudaMemcpy(status.data(),nc.status,status.size()*4,cudaMemcpyDeviceToHost));for(auto s:status)invalid+=s!=bitserial::InputStatus::Valid;if(invalid)exit(3);}
    if(cont&&kind=="head"){std::vector<int> ht(r);CUDA(cudaMemcpy(ht.data(),tokens,r*4,cudaMemcpyDeviceToHost));for(int t=0;t<r;t++){int expected=std::max_element(href.begin()+size_t(t)*n,href.begin()+size_t(t+1)*n)-(href.begin()+size_t(t)*n);if(ht[t]!=expected){fprintf(stderr,"head argmax parity mismatch row%d\n",t);exit(3);}}}
    fprintf(file,"{\"kind\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"route\":\"%s\",\"boundary\":\"%s\",\"cfg\":%d,\"splits\":%d,\"warm_us\":%.6f,\"cold_us\":%.6f,\"max_error\":%.9g,\"relative_rms\":%.9g,\"same_plan_four_digit_bitexact\":%s,\"residual_max_error\":%.9g,\"residual_relative_rms\":%.9g,\"next_producer_invalid_groups\":%u}\n",kind.c_str(),n,k,r,route,boundary,cfg,split,warm,cold,max,rel,exact?"true":"false",hmax,hrel,invalid);fflush(file);destroy(g);
  };
  for(int boundary=0;boundary<4;boundary++){
    const char* name=boundary==0?"project":boundary==1?"encode_project":boundary==2?"producer_encode_project":"producer_encode_project_residual_next_a8";bool incl=boundary>=2,cont=boundary==3;
    record("dp4a_given8",name,-1,-1,false,cont,[&](){if(incl)CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));dp.run(signs,ws,c,y,st);if(cont)continuation();});
    for(int cfg=0;cfg<5;cfg++)for(int split:{1,4,8,16}){
      signed_fp4::Plan p{n,k,r,split,signed_fp4::Schedule(cfg),signed_fp4::WeightLayout::Packed1};
      record("exact3_given8",name,cfg,split,false,cont,[&](){if(incl)CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));if(boundary>=1)CUDA(signed_fp4::encode(c,k,r,d,st));CUDA(signed_fp4::project(p,signs,ws,c,d,y,n,partial,out*16,1,0,st));if(cont)continuation();});
      a8_fp4::Plan p4{n,k,r,split,a8_fp4::Schedule(cfg),a8_fp4::WeightLayout::Packed1};
      record("four_digits_given8",name,cfg,split,false,cont,[&](){if(incl)CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));if(boundary>=1)CUDA(a8_fp4::encode(c,k,r,d4,st));CUDA(a8_fp4::project(p4,signs,ws,c,d4,y,n,partial,out*16,1,0,st));if(cont)continuation();});
    }
  }
  fclose(file);printf("complete exact3 %s R%d\n",kind.c_str(),r);return 0;
}
int main(int argc,char** argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"error: %s\n",e.what());return 2;}}
