#include "digits.cuh"
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
void witness(const uint8_t* host,int k,cudaStream_t st,FILE* file,a8_fp4::WeightLayout layout){
  constexpr int n=96,ld=128,prefix=32;Memory m;auto* raw=m.alloc<uint8_t>(size_t(n)*k/128*18);auto* w=m.alloc<uint8_t>(size_t(n)*k/8);auto* s=m.alloc<uint16_t>(size_t(n)*k/128);auto* pw=m.alloc<uint32_t>(size_t(n)*k/128*4);auto* ps=m.alloc<__half>(size_t(n)*k/128);
  CUDA(cudaMemcpy(raw,host,size_t(n)*k/128*18,cudaMemcpyHostToDevice));CUDA(h5::launch_repack_q1_fp4(raw,k,n,w,s,n/32,1,0,true,st));CUDA(bitserial::prepare(raw,pw,ps,n,k,st));
  const void* weights=layout==a8_fp4::WeightLayout::WF8?(const void*)w:(const void*)pw;
  const void* scales=layout==a8_fp4::WeightLayout::WF8?(const void*)s:(const void*)ps;
  for(int r:{5,8,16,32}){
    auto c=carrier(m,k,r);size_t sz=size_t(r)*k;std::vector<int8_t> q(sz);for(size_t i=0;i<sz;i++)q[i]=int8_t(i%256-128);std::vector<float> sx(sz/128);for(size_t i=0;i<sx.size();i++)sx[i]=float(i%17+1)/1024.f;
    CUDA(cudaMemcpy(c.q,q.data(),sz,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(c.scales,sx.data(),sx.size()*4,cudaMemcpyHostToDevice));
    a8_fp4::Digits d{m.alloc<uint8_t>(a8_fp4::digit_bytes(k,r)),a8_fp4::digit_bytes(k,r)};CUDA(a8_fp4::encode(c,k,r,d,st));
    size_t count=size_t(r+2)*ld;auto* y=m.alloc<float>(count);auto* ref=m.alloc<float>(count);auto* initial=m.alloc<float>(count);auto* partial=m.alloc<float>(size_t(r)*n*8);
    std::vector<float> seed(count,-9876.25f),href(count),hy(count);for(int t=0;t<r;t++)for(int j=0;j<n;j++)seed[prefix+size_t(t)*ld+j]=float(int(mix(t*n+j)%1024)-512)/1024;
    CUDA(cudaMemcpy(initial,seed.data(),count*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(ref,seed.data(),count*4,cudaMemcpyHostToDevice));raw_reference<<<dim3((n+3)/4,r),128,0,st>>>(raw,c,ref+prefix,initial+prefix,n,k,r,ld,.75f,.25f);CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(href.data(),ref,count*4,cudaMemcpyDeviceToHost));
    for(int cfg=0;cfg<5;cfg++)for(int split:{1,4,8}){
      a8_fp4::Plan p{n,k,r,split,a8_fp4::Schedule(cfg),layout};CUDA(cudaMemcpyAsync(y,initial,count*4,cudaMemcpyDeviceToDevice,st));CUDA(a8_fp4::project(p,weights,scales,c,d,y+prefix,ld,partial,size_t(r)*n*8,.75f,.25f,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hy.data(),y,count*4,cudaMemcpyDeviceToHost));double max,rel;
      std::vector<float> active,expected;for(int t=0;t<r;t++)for(int j=0;j<n;j++){active.push_back(hy[prefix+size_t(t)*ld+j]);expected.push_back(href[prefix+size_t(t)*ld+j]);}check(active,expected,&max,&rel);
      for(size_t i=0;i<count;i++){bool live=i>=prefix&&i<prefix+size_t(r)*ld&&(i-prefix)%ld<size_t(n);if(!live&&hy[i]!=seed[i]){fprintf(stderr,"guard overwritten\n");exit(3);}}
      fprintf(file,"{\"kind\":\"witness\",\"layout\":\"%s\",\"r\":%d,\"cfg\":%d,\"splits\":%d,\"max_error\":%.9g,\"relative_rms\":%.9g,\"q_range\":[-128,127],\"alpha\":0.75,\"beta\":0.25,\"stride\":128,\"guards\":true}\n",layout==a8_fp4::WeightLayout::WF8?"wf8":"packed1",r,cfg,split,max,rel);
    }
    auto status=bitserial::InputStatus::NonFinite;CUDA(cudaMemcpy(c.status,&status,sizeof(status),cudaMemcpyHostToDevice));a8_fp4::Plan p{n,k,r,4,a8_fp4::Schedule::F4G2T32,layout};CUDA(a8_fp4::project(p,weights,scales,c,d,y+prefix,ld,partial,size_t(r)*n*8,1,0,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hy.data(),y,count*4,cudaMemcpyDeviceToHost));for(int j=0;j<n;j++)if(!std::isnan(hy[prefix+j])){fprintf(stderr,"invalidstatus not NaN\n");exit(3);}
    auto* x=m.alloc<float>(sz);std::vector<float> hx(sz,1.f);hx[0]=NAN;hx[128]=1e31f;CUDA(cudaMemcpy(x,hx.data(),sz*4,cudaMemcpyHostToDevice));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(a8_fp4::encode(c,k,r,d,st));CUDA(a8_fp4::project(p,weights,scales,c,d,y+prefix,ld,partial,size_t(r)*n*8,1,0,st));CUDA(cudaStreamSynchronize(st));bitserial::InputStatus error;CUDA(cudaMemcpy(&error,c.error,sizeof(error),cudaMemcpyDeviceToHost));if(error!=bitserial::InputStatus::ScaleRange)exit(3);CUDA(cudaMemcpy(hy.data(),y,count*4,cudaMemcpyDeviceToHost));for(int j=0;j<n;j++)if(!std::isnan(hy[prefix+j]))exit(3);
    CUDA(cudaMemset(x,0,sz*4));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(a8_fp4::encode(c,k,r,d,st));CUDA(a8_fp4::project(p,weights,scales,c,d,y+prefix,ld,partial,size_t(r)*n*8,1,0,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hy.data(),y,count*4,cudaMemcpyDeviceToHost));for(int t=0;t<r;t++)for(int j=0;j<n;j++)if(hy[prefix+size_t(t)*ld+j]!=0)exit(3);CUDA(cudaMemcpy(&error,c.error,sizeof(error),cudaMemcpyDeviceToHost));if(error!=bitserial::InputStatus::ScaleRange)exit(3);
  }
  fflush(file);puts("witness passed q[-128,127], R5/8/16/32, alpha/beta,stride,tails,status,error,zero");
}
int run(int argc,char** argv){
  if(argc!=6){fprintf(stderr,"bench MODEL gu|down|head|witness R OUTPUT wf8|packed1\n");return 1;}std::string kind=argv[2];int r=atoi(argv[3]);if((r<1||r>32)||(kind!="gu"&&kind!="down"&&kind!="head"&&kind!="witness"))return 1;
  a8_fp4::WeightLayout layout;
  if(std::string(argv[5])=="wf8")layout=a8_fp4::WeightLayout::WF8;else if(std::string(argv[5])=="packed1")layout=a8_fp4::WeightLayout::Packed1;else return 1;
  for(int cfg=0;cfg<5;cfg++)CUDA(a8_fp4::configure(a8_fp4::Schedule(cfg)));
  h5::Gguf model;model.open(argv[1]);auto tensor=model.t(kind=="gu"||kind=="witness"?"blk.0.ffn_gate.weight":kind=="down"?"blk.0.ffn_down.weight":"output.weight");std::vector<uint8_t> joined;
  if(kind=="gu"){auto up=model.t("blk.0.ffn_up.weight");joined.assign(tensor.data,tensor.data+tensor.bytes);joined.insert(joined.end(),up.data,up.data+up.bytes);tensor.data=joined.data();tensor.bytes=joined.size();tensor.ne[1]+=up.ne[1];}
  int n=tensor.ne[1],k=tensor.ne[0];cudaStream_t st;CUDA(cudaStreamCreate(&st));FILE* file=fopen(argv[4],"w");if(!file)return 2;if(kind=="witness"){witness(tensor.data,k,st,file,layout);fclose(file);return 0;}
  Memory m;size_t blocks=size_t(n)*k/128,out=size_t(n)*r;auto* raw=m.alloc<uint8_t>(blocks*18);auto* w=m.alloc<uint8_t>(size_t(n)*k/8);auto* s=m.alloc<uint16_t>(blocks);auto* signs=m.alloc<uint32_t>(blocks*4);auto* ws=m.alloc<__half>(blocks);
  CUDA(cudaMemcpy(raw,tensor.data,blocks*18,cudaMemcpyHostToDevice));CUDA(h5::launch_repack_q1_fp4(raw,k,n,w,s,n/32,1,0,true,st));CUDA(bitserial::prepare(raw,signs,ws,n,k,st));
  const void* weights=layout==a8_fp4::WeightLayout::WF8?(const void*)w:(const void*)signs;
  const void* scales=layout==a8_fp4::WeightLayout::WF8?(const void*)s:(const void*)ws;
  auto c=carrier(m,k,r);a8_fp4::Digits d{m.alloc<uint8_t>(a8_fp4::digit_bytes(k,r)),a8_fp4::digit_bytes(k,r)};auto* x=m.alloc<float>(size_t(r)*k);auto* y=m.alloc<float>(out);auto* ref=m.alloc<float>(out);auto* partial=m.alloc<float>(out*16);auto* residual=m.alloc<float>(out);auto* h=m.alloc<float>(out);auto* tokens=m.alloc<int>(r);int cn=kind=="gu"?n/2:n;auto nc=carrier(m,cn,r);
  std::vector<float> hx(size_t(r)*k),hr(out),hy(out),href(out);for(size_t i=0;i<hx.size();i++)hx[i]=float(int(mix(i+999)%65536)-32768)/32761;for(size_t i=0;i<hr.size();i++)hr[i]=float(int(mix(i+21)%65536)-32768)/32761;
  CUDA(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(residual,hr.data(),out*4,cudaMemcpyHostToDevice));Dp dp(m,n,k,r);
  CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));dp.run(signs,ws,c,ref,st);CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(href.data(),ref,out*4,cudaMemcpyDeviceToHost));size_t trashn=128*1024*1024/4;auto* trash=m.alloc<unsigned>(trashn);
  auto continuation=[&](){if(kind=="head"){h5::launch_argmax(y,n,tokens,r,st);return;}h5::pdl(next,dim3((size_t(cn)*r+255)/256),dim3(256),0,st,(const float*)y,(const float*)residual,h,n,r,kind=="gu");CUDA(bitserial::produce(bitserial::Route::Dp4a01,h,cn,r,nc,st));};
  auto record=[&](const char* route,int cfg,int splits,auto body){Graph g=capture(st,body);float warm=timed(g,st,trash,trashn,false),cold=timed(g,st,trash,trashn,true);CUDA(cudaMemcpy(hy.data(),y,out*4,cudaMemcpyDeviceToHost));double max,rel;check(hy,href,&max,&rel);
    bool parity=false;
    if(cfg>=0&&layout==a8_fp4::WeightLayout::Packed1){
      a8_fp4::Plan control{n,k,r,splits,a8_fp4::Schedule(cfg),a8_fp4::WeightLayout::WF8};
      CUDA(a8_fp4::project(control,w,s,c,d,y,n,partial,out*16,1,0,st));CUDA(cudaStreamSynchronize(st));
      std::vector<float> hc(out);CUDA(cudaMemcpy(hc.data(),y,out*4,cudaMemcpyDeviceToHost));
      for(size_t i=0;i<out;i++)if(hc[i]!=hy[i]){fprintf(stderr,"WF8 Packed1 parity mismatch cfg%d split%d at%zu\n",cfg,splits,i);exit(3);}parity=true;
    }
    fprintf(file,"{\"kind\":\"%s\",\"layout\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"route\":\"%s\",\"cfg\":%d,\"splits\":%d,\"inclusive_warm_us\":%.6f,\"inclusive_cold_us\":%.6f,\"max_error\":%.9g,\"relative_rms\":%.9g,\"wf8_packed1_bitexact\":%s}\n",kind.c_str(),argv[5],n,k,r,route,cfg,splits,warm,cold,max,rel,parity?"true":"false");fflush(file);destroy(g);};
  record("dp4a",-1,-1,[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));dp.run(signs,ws,c,y,st);continuation();});
  for(int cfg=0;cfg<5;cfg++)for(int split:{1,4,8,16}){a8_fp4::Plan p{n,k,r,split,a8_fp4::Schedule(cfg),layout};record("fp4_digits",cfg,split,[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(a8_fp4::encode(c,k,r,d,st));CUDA(a8_fp4::project(p,weights,scales,c,d,y,n,partial,out*16,1,0,st));continuation();});}
  fclose(file);printf("complete %s R%d\n",kind.c_str(),r);return 0;
}
int main(int argc,char** argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"error: %s\n",e.what());return 2;}}
