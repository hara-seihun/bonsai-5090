#include "whole_ffn.cuh"
#include "../binary-composition/composition.cuh"
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
void check(const std::vector<float>& x,const std::vector<float>& y,double* maximum,double* relative){double sq=0,ss=0;*maximum=0;for(size_t i=0;i<x.size();i++){if(!std::isfinite(x[i])||!std::isfinite(y[i]))exit(3);double d=double(x[i])-y[i];*maximum=std::max(*maximum,std::abs(d));sq+=d*d;ss+=double(y[i])*y[i];}*relative=sqrt(sq/std::max(ss,1e-30));if(*relative>1e-2){fprintf(stderr,"rms %.9g max %.9g\n",*relative,*maximum);exit(3);}}


int controls(int argc,char** argv){
  if(argc!=4)return 1;int r=atoi(argv[2]);if(r!=5&&r!=16&&r!=32)return 1;
  using signed_fp4::Schedule;using signed_fp4::WeightLayout;
  for(int cfg=0;cfg<5;cfg++){CUDA(signed_fp4::configure(Schedule(cfg)));CUDA(a8_fp4::configure(a8_fp4::Schedule(cfg)));if(cfg<4)CUDA(whole_ffn::configure(Schedule(cfg)));}
  h5::Gguf model;model.open(argv[1]);auto gate=model.t("blk.0.ffn_gate.weight"),up=model.t("blk.0.ffn_up.weight"),down=model.t("blk.0.ffn_down.weight");
  int k=gate.ne[0],h=gate.ne[1],n=down.ne[1];std::vector<uint8_t> joined(gate.data,gate.data+gate.bytes);joined.insert(joined.end(),up.data,up.data+up.bytes);
  Memory m;cudaStream_t st;CUDA(cudaStreamCreate(&st));FILE* f=fopen(argv[3],"w");if(!f)return 2;
  auto* gr=m.alloc<uint8_t>(joined.size());auto* dr=m.alloc<uint8_t>(down.bytes);CUDA(cudaMemcpy(gr,joined.data(),joined.size(),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dr,down.data,down.bytes,cudaMemcpyHostToDevice));
  size_t gb=size_t(2*h)*k/128,db=size_t(n)*h/128;
  auto* gw=m.alloc<uint32_t>(gb*4);auto* gs=m.alloc<__half>(gb);auto* cw=m.alloc<uint32_t>(gb*4);auto* cs=m.alloc<__half>(gb);auto* dw=m.alloc<uint32_t>(db*4);auto* ds=m.alloc<__half>(db);
  CUDA(bitserial::prepare(gr,gw,gs,2*h,k,st));CUDA(bitserial::prepare(dr,dw,ds,n,h,st));CUDA(whole_ffn::prepare(gr,h,k,cw,cs,st));
  auto c=carrier(m,k,r),nc=carrier(m,h,r);
  a8_fp4::Digits input{m.alloc<uint8_t>(a8_fp4::digit_bytes(k,r)),a8_fp4::digit_bytes(k,r)},hd{m.alloc<uint8_t>(a8_fp4::digit_bytes(h,r)),a8_fp4::digit_bytes(h,r)};
  signed_fp4::Digits in3{m.alloc<uint8_t>(signed_fp4::digit_bytes(k,r)),signed_fp4::digit_bytes(k,r),signed_fp4::Encoding::ExactA8},out3{m.alloc<uint8_t>(signed_fp4::digit_bytes(h,r)),signed_fp4::digit_bytes(h,r),signed_fp4::Encoding::ExactA8};
  auto* x=m.alloc<float>(size_t(k)*r);auto* gu=m.alloc<float>(size_t(2*h)*r);auto* hidden=m.alloc<float>(size_t(h)*r);auto* y=m.alloc<float>(size_t(n)*r);auto* reference=m.alloc<float>(size_t(n)*r);
  auto* gp=m.alloc<float>(size_t(2*h)*r*16);auto* dp=m.alloc<float>(size_t(n)*r*16);auto* cp=m.alloc<float>(size_t(2*h)*16*((k/128+3)/4));
  std::vector<float> hx(size_t(k)*r);for(size_t i=0;i<hx.size();i++)hx[i]=float(int(mix(i+999)%65536)-32768)/32761;
  CUDA(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));
  size_t trashn=128*1024*1024/4;auto* trash=m.alloc<unsigned>(trashn);Dp dgu(m,2*h,k,r),ddown(m,n,h,r);
  auto tail=[&](){h5::pdl(next,dim3((size_t(h)*r+255)/256),dim3(256),0,st,(const float*)gu,(const float*)gu,hidden,2*h,r,true);CUDA(bitserial::produce(bitserial::Route::Dp4a01,hidden,h,r,nc,st));};
  auto four=[&](int gc,int gspl,int dc,int dspl){
    CUDA(a8_fp4::encode(c,k,r,input,st));a8_fp4::Plan g{2*h,k,r,gspl,a8_fp4::Schedule(gc),a8_fp4::WeightLayout::Packed1};CUDA(a8_fp4::project(g,gw,gs,c,input,gu,2*h,gp,size_t(2*h)*r*16,1,0,st));tail();CUDA(a8_fp4::encode(nc,h,r,hd,st));
    a8_fp4::Plan d{n,h,r,dspl,a8_fp4::Schedule(dc),a8_fp4::WeightLayout::Packed1};CUDA(a8_fp4::project(d,dw,ds,nc,hd,y,n,dp,size_t(n)*r*16,1,0,st));
  };
  int down_chunk=4,down_threads=128;
  auto down_dp=[&](){int t=0;for(size_t i=0;i<ddown.p.size();i++){auto p=ddown.p[i];p.groups_per_split=down_chunk;p.threads=down_threads;CUDA(bitserial::project_carrier(p,dw,ds,view(nc,h,t),y+size_t(t)*n,ddown.partial[i],1,0,st));t+=p.r;}};
  auto fused_dp=[&](int tile,int chunk){for(int t=0;t<r;t+=16){int rows=std::min(16,r-t);CUDA(composition::fused({h,k,rows,tile,chunk},gw,gs,view(c,k,t),view(nc,h,t),cp,st));}down_dp();};
  auto compound=[&](int cfg,int downcfg,int splits){auto qless=nc;qless.q=nullptr;
    whole_ffn::Scratch scratch{in3,qless,out3,dp,size_t(n)*r*16};signed_fp4::Plan d{n,h,r,splits,Schedule(downcfg),WeightLayout::Packed1};CUDA(whole_ffn::project({h,k,r,Schedule(cfg)},d,{cw,cs,dw,ds},c,scratch,y,n,1,0,st));
  };
  auto materialized3=[&](int cfg,int split,int downcfg,int downsplit){
    CUDA(signed_fp4::encode(c,k,r,in3,st));signed_fp4::Plan g{2*h,k,r,split,Schedule(cfg),WeightLayout::Packed1};CUDA(signed_fp4::project(g,gw,gs,c,in3,gu,2*h,gp,size_t(2*h)*r*16,1,0,st));tail();CUDA(signed_fp4::encode(nc,h,r,out3,st));
    signed_fp4::Plan d{n,h,r,downsplit,Schedule(downcfg),WeightLayout::Packed1};CUDA(signed_fp4::project(d,dw,ds,nc,out3,y,n,dp,size_t(n)*r*16,1,0,st));
  };
  // Original row-major integer-dot GU, native SiLU/A8, original row-major trained down.
  raw_reference<<<dim3((2*h+3)/4,r),128,0,st>>>(gr,c,gu,gu,2*h,k,r,2*h,1,0);tail();raw_reference<<<dim3((n+3)/4,r),128,0,st>>>(dr,nc,reference,reference,n,h,r,n,1,0);CUDA(cudaStreamSynchronize(st));
  std::vector<float> ref(size_t(n)*r);CUDA(cudaMemcpy(ref.data(),reference,ref.size()*4,cudaMemcpyDeviceToHost));
  for(int boundary=0;boundary<2;boundary++){
    auto record=[&](const char* route,int gc,int gspl,int dc,int dspl,int tile,int chunk,auto body){
      Graph g=capture(st,[&](){if(boundary)CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));body();});
      float warm=timed(g,st,trash,trashn,false),cold=timed(g,st,trash,trashn,true);std::vector<float> actual(ref.size());CUDA(cudaMemcpy(actual.data(),y,actual.size()*4,cudaMemcpyDeviceToHost));double max,rel;check(actual,ref,&max,&rel);
      fprintf(f,"{\"kind\":\"complete_controls\",\"route\":\"%s\",\"boundary\":\"%s\",\"r\":%d,\"k\":%d,\"hidden\":%d,\"down_n\":%d,\"gu_cfg\":%d,\"gu_splits\":%d,\"down_cfg\":%d,\"down_splits\":%d,\"dp_tile\":%d,\"dp_chunk\":%d,\"dp_down_chunk\":%d,\"dp_down_threads\":%d,\"warm_us\":%.6f,\"cold_us\":%.6f,\"integer_reference_max_error\":%.9g,\"integer_reference_relative_rms\":%.9g}\n",route,boundary?"f32_producer_encode":"given_a8_encode",r,k,h,n,gc,gspl,dc,dspl,tile,chunk,route[0]=='d'?down_chunk:0,route[0]=='d'?down_threads:0,warm,cold,max,rel);fflush(f);destroy(g);
    };
    for(int gc=0;gc<5;gc++)for(int gspl:{1,4,8})for(int dc:{1,2})for(int dspl:{4,8,16})record("four_digits",gc,gspl,dc,dspl,0,0,[&](){four(gc,gspl,dc,dspl);});
    for(int chunk_down:{4,16,32,h/128})for(int threads_down:{64,128,256}){
      down_chunk=chunk_down;down_threads=threads_down;
      record("dp4a_materialized",-1,-1,-1,-1,0,0,[&](){dgu.run(gw,gs,c,gu,st);tail();down_dp();});
      for(int tile:r==5?std::vector<int>{1}:std::vector<int>{4,8})for(int chunk:{4,16,k/128})record("dp4a_fused_gu_to_a8",-1,-1,-1,-1,tile,chunk,[&](){fused_dp(tile,chunk);});
    }
    if(r==5){
      for(int cfg=0;cfg<3;cfg++)for(int split:{1,4,8})for(int dc:{0,1})for(int dspl:{4,8,16})record("materialized_exact3",cfg,split,dc,dspl,0,0,[&](){materialized3(cfg,split,dc,dspl);});
      for(int cfg:{0,1})for(int dc:{0,1})for(int dspl:{4,8,16})record("compound_exact3",cfg,1,dc,dspl,0,0,[&](){compound(cfg,dc,dspl);});
    }else record("compound_exact3",r==16?0:1,1,1,r==16?16:8,0,0,[&](){compound(r==16?0:1,1,r==16?16:8);});
  }
  fclose(f);printf("complete controls R%d passed\n",r);return 0;
}
int main(int argc,char** argv){try{return controls(argc,argv);}catch(const std::exception& e){fprintf(stderr,"error: %s\n",e.what());return 2;}}
