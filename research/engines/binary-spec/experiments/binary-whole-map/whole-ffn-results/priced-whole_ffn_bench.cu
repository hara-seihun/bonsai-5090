#include "whole_ffn.cuh"
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

struct Arena {
  signed_fp4::Digits d;
  Arena(Memory& m,int k,int r):d{m.alloc<uint8_t>(signed_fp4::digit_bytes(k,r)),signed_fp4::digit_bytes(k,r),signed_fp4::Encoding::ExactA8}{}
};
void equal_carrier(bitserial::Carrier c,bitserial::Carrier ref,signed_fp4::Digits d,signed_fp4::Digits rd,int h,int r,bool compare_q){
  auto compare=[&](const void* a,const void* b,size_t bytes,const char* name){
    std::vector<uint8_t> x(bytes),y(bytes);CUDA(cudaMemcpy(x.data(),a,bytes,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(y.data(),b,bytes,cudaMemcpyDeviceToHost));
    if(x!=y){size_t i=0;while(i<bytes&&x[i]==y[i])i++;fprintf(stderr,"%s mismatch byte %zu: %u != %u\n",name,i,unsigned(x[i]),unsigned(y[i]));exit(3);}
  };
  if(compare_q)compare(c.q,ref.q,size_t(h)*r,"A8 q");
  compare(c.scales,ref.scales,size_t(h)*r/128*4,"A8 scales");
  compare(c.dc,ref.dc,size_t(h)*r/128*4,"A8 DC");
  compare(c.status,ref.status,size_t(h)*r/128*4,"A8 status");
  compare(d.data,rd.data,d.capacity,"three-plane bytes including padded tails");
}
int run(int argc,char** argv){
  if(argc!=4){fprintf(stderr,"whole_ffn_bench MODEL R OUTPUT\n");return 1;}
  int r=atoi(argv[2]);if(r<1||r>65)return 1;
  using signed_fp4::Schedule;using signed_fp4::WeightLayout;
  for(int cfg=0;cfg<4;cfg++){CUDA(signed_fp4::configure(Schedule(cfg)));CUDA(whole_ffn::configure(Schedule(cfg)));}
  h5::Gguf model;model.open(argv[1]);auto gate=model.t("blk.0.ffn_gate.weight"),up=model.t("blk.0.ffn_up.weight"),down=model.t("blk.0.ffn_down.weight");
  int k=gate.ne[0],h=gate.ne[1],n=down.ne[1];if(down.ne[0]!=h||up.bytes!=gate.bytes)return 1;
  std::vector<uint8_t> joined(gate.data,gate.data+gate.bytes);joined.insert(joined.end(),up.data,up.data+up.bytes);
  Memory m;cudaStream_t st;CUDA(cudaStreamCreate(&st));FILE* f=fopen(argv[3],"w");if(!f)return 2;
  auto* gr=m.alloc<uint8_t>(joined.size());auto* dr=m.alloc<uint8_t>(down.bytes);
  CUDA(cudaMemcpy(gr,joined.data(),joined.size(),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dr,down.data,down.bytes,cudaMemcpyHostToDevice));
  size_t gb=size_t(2*h)*k/128,db=size_t(n)*h/128;
  auto* gw=m.alloc<uint32_t>(gb*4);auto* gs=m.alloc<__half>(gb);auto* cw=m.alloc<uint32_t>(gb*4);auto* cs=m.alloc<__half>(gb);auto* dw=m.alloc<uint32_t>(db*4);auto* ds=m.alloc<__half>(db);
  CUDA(bitserial::prepare(gr,gw,gs,2*h,k,st));CUDA(bitserial::prepare(dr,dw,ds,n,h,st));
  cudaEvent_t pa,pb;CUDA(cudaEventCreate(&pa));CUDA(cudaEventCreate(&pb));CUDA(cudaEventRecord(pa,st));CUDA(whole_ffn::prepare(gr,h,k,cw,cs,st));CUDA(cudaEventRecord(pb,st));CUDA(cudaEventSynchronize(pb));float prep;CUDA(cudaEventElapsedTime(&prep,pa,pb));
  fprintf(f,"{\"kind\":\"preparation\",\"static_relabel_us\":%.6f,\"candidate_gu_resident_bytes\":%zu,\"candidate_down_resident_bytes\":%zu,\"extra_static_bytes_vs_packed1\":0,\"k\":%d,\"hidden\":%d,\"down_n\":%d,\"r\":%d}\n",prep*1000,gb*18,db*18,k,h,n,r);
  auto c=carrier(m,k,r),nc=carrier(m,h,r),refc=carrier(m,h,r);Arena input(m,k,r),candidate(m,h,r),reference(m,h,r);
  auto* x=m.alloc<float>(size_t(k)*r);auto* gu=m.alloc<float>(size_t(2*h)*r);auto* hidden=m.alloc<float>(size_t(h)*r);auto* y=m.alloc<float>(size_t(n)*r);auto* ry=m.alloc<float>(size_t(n)*r);
  auto* gp=m.alloc<float>(size_t(2*h)*r*8);auto* dp=m.alloc<float>(size_t(n)*r*16);
  std::vector<float> hx(size_t(k)*r);for(size_t i=0;i<hx.size();i++)hx[i]=float(int(mix(i+999)%65536)-32768)/32761;
  CUDA(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));if(r!=16&&r!=32){
    std::vector<int8_t> fullq(size_t(k)*r);std::vector<int32_t> dc(size_t(k)*r/128);
    for(size_t i=0;i<fullq.size();i++){fullq[i]=int8_t(i%256-128);dc[i/128]+=fullq[i];}
    CUDA(cudaMemcpyAsync(c.q,fullq.data(),fullq.size(),cudaMemcpyHostToDevice,st));CUDA(cudaMemcpyAsync(c.dc,dc.data(),dc.size()*4,cudaMemcpyHostToDevice,st));
  }
  CUDA(signed_fp4::encode(c,k,r,input.d,st));
  size_t trashn=128*1024*1024/4;auto* trash=m.alloc<unsigned>(trashn);
  auto materialize=[&](int cfg,int split,bitserial::Carrier out,signed_fp4::Digits digits){
    signed_fp4::Plan p{2*h,k,r,split,Schedule(cfg),WeightLayout::Packed1};CUDA(signed_fp4::project(p,gw,gs,c,input.d,gu,2*h,gp,size_t(2*h)*r*8,1,0,st));
    h5::pdl(next,dim3((size_t(h)*r+255)/256),dim3(256),0,st,(const float*)gu,(const float*)gu,hidden,2*h,r,true);
    CUDA(bitserial::produce(bitserial::Route::Dp4a01,hidden,h,r,out,st));CUDA(signed_fp4::encode(out,h,r,digits,st));
  };
  auto project_down=[&](int cfg,int split,bitserial::Carrier c,signed_fp4::Digits d,float* out){
    signed_fp4::Plan p{n,h,r,split,Schedule(cfg),WeightLayout::Packed1};CUDA(signed_fp4::project(p,dw,ds,c,d,out,n,dp,size_t(n)*r*16,1,0,st));
  };
  // Real trained GU/down witness against the identical native dot/scale/SFU association.
  for(int cfg=0;cfg<4;cfg++){
    materialize(cfg,1,refc,reference.d);project_down(1,8,refc,reference.d,ry);
    CUDA(whole_ffn::gate_up({h,k,r,Schedule(cfg)},cw,cs,c,input.d,nc,candidate.d,st));project_down(1,8,nc,candidate.d,y);CUDA(cudaStreamSynchronize(st));
    equal_carrier(nc,refc,candidate.d,reference.d,h,r,true);
    std::vector<float> a(size_t(n)*r),b(a.size());CUDA(cudaMemcpy(a.data(),y,a.size()*4,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(b.data(),ry,b.size()*4,cudaMemcpyDeviceToHost));if(a!=b){fprintf(stderr,"complete down not bitexact same plan\n");return 3;}
    fprintf(f,"{\"kind\":\"witness\",\"cfg\":%d,\"r\":%d,\"same_plan_a8_scales_dc_status_digits_bitexact\":true,\"complete_trained_down_bitexact\":true}\n",cfg,r);fflush(f);
  }
  // Exercise the complete API with residual and non-tight row stride, preserving both guards.
  {
    int cfg=r<=16?0:1,split=r<=16?16:8,ld=n+32,prefix=32;
    size_t count=size_t(r+2)*ld;
    auto* actual=m.alloc<float>(count);auto* expected=m.alloc<float>(count);
    std::vector<float> seed(count,-9876.25f),ha(count),he(count);
    for(int t=0;t<r;t++)for(int j=0;j<n;j++)seed[prefix+size_t(t)*ld+j]=float(int(mix(t*n+j+41)%1024)-512)/1024;
    CUDA(cudaMemcpyAsync(actual,seed.data(),count*4,cudaMemcpyHostToDevice,st));CUDA(cudaMemcpyAsync(expected,seed.data(),count*4,cudaMemcpyHostToDevice,st));
    materialize(cfg,1,refc,reference.d);
    signed_fp4::Plan p{n,h,r,split,Schedule::F4G2T16,WeightLayout::Packed1};
    CUDA(signed_fp4::project(p,dw,ds,refc,reference.d,expected+prefix,ld,dp,size_t(n)*r*16,1,1,st));
    auto direct=nc;direct.q=nullptr;
    whole_ffn::Scratch scratch{input.d,direct,candidate.d,dp,size_t(n)*r*16};
    CUDA(whole_ffn::project({h,k,r,Schedule(cfg)},p,{cw,cs,dw,ds},c,scratch,actual+prefix,ld,1,1,st));CUDA(cudaStreamSynchronize(st));
    CUDA(cudaMemcpy(ha.data(),actual,count*4,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(he.data(),expected,count*4,cudaMemcpyDeviceToHost));if(ha!=he){fprintf(stderr,"complete wrapper residual/stride/guard parity failed R%d\n",r);return 3;}
    equal_carrier(direct,refc,candidate.d,reference.d,h,r,false);
    fprintf(f,"{\"kind\":\"complete_wrapper_witness\",\"r\":%d,\"gu_cfg\":%d,\"down_cfg\":1,\"down_splits\":%d,\"alpha\":1,\"beta\":1,\"row_stride\":%d,\"residual_and_guards_bitexact\":true,\"null_q\":true,\"next_carrier_and_all_digit_padding_bitexact\":true}\n",r,cfg,split,ld);fflush(f);
  }
  // Independent original row-major integer-dot GU reference, followed by native SFU/A8 and trained down.
  raw_reference<<<dim3((2*h+3)/4,r),128,0,st>>>(gr,c,gu,gu,2*h,k,r,2*h,1,0);
  h5::pdl(next,dim3((size_t(h)*r+255)/256),dim3(256),0,st,(const float*)gu,(const float*)gu,hidden,2*h,r,true);
  CUDA(bitserial::produce(bitserial::Route::Dp4a01,hidden,h,r,refc,st));
  raw_reference<<<dim3((n+3)/4,r),128,0,st>>>(dr,refc,ry,ry,n,h,r,n,1,0);CUDA(cudaStreamSynchronize(st));
  std::vector<float> integer_reference(size_t(n)*r);CUDA(cudaMemcpy(integer_reference.data(),ry,integer_reference.size()*4,cudaMemcpyDeviceToHost));
  // No global A8-byte tensor on the candidate recurring path.
  auto no_q=nc;no_q.q=nullptr;
  if(r==16||r==32)for(int boundary=0;boundary<2;boundary++)for(bool complete:{false,true}){
    const char* name=boundary?"f32_producer_encode":"given_a8_encode";
    auto record=[&](const char* route,int cfg,int split,int downcfg,int downsplit,auto body){
      Graph g=capture(st,body);float warm=timed(g,st,trash,trashn,false),cold=timed(g,st,trash,trashn,true);
      double maximum=0,relative=0;if(complete){std::vector<float> a(integer_reference.size());CUDA(cudaMemcpy(a.data(),y,a.size()*4,cudaMemcpyDeviceToHost));check(a,integer_reference,&maximum,&relative);}
      fprintf(f,"{\"kind\":\"price\",\"route\":\"%s\",\"boundary\":\"%s\",\"complete_down\":%s,\"r\":%d,\"gu_cfg\":%d,\"gu_splits\":%d,\"down_cfg\":%d,\"down_splits\":%d,\"warm_us\":%.6f,\"cold_us\":%.6f,\"integer_reference_max_error\":%.9g,\"integer_reference_relative_rms\":%.9g}\n",route,name,complete?"true":"false",r,cfg,split,downcfg,downsplit,warm,cold,maximum,relative);fflush(f);destroy(g);
    };
    for(int cfg=0;cfg<4;cfg++)for(int split:{1,4,8})for(int dc:{1,2})for(int ds:{4,8,16}){
      if(!complete&&(dc!=1||ds!=8))continue;
      record("materialized_exact3",cfg,split,dc,ds,[&](){if(boundary)CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(signed_fp4::encode(c,k,r,input.d,st));materialize(cfg,split,nc,candidate.d);if(complete)project_down(dc,ds,nc,candidate.d,y);});
    }
    for(int cfg=0;cfg<4;cfg++)for(int dc:{1,2})for(int ds:{4,8,16}){
      if(!complete&&(dc!=1||ds!=8))continue;
      record("compound_exact3",cfg,1,dc,ds,[&](){if(boundary)CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(signed_fp4::encode(c,k,r,input.d,st));CUDA(whole_ffn::gate_up({h,k,r,Schedule(cfg)},cw,cs,c,input.d,no_q,candidate.d,st));if(complete)project_down(dc,ds,no_q,candidate.d,y);});
    }
  }
  // Zero and typed-invalid/status propagation on complete original trained down.
  CUDA(cudaMemsetAsync(x,0,size_t(k)*r*4,st));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(signed_fp4::encode(c,k,r,input.d,st));
  CUDA(whole_ffn::gate_up({h,k,r,Schedule::F4G2T16},cw,cs,c,input.d,no_q,candidate.d,st));project_down(1,8,no_q,candidate.d,y);CUDA(cudaStreamSynchronize(st));
  std::vector<float> a(size_t(n)*r);CUDA(cudaMemcpy(a.data(),y,a.size()*4,cudaMemcpyDeviceToHost));for(float v:a)if(v!=0)return 3;
  auto invalid=bitserial::InputStatus::NonFinite;CUDA(cudaMemcpyAsync(c.status,&invalid,4,cudaMemcpyHostToDevice,st));CUDA(whole_ffn::gate_up({h,k,r,Schedule::F4G2T16},cw,cs,c,input.d,no_q,candidate.d,st));project_down(1,8,no_q,candidate.d,y);CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(a.data(),y,a.size()*4,cudaMemcpyDeviceToHost));for(int j=0;j<n;j++)if(!std::isnan(a[j]))return 3;
  bitserial::InputStatus error;CUDA(cudaMemcpy(&error,nc.error,4,cudaMemcpyDeviceToHost));if(error!=invalid)return 3;
  CUDA(cudaMemsetAsync(c.status,0,size_t(k)*r/128*4,st));CUDA(whole_ffn::gate_up({h,k,r,Schedule::F4G2T16},cw,cs,c,input.d,no_q,candidate.d,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(&error,nc.error,4,cudaMemcpyDeviceToHost));if(error!=invalid)return 3;
  if(whole_ffn::gate_up({h,k,r,Schedule::F4G2T16},cw,cs,c,input.d,no_q,{candidate.d.data,candidate.d.capacity-1,candidate.d.encoding},st)!=cudaErrorInvalidValue)return 3;
  hx.assign(size_t(k)*r,1.f);hx[0]=NAN;hx[128]=1e31f;
  CUDA(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));
  CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(signed_fp4::encode(c,k,r,input.d,st));
  CUDA(whole_ffn::gate_up({h,k,r,Schedule::F4G2T16},cw,cs,c,input.d,no_q,candidate.d,st));project_down(1,8,no_q,candidate.d,y);CUDA(cudaStreamSynchronize(st));
  CUDA(cudaMemcpy(&error,c.error,4,cudaMemcpyDeviceToHost));if(error!=bitserial::InputStatus::ScaleRange)return 3;
  CUDA(cudaMemcpy(a.data(),y,a.size()*4,cudaMemcpyDeviceToHost));for(int j=0;j<n;j++)if(!std::isnan(a[j]))return 3;
  fprintf(f,"{\"kind\":\"contracts\",\"r\":%d,\"zero_complete_down\":true,\"invalid_complete_down_nan\":true,\"sticky_error\":true,\"source_scale_range_error_preserved\":true,\"null_q_path\":true,\"undersized_digits_rejected\":true}\n",r);fclose(f);printf("whole FFN complete R%d\n",r);return 0;
}
int main(int argc,char** argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"error: %s\n",e.what());return 2;}}
