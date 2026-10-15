#include "composition.cuh"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(c) do {cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}} while(0)
using bitserial::Carrier;
struct DeviceWeights {uint32_t* signs;__half* scales;};
DeviceWeights weights(const std::vector<uint8_t>& bytes,int n,int k,cudaStream_t st) {
  DeviceWeights w;uint8_t* raw;
  CK(cudaMalloc(&raw,bytes.size()));CK(cudaMalloc(&w.signs,bitserial::sign_bytes(n,k)));CK(cudaMalloc(&w.scales,bitserial::scale_bytes(n,k)));
  CK(cudaMemcpyAsync(raw,bytes.data(),bytes.size(),cudaMemcpyHostToDevice,st));CK(bitserial::prepare(raw,w.signs,w.scales,n,k,st));
  CK(cudaStreamSynchronize(st));CK(cudaFree(raw));return w;
}
Carrier carrier(int k,int r) {
  Carrier c{};CK(cudaMalloc(&c.q,size_t(k)*r));CK(cudaMalloc(&c.scales,size_t(k/128)*r*4));CK(cudaMalloc(&c.dc,size_t(k/128)*r*4));
  CK(cudaMalloc(&c.status,size_t(k/128)*r*4));CK(cudaMalloc(&c.error,4));CK(cudaMemset(c.error,0,4));return c;
}
struct HostCarrier {std::vector<int8_t> q;std::vector<float> sx;std::vector<int32_t> dc;std::vector<bitserial::InputStatus> status;};
HostCarrier download(Carrier c,int k,int r) {
  HostCarrier h;h.q.resize(size_t(k)*r);h.sx.resize(size_t(k/128)*r);h.dc.resize(h.sx.size());h.status.resize(h.sx.size());
  CK(cudaMemcpy(h.q.data(),c.q,h.q.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(h.sx.data(),c.scales,h.sx.size()*4,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.dc.data(),c.dc,h.dc.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(h.status.data(),c.status,h.status.size()*4,cudaMemcpyDeviceToHost));return h;
}
void equal(const HostCarrier& a,const HostCarrier& b) {
  if(a.q!=b.q||a.dc!=b.dc||a.status!=b.status||a.sx.size()!=b.sx.size()||memcmp(a.sx.data(),b.sx.data(),a.sx.size()*4)) {
    size_t dq=0,ds=0;for(size_t i=0;i<a.q.size();i++)dq+=a.q[i]!=b.q[i];
    for(size_t i=0;i<a.sx.size();i++)ds+=memcmp(&a.sx[i],&b.sx[i],4)!=0;
    fprintf(stderr,"carrier failure q=%zu scales=%zu\n",dq,ds);exit(3);
  }
  for(auto s:a.status)if(s!=bitserial::InputStatus::Valid){fprintf(stderr,"invalid trained probe continuation\n");exit(3);}
}
struct Graph {cudaGraph_t g;cudaGraphExec_t exec;};
template<class F> Graph capture(cudaStream_t st,F f) {
  Graph g;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));f();CK(cudaStreamEndCapture(st,&g.g));CK(cudaGraphInstantiate(&g.exec,g.g,0));return g;
}
__global__ void evict(unsigned* p,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]++;}
float time_us(Graph g,cudaStream_t st,unsigned* trash,size_t count,bool cold) {
  cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));std::vector<float> t;
  for(int i=0;i<18;i++){
    if(cold)evict<<<(count+255)/256,256,0,st>>>(trash,count);
    CK(cudaEventRecord(a,st));CK(cudaGraphLaunch(g.exec,st));CK(cudaEventRecord(b,st));CK(cudaEventSynchronize(b));
    float ms;CK(cudaEventElapsedTime(&ms,a,b));if(i>=3)t.push_back(ms*1000);
  }
  CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));std::sort(t.begin(),t.end());return t[t.size()/2];
}
unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
int run(int argc,char** argv) {
  if(argc!=4){fprintf(stderr,"usage: bench MODEL R PREFIX\n");return 1;}
  char* end;long rows=strtol(argv[2],&end,10);if(*end||(rows!=1&&rows!=8&&rows!=16))return 1;int r=rows;
  h5::Gguf model;model.open(argv[1]);const auto& g=model.t("blk.0.ffn_gate.weight");const auto& u=model.t("blk.0.ffn_up.weight");const auto& d=model.t("blk.0.ffn_down.weight");
  if(g.type!=41||u.type!=41||d.type!=41||g.ne.size()!=2||u.ne!=g.ne||d.ne.size()!=2||d.ne[0]!=g.ne[1]||d.ne[1]!=g.ne[0])return 1;
  int n=g.ne[1],k=g.ne[0];if(n%128||k%128)return 1;
  std::vector<uint8_t> gu(g.data,g.data+g.bytes);gu.insert(gu.end(),u.data,u.data+u.bytes);std::vector<uint8_t> dw(d.data,d.data+d.bytes);
  cudaStream_t st;CK(cudaStreamCreate(&st));auto t0=std::chrono::steady_clock::now();
  auto w=weights(gu,2*n,k,st);auto down=weights(dw,k,n,st);
  double prep_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
  Carrier input=carrier(k,r),next=carrier(n,r),reference=carrier(n,r);
  float *x,*ygu,*h,*partials,*ydown,*downpartials;
  CK(cudaMalloc(&x,size_t(r)*k*4));CK(cudaMalloc(&ygu,size_t(r)*2*n*4));CK(cudaMalloc(&h,size_t(r)*n*4));
  CK(cudaMalloc(&partials,size_t(r)*2*n*(k/128)*4));CK(cudaMalloc(&ydown,size_t(r)*k*4));CK(cudaMalloc(&downpartials,size_t(r)*k*(n/128)*4));
  size_t count=128*1024*1024/4;unsigned* trash;CK(cudaMalloc(&trash,count*4));CK(cudaMemset(trash,0,count*4));
  std::vector<float> hx(size_t(r)*k);for(size_t i=0;i<hx.size();i++)hx[i]=(int(mix(unsigned(i)+999)%65536)-32768)/32761.f;
  CK(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));CK(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,input,st));
  bitserial::Plan baseline,dp;CK(bitserial::select(2*n,k,r,&baseline));CK(bitserial::select(k,n,r,&dp));
  FILE* out=fopen((std::string(argv[3])+".jsonl").c_str(),"w");if(!out)return 2;
  auto record=[&](const char* route,int tile,int chunk,Graph graph,bool with_down){
    float warm=time_us(graph,st,trash,count,false),cold=time_us(graph,st,trash,count,true);
    fprintf(out,"{\"route\":\"%s\",\"r\":%d,\"n\":%d,\"k\":%d,\"tile\":%d,\"chunk\":%d,\"down_included\":%s,\"warm_us\":%.6f,\"cold_us\":%.6f,\"static_prepare_ms\":%.6f,\"input\":\"deterministic_generic_f32\",\"weights\":\"trained_blk0\"}\n",route,r,n,k,tile,chunk,with_down?"true":"false",warm,cold,prep_ms);fflush(out);
    printf("%s R%d tile%d chunk%d down%d warm%.3f cold%.3f us\n",route,r,tile,chunk,with_down,warm,cold);
    CK(cudaGraphExecDestroy(graph.exec));CK(cudaGraphDestroy(graph.g));
  };
  auto down_call=[&](Carrier c){CK(bitserial::project_carrier(dp,down.signs,down.scales,c,ydown,downpartials,1,0,st));};
  auto conventional=[&](bitserial::Plan p,Carrier c){CK(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,input,st));CK(bitserial::project_carrier(p,w.signs,w.scales,input,ygu,partials,1,0,st));CK(composition::unfused_tail(ygu,n,r,c,h,st));};
  for(bool incl:{false,true}) {
    record("unfused_selected",r,baseline.groups_per_split,capture(st,[&]{conventional(baseline,next);if(incl)down_call(next);}),incl);
    for(int chunk:{4,16,k/128}) {
      auto p=baseline;p.groups_per_split=chunk;p.threads=128;
      conventional(p,reference);CK(cudaStreamSynchronize(st));auto expected=download(reference,n,r);
      down_call(reference);CK(cudaStreamSynchronize(st));std::vector<float> expected_down(size_t(r)*k);CK(cudaMemcpy(expected_down.data(),ydown,expected_down.size()*4,cudaMemcpyDeviceToHost));
      record("unfused_matching",r,chunk,capture(st,[&]{conventional(p,next);if(incl)down_call(next);}),incl);
      for(int tile:{1,4,8}) {
        if(r%tile)continue;composition::Plan cp{n,k,r,tile,chunk};
        auto fused=[&]{CK(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,input,st));CK(composition::fused(cp,w.signs,w.scales,input,next,partials,st));if(incl)down_call(next);};
        fused();CK(cudaStreamSynchronize(st));equal(expected,download(next,n,r));
        if(incl){std::vector<float> actual(expected_down.size());CK(cudaMemcpy(actual.data(),ydown,actual.size()*4,cudaMemcpyDeviceToHost));if(memcmp(actual.data(),expected_down.data(),actual.size()*4)){fprintf(stderr,"down bit mismatch\n");return 3;}}
        record("fused",tile,chunk,capture(st,fused),incl);
      }
    }
  }
  printf("PASS carrier and downstream bitwise equality for all priced same-chunk continuations R%d; q=%zu scale=%zu\n",r,size_t(r)*n,size_t(r)*n/128);
  fclose(out);return 0;
}
int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"model error: %s\n",e.what());return 2;}}
