#include "batch_mma.cuh"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(2); } } while(0)
static unsigned mix(unsigned x) { x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; return x^(x>>16); }
static constexpr float guard=-9876.25f;
static auto deadline=std::chrono::steady_clock::time_point::max();
static void budget() { if(std::chrono::steady_clock::now()>deadline) { fprintf(stderr,"compute/timing budget exceeded (50 seconds)\n"); std::exit(4); } }
struct Memory {
  std::vector<void*> allocations;
  template<class T> T* alloc(size_t count) { T* p=nullptr; CUDA(cudaMalloc(&p,std::max(size_t(1),count)*sizeof(T))); allocations.push_back(p); return p; }
  ~Memory() { for(auto p:allocations) cudaFree(p); }
};
struct Stream { cudaStream_t value; Stream(){CUDA(cudaStreamCreate(&value));} ~Stream(){cudaStreamDestroy(value);} };
struct Graph {
  cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
  Graph()=default; Graph(const Graph&)=delete; Graph& operator=(const Graph&)=delete;
  Graph(Graph&& other):graph(other.graph),exec(other.exec){other.graph=nullptr;other.exec=nullptr;}
  ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
};
template<class F> static Graph capture(cudaStream_t st,F body) {
  Graph g; CUDA(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal)); body();
  CUDA(cudaStreamEndCapture(st,&g.graph)); CUDA(cudaGraphInstantiate(&g.exec,g.graph,0)); return g;
}
__global__ void evict(unsigned* p,size_t count) { size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i<count)p[i]+=1; }
static float median(const Graph& graph,cudaStream_t st,unsigned* trash,size_t count,bool cold) {
  cudaEvent_t a,b; CUDA(cudaEventCreate(&a)); CUDA(cudaEventCreate(&b)); std::vector<float> samples;
  for(int i=0;i<9;i++) {
    budget();
    if(cold) { evict<<<unsigned((count+255)/256),256,0,st>>>(trash,count); CUDA(cudaGetLastError()); }
    CUDA(cudaEventRecord(a,st)); CUDA(cudaGraphLaunch(graph.exec,st)); CUDA(cudaEventRecord(b,st));
    CUDA(cudaEventSynchronize(b)); float ms; CUDA(cudaEventElapsedTime(&ms,a,b)); if(i>=2)samples.push_back(ms*1000);
  }
  CUDA(cudaEventDestroy(a)); CUDA(cudaEventDestroy(b)); std::sort(samples.begin(),samples.end()); return samples[3];
}
static bitserial::Carrier allocate_carrier(Memory& m,int k,int r) {
  bitserial::Carrier c{}; c.q=m.alloc<int8_t>(size_t(r)*k); c.scales=m.alloc<float>(size_t(r)*(k/128));
  c.dc=m.alloc<int32_t>(size_t(r)*(k/128)); c.status=m.alloc<bitserial::InputStatus>(size_t(r)*(k/128)); return c;
}
static bitserial::Carrier view(bitserial::Carrier c,int k,int row) {
  c.q+=size_t(row)*k; c.scales+=size_t(row)*(k/128); c.dc+=size_t(row)*(k/128); c.status+=size_t(row)*(k/128); return c;
}
struct Dp4a {
  std::vector<bitserial::Plan> plans; std::vector<float*> partials; int k,n,r;
  Dp4a(Memory& m,int nn,int kk,int rr):k(kk),n(nn),r(rr) {
    for(int t=0;t<r;t+=16) {
      bitserial::Plan p{}; CUDA(bitserial::select(n,k,std::min(16,r-t),bitserial::Route::Dp4a01,&p));
      plans.push_back(p); partials.push_back(m.alloc<float>((bitserial::partial_bytes(p)+3)/4));
    }
  }
  void consume(const uint32_t* signs,const __half* scales,bitserial::Carrier c,float* y,cudaStream_t st) const {
    int t=0; for(size_t i=0;i<plans.size();i++) {
      CUDA(bitserial::project_carrier(plans[i],signs,scales,view(c,k,t),y+size_t(t)*n,partials[i],1,0,st)); t+=plans[i].r;
    }
  }
};
struct Error { double maxabs,rms,relative_rms,maxref; };
static Error compare(const std::vector<float>& actual,const std::vector<float>& expected,int n,int r,int stride) {
  Error e{}; double square=0,reference_square=0;
  for(int t=0;t<r;t++)for(int j=0;j<n;j++) {
    double a=actual[size_t(t)*stride+j],b=expected[size_t(t)*stride+j];
    if(!std::isfinite(a)||!std::isfinite(b)){fprintf(stderr,"nonfinite output row%d column%d\n",t,j);std::exit(3);}
    double d=a-b; e.maxabs=std::max(e.maxabs,std::abs(d)); e.maxref=std::max(e.maxref,std::abs(b)); square+=d*d; reference_square+=b*b;
  }
  e.rms=std::sqrt(square/(size_t(n)*r)); e.relative_rms=std::sqrt(square/std::max(reference_square,1e-30));
  if(e.relative_rms>1e-5 || e.maxabs>1e-4+3e-5*e.maxref) {fprintf(stderr,"association tolerance exceeded: maxabs %.9g rms %.9g relative_rms %.9g\n",e.maxabs,e.rms,e.relative_rms);std::exit(3);} return e;
}
// Independent raw Q1_0 map, only for the N96 strided alpha/beta contract case.
__global__ void exact_a8(const uint8_t* raw,bitserial::Carrier c,const float* initial,float* out,int n,int k,int r,int stride) {
  int row=int(blockIdx.x)*4+int(threadIdx.x)/32,lane=int(threadIdx.x)%32,t=int(blockIdx.y); if(row>=n||t>=r)return;
  float total=0;
  for(int g=0;g<k/128;g++) {
    const uint8_t* b=raw+(size_t(row)*(k/128)+g)*18; int dot=0;
    for(int j=lane;j<128;j+=32) { int q=c.q[(size_t(t)*(k/128)+g)*128+j]; dot+=((b[2+j/8]>>(j%8))&1)?q:-q; }
    for(int d=16;d;d/=2)dot+=__shfl_down_sync(0xffffffff,dot,d);
    total=fmaf(float(dot),__half2float(*reinterpret_cast<const __half*>(b))*c.scales[size_t(t)*(k/128)+g],total);
  }
  if(lane==0)out[size_t(t)*stride+row]=fmaf(.75f,total,.25f*initial[size_t(t)*stride+row]);
}
__global__ void strided_continuation(const float* tight,const float* initial,float* out,int n,int r,int stride) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i>=size_t(r)*n)return; size_t j=(i/n)*stride+i%n;
  out[j]=fmaf(.75f,tight[i],.25f*initial[j]);
}
static void check_status(bitserial::Carrier c,int k,int r) {
  std::vector<bitserial::InputStatus> status(size_t(r)*(k/128)); CUDA(cudaMemcpy(status.data(),c.status,status.size()*sizeof(status[0]),cudaMemcpyDeviceToHost));
  for(auto s:status)if(s!=bitserial::InputStatus::Valid){fprintf(stderr,"invalid producer status %u\n",unsigned(s));std::exit(3);}
}
static void guarded_case(FILE* out,const char* tensor,const uint8_t* host,int k,int r,cudaStream_t st) {
  constexpr int n=96,stride=128,prefix=32; Memory m; size_t count=size_t(r+2)*stride+64;
  auto raw=m.alloc<uint8_t>(size_t(n)*(k/128)*18); auto signs=m.alloc<uint32_t>(size_t(n)*(k/128)*4); auto scales=m.alloc<__half>(size_t(n)*(k/128));
  auto x=m.alloc<float>(size_t(r)*k); auto c=allocate_carrier(m,k,r); auto ybase=m.alloc<float>(count); auto refbase=m.alloc<float>(count);
  auto initial=m.alloc<float>(count); auto tight=m.alloc<float>(size_t(r)*n);
  std::vector<float> hx(size_t(r)*k),seed(count,guard),hy(count),href(count);
  for(size_t i=0;i<hx.size();i++)hx[i]=(int(mix(unsigned(i)+1771)%65536)-32768)/32761.f;
  for(int t=0;t<r;t++)for(int j=0;j<n;j++)seed[prefix+size_t(t)*stride+j]=(int(mix(unsigned(t*n+j)+71)%1024)-512)/1024.f;
  CUDA(cudaMemcpyAsync(raw,host,size_t(n)*(k/128)*18,cudaMemcpyHostToDevice,st)); CUDA(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));
  CUDA(cudaMemcpyAsync(initial,seed.data(),count*4,cudaMemcpyHostToDevice,st)); CUDA(cudaMemcpyAsync(refbase,initial,count*4,cudaMemcpyDeviceToDevice,st));
  CUDA(bitserial::prepare(raw,signs,scales,n,k,st)); CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));
  exact_a8<<<dim3((n+3)/4,r),128,0,st>>>(raw,c,initial+prefix,refbase+prefix,n,k,r,stride); CUDA(cudaGetLastError());
  CUDA(cudaStreamSynchronize(st)); check_status(c,k,r); CUDA(cudaMemcpy(href.data(),refbase,count*4,cudaMemcpyDeviceToHost));
  Dp4a dp(m,n,k,r); dp.consume(signs,scales,c,tight,st); CUDA(cudaMemcpyAsync(ybase,initial,count*4,cudaMemcpyDeviceToDevice,st));
  strided_continuation<<<unsigned((size_t(r)*n+255)/256),256,0,st>>>(tight,initial+prefix,ybase+prefix,n,r,stride); CUDA(cudaGetLastError()); CUDA(cudaStreamSynchronize(st));
  CUDA(cudaMemcpy(hy.data(),ybase,count*4,cudaMemcpyDeviceToHost));
  std::vector<float> ref(href.begin()+prefix,href.begin()+prefix+size_t(r)*stride),actual(hy.begin()+prefix,hy.begin()+prefix+size_t(r)*stride);
  compare(actual,ref,n,r,stride);
  for(int tile:{32,64,128})for(int chunk:{4,16,k/128}) {
    budget(); batch_mma::Plan p{n,k,r,chunk,tile}; CUDA(batch_mma::validate(p)); auto partial=m.alloc<float>((batch_mma::partial_bytes(p)+3)/4);
    CUDA(cudaMemcpyAsync(ybase,initial,count*4,cudaMemcpyDeviceToDevice,st));
    CUDA(batch_mma::project(p,signs,scales,x,c,ybase+prefix,stride,partial,.75f,.25f,st)); CUDA(cudaStreamSynchronize(st)); check_status(c,k,r);
    CUDA(cudaMemcpy(hy.data(),ybase,count*4,cudaMemcpyDeviceToHost)); actual.assign(hy.begin()+prefix,hy.begin()+prefix+size_t(r)*stride);
    Error e=compare(actual,ref,n,r,stride);
    for(size_t i=0;i<count;i++) {
      bool active=i>=prefix&&i<prefix+size_t(r)*stride&&(i-prefix)%stride<size_t(n);
      if(!active&&hy[i]!=guard){fprintf(stderr,"tail guard overwritten R%d tile%d chunk%d index%zu\n",r,tile,chunk,i);std::exit(3);}
    }
    fprintf(out,"{\"kind\":\"correctness\",\"tensor\":\"%s\",\"n\":96,\"k\":%d,\"r\":%d,\"output_stride\":128,\"alpha\":0.75,\"beta\":0.25,\"n_tile\":%d,\"groups_per_split\":%d,\"maxabs\":%.9g,\"rms\":%.9g,\"relative_rms\":%.9g,\"guards\":true}\n",tensor,k,r,tile,chunk,e.maxabs,e.rms,e.relative_rms); fflush(out);
  }
}
static void benchmark(FILE* out,const char* tensor,const uint8_t* host,int n,int k,int r,cudaStream_t st) {
  Memory m; size_t blocks=size_t(n)*(k/128),output_count=size_t(r)*n;
  auto raw=m.alloc<uint8_t>(blocks*18); auto signs=m.alloc<uint32_t>(blocks*4); auto scales=m.alloc<__half>(blocks); auto x=m.alloc<float>(size_t(r)*k);
  auto reference=m.alloc<float>(output_count); auto y=m.alloc<float>(output_count); auto c=allocate_carrier(m,k,r);
  const size_t trashcount=128*1024*1024/sizeof(unsigned); auto trash=m.alloc<unsigned>(trashcount); CUDA(cudaMemsetAsync(trash,0,trashcount*4,st));
  std::vector<float> hx(size_t(r)*k),href(output_count),hy(output_count); for(size_t i=0;i<hx.size();i++)hx[i]=(int(mix(unsigned(i)+999)%65536)-32768)/32761.f;
  CUDA(cudaMemcpyAsync(raw,host,blocks*18,cudaMemcpyHostToDevice,st)); CUDA(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));
  CUDA(bitserial::prepare(raw,signs,scales,n,k,st)); CUDA(cudaStreamSynchronize(st));
  Dp4a dp(m,n,k,r); CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st)); dp.consume(signs,scales,c,reference,st); CUDA(cudaStreamSynchronize(st)); check_status(c,k,r);
  CUDA(cudaMemcpy(href.data(),reference,output_count*4,cudaMemcpyDeviceToHost));
  auto dp_carrier=[&](){dp.consume(signs,scales,c,y,st);};
  auto dp_inclusive=[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));dp_carrier();};
  auto report=[&](const char* route,int tile,int chunk,const Graph& inclusive,const Graph& carrier,Error e) {
    float warm=median(inclusive,st,trash,trashcount,false),cold=median(inclusive,st,trash,trashcount,true);
    float cwarm=median(carrier,st,trash,trashcount,false),ccold=median(carrier,st,trash,trashcount,true);
    fprintf(out,"{\"kind\":\"timing\",\"tensor\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"route\":\"%s\",\"n_tile\":%d,\"groups_per_split\":%d,\"inclusive_warm_us\":%.6f,\"inclusive_cold_us\":%.6f,\"carrier_warm_us\":%.6f,\"carrier_cold_us\":%.6f,\"maxabs\":%.9g,\"rms\":%.9g,\"relative_rms\":%.9g,\"median_samples\":7,\"discarded_samples\":2,\"eviction_bytes\":134217728,\"producer_calls\":1,\"dp4a_row_chunk_limit\":16}\n",tensor,n,k,r,route,tile,chunk,warm,cold,cwarm,ccold,e.maxabs,e.rms,e.relative_rms); fflush(out);
    printf("%s N%d K%d R%d tile%d chunk%d inclusive cold %.3fus carrier cold %.3fus maxabs %.3g rms %.3g\n",route,n,k,r,tile,chunk,cold,ccold,e.maxabs,e.rms);
  };
  auto di=capture(st,dp_inclusive),dc=capture(st,dp_carrier); report("stable_dp4a01",0,dp.plans.front().groups_per_split,di,dc,Error{});
  for(int tile:{32,64,128})for(int chunk:{4,16,k/128}) {
    budget(); batch_mma::Plan p{n,k,r,chunk,tile}; CUDA(batch_mma::validate(p)); auto partial=m.alloc<float>((batch_mma::partial_bytes(p)+3)/4);
    auto carrier=[&](){CUDA(batch_mma::project_carrier(p,signs,scales,c,y,n,partial,1,0,st));};
    auto inclusive=[&](){CUDA(batch_mma::project(p,signs,scales,x,c,y,n,partial,1,0,st));};
    inclusive(); CUDA(cudaStreamSynchronize(st)); check_status(c,k,r); CUDA(cudaMemcpy(hy.data(),y,output_count*4,cudaMemcpyDeviceToHost)); Error e=compare(hy,href,n,r,n);
    auto gi=capture(st,inclusive),gc=capture(st,carrier); report("batch_mma",tile,chunk,gi,gc,e);
  }
}
static bool number(const char* text,int* out) { char* end=nullptr; long n=std::strtol(text,&end,10); if(!*text||*end||n<0||n>std::numeric_limits<int>::max())return false; *out=int(n); return true; }
static int run(int argc,char** argv) {
  if(argc!=2&&argc!=6){fprintf(stderr,"usage: bench MODEL [TENSOR R(5|8|16|32|all) N_ROWS(0=full) PREFIX]\nTENSOR: gu | down | head | GGUF tensor name\n");return 1;}
  h5::Gguf model; model.open(argv[1]);
  if(argc==2){for(const auto& t:model.tensors)if(t.type==41&&t.ne.size()==2)printf("%s %lu %lu\n",t.name.c_str(),t.ne[1],t.ne[0]);return 0;}
  std::string requested=argv[2]; std::vector<std::string> names;
  if(requested=="gu")names={"blk.0.ffn_gate.weight","blk.0.ffn_up.weight"};
  else if(requested=="down")names={"blk.0.ffn_down.weight"}; else if(requested=="head")names={"output.weight"}; else names={requested};
  h5::GTensor tensor=model.t(names.front());
  if(tensor.type!=41||tensor.ne.size()!=2){fprintf(stderr,"tensor must be Q1_0 rank2\n");return 1;}
  std::vector<uint8_t> joined;
  if(names.size()>1) {
    joined.assign(tensor.data,tensor.data+tensor.bytes);
    for(size_t i=1;i<names.size();i++){const auto& t=model.t(names[i]);if(t.type!=41||t.ne.size()!=2||t.ne[0]!=tensor.ne[0]){fprintf(stderr,"incompatible joined tensor\n");return 1;}joined.insert(joined.end(),t.data,t.data+t.bytes);tensor.ne[1]+=t.ne[1];}
    tensor.data=joined.data();tensor.bytes=joined.size();
  }
  int limit; if(!number(argv[4],&limit)||tensor.ne[0]>unsigned(std::numeric_limits<int>::max())||tensor.ne[1]>unsigned(std::numeric_limits<int>::max()))return 1;
  int n=limit?std::min(limit,int(tensor.ne[1])):int(tensor.ne[1]),k=int(tensor.ne[0]);
  if(n<96||n%32||k<=0||k%128||tensor.bytes<size_t(n)*(k/128)*18){fprintf(stderr,"invalid tensor geometry or payload\n");return 1;}
  std::vector<int> rows; if(std::string(argv[3])=="all")rows={5,8,16,32}; else {int r;if(!number(argv[3],&r)||(r!=5&&r!=8&&r!=16&&r!=32))return 1;rows={r};}
  FILE* out=std::fopen((std::string(argv[5])+".jsonl").c_str(),"w"); if(!out){std::perror("output");return 2;}
  Stream st; deadline=std::chrono::steady_clock::now()+std::chrono::seconds(50);
  guarded_case(out,requested.c_str(),tensor.data,k,5,st.value); guarded_case(out,requested.c_str(),tensor.data,k,32,st.value);
  for(int r:rows)benchmark(out,requested.c_str(),tensor.data,n,k,r,st.value);
  budget(); if(std::fclose(out)){std::perror("close output");return 2;} return 0;
}
int main(int argc,char** argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"modelerror: %s\n",e.what());return 2;}}
