#include "fp8.cuh"
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
static void budget() { if(std::chrono::steady_clock::now()>deadline) { fprintf(stderr,"shared GPU-round budget exceeded (50 seconds)\n"); std::exit(4); } }
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
static bitserial::Carrier allocate_a8(Memory& m,int k,int r) {
  bitserial::Carrier c{}; c.q=m.alloc<int8_t>(size_t(r)*k); c.scales=m.alloc<float>(size_t(r)*(k/128));
  c.dc=m.alloc<int32_t>(size_t(r)*(k/128)); c.status=m.alloc<bitserial::InputStatus>(size_t(r)*(k/128)); return c;
}
static binary_fp8::Carrier allocate_fp8(Memory& m,int k,int r) {
  return {m.alloc<uint8_t>(size_t(r)*k),m.alloc<float>(size_t(r)*(k/128)),m.alloc<bitserial::InputStatus>(size_t(r)*(k/128)),nullptr};
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
static Error compare(const std::vector<float>& actual,const std::vector<float>& expected,int n,int r,int stride,bool gate) {
  Error e{}; double square=0,reference_square=0;
  for(int t=0;t<r;t++)for(int j=0;j<n;j++) {
    double a=actual[size_t(t)*stride+j],b=expected[size_t(t)*stride+j];
    if(!std::isfinite(a)||!std::isfinite(b)){fprintf(stderr,"nonfinite output row%d column%d\n",t,j);std::exit(3);}
    double d=a-b; e.maxabs=std::max(e.maxabs,std::abs(d)); e.maxref=std::max(e.maxref,std::abs(b)); square+=d*d; reference_square+=b*b;
  }
  e.rms=std::sqrt(square/(size_t(n)*r)); e.relative_rms=std::sqrt(square/std::max(reference_square,1e-30));
  if(gate&&(e.relative_rms>1e-5 || e.maxabs>1e-4+3e-5*e.maxref)) {fprintf(stderr,"association tolerance exceeded: maxabs %.9g rms %.9g relative_rms %.9g\n",e.maxabs,e.rms,e.relative_rms);std::exit(3);} return e;
}
static void error_record(FILE* out,const char* tensor,const char* route,const char* oracle,const char* distribution,int n,int k,int r,int samples,bool gate,Error e) {
  fprintf(out,"{\"kind\":\"numerical_error\",\"tensor\":\"%s\",\"route\":\"%s\",\"oracle\":\"%s\",\"input_distribution\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"columns_compared\":%d,\"full_output\":%s,\"correctness_gate\":%s,\"maxabs\":%.9g,\"rms\":%.9g,\"relative_rms\":%.9g}\n",tensor,route,oracle,distribution,n,k,r,samples,samples==n?"true":"false",gate?"true":"false",e.maxabs,e.rms,e.relative_rms); fflush(out);
}
// Independent raw Q1 oracle: reads original 18-byte groups, never prepared signs/scales.
__global__ void exact_fp8(const uint8_t* raw,binary_fp8::Carrier c,const float* initial,float* out,int n,int k,int r,int stride,float alpha,float beta) {
  int row=int(blockIdx.x)*4+int(threadIdx.x)/32,lane=int(threadIdx.x)%32,t=int(blockIdx.y); if(row>=n||t>=r)return;
  float total=0;
  for(int g=0;g<k/128;g++) {
    const uint8_t* b=raw+(size_t(row)*(k/128)+g)*18; float dot=0;
    for(int j=lane;j<128;j+=32) {
      __nv_fp8_e4m3 q; q.__x=c.q[size_t(t)*k+g*128+j]; float v=float(q);
      dot+=((b[2+j/8]>>(j%8))&1)?v:-v;
    }
    for(int d=16;d;d/=2)dot+=__shfl_down_sync(0xffffffff,dot,d);
    if(lane==0)total=fmaf(dot,__half2float(*reinterpret_cast<const __half*>(b))*c.scales[size_t(t)*(k/128)+g],total);
  }
  if(lane==0)out[size_t(t)*stride+row]=alpha*total+(beta==0?0:beta*initial[size_t(t)*stride+row]);
}
__global__ void exact_a8(const uint8_t* raw,bitserial::Carrier c,float* out,int n,int k,int r) {
  int row=int(blockIdx.x)*4+int(threadIdx.x)/32,lane=int(threadIdx.x)%32,t=int(blockIdx.y); if(row>=n||t>=r)return;
  float total=0;
  for(int g=0;g<k/128;g++) {
    const uint8_t* b=raw+(size_t(row)*(k/128)+g)*18; int dot=0;
    for(int j=lane;j<128;j+=32) {int v=c.q[size_t(t)*k+g*128+j];dot+=((b[2+j/8]>>(j%8))&1)?v:-v;}
    for(int d=16;d;d/=2)dot+=__shfl_down_sync(0xffffffff,dot,d);
    if(lane==0)total=fmaf(float(dot),__half2float(*reinterpret_cast<const __half*>(b))*c.scales[size_t(t)*(k/128)+g],total);
  }
  if(lane==0)out[size_t(t)*n+row]=total;
}
// FP64 accumulation is an independent reference for the original generic FP32 input.
__global__ void raw_fp32_sample(const uint8_t* raw,const float* x,float* out,int n,int k,int r,int samples) {
  int s=int(blockIdx.x)*4+int(threadIdx.x)/32,lane=int(threadIdx.x)%32,t=int(blockIdx.y); if(s>=samples||t>=r)return;
  int row=samples==1?0:int((int64_t(s)*(n-1))/(samples-1)); double total=0;
  for(int g=0;g<k/128;g++) {
    const uint8_t* b=raw+(size_t(row)*(k/128)+g)*18; double w=__half2float(*reinterpret_cast<const __half*>(b));
    for(int j=lane;j<128;j+=32) {double v=x[size_t(t)*k+g*128+j];total+=((b[2+j/8]>>(j%8))&1)?v*w:-v*w;}
  }
  for(int d=16;d;d/=2)total+=__shfl_down_sync(0xffffffff,total,d);
  if(lane==0)out[size_t(t)*samples+s]=float(total);
}
__global__ void strided_continuation(const float* tight,const float* initial,float* out,int n,int r,int stride) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i>=size_t(r)*n)return; size_t j=(i/n)*stride+i%n;
  out[j]=fmaf(.75f,tight[i],.25f*initial[j]);
}
static void check_status(const bitserial::InputStatus* ptr,int k,int r) {
  std::vector<bitserial::InputStatus> status(size_t(r)*(k/128)); CUDA(cudaMemcpy(status.data(),ptr,status.size()*sizeof(status[0]),cudaMemcpyDeviceToHost));
  for(auto s:status)if(s!=bitserial::InputStatus::Valid){fprintf(stderr,"invalid producer status %u\n",unsigned(s));std::exit(3);}
}
static void check_guards(const std::vector<float>& y,int n,int r,int stride,int prefix) {
  for(size_t i=0;i<y.size();i++) {
    bool active=i>=size_t(prefix)&&i<size_t(prefix)+size_t(r)*stride&&(i-prefix)%stride<size_t(n);
    if(!active&&y[i]!=guard){fprintf(stderr,"output guard overwritten R%d index%zu\n",r,i);std::exit(3);}
  }
}
static std::vector<float> input(int k,int r,bool gaussian) {
  std::vector<float> x(size_t(r)*k);
  for(size_t i=0;i<x.size();i++) {
    float u=(float(mix(unsigned(i)+999)%65536)+.5f)/65536.f;
    float v=(float(mix(unsigned(i)+7171)%65536)+.5f)/65536.f;
    x[i]=gaussian?std::sqrt(-2.f*std::log(u))*std::cos(6.283185307179586f*v):2.f*u-1.f;
  } return x;
}
static void guarded_case(FILE* out,const char* tensor,const uint8_t* host,int k,int r,bool gaussian,cudaStream_t st) {
  budget(); constexpr int n=96,stride=128,prefix=32; Memory m; size_t count=size_t(r+2)*stride+64;
  auto raw=m.alloc<uint8_t>(size_t(n)*(k/128)*18); auto signs=m.alloc<uint32_t>(size_t(n)*(k/128)*4); auto scales=m.alloc<__half>(size_t(n)*(k/128));
  auto x=m.alloc<float>(size_t(r)*k); auto c=allocate_fp8(m,k,r); auto a8=allocate_a8(m,k,r);
  auto ybase=m.alloc<float>(count); auto refbase=m.alloc<float>(count); auto initial=m.alloc<float>(count); auto tight=m.alloc<float>(size_t(r)*n); auto a8ref=m.alloc<float>(size_t(r)*n);
  auto hx=input(k,r,gaussian); std::vector<float> seed(count,guard),hy(count),href(count);
  for(int t=0;t<r;t++)for(int j=0;j<n;j++)seed[prefix+size_t(t)*stride+j]=(int(mix(unsigned(t*n+j)+71)%1024)-512)/1024.f;
  CUDA(cudaMemcpyAsync(raw,host,size_t(n)*(k/128)*18,cudaMemcpyHostToDevice,st)); CUDA(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));
  CUDA(cudaMemcpyAsync(initial,seed.data(),count*4,cudaMemcpyHostToDevice,st)); CUDA(cudaMemcpyAsync(refbase,initial,count*4,cudaMemcpyDeviceToDevice,st));
  CUDA(bitserial::prepare(raw,signs,scales,n,k,st)); CUDA(binary_fp8::produce(x,k,r,c,st));
  exact_fp8<<<dim3((n+3)/4,r),128,0,st>>>(raw,c,initial+prefix,refbase+prefix,n,k,r,stride,.75f,.25f); CUDA(cudaGetLastError());
  CUDA(cudaStreamSynchronize(st)); check_status(c.status,k,r); CUDA(cudaMemcpy(href.data(),refbase,count*4,cudaMemcpyDeviceToHost));
  std::vector<float> ref(href.begin()+prefix,href.begin()+prefix+size_t(r)*stride),actual;
  Dp4a dp(m,n,k,r); CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,a8,st)); dp.consume(signs,scales,a8,tight,st);
  exact_a8<<<dim3((n+3)/4,r),128,0,st>>>(raw,a8,a8ref,n,k,r); CUDA(cudaGetLastError());
  CUDA(cudaMemcpyAsync(refbase,initial,count*4,cudaMemcpyDeviceToDevice,st)); CUDA(cudaMemcpyAsync(ybase,initial,count*4,cudaMemcpyDeviceToDevice,st));
  strided_continuation<<<unsigned((size_t(r)*n+255)/256),256,0,st>>>(tight,initial+prefix,ybase+prefix,n,r,stride); CUDA(cudaGetLastError());
  strided_continuation<<<unsigned((size_t(r)*n+255)/256),256,0,st>>>(a8ref,initial+prefix,refbase+prefix,n,r,stride); CUDA(cudaGetLastError());
  CUDA(cudaStreamSynchronize(st)); check_status(a8.status,k,r); CUDA(cudaMemcpy(hy.data(),ybase,count*4,cudaMemcpyDeviceToHost)); CUDA(cudaMemcpy(href.data(),refbase,count*4,cudaMemcpyDeviceToHost)); check_guards(hy,n,r,stride,prefix);
  actual.assign(hy.begin()+prefix,hy.begin()+prefix+size_t(r)*stride); std::vector<float> dref(href.begin()+prefix,href.begin()+prefix+size_t(r)*stride);
  Error de=compare(actual,dref,n,r,stride,true);
  fprintf(out,"{\"kind\":\"correctness\",\"tensor\":\"%s\",\"route\":\"stable_dp4a01\",\"n\":96,\"k\":%d,\"r\":%d,\"output_stride\":128,\"alpha\":0.75,\"beta\":0.25,\"maxabs\":%.9g,\"relative_rms\":%.9g,\"guards\":true,\"oracle\":\"raw_q1_a8\"}\n",tensor,k,r,de.maxabs,de.relative_rms);
  for(int tile:{32,64,128}) {
    budget(); binary_fp8::Plan p{n,k,r,16,tile}; CUDA(binary_fp8::validate(p)); auto partial=m.alloc<float>((binary_fp8::partial_bytes(p)+3)/4);
    for(bool inclusive:{false,true}) {
      CUDA(cudaMemcpyAsync(ybase,initial,count*4,cudaMemcpyDeviceToDevice,st));
      if(inclusive)CUDA(binary_fp8::project(p,signs,scales,x,c,ybase+prefix,stride,partial,.75f,.25f,st));
      else CUDA(binary_fp8::project_carrier(p,signs,scales,c,ybase+prefix,stride,partial,.75f,.25f,st));
      CUDA(cudaStreamSynchronize(st)); check_status(c.status,k,r);
      CUDA(cudaMemcpy(hy.data(),ybase,count*4,cudaMemcpyDeviceToHost)); actual.assign(hy.begin()+prefix,hy.begin()+prefix+size_t(r)*stride);
      Error e=compare(actual,ref,n,r,stride,true); check_guards(hy,n,r,stride,prefix);
      fprintf(out,"{\"kind\":\"correctness\",\"tensor\":\"%s\",\"route\":\"native_e4m3\",\"inclusive\":%s,\"n\":96,\"k\":%d,\"r\":%d,\"output_stride\":128,\"alpha\":0.75,\"beta\":0.25,\"n_tile\":%d,\"groups_per_split\":16,\"input_distribution\":\"%s\",\"maxabs\":%.9g,\"rms\":%.9g,\"relative_rms\":%.9g,\"guards\":true,\"oracle\":\"raw_q1_decoded_e4m3\"}\n",tensor,inclusive?"true":"false",k,r,tile,gaussian?"gaussian":"uniform",e.maxabs,e.rms,e.relative_rms); fflush(out);
    }
  }
}
static Error sampled_error(const std::vector<float>& actual,const std::vector<float>& reference,int n,int r,int samples) {
  std::vector<float> subset(size_t(r)*samples);
  for(int t=0;t<r;t++)for(int s=0;s<samples;s++) {int row=samples==1?0:int((int64_t(s)*(n-1))/(samples-1));subset[size_t(t)*samples+s]=actual[size_t(t)*n+row];}
  return compare(subset,reference,samples,r,samples,false);
}
static void benchmark(FILE* out,const char* tensor,const uint8_t* host,int n,int k,int r,cudaStream_t st) {
  budget(); Memory m; size_t blocks=size_t(n)*(k/128),output_count=size_t(r)*n; int samples=std::min(n,256);
  auto raw=m.alloc<uint8_t>(blocks*18); auto signs=m.alloc<uint32_t>(blocks*4); auto scales=m.alloc<__half>(blocks); auto x=m.alloc<float>(size_t(r)*k);
  auto reference=m.alloc<float>(output_count); auto dpout=m.alloc<float>(output_count); auto y=m.alloc<float>(output_count); auto rawref=m.alloc<float>(size_t(r)*samples);
  auto c=allocate_fp8(m,k,r); auto a8=allocate_a8(m,k,r);
  const size_t trashcount=128*1024*1024/sizeof(unsigned); auto trash=m.alloc<unsigned>(trashcount); CUDA(cudaMemsetAsync(trash,0,trashcount*4,st));
  auto hx=input(k,r,false); std::vector<float> href(output_count),hdp(output_count),hy(output_count),hraw(size_t(r)*samples);
  CUDA(cudaMemcpyAsync(raw,host,blocks*18,cudaMemcpyHostToDevice,st)); CUDA(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));
  CUDA(bitserial::prepare(raw,signs,scales,n,k,st)); CUDA(binary_fp8::produce(x,k,r,c,st)); CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,a8,st));
  exact_fp8<<<dim3((n+3)/4,r),128,0,st>>>(raw,c,nullptr,reference,n,k,r,n,1,0); CUDA(cudaGetLastError());
  raw_fp32_sample<<<dim3((samples+3)/4,r),128,0,st>>>(raw,x,rawref,n,k,r,samples); CUDA(cudaGetLastError());
  Dp4a dp(m,n,k,r); dp.consume(signs,scales,a8,dpout,st); CUDA(cudaStreamSynchronize(st)); check_status(c.status,k,r); check_status(a8.status,k,r);
  CUDA(cudaMemcpy(href.data(),reference,output_count*4,cudaMemcpyDeviceToHost)); CUDA(cudaMemcpy(hdp.data(),dpout,output_count*4,cudaMemcpyDeviceToHost)); CUDA(cudaMemcpy(hraw.data(),rawref,hraw.size()*4,cudaMemcpyDeviceToHost));
  error_record(out,tensor,"stable_dp4a01","raw_q1_fp32_sample_fp64_accumulation","uniform",n,k,r,samples,false,sampled_error(hdp,hraw,n,r,samples));
  fprintf(out,"{\"kind\":\"residency\",\"tensor\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"q1_raw_bytes\":%zu,\"prepared_sign_bytes\":%zu,\"original_scale_bytes\":%zu,\"weight_fp8_resident_bytes\":0,\"weight_fp8_conversion\":\"transient_exact_signed_one\",\"carrier_q_bytes\":%zu,\"carrier_scale_status_bytes\":%zu,\"dp4a_q_bytes\":%zu,\"dp4a_scale_dc_status_bytes\":%zu,\"static_prepare_in_recurring\":false,\"host_device_transfer_in_recurring\":false}\n",tensor,n,k,r,blocks*18,blocks*16,blocks*2,size_t(r)*k,size_t(r)*(k/128)*8,size_t(r)*k,size_t(r)*(k/128)*12); fflush(out);
  auto dp_carrier=[&](){dp.consume(signs,scales,a8,y,st);};
  auto dp_inclusive=[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,a8,st));dp_carrier();};
  auto report=[&](const char* route,int tile,int chunk,const Graph& inclusive,const Graph& carrier,Error e) {
    float warm=median(inclusive,st,trash,trashcount,false),cold=median(inclusive,st,trash,trashcount,true);
    float cwarm=median(carrier,st,trash,trashcount,false),ccold=median(carrier,st,trash,trashcount,true);
    fprintf(out,"{\"kind\":\"timing\",\"tensor\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"route\":\"%s\",\"n_tile\":%d,\"groups_per_split\":%d,\"input_distribution\":\"uniform\",\"inclusive_warm_us\":%.6f,\"inclusive_cold_us\":%.6f,\"carrier_warm_us\":%.6f,\"carrier_cold_us\":%.6f,\"carrier_oracle_maxabs\":%.9g,\"carrier_oracle_relative_rms\":%.9g,\"median_samples\":7,\"discarded_samples\":2,\"eviction_bytes\":134217728,\"producer_calls\":1,\"dp4a_row_chunk_limit\":16,\"full_output_elements_compared\":%zu}\n",tensor,n,k,r,route,tile,chunk,warm,cold,cwarm,ccold,e.maxabs,e.relative_rms,output_count); fflush(out);
    printf("%s %s N%d K%d R%d tile%d split%d inclusive cold %.3fus carrier cold %.3fus\n",tensor,route,n,k,r,tile,chunk,cold,ccold);
  };
  // DP4A has its own exact A8 oracle, not the numerically different FP8 carrier.
  exact_a8<<<dim3((n+3)/4,r),128,0,st>>>(raw,a8,reference,n,k,r); CUDA(cudaGetLastError()); CUDA(cudaStreamSynchronize(st));
  CUDA(cudaMemcpy(hy.data(),reference,output_count*4,cudaMemcpyDeviceToHost)); Error de=compare(hdp,hy,n,r,n,true);
  auto di=capture(st,dp_inclusive),dg=capture(st,dp_carrier); report("stable_dp4a01",0,dp.plans.front().groups_per_split,di,dg,de);
  for(int tile:{32,64,128}) {
    budget(); binary_fp8::Plan p{n,k,r,16,tile}; CUDA(binary_fp8::validate(p)); auto partial=m.alloc<float>((binary_fp8::partial_bytes(p)+3)/4);
    auto carrier=[&](){CUDA(binary_fp8::project_carrier(p,signs,scales,c,y,n,partial,1,0,st));};
    auto inclusive=[&](){CUDA(binary_fp8::project(p,signs,scales,x,c,y,n,partial,1,0,st));};
    inclusive(); CUDA(cudaStreamSynchronize(st)); check_status(c.status,k,r); CUDA(cudaMemcpy(hy.data(),y,output_count*4,cudaMemcpyDeviceToHost)); Error e=compare(hy,href,n,r,n,true);
    error_record(out,tensor,"native_e4m3","raw_q1_fp32_sample_fp64_accumulation","uniform",n,k,r,samples,false,sampled_error(hy,hraw,n,r,samples));
    error_record(out,tensor,"native_e4m3","stable_dp4a01_full","uniform",n,k,r,n,false,compare(hy,hdp,n,r,n,false));
    error_record(out,tensor,"native_e4m3","raw_q1_decoded_e4m3_full","uniform",n,k,r,n,true,e);
    carrier(); CUDA(cudaStreamSynchronize(st)); CUDA(cudaMemcpy(hy.data(),y,output_count*4,cudaMemcpyDeviceToHost)); compare(hy,href,n,r,n,true);
    auto gi=capture(st,inclusive),gc=capture(st,carrier); report("native_e4m3",tile,16,gi,gc,e);
  }
  auto producer=[&](){CUDA(binary_fp8::produce(x,k,r,c,st));}; auto a8producer=[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,a8,st));};
  auto pg=capture(st,producer),ag=capture(st,a8producer);
  auto producer_report=[&](const char* stage,const Graph& g) {
    float warm=median(g,st,trash,trashcount,false),cold=median(g,st,trash,trashcount,true);
    fprintf(out,"{\"kind\":\"producer_timing\",\"tensor\":\"%s\",\"k\":%d,\"r\":%d,\"stage\":\"%s\",\"warm_us\":%.6f,\"cold_us\":%.6f,\"median_samples\":7,\"discarded_samples\":2,\"eviction_bytes\":134217728}\n",tensor,k,r,stage,warm,cold); fflush(out);
  };
  producer_report("maxabs448_e4m3_rn",pg); producer_report("original_maxabs127_a8_rn",ag);
}
static bool number(const char* text,int* out) { char* end=nullptr; long n=std::strtol(text,&end,10); if(!*text||*end||n<0||n>std::numeric_limits<int>::max())return false; *out=int(n); return true; }
static void tensor_case(FILE* out,h5::Gguf& model,const std::string& requested,int limit,const std::vector<int>& rows,cudaStream_t st) {
  budget(); std::vector<std::string> names;
  if(requested=="gu")names={"blk.0.ffn_gate.weight","blk.0.ffn_up.weight"};
  else if(requested=="down")names={"blk.0.ffn_down.weight"}; else if(requested=="head"||requested=="head65536"||requested=="fullhead")names={"output.weight"}; else names={requested};
  h5::GTensor tensor=model.t(names.front());
  if(tensor.type!=41||tensor.ne.size()!=2){fprintf(stderr,"tensor must be Q1_0 rank2\n");std::exit(1);}
  std::vector<uint8_t> joined;
  if(names.size()>1) {
    joined.assign(tensor.data,tensor.data+tensor.bytes);
    for(size_t i=1;i<names.size();i++){const auto& t=model.t(names[i]);if(t.type!=41||t.ne.size()!=2||t.ne[0]!=tensor.ne[0]){fprintf(stderr,"incompatible joined tensor\n");std::exit(1);}joined.insert(joined.end(),t.data,t.data+t.bytes);tensor.ne[1]+=t.ne[1];}
    tensor.data=joined.data();tensor.bytes=joined.size();
  }
  if(tensor.ne[0]>unsigned(std::numeric_limits<int>::max())||tensor.ne[1]>unsigned(std::numeric_limits<int>::max())){fprintf(stderr,"tensor dimensions exceed int range\n");std::exit(1);}
  int n=limit?std::min(limit,int(tensor.ne[1])):int(tensor.ne[1]),k=int(tensor.ne[0]);
  if(n<96||n%32||k<=0||k%128||tensor.bytes<size_t(n)*(k/128)*18){fprintf(stderr,"invalid tensor geometry or payload\n");std::exit(1);}
  fprintf(out,"{\"kind\":\"tensor_selection\",\"tensor\":\"%s\",\"selected_rows\":%d,\"original_rows\":%lu,\"row_selection\":\"%s\",\"pruned_vocabulary_claim\":false}\n",requested.c_str(),n,tensor.ne[1],n<int(tensor.ne[1])?"contiguous_prefix":"full_tensor"); fflush(out);
  guarded_case(out,requested.c_str(),tensor.data,k,5,false,st); guarded_case(out,requested.c_str(),tensor.data,k,32,true,st);
  for(int r:rows)benchmark(out,requested.c_str(),tensor.data,n,k,r,st);
}
static int run(int argc,char** argv) {
  bool round=argc==4&&std::string(argv[2])=="round";
  if(argc!=2&&argc!=6&&!round){fprintf(stderr,"usage: bench MODEL round PREFIX\n       bench MODEL [TENSOR R(5|8|16|32|all) N_ROWS(0=full) PREFIX]\nTENSOR: gu | down | head | GGUF tensor name\nround: GU/down/head65536/fullhead R32, shared 50s GPU budget\n");return 1;}
  h5::Gguf model; model.open(argv[1]);
  if(argc==2){for(const auto& t:model.tensors)if(t.type==41&&t.ne.size()==2)printf("%s %lu %lu\n",t.name.c_str(),t.ne[1],t.ne[0]);return 0;}
  int limit=0; std::vector<int> rows;
  if(!round) {
    if(!number(argv[4],&limit))return 1;
    if(std::string(argv[3])=="all")rows={5,8,16,32}; else {int r;if(!number(argv[3],&r)||(r!=5&&r!=8&&r!=16&&r!=32))return 1;rows={r};}
  }
  FILE* out=std::fopen((std::string(round?argv[3]:argv[5])+".jsonl").c_str(),"w"); if(!out){std::perror("output");return 2;}
  auto start=std::chrono::steady_clock::now(); deadline=start+std::chrono::seconds(50); Stream st;
  fprintf(out,"{\"kind\":\"scope\",\"activations\":\"generic_nonconstant_fp32_not_real_engine_activations\",\"weight_source\":\"original_binary_trained_q1_0\",\"native_instruction\":\"e4m3_x_e4m3_mma\",\"fp4_x_fp8\":\"unsupported_compile\",\"fp8_is_exact_a8\":false,\"fp8_quantization\":\"maxabs448_e4m3_rn_per_k128\",\"raw_fp32_error_is_failure\":false,\"raw_fp32_sample_columns\":256,\"raw_fp32_sample_map\":\"floor(s*(n-1)/(samples-1))\",\"carrier_gate_relative_rms\":1e-5,\"carrier_gate_maxabs\":\"1e-4+3e-5*maxref\"}\n"); fflush(out);
  if(round) {
    tensor_case(out,model,"gu",0,{32},st.value); tensor_case(out,model,"down",0,{32},st.value);
    tensor_case(out,model,"head65536",65536,{32},st.value); tensor_case(out,model,"fullhead",0,{32},st.value);
  } else tensor_case(out,model,argv[2],limit,rows,st.value);
  budget(); double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  fprintf(out,"{\"kind\":\"round_complete\",\"shared_gpu_budget_seconds\":50,\"elapsed_seconds\":%.6f}\n",elapsed);
  if(std::fclose(out)){std::perror("close output");return 2;} return 0;
}
int main(int argc,char** argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"modelerror: %s\n",e.what());return 2;}}
