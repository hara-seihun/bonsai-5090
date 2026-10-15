// The generated prefix retains the native helpers/body byte-for-byte, excluding registries.
#ifndef H5_PRESSURE_NATIVE_BODY
#error "Build with bench/hproj_pressure.mk and an explicit SOURCE_ROOT"
#endif
#include H5_PRESSURE_NATIVE_BODY
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#if defined(EXP_NOWAIT) || defined(EXP_NOMMA) || defined(H5_PROFILE_GEMM_STREAM) || defined(H5_WARP_MICROSCOPE) || defined(H5_PDLCLOCK) || defined(H5_SMLIT)
#error "The reproducer requires unabbreviated, uninstrumented native work"
#endif
using namespace h5;
#if H5_PRESSURE_CTA_TRACE
#define TRACE_PARAMETER , unsigned long long* cta_times
#define TRACE_LAUNCH_ARGUMENT , times.p
__device__ __forceinline__ unsigned long long probe_clock() {
  unsigned long long clock;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(clock) :: "memory");return clock;
}
#define TRACE_BEGIN if(threadIdx.x==0) cta_times[2*(blockIdx.x+gridDim.x*blockIdx.y)]=probe_clock()
#define TRACE_END if(threadIdx.x==0) cta_times[2*(blockIdx.x+gridDim.x*blockIdx.y)+1]=probe_clock()
#else
#define TRACE_PARAMETER
#define TRACE_LAUNCH_ARGUMENT
#define TRACE_BEGIN
#define TRACE_END
#endif
extern "C" __global__ __launch_bounds__(160) void hproj_probe(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ AS, const float* __restrict__ AG,
    int R, int K, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N,
    float* __restrict__ out, int ldo, const float* __restrict__ RS TRACE_PARAMETER) {
  TRACE_BEGIN;
  gemm_fp4_body<4,2,32,0,1>(A,AS,AG,R,K,W,S,N,out,ldo,RS,FuseArgs{});
  TRACE_END;
}
extern "C" __global__ __launch_bounds__(288) void head_pressure_probe(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ AS, const float* __restrict__ AG,
    int R, int K, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N,
    float* __restrict__ out, int ldo, const float* __restrict__ RS TRACE_PARAMETER) {
  TRACE_BEGIN;
  gemm_fp4_body<8,1,64,0,2>(A,AS,AG,R,K,W,S,N,out,ldo,RS,FuseArgs{});
  TRACE_END;
}
namespace {
enum class Failure { None, Invalid, Cuda, Io, NonfiniteBaseline, Foreign, NoOverlap };
struct Status {
  Failure kind; const char* stage; cudaError_t cuda;
  explicit operator bool() const { return kind!=Failure::None; }
};
constexpr Status ok{Failure::None,"",cudaSuccess};
int cleanup_errors=0;
Status checked(cudaError_t e,const char* stage) { return e==cudaSuccess ? ok : Status{Failure::Cuda,stage,e}; }
#define TRY(call) do { const Status error=(call);if(error)return error; } while(0)
#define CUDA(call) TRY(checked((call),#call))
void cleanup(cudaError_t e,const char* stage) {
  if(e!=cudaSuccess) { cleanup_errors++;std::fprintf(stderr,"CUDA cleanup %s: %d %s\n",stage,int(e),cudaGetErrorString(e)); }
}
std::string quote(const std::string& x) {
  std::string out="\"";for(unsigned char c:x) { if(c=='"'||c=='\\') {out+='\\';out+=char(c);}else if(c<32) {char s[7];std::snprintf(s,sizeof(s),"\\u%04x",c);out+=s;}else out+=char(c); }return out+'"';
}
struct Stream {
  cudaStream_t p=nullptr;
  Stream()=default;Stream(const Stream&)=delete;Stream& operator=(const Stream&)=delete;
  Status create() { if(p)return {Failure::Invalid,"stream already set",cudaSuccess};return checked(cudaStreamCreateWithFlags(&p,cudaStreamNonBlocking),"stream create"); }
  ~Stream() { if(p)cleanup(cudaStreamDestroy(p),"stream destroy"); }
};
struct Graph {
  cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
  Graph()=default;Graph(const Graph&)=delete;Graph& operator=(const Graph&)=delete;
  ~Graph() { if(exec)cleanup(cudaGraphExecDestroy(exec),"graph exec destroy");if(graph)cleanup(cudaGraphDestroy(graph),"graph destroy"); }
  Status capture(cudaStream_t st,const std::function<Status(cudaStream_t)>& body) {
    if(graph||exec)return {Failure::Invalid,"graph already set",cudaSuccess};
    CUDA(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
    const Status body_error=body(st);
    const auto end=cudaStreamEndCapture(st,&graph);
    if(body_error)return body_error;
    CUDA(end);CUDA(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));return ok;
  }
  Status launch(cudaStream_t st) { return checked(cudaGraphLaunch(exec,st),"graph launch"); }
};
template<class T> struct Buffer {
  T* p=nullptr;size_t n=0;
  Buffer()=default;Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
  Status create(size_t count) { if(p||!count)return {Failure::Invalid,"buffer create state",cudaSuccess};CUDA(cudaMalloc(&p,count*sizeof(T)));n=count;return ok; }
  Status upload(const std::vector<T>& host) {
    if(host.size()!=n)return {Failure::Invalid,"buffer upload size",cudaSuccess};
    CUDA(cudaMemcpy(p,host.data(),n*sizeof(T),cudaMemcpyHostToDevice));return ok;
  }
  ~Buffer() { if(p)cleanup(cudaFree(p),"buffer free"); }
};
uint32_t hash32(uint32_t x) { x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16); }
uint64_t bits_hash(const std::vector<uint32_t>& v) { uint64_t h=14695981039346656037ULL;for(auto x:v)for(int b=0;b<4;b++){h^=(x>>(b*8))&255;h*=1099511628211ULL;}return h; }
float number(uint32_t bits) { float x;std::memcpy(&x,&bits,4);return x; }
constexpr size_t GUARD=64;
constexpr uint32_t GUARD_BITS=0xa53c96e1u;
enum class WorkKind { Hproj, Pressure };
struct Work {
  WorkKind kind;int rows,columns;static constexpr int k=5120;
  Buffer<uint8_t> a,as,w,s;Buffer<float> ag,rs;Buffer<uint32_t> storage;Buffer<unsigned long long> times;
  std::vector<uint32_t> baseline;
  explicit Work(WorkKind type):kind(type),rows(type==WorkKind::Hproj?448:128),columns(type==WorkKind::Hproj?256:65536){}
  Status create(unsigned seed) {
    const size_t padded=(size_t(rows)+63)&~size_t(63),chunks=size_t(columns/32)*(k/128);
    CUDA(cudaFuncSetAttribute(kind==WorkKind::Hproj ? hproj_probe : head_pressure_probe,
      cudaFuncAttributeMaxDynamicSharedMemorySize,kind==WorkKind::Hproj ? WsGeom<4,2,32,1>::SMEM : WsGeom<8,1,64,2>::SMEM));
    TRY(a.create(padded*k/2));TRY(as.create(padded*k/16));TRY(ag.create(padded));TRY(rs.create(columns));
    TRY(w.create(chunks*(kind==WorkKind::Hproj?2048:1024)));TRY(s.create(chunks*(kind==WorkKind::Hproj?256:64)));
    TRY(storage.create(size_t(rows)*columns+2*GUARD));
#if H5_PRESSURE_CTA_TRACE
    TRY(times.create(size_t(kind==WorkKind::Hproj?14:512)*2));
#endif
    std::vector<uint8_t> bytes(a.n);for(size_t i=0;i<bytes.size();i++)bytes[i]=uint8_t(hash32(uint32_t(i)^seed));TRY(a.upload(bytes));
    bytes.resize(as.n);for(size_t i=0;i<bytes.size();i++)bytes[i]=uint8_t(0x20+hash32(uint32_t(i)^(seed+1))%33);TRY(as.upload(bytes));
    bytes.resize(w.n);for(size_t i=0;i<bytes.size();i++)bytes[i]=uint8_t(hash32(uint32_t(i)^(seed+2)));TRY(w.upload(bytes));
    bytes.resize(s.n);for(size_t i=0;i<bytes.size();i++)bytes[i]=uint8_t(0x28+hash32(uint32_t(i)^(seed+3))%25);TRY(s.upload(bytes));
    std::vector<float> floats(ag.n);for(size_t i=0;i<floats.size();i++)floats[i]=float(16+hash32(uint32_t(i)^(seed+4))%241)/65536.f;TRY(ag.upload(floats));
    floats.resize(rs.n);for(size_t i=0;i<floats.size();i++)floats[i]=float(16+hash32(uint32_t(i)^(seed+5))%241)/131072.f;TRY(rs.upload(floats));
    std::printf("{\"event\":\"work\",\"kind\":\"%s\",\"seed\":%u,\"rows\":%d,\"n\":%d,\"k\":%d,\"device_bytes\":%zu}\n",kind==WorkKind::Hproj?"hproj":"pressure",seed,rows,columns,k,a.n+as.n+w.n+s.n+(ag.n+rs.n+storage.n)*4+times.n*sizeof(unsigned long long));return ok;
  }
  Status reset(cudaStream_t st) {
    std::vector<uint32_t> words(storage.n,0x7fc01234u);
    std::fill(words.begin(),words.begin()+GUARD,GUARD_BITS);std::fill(words.end()-GUARD,words.end(),GUARD_BITS);
    // The synchronous upload completes before any concurrent/PDL/graph test arm.
    CUDA(cudaMemcpyAsync(storage.p,words.data(),words.size()*4,cudaMemcpyHostToDevice,st));
#if H5_PRESSURE_CTA_TRACE
    CUDA(cudaMemsetAsync(times.p,0,times.n*sizeof(unsigned long long),st));
#endif
    CUDA(cudaStreamSynchronize(st));return ok;
  }
  Status launch(cudaStream_t st) {
    auto fn=kind==WorkKind::Hproj ? hproj_probe : head_pressure_probe;
    const int tile=kind==WorkKind::Hproj?32:64,threads=kind==WorkKind::Hproj?160:288;
    const int shared=kind==WorkKind::Hproj?WsGeom<4,2,32,1>::SMEM:WsGeom<8,1,64,2>::SMEM;
    pdl(fn,dim3((rows+tile-1)/tile,(columns/32+7)/8,1),dim3(threads),shared,st,
      a.p,as.p,ag.p,rows,k,w.p,s.p,columns,reinterpret_cast<float*>(storage.p+GUARD),columns,rs.p TRACE_LAUNCH_ARGUMENT);
    CUDA(cudaPeekAtLastError());return ok;
  }
  Status download(std::vector<uint32_t>& core,size_t& guard_differences) {
    std::vector<uint32_t> all(storage.n);CUDA(cudaMemcpy(all.data(),storage.p,all.size()*4,cudaMemcpyDeviceToHost));
    guard_differences=0;for(size_t i=0;i<GUARD;i++)guard_differences+=(all[i]!=GUARD_BITS)+(all[all.size()-1-i]!=GUARD_BITS);
    core.assign(all.begin()+GUARD,all.end()-GUARD);return ok;
  }
};
enum class Backend { Eager, Graphs, Both };
enum class Arm { IsolatedH, IsolatedP, SerialHP, SerialPH, ConcurrentHP, ConcurrentPH, ChainHP, ChainPH };
constexpr std::array<Arm,8> arms{{Arm::IsolatedH,Arm::IsolatedP,Arm::SerialHP,Arm::SerialPH,Arm::ConcurrentHP,Arm::ConcurrentPH,Arm::ChainHP,Arm::ChainPH}};
const char* name(Arm arm) {
  switch(arm) { case Arm::IsolatedH:return "isolated_hproj";case Arm::IsolatedP:return "isolated_pressure";case Arm::SerialHP:return "serial_hp_pressure";case Arm::SerialPH:return "serial_pressure_hp";case Arm::ConcurrentHP:return "concurrent_hp_first";case Arm::ConcurrentPH:return "concurrent_pressure_first";case Arm::ChainHP:return "same_stream_hp_pressure";case Arm::ChainPH:return "same_stream_pressure_hp"; }std::abort();
}
struct Options { int repeats,pressure_calls,device,pdl;Backend backend;bool dump;std::string prefix; };
void usage() {
  std::puts("hproj-pressure --repeats 1..64 --pressure-calls 1..8 --backend eager|graphs|both\n"
    " --pdl 0|1 --device N --dump none|first-difference --output-prefix PATH\n"
    "All seven options required. Hproj: WF1 cfg3 R448 N256 K5120 ks1 MODE0.\n"
    "Pressure: independent WF2 cfg1 R128 N65536 K5120 ks1 MODE0 per call.\n"
    "Checks all float bits and output guards; does not load models or reproduce their lifetime path.");
}
Status integer(const std::string& text,int low,int high,int& result) {
  errno=0;char* end=nullptr;long n=std::strtol(text.c_str(),&end,10);
  if(errno||end==text.c_str()||*end||n<low||n>high)return {Failure::Invalid,"integer range",cudaSuccess};
  result=int(n);return ok;
}
Status parse(int argc,char** argv,Options& o) {
  std::map<std::string,std::string> v;
  for(int i=1;i<argc;i+=2) {
    if(i+1==argc)return {Failure::Invalid,"incomplete option",cudaSuccess};
    const std::string key=argv[i];
    if((key!="--repeats"&&key!="--pressure-calls"&&key!="--backend"&&key!="--pdl"&&key!="--device"&&key!="--dump"&&key!="--output-prefix")||!v.emplace(key,argv[i+1]).second)return {Failure::Invalid,"unknown/duplicate option",cudaSuccess};
  }
  if(v.size()!=7)return {Failure::Invalid,"all seven options required",cudaSuccess};
  TRY(integer(v.at("--repeats"),1,64,o.repeats));TRY(integer(v.at("--pressure-calls"),1,8,o.pressure_calls));TRY(integer(v.at("--device"),0,64,o.device));TRY(integer(v.at("--pdl"),0,1,o.pdl));
  const auto b=v.at("--backend");if(b=="eager")o.backend=Backend::Eager;else if(b=="graphs")o.backend=Backend::Graphs;else if(b=="both")o.backend=Backend::Both;else return {Failure::Invalid,"backend",cudaSuccess};
  const auto d=v.at("--dump");if(d=="none")o.dump=false;else if(d=="first-difference")o.dump=true;else return {Failure::Invalid,"dump",cudaSuccess};
  o.prefix=v.at("--output-prefix");if(o.prefix.empty())return {Failure::Invalid,"output prefix",cudaSuccess};return ok;
}
Status write_words(const std::string& path,const std::vector<uint32_t>& words) {
  std::ofstream file(path,std::ios::binary);if(!file)return {Failure::Io,"open output dump",cudaSuccess};file.write(reinterpret_cast<const char*>(words.data()),words.size()*4);file.close();return file ? ok : Status{Failure::Io,"write output dump",cudaSuccess};
}
struct Comparison { bool exact; };
Status compare(Work& work,const Options& o,const char* backend,const char* arm,int repetition,int index,bool& dumped,Comparison& result) {
  std::vector<uint32_t> current;size_t guards;TRY(work.download(current,guards));
  size_t unequal=0,nonfinite=0;long first=-1;double max_abs=0,squared=0;
  for(size_t i=0;i<current.size();i++) {
    if(current[i]!=work.baseline[i]){unequal++;if(first<0)first=long(i);}
    const double a=number(work.baseline[i]),b=number(current[i]);
    if(!std::isfinite(a)||!std::isfinite(b))nonfinite++;else{const double delta=b-a;max_abs=std::max(max_abs,std::abs(delta));squared+=delta*delta;}
  }
  result={unequal==0&&nonfinite==0&&guards==0};
  const std::string stem=o.prefix+"."+backend+"."+arm+"."+std::to_string(repetition)+"."+std::to_string(index);
  std::string dump="none";
  if(!result.exact&&o.dump&&!dumped){TRY(write_words(stem+".baseline.u32",work.baseline));TRY(write_words(stem+".current.u32",current));dumped=true;dump=stem;}
  std::printf("{\"event\":\"comparison\",\"backend\":\"%s\",\"arm\":\"%s\",\"repeat\":%d,\"work\":%d,\"elements\":%zu,\"bit_differences\":%zu,\"first_element\":%ld,\"guard_differences\":%zu,\"nonfinite_pairs\":%zu,\"max_abs\":%.9g,\"rms\":%.9g,\"baseline_hash\":\"%llu\",\"current_hash\":\"%llu\",\"dump_prefix\":%s,\"exact\":%s}\n",
    backend,arm,repetition,index,current.size(),unequal,first,guards,nonfinite,max_abs,std::sqrt(squared/current.size()),(unsigned long long)bits_hash(work.baseline),(unsigned long long)bits_hash(current),quote(dump).c_str(),result.exact?"true":"false");return ok;
}
#if H5_PRESSURE_CTA_TRACE
Status trace_overlap(Work& hp,const std::vector<std::unique_ptr<Work>>& pressure,const char* backend,Arm arm,int repetition,bool& exposed) {
  std::vector<unsigned long long> h(hp.times.n);CUDA(cudaMemcpy(h.data(),hp.times.p,h.size()*sizeof(h[0]),cudaMemcpyDeviceToHost));
  size_t pairs=0;unsigned long long max_overlap=0;
  for(const auto& work:pressure) {
    std::vector<unsigned long long> p(work->times.n);CUDA(cudaMemcpy(p.data(),work->times.p,p.size()*sizeof(p[0]),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<h.size();i+=2) {
      if(!h[i]||h[i+1]<h[i])return {Failure::Invalid,"hproj CTA trace",cudaSuccess};
      for(size_t j=0;j<p.size();j+=2) {
        if(!p[j]||p[j+1]<p[j])return {Failure::Invalid,"pressure CTA trace",cudaSuccess};
        const auto begin=std::max(h[i],p[j]),end=std::min(h[i+1],p[j+1]);
        if(end>begin){pairs++;max_overlap=std::max(max_overlap,end-begin);}
      }
    }
  }
  exposed=pairs>0;
  std::printf("{\"event\":\"cta_overlap\",\"backend\":\"%s\",\"arm\":\"%s\",\"repeat\":%d,\"overlapping_cta_pairs\":%zu,\"max_overlap_ns\":%llu,\"pressure_exposed\":%s}\n",backend,name(arm),repetition,pairs,max_overlap,exposed?"true":"false");return ok;
}
#endif
Status run(const Options& o,bool& exact) {
  if(setenv("H5_PDL",o.pdl?"1":"0",1))return {Failure::Invalid,"set H5_PDL",cudaSuccess};
  CUDA(cudaSetDevice(o.device));cudaDeviceProp prop{};CUDA(cudaGetDeviceProperties(&prop,o.device));
  if(prop.major!=12||prop.minor!=0)return {Failure::Invalid,"requires sm120 device",cudaSuccess};
  std::printf("{\"event\":\"config\",\"device\":%d,\"gpu\":%s,\"repeats\":%d,\"pressure_calls\":%d,\"pdl\":%d,\"native_source\":%s,\"build_identity\":%s,\"ring_release\":%s,\"scope\":\"native-kernel-isolation-not-model-lifetime\"}\n",o.device,quote(prop.name).c_str(),o.repeats,o.pressure_calls,o.pdl,quote(H5_PRESSURE_NATIVE_SOURCE).c_str(),quote(H5_PRESSURE_BUILD_ID).c_str(),quote(H5_PRESSURE_RING_RELEASE).c_str());
  Stream hs,ps;TRY(hs.create());TRY(ps.create());Work hp(WorkKind::Hproj);TRY(hp.create(0x7193));
  std::vector<std::unique_ptr<Work>> pressure;for(int i=0;i<o.pressure_calls;i++){pressure.emplace_back(new Work(WorkKind::Pressure));TRY(pressure.back()->create(0x181d+unsigned(i)*131));}
  auto emit_hp=[&](cudaStream_t st){return hp.launch(st);};
  auto emit_pressure=[&](cudaStream_t st)->Status {for(auto& w:pressure)TRY(w->launch(st));return ok;};
  auto hp_then_pressure=[&](cudaStream_t st)->Status {TRY(emit_hp(st));return emit_pressure(st);};
  auto pressure_then_hp=[&](cudaStream_t st)->Status {TRY(emit_pressure(st));return emit_hp(st);};
  TRY(hp.reset(hs.p));TRY(emit_hp(hs.p));CUDA(cudaStreamSynchronize(hs.p));size_t guards;TRY(hp.download(hp.baseline,guards));
  if(guards||std::any_of(hp.baseline.begin(),hp.baseline.end(),[](uint32_t x){return !std::isfinite(number(x));}))return {Failure::NonfiniteBaseline,"hproj baseline/guards",cudaSuccess};
  for(auto& w:pressure){TRY(w->reset(ps.p));TRY(w->launch(ps.p));CUDA(cudaStreamSynchronize(ps.p));TRY(w->download(w->baseline,guards));if(guards||std::any_of(w->baseline.begin(),w->baseline.end(),[](uint32_t x){return !std::isfinite(number(x));}))return {Failure::NonfiniteBaseline,"pressure baseline/guards",cudaSuccess};}
  cudaFuncAttributes attr{};int blocks;CUDA(cudaFuncGetAttributes(&attr,hproj_probe));CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,hproj_probe,160,WsGeom<4,2,32,1>::SMEM));
  std::printf("{\"event\":\"resources\",\"work\":\"hproj\",\"ctas\":14,\"registers\":%d,\"shared_bytes\":%d,\"blocks_per_sm\":%d}\n",attr.numRegs,WsGeom<4,2,32,1>::SMEM,blocks);
  CUDA(cudaFuncGetAttributes(&attr,head_pressure_probe));CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,head_pressure_probe,288,WsGeom<8,1,64,2>::SMEM));
  std::printf("{\"event\":\"resources\",\"work\":\"pressure\",\"ctas_per_call\":512,\"registers\":%d,\"shared_bytes\":%d,\"blocks_per_sm\":%d}\n",attr.numRegs,WsGeom<8,1,64,2>::SMEM,blocks);
  Graph gh,gp,ghp,gph;
  if(o.backend!=Backend::Eager){TRY(gh.capture(hs.p,emit_hp));TRY(gp.capture(ps.p,emit_pressure));TRY(ghp.capture(hs.p,hp_then_pressure));TRY(gph.capture(hs.p,pressure_then_hp));}
  bool dumped=false;exact=true;
#if H5_PRESSURE_CTA_TRACE
  size_t exposed_runs=0;
#endif
  for(int backend=0;backend<2;backend++) {
    const bool graphs=backend==1;if((graphs&&o.backend==Backend::Eager)||(!graphs&&o.backend==Backend::Graphs))continue;
    const char* label=graphs?"graphs":"eager";
    auto h=[&](cudaStream_t st){return graphs?gh.launch(st):emit_hp(st);};
    auto p=[&](cudaStream_t st){return graphs?gp.launch(st):emit_pressure(st);};
    for(int repetition=0;repetition<o.repeats;repetition++)for(Arm arm:arms) {
      TRY(hp.reset(hs.p));for(auto& w:pressure)TRY(w->reset(ps.p));
      const auto begin=std::chrono::steady_clock::now();
      switch(arm) {
        case Arm::IsolatedH:TRY(h(hs.p));break;
        case Arm::IsolatedP:TRY(p(ps.p));break;
        case Arm::SerialHP:TRY(h(hs.p));CUDA(cudaStreamSynchronize(hs.p));TRY(p(ps.p));break;
        case Arm::SerialPH:TRY(p(ps.p));CUDA(cudaStreamSynchronize(ps.p));TRY(h(hs.p));break;
        case Arm::ConcurrentHP:TRY(h(hs.p));TRY(p(ps.p));break;
        case Arm::ConcurrentPH:TRY(p(ps.p));TRY(h(hs.p));break;
        case Arm::ChainHP:TRY(graphs?ghp.launch(hs.p):hp_then_pressure(hs.p));break;
        case Arm::ChainPH:TRY(graphs?gph.launch(hs.p):pressure_then_hp(hs.p));break;
      }
      CUDA(cudaStreamSynchronize(hs.p));CUDA(cudaStreamSynchronize(ps.p));
      const double elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
      std::printf("{\"event\":\"timing\",\"backend\":\"%s\",\"arm\":\"%s\",\"repeat\":%d,\"wall_us\":%.3f}\n",label,name(arm),repetition,elapsed);
#if H5_PRESSURE_CTA_TRACE
      if(arm==Arm::ConcurrentHP||arm==Arm::ConcurrentPH||arm==Arm::ChainHP||arm==Arm::ChainPH) {
        bool exposed=false;TRY(trace_overlap(hp,pressure,label,arm,repetition,exposed));exposed_runs+=exposed;
      }
#endif
      Comparison match{true};if(arm!=Arm::IsolatedP){TRY(compare(hp,o,label,name(arm),repetition,0,dumped,match));exact&=match.exact;}
      if(arm!=Arm::IsolatedH)for(size_t i=0;i<pressure.size();i++){TRY(compare(*pressure[i],o,label,name(arm),repetition,int(i)+1,dumped,match));exact&=match.exact;}
      std::fflush(stdout);
    }
  }
#if H5_PRESSURE_CTA_TRACE
  std::printf("{\"event\":\"result\",\"exact\":%s,\"cta_trace\":true,\"exposed_runs\":%zu}\n",exact?"true":"false",exposed_runs);
  if(exact&&!exposed_runs)return {Failure::NoOverlap,"no observed CTA pressure",cudaSuccess};
#else
  std::printf("{\"event\":\"result\",\"exact\":%s,\"cta_trace\":false,\"pressure_exposed\":null}\n",exact?"true":"false");
#endif
  return ok;
}
}
int main(int argc,char** argv) {
  if(argc==2&&std::string(argv[1])=="--help"){usage();return 0;}
  Status status=ok;bool exact=false;
  try { Options o{};status=parse(argc,argv,o);if(!status)status=run(o,exact); }
  catch(const std::exception&){status={Failure::Foreign,"host standard-library exception",cudaSuccess};}
  if(status){std::fprintf(stderr,"{\"event\":\"error\",\"kind\":%d,\"stage\":%s,\"cuda_code\":%d,\"cuda_name\":%s,\"cuda_message\":%s}\n",int(status.kind),quote(status.stage).c_str(),int(status.cuda),quote(cudaGetErrorName(status.cuda)).c_str(),quote(cudaGetErrorString(status.cuda)).c_str());return status.kind==Failure::Invalid?2:status.kind==Failure::NoOverlap?4:1;}
  if(cleanup_errors)return 1;
  return exact?0:3;
}
