// Standalone source-identical prep -> dense5 GEMM control/observer pair.
#define H5_WARP_MICROSCOPE
#define H5_WARP_MICROSCOPE_STANDALONE
#include "../src/fp4.cu"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
namespace h5 { __global__ void k_prep2(PrepArgs); }

static void fail(const std::string& message) { fprintf(stderr, "%s\n", message.c_str()); exit(1); }
static void ck(cudaError_t e) { if (e != cudaSuccess) fail(cudaGetErrorString(e)); }
static int integer(const char* text, int low, int high) {
  char* end; errno = 0; long n = strtol(text, &end, 10);
  if (errno || end == text || *end || n < low || n > high) fail("invalid bounded integer: " + std::string(text));
  return int(n);
}
struct Memory {
  std::vector<void*> pointers;
  template<class T> T* allocate(size_t n) { T* p; ck(cudaMalloc(&p, n * sizeof(T))); pointers.push_back(p); return p; }
  template<class T> T* copy(const std::vector<T>& v) {
    T* p = allocate<T>(v.size()); ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
  }
  ~Memory() { for (void* p : pointers) cudaFree(p); }
};
struct Shape { int n, k, fw, wg, pipe, mode, cfg; };
struct Resources { cudaFuncAttributes attributes; int active_blocks, dynamic_smem; };
static Resources resources[2];
template<int FW, int WG, int MODE, int PIPE, bool TRACE>
static void gemm(h5::PrepArgs a, int rows, int n, int ks, const uint8_t* w, const uint8_t* s,
                 float* out, int ldo, cudaStream_t stream) {
  using G = h5::WsGeom<FW, WG, 8, 3, PIPE>;
  auto f = h5::k_gemm_fp4<FW, WG, 8, MODE, 3, PIPE, TRACE>;
  static bool configured = false;
  if (!configured) {
    ck(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    auto& r = resources[TRACE]; r.dynamic_smem = G::SMEM;
    ck(cudaFuncGetAttributes(&r.attributes, f));
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,f,(FW+1)*32,G::SMEM));
    configured = true;
  }
  h5::pdl(f, dim3(1, (n/32 + FW*WG-1)/(FW*WG), ks), dim3((FW+1)*32), G::SMEM, stream,
          a.q4, a.s4, a.g4, rows, a.n, w, s, n, out, ldo, (const float*)nullptr);
  ck(cudaGetLastError());
}
static void dispatch(bool trace, const Shape& s, h5::PrepArgs a, int rows, int ks,
                     const uint8_t* w, const uint8_t* ws, float* out, int ldo, cudaStream_t st) {
#define RUN(FW, WG, MODE, PIPE) do { if(trace) gemm<FW,WG,MODE,PIPE,true>(a,rows,s.n,ks,w,ws,out,ldo,st); \
  else gemm<FW,WG,MODE,PIPE,false>(a,rows,s.n,ks,w,ws,out,ldo,st); } while(0)
  if (s.mode == 3) RUN(4,2,3,0);
  else if (ks > 1) RUN(8,1,2,4);
  else if (s.mode == 1) RUN(8,1,1,4);
  else RUN(8,1,0,4);
#undef RUN
}
int main(int argc, char** argv) {
  if(argc != 8) fail("usage: warp-microscope gu|input|output ROWS KS SAMPLED_CTAS CHAINS REPEATS OUTPUT_PREFIX\nrows=1|8; ks: gu1, matched input4/output6; samples1..64, chains1..64, repeats1..15; H5_PDL=0|1 required");
  std::string family = argv[1];
  Shape shape;
  if(family == "gu") shape = {34816,5120,4,2,0,3,19};
  else if(family == "input") shape = {16384,5120,8,1,4,0,26};
  else if(family == "output") shape = {5120,6144,8,1,4,1,26};
  else fail("unknown family");
  int rows = integer(argv[2],1,8), ks = integer(argv[3],1,32);
  int samples = integer(argv[4],1,64), chains = integer(argv[5],1,64), repeats = integer(argv[6],1,15);
  if(rows != 1 && rows != 8) fail("rows must be1 or8");
  if(family == "gu" && ks != 1) fail("fused GU requires ks1");
  const char* pdl = getenv("H5_PDL");
  if(!pdl || (strcmp(pdl,"0") && strcmp(pdl,"1"))) fail("H5_PDL=0|1 must be explicit");
  std::string prefix = argv[7];
  for(const char* suffix : {"-timing.csv", "-spans.csv", "-phases.csv", "-meta.json"})
    if(std::ifstream(prefix + suffix).good()) fail("output already exists: " + prefix + suffix);
  std::ofstream timing(prefix+"-timing.csv"), spans(prefix+"-spans.csv"), phases(prefix+"-phases.csv"), meta(prefix+"-meta.json");
  if(!timing || !spans || !phases || !meta) fail("cannot create output files");
  Memory memory;
  cudaDeviceProp prop; ck(cudaGetDeviceProperties(&prop,0));
  cudaStream_t st; ck(cudaStreamCreate(&st));
  const int n = shape.n, k = shape.k, ng = n/32, nkb = k/128;
  int ldo = family == "gu" ? n/2 : n;
  std::mt19937 rng(20261007);
  std::normal_distribution<float> normal(0,1);
  std::vector<float> x(rows*k), norm(k,1.f), sign(k), z(rows*k,1.f), gnorm(128,1.f);
  for(float& v:x) v=normal(rng);
  for(float& v:sign) v=(rng()&1)?1.f:-1.f;
  std::vector<uint8_t> weights(size_t(nkb)*ng*832);
  std::vector<__half> ws(size_t(nkb)*ng*32);
  for(int kb=0;kb<nkb;kb++) for(int g=0;g<ng;g++) {
    uint8_t* chunk = weights.data() + (size_t(kb)*ng+g)*832;
    for(int lane=0;lane<32;lane++) {
      int8_t trits[128]; for(int8_t& t:trits) t=int(rng()%3)-1;
      dense5::pack_lane(chunk,lane,trits);
    }
    for(int i=0;i<32;i++) ws[(size_t(kb)*ng+g)*32+i]=__float2half(.0125f + float(rng()%100)*.0001f);
  }
  const uint8_t* w = memory.copy(weights);
  const uint8_t* s = (const uint8_t*)memory.copy(ws);
  h5::PrepArgs a{};
  a.x=memory.copy(x); a.n=k; a.ldx=k; a.ldo=k;
  a.norm_w=memory.copy(norm); a.signs=memory.copy(sign);
  a.flags=h5::P_SIGN|h5::P_HADAMARD|h5::P_QUANT;
  if(family == "output") {
    a.flags |= h5::P_GDN_NORM|h5::P_PERMUTE_GDN;
    a.gnorm_w=memory.copy(gnorm); a.gz=memory.copy(z); a.gz_ld=k;
  } else a.flags |= h5::P_NORM;
  a.q4=memory.allocate<uint8_t>(size_t(64)*k/2);
  a.s4=memory.allocate<uint8_t>(size_t(64)*k/16);
  a.g4=memory.allocate<float>(rows);
  // Real prep writes live rows only; TMA reads a whole token-eight tile.
  ck(cudaMemset(a.q4,0,size_t(64)*k/2)); ck(cudaMemset(a.s4,0,size_t(64)*k/16));
  float* out=memory.allocate<float>(size_t(rows)*ldo);
  unsigned ctas=(ng+shape.fw*shape.wg-1)/(shape.fw*shape.wg)*ks;
  samples=std::min(unsigned(samples),ctas);
  h5::micro::Config config{memory.allocate<h5::micro::Span>(samples*(shape.fw+1)),
    memory.allocate<h5::micro::Interval>(size_t(samples)*(shape.fw+1)*((nkb+ks-1)/ks+1)*h5::micro::PhaseCount),
    ctas,unsigned(samples),unsigned(shape.fw+1),unsigned((nkb+ks-1)/ks+1)};
  ck(cudaMemset(config.spans,0,size_t(samples)*config.warps*sizeof(h5::micro::Span)));
  size_t interval_count=size_t(samples)*config.warps*config.slots*h5::micro::PhaseCount;
  ck(cudaMemset(config.intervals,0,interval_count*sizeof(h5::micro::Interval)));
  ck(cudaMemcpyToSymbol(h5::micro::config,&config,sizeof(config)));
  cudaGraph_t graph[2]; cudaGraphExec_t exec[2];
  for(int trace=0;trace<2;trace++) {
    // Attribute preparation outside capture and timing.
    ck(cudaMemsetAsync(out,0,size_t(rows)*ldo*4,st));
    h5::pdl(h5::k_prep2,dim3(rows),dim3(32*k/1024),0,st,a); ck(cudaGetLastError());
    dispatch(trace,shape,a,rows,ks,w,s,out,ldo,st); ck(cudaStreamSynchronize(st));
    ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<chains;i++) {
      ck(cudaMemsetAsync(out,0,size_t(rows)*ldo*4,st));
      h5::pdl(h5::k_prep2,dim3(rows),dim3(32*k/1024),0,st,a); ck(cudaGetLastError());
      dispatch(trace,shape,a,rows,ks,w,s,out,ldo,st);
    }
    ck(cudaStreamEndCapture(st,&graph[trace])); ck(cudaGraphInstantiate(&exec[trace],graph[trace],0));
    for(int i=0;i<2;i++) ck(cudaGraphLaunch(exec[trace],st)); ck(cudaStreamSynchronize(st));
  }
  std::vector<float> control(rows*ldo), observed(rows*ldo);
  ck(cudaGraphLaunch(exec[0],st)); ck(cudaStreamSynchronize(st));
  ck(cudaMemcpy(control.data(),out,control.size()*4,cudaMemcpyDeviceToHost));
  ck(cudaGraphLaunch(exec[1],st)); ck(cudaStreamSynchronize(st));
  ck(cudaMemcpy(observed.data(),out,observed.size()*4,cudaMemcpyDeviceToHost));
  size_t different=0; float max_abs=0, max_scaled=0;
  for(size_t i=0;i<control.size();i++) {
    if(!std::isfinite(control[i]) || !std::isfinite(observed[i])) fail("nonfinite output");
    different += memcmp(&control[i],&observed[i],4)!=0;
    float delta=std::fabs(control[i]-observed[i]); max_abs=std::max(max_abs,delta);
    max_scaled=std::max(max_scaled,delta/(1.f+std::fabs(control[i])));
  }
  if(ks==1 && different) fail("observer changed unsplit math (bitwise mismatch)");
  if(ks>1 && max_scaled>2e-5f) fail("observer/control split-K mismatch beyond explicit scaled tolerance2e-5");
  cudaEvent_t start,stop; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
  timing << "repeat,arm,position,chain_us\n" << std::setprecision(9);
  std::vector<float> timings[2];
  for(int rep=0;rep<repeats;rep++) for(int pos=0;pos<4;pos++) {
    int arm = pos==1 || pos==2;
    ck(cudaEventRecord(start,st)); ck(cudaGraphLaunch(exec[arm],st));
    ck(cudaEventRecord(stop,st)); ck(cudaEventSynchronize(stop));
    float ms; ck(cudaEventElapsedTime(&ms,start,stop)); float us=ms*1000/chains;
    timings[arm].push_back(us); timing<<rep<<','<<(arm?"observer":"control")<<','<<pos<<','<<us<<'\n';
  }
  // One bounded observed chain supplies a coherent critical-end snapshot, not accumulated graph visits.
  ck(cudaMemsetAsync(out,0,size_t(rows)*ldo*4,st));
  h5::pdl(h5::k_prep2,dim3(rows),dim3(32*k/1024),0,st,a);
  dispatch(true,shape,a,rows,ks,w,s,out,ldo,st); ck(cudaStreamSynchronize(st));
  std::vector<h5::micro::Span> host_spans(samples*config.warps);
  std::vector<h5::micro::Interval> host_intervals(interval_count);
  ck(cudaMemcpy(host_spans.data(),config.spans,host_spans.size()*sizeof(host_spans[0]),cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(host_intervals.data(),config.intervals,interval_count*sizeof(host_intervals[0]),cudaMemcpyDeviceToHost));
  spans << "sample,cta,x,y,z,warp,role,sm,begin_ns,end_ns,begin_cycles,end_cycles\n";
  phases << "sample,cta,x,y,z,warp,role,sm,kb_local,kb_global,ring_stage,phase,kind,begin_ns,end_ns,cycles\n";
  int stages=shape.pipe?shape.pipe:h5::WsGeom<4,2,8,3>::STAGES;
  unsigned gy=ctas/ks;
  for(unsigned sample=0;sample<config.samples;sample++) {
    unsigned cta=samples==1?ctas-1:uint64_t(sample)*(ctas-1)/(samples-1), y=cta%gy, split=cta/gy;
    int kb0=nkb*split/ks, kb1=nkb*(split+1)/ks;
    for(unsigned warp=0;warp<config.warps;warp++) {
      size_t index=sample*config.warps+warp; auto span=host_spans[index];
      if(!span.begin_ns || span.end_ns<span.begin_ns) fail("missing/invalid sampled warp span");
      const char* role=warp==unsigned(shape.fw)?"producer":"consumer";
      spans<<sample<<','<<cta<<",0,"<<y<<','<<split<<','<<warp<<','<<role<<','<<span.sm<<','<<span.begin_ns<<','<<span.end_ns<<','<<span.begin_cycles<<','<<span.end_cycles<<'\n';
      for(int kb=-1;kb<kb1-kb0;kb++) for(unsigned phase=0;phase<h5::micro::PhaseCount;phase++) {
        const auto& r=host_intervals[(index*config.slots+kb+1)*h5::micro::PhaseCount+phase];
        if(!r.visits) continue;
        bool wait=phase==h5::micro::Predecessor || phase==h5::micro::FinalPredecessor || phase==h5::micro::EmptyWait || phase==h5::micro::FullWait;
        phases<<sample<<','<<cta<<",0,"<<y<<','<<split<<','<<warp<<','<<role<<','<<span.sm<<','<<kb<<','<<(kb<0?-1:kb0+kb)<<','<<(kb<0?-1:kb%stages)<<','<<h5::micro::phase_names[phase]<<','<<(wait?"wait_region":"work_region")<<','<<r.begin_ns<<','<<r.end_ns<<','<<r.cycles<<'\n';
      }
    }
  }
  auto median=[](std::vector<float> v){std::sort(v.begin(),v.end());return (v[(v.size()-1)/2]+v[v.size()/2])*.5f;};
  float control_us=median(timings[0]), observer_us=median(timings[1]);
  meta<<std::setprecision(9)<<"{\"family\":\""<<family<<"\",\"rows\":"<<rows<<",\"n\":"<<n<<",\"k\":"<<k
    <<",\"cfg\":"<<shape.cfg<<",\"ks\":"<<ks<<",\"stages\":"<<stages<<",\"ctas\":"<<ctas<<",\"sampled_ctas\":"<<samples
    <<",\"chains\":"<<chains<<",\"repeats\":"<<repeats<<",\"pdl\":"<<pdl<<",\"compute_major\":"<<prop.major
    <<",\"control_us\":"<<control_us<<",\"observer_us\":"<<observer_us<<",\"perturbation_percent\":"<<100*(observer_us/control_us-1)
    <<",\"bitwise_different\":"<<different<<",\"max_abs\":"<<max_abs<<",\"max_scaled\":"<<max_scaled
    <<",\"control_resources\":{\"registers_per_thread\":"<<resources[0].attributes.numRegs<<",\"local_bytes\":"<<resources[0].attributes.localSizeBytes<<",\"dynamic_smem\":"<<resources[0].dynamic_smem<<",\"active_blocks_per_sm\":"<<resources[0].active_blocks<<'}'
    <<",\"observer_resources\":{\"registers_per_thread\":"<<resources[1].attributes.numRegs<<",\"local_bytes\":"<<resources[1].attributes.localSizeBytes<<",\"dynamic_smem\":"<<resources[1].dynamic_smem<<",\"active_blocks_per_sm\":"<<resources[1].active_blocks<<'}'
#ifdef H5_NO_L2PF
    <<",\"l2_prefetch\":false"
#else
    <<",\"l2_prefetch\":true"
#endif
#ifdef H5_PDL_EARLY_GEMM_RELEASE
    <<",\"early_release\":true"
#else
    <<",\"early_release\":false"
#endif
    <<",\"scope\":\"seeded synthetic dense5 weights/inputs, real unmodified k_prep2 -> integrated GEMM; warm graph with output clear; software intervals, not hardware stall counters\"}\n";
  timing.close();spans.close();phases.close();meta.close();
  if(!timing || !spans || !phases || !meta) fail("output write failed");
  printf("%s rows%d ks%d: control%.3fus observer%.3fus perturbation%+.2f%%; %s-{timing,spans,phases}.csv\n",family.c_str(),rows,ks,control_us,observer_us,100*(observer_us/control_us-1),prefix.c_str());
  ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
  for(int i=0;i<2;i++){ck(cudaGraphExecDestroy(exec[i]));ck(cudaGraphDestroy(graph[i]));}
  ck(cudaStreamDestroy(st));
}
