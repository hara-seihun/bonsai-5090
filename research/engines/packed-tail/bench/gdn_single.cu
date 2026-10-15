#include "gdn_single.h"
#include <cuda_profiler_api.h>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace h5;
static_assert(gdn_single_eligible(Seg{9,3,1,0,0,1,1}));
static_assert(!gdn_single_eligible(Seg{9,3,1,2,0,1,1}));
static_assert(!gdn_single_eligible(Seg{9,3,1,0,1,1,1}));
static_assert(!gdn_single_eligible(Seg{9,3,2,0,0,1,1}));
static_assert(!gdn_single_eligible(Seg{9,3,0,0,0,1,1}));
namespace h5 {
void launch_conv(const ConvArgs&, int, StateFormat, cudaStream_t);
void launch_gdn(const GdnArgs&, int, StateFormat, cudaStream_t);
void launch_conv_gdn(const ConvArgs&, const GdnArgs&, int, StateFormat, cudaStream_t);
}
static void ck(cudaError_t e) {
  if (e != cudaSuccess) { fprintf(stderr, "error=cuda code=%d detail=%s\n", (int)e, cudaGetErrorString(e)); exit(2); }
}
template <class T> T* alloc(size_t n) { T* p; ck(cudaMalloc(&p, n * sizeof(T))); return p; }
template <class T> __device__ T cvt(float x) { return T(x); }
template <> __device__ __half cvt(float x) { return __float2half_rn(x); }
template <> __device__ __nv_bfloat16 cvt(float x) { return __float2bfloat16_rn(x); }
static constexpr size_t MS = HV * 128 * 128, CS = 3 * CONV_CH;
static constexpr int GL = 3;
template <class T>
__global__ void init(T* a, T* b, float* ca, float* cb, int streams, int banks, int layers = 1) {
  const size_t ti = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  const int first = layers == 1 ? GL : 0;
  for (size_t i = ti; i < (size_t)streams * layers * MS; i += stride) {
    const size_t s = i / (layers * MS), gl = first + i / MS % layers, d = i % MS, index = (s * NGDN + gl) * MS + d;
    const T v = cvt<T>(((int)(i % 251) - 125) * .001f);
    a[index] = v; b[index] = v;
  }
  for (size_t i = ti; i < (size_t)streams * layers * CS; i += stride) {
    const size_t s = i / (layers * CS), gl = first + i / CS % layers, d = i % CS;
    const float v = ((int)(i % 239) - 119) * .002f;
    ca[(s * NGDN + gl) * CS + d] = v;
    for (int k = 0; k < banks; ++k) cb[((s * NGDN + gl) * banks + k) * CS + d] = v;
  }
}
__global__ void inputs(float* proj, float* w, float* ab, float* sa, float* dt, int streams) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (int i = t; i < streams * GDN_IN; i += stride) proj[i] = sinf(i * .19f) * .4f;
  for (int i = t; i < streams * 96; i += stride) ab[i] = cosf(i * .1f) * .9f;
  for (int i = t; i < CONV_CH * 4; i += stride) w[i] = sinf(i * .015f) * .2f;
  for (int i = t; i < HV; i += stride) { sa[i] = -.1f - i * .002f; dt[i] = -.4f + i * .004f; }
}
// Emulates the upstream projection's acquire before it releases the state reader.
// Without this boundary a graph of repeated fused updates could read its own
// preceding update before its PDL wait, a chain the engine never constructs.
__global__ void boundary(Seg* segs, int streams, int parity) {
  pdl_wait();
  const int t = threadIdx.x;
  if (t < streams) segs[t] = Seg{streams - 1 - t, t, 1, 0, 0, 0, parity};
  __syncthreads(); pdl_launch();
}
struct Error {
  double sq = 0, refsq = 0; float max_abs = 0; size_t mismatches = 0, count = 0; bool finite = true;
  void add(float a, float b) {
    finite &= std::isfinite(a) && std::isfinite(b);
    max_abs = std::max(max_abs, fabsf(a - b));
    sq += double(a - b) * double(a - b); refsq += double(a) * a;
    mismatches += a != b; ++count;
  }
  void show(const char* mode, const char* what, int steps) const {
    printf("error mode=%s what=%s steps=%d max_abs=%.9g rel_rms=%.9g different=%zu/%zu finite=%d\n",
        mode, what, steps, max_abs, sqrt(sq / std::max(refsq, 1e-300)), mismatches, count, finite);
  }
};
template <class T> static std::vector<T> copy(const T* p, size_t n) {
  std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
struct Fixture {
  int streams, es, banks = 1; StateFormat format;
  void *s0, *s1; float *c0, *c1, *y0, *y1;
  float *proj, *w, *ab, *sa, *dt, *co, *gates;
  Seg *segs; cudaStream_t stream;
  ConvArgs c{}, cf{}; GdnArgs a{}, af{};
  Fixture(int n, StateFormat f) : streams(n), es(f == StateFormat::F32 ? 4 : 2), format(f) {
    ck(cudaStreamCreate(&stream));
    s0 = alloc<unsigned char>((size_t)n * NGDN * MS * es); s1 = alloc<unsigned char>((size_t)n * NGDN * MS * es);
    c0 = alloc<float>((size_t)n * NGDN * CS); c1 = alloc<float>((size_t)n * NGDN * 2 * CS);
    y0 = alloc<float>(n * VDIM); y1 = alloc<float>(n * VDIM);
    proj = alloc<float>(n * GDN_IN); w = alloc<float>(CONV_CH * 4); ab = alloc<float>(n * 96);
    sa = alloc<float>(HV); dt = alloc<float>(HV); co = alloc<float>(n * CONV_CH); gates = alloc<float>(n * 96);
    segs = alloc<Seg>(n);
    c = ConvArgs{proj,w,c0,GL,co,segs,s0,ab,sa,dt,gates}; cf = c; cf.state = c1;
    a = GdnArgs{co,proj,ab,sa,dt,nullptr,s0,GL,y0,segs,s0,gates}; af = a; af.state = s1; af.y = y1;
    inputs<<<128,256,0,stream>>>(proj,w,ab,sa,dt,n); ck(cudaStreamSynchronize(stream));
  }
  ~Fixture() {
    for (void* p : {s0,s1,(void*)c0,(void*)c1,(void*)y0,(void*)y1,(void*)proj,(void*)w,(void*)ab,(void*)sa,(void*)dt,(void*)co,(void*)gates,(void*)segs}) ck(cudaFree(p));
    ck(cudaStreamDestroy(stream));
  }
  void reset(int b, int layers = 1) {
    banks = b;
    if (format == StateFormat::F16) init<<<256,256,0,stream>>>((__half*)s0,(__half*)s1,c0,c1,streams,b,layers);
    else if (format == StateFormat::BF16) init<<<256,256,0,stream>>>((__nv_bfloat16*)s0,(__nv_bfloat16*)s1,c0,c1,streams,b,layers);
    else init<<<256,256,0,stream>>>((float*)s0,(float*)s1,c0,c1,streams,b,layers);
    ck(cudaStreamSynchronize(stream));
  }
  void visit(int mode, int step) {
    pdl(boundary, dim3(1), dim3(128), 0, stream, segs, streams, step & 1);
    if (mode == 0) { launch_conv(c,streams,format,stream); launch_gdn(a,streams,format,stream); }
    else if (mode == 1) { launch_conv(cf,streams,format,stream); launch_gdn_single(af,streams,format,stream); }
    else if (mode == 2) launch_conv_gdn_single(cf,af,streams,format,stream);
    else if (mode == 3) launch_conv_gdn(cf,af,streams,format,stream);
    else abort();
  }
  bool compare(const char* mode, int steps) {
    Error output, state, conv;
    const auto ya = copy(y0,(size_t)streams * VDIM), yb = copy(y1,(size_t)streams * VDIM);
    for (size_t i = 0; i < ya.size(); ++i) output.add(ya[i],yb[i]);
    for (int slot = 0; slot < streams; ++slot) {
      const size_t off = ((size_t)slot * NGDN + GL) * MS;
      if (format == StateFormat::F32) {
        const auto aa = copy((float*)s0+off,MS), bb = copy((float*)s1+off,MS);
        for (size_t i = 0; i < MS; ++i) state.add(aa[i],bb[i]);
      } else if (format == StateFormat::F16) {
        const auto aa = copy((__half*)s0+off,MS), bb = copy((__half*)s1+off,MS);
        for (size_t i = 0; i < MS; ++i) state.add(__half2float(aa[i]),__half2float(bb[i]));
      } else {
        const auto aa = copy((__nv_bfloat16*)s0+off,MS), bb = copy((__nv_bfloat16*)s1+off,MS);
        for (size_t i = 0; i < MS; ++i) state.add(__bfloat162float(aa[i]),__bfloat162float(bb[i]));
      }
      const auto ca = copy(c0 + ((size_t)slot * NGDN + GL) * CS,CS);
      const auto cb = copy(c1 + (((size_t)slot * NGDN + GL) * banks + (banks == 2 ? steps & 1 : 0)) * CS,CS);
      for (size_t i = 0; i < CS; ++i) conv.add(ca[i],cb[i]);
    }
    output.show(mode,"output",steps); state.show(mode,"state",steps); conv.show(mode,"conv",steps);
    return output.finite && state.finite && conv.finite && conv.mismatches == 0;
  }
  cudaGraphExec_t graph(int mode, int count) {
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for (int i = 0; i < count; ++i) visit(mode,i);
    cudaGraph_t g; ck(cudaStreamEndCapture(stream,&g)); cudaGraphExec_t e;
    ck(cudaGraphInstantiate(&e,g,nullptr,nullptr,0)); ck(cudaGraphDestroy(g)); return e;
  }
  float time(cudaGraphExec_t e, int count) {
    cudaEvent_t a,b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a,stream)); ck(cudaGraphLaunch(e,stream)); ck(cudaEventRecord(b,stream));
    ck(cudaEventSynchronize(b)); float ms; ck(cudaEventElapsedTime(&ms,a,b));
    ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b)); return ms * 1000 / count;
  }
};
#include "gdn_fused_causal.cuh"
int main(int argc, char** argv) {
  if (argc > 1 && (!strcmp(argv[1], "--fused-causal") || !strcmp(argv[1], "--fused-probe")))
    return fused_causal_main(argc, argv);
  if (argc < 3 || argc > 5) { fprintf(stderr,"usage: gdn-single-bench 1..64 f16|bf16|f32 [even graph visits, default 100] [--pad-abba]\n"); return 1; }
  const bool pad_abba = argc == 5 && !strcmp(argv[4], "--pad-abba");
  if (argc == 5 && !pad_abba) { fprintf(stderr,"fifth argument must be --pad-abba\n"); return 1; }
  const int n = atoi(argv[1]), count = argc >= 4 ? atoi(argv[3]) : 100;
  if (n < 1 || n > 64 || count < 2 || count > 2000 || count % 2) {
    fprintf(stderr,"invalid benchmark shape: streams must be1..64 and graph visits an even integer2..2000\n"); return 1;
  }
  StateFormat f;
  if (!strcmp(argv[2],"f16")) f = StateFormat::F16;
  else if (!strcmp(argv[2],"bf16")) f = StateFormat::BF16;
  else if (!strcmp(argv[2],"f32")) f = StateFormat::F32;
  else return 1;
  try {
    auto pad = [](const char* kib) { if (setenv("H5_GDN_SINGLE_PAD_KIB",kib,1)) throw std::runtime_error("setting padding environment failed"); };
    if (pad_abba) pad("40");
    validate_gdn_single(GdnSingleMode::Separate,f,false,false,0);
    Fixture b(n,f);
    const char* names[] = {"generic","single-separate","single-fused","generic-fused"};
    printf("config streams=%d dtype=%s cg=%s cg_segs=%s pdl=%d visits=%d includes_acquire_boundary=1\n", n,argv[2],
        getenv("H5_GDN_CG") ? getenv("H5_GDN_CG") : "4",getenv("H5_GDN_CG_SEGS") ? getenv("H5_GDN_CG_SEGS") : "0",pdl_enabled(),count);
    printf("experiment=%s dynamic_shared_bytes=%zu carveout=%d\n",pad_abba ? "same-kernel-pad0-vs40" : "generic-vs-specialized",gdn_single_pad_bytes(),launch_carveout());
    for (int mode = 1; mode <= (pad_abba ? 2 : 3); ++mode) {
      if (pad_abba) pad("40");
      b.reset(mode == 1 ? 1 : 2);
      for (int i = 0; i < 16; ++i) {
        b.visit(0,i); b.visit(mode,i);
        ck(cudaStreamSynchronize(b.stream));
        if ((i == 0 || i == 15) && !b.compare(names[mode],i+1)) return 3;
      }
      b.reset(mode == 1 ? 1 : 2);
      if (pad_abba) pad("0");
      const auto a = b.graph(pad_abba ? mode : 0,count);
      if (pad_abba) pad("40");
      const auto c = b.graph(mode,count);
      for (int i = 0; i < 4; ++i) { ck(cudaGraphLaunch(a,b.stream)); ck(cudaGraphLaunch(c,b.stream)); }
      ck(cudaStreamSynchronize(b.stream));
      std::vector<float> refs, opts, deltas;
      for (int i = 0; i < 10; ++i) {
        const float a1 = b.time(a,count), b1 = b.time(c,count), b2 = b.time(c,count), a2 = b.time(a,count);
        const float r = (a1+a2)*.5f, s = (b1+b2)*.5f;
        refs.push_back(r); opts.push_back(s); deltas.push_back(r-s);
      }
      std::sort(refs.begin(),refs.end()); std::sort(opts.begin(),opts.end()); std::sort(deltas.begin(),deltas.end());
      printf("timing mode=%s experiment=%s streams=%d ref_us=%.6f candidate_us=%.6f speedup=%.6f paired_saved_us=%.6f\n",
          names[mode],pad_abba ? "pad0-vs40" : "generic-vs-specialized",n,refs[5],opts[5],refs[5]/opts[5],deltas[5]); fflush(stdout);
      ck(cudaGraphExecDestroy(a)); ck(cudaGraphExecDestroy(c));
    }
  } catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
