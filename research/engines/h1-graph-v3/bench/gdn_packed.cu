#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using namespace h5;
namespace h5 { void launch_gdn(const GdnArgs&, int, StateFormat, cudaStream_t); }
#define CHECK(x) do { const cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(2); } } while (0)
constexpr size_t MS = (size_t)HV * 128 * 128;
struct Metric {
  double squared, reference_squared, max_abs;
  float max_reference;
  unsigned long long nonfinite_reference, nonfinite_candidate, unequal, count, finite_pairs;
};
template<class T> T* alloc(size_t n) { T* p = nullptr; CHECK(cudaMalloc(&p, n * sizeof(T))); return p; }
template<class T> __global__ void init_state(T* a, T* b, size_t n, float scale) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t at = (i / MS) * NGDN * MS + i % MS;
    const T v = T(scale * .03f * sinf((float)(i % 8191) * .17f));
    a[at] = v; b[at] = v;
  }
}
template<class T> __global__ void compare(const T* a, const T* b, size_t per_slot, size_t stride, int slots, Metric* result) {
  Metric m{};
  const size_t n = per_slot * slots;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const size_t at = (i / per_slot) * stride + i % per_slot;
    const float x = float(a[at]), y = float(b[at]);
    m.count++; m.unequal += x != y;
    m.nonfinite_reference += !isfinite(x); m.nonfinite_candidate += !isfinite(y);
    if (isfinite(x) && isfinite(y)) {
      m.finite_pairs++;
      const double d = fabs((double)x-y);
      m.squared += d * d; m.reference_squared += (double)x * x;
      m.max_abs = fmax(m.max_abs, d); m.max_reference = fmaxf(m.max_reference, fabsf(x));
    }
  }
  __shared__ Metric sm[256]; sm[threadIdx.x] = m; __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) {
      Metric& x = sm[threadIdx.x]; const Metric& y = sm[threadIdx.x+d];
      x.squared += y.squared; x.reference_squared += y.reference_squared;
      x.max_abs = fmax(x.max_abs, y.max_abs); x.max_reference = fmaxf(x.max_reference, y.max_reference);
      x.nonfinite_reference += y.nonfinite_reference; x.nonfinite_candidate += y.nonfinite_candidate;
      x.unequal += y.unequal; x.count += y.count; x.finite_pairs += y.finite_pairs;
    }
    __syncthreads();
  }
  if (!threadIdx.x) result[blockIdx.x] = sm[0];
}
template<class T> void report(const char* mode, const char* name, int step, const T* a, const T* b,
                             size_t per_slot, size_t stride, int slots, Metric* scratch, cudaStream_t stream) {
  compare<<<256,256,0,stream>>>(a,b,per_slot,stride,slots,scratch); CHECK(cudaGetLastError());
  std::vector<Metric> blocks(256); CHECK(cudaMemcpyAsync(blocks.data(),scratch,blocks.size()*sizeof(Metric),cudaMemcpyDeviceToHost,stream));
  CHECK(cudaStreamSynchronize(stream)); Metric m{};
  for (const Metric& x : blocks) {
    m.squared += x.squared; m.reference_squared += x.reference_squared;
    m.max_abs = std::max(m.max_abs,x.max_abs); m.max_reference = std::max(m.max_reference,x.max_reference);
    m.nonfinite_reference += x.nonfinite_reference; m.nonfinite_candidate += x.nonfinite_candidate;
    m.unequal += x.unequal; m.count += x.count; m.finite_pairs += x.finite_pairs;
  }
  char rms[64], relative[64];
  if (m.finite_pairs) snprintf(rms,sizeof(rms),"%.9g",sqrt(m.squared/m.finite_pairs));
  else strcpy(rms,"null");
  if (m.reference_squared) snprintf(relative,sizeof(relative),"%.9g",sqrt(m.squared/m.reference_squared));
  else strcpy(relative,"null");
  printf("{\"kind\":\"quality\",\"mode\":\"%s\",\"tensor\":\"%s\",\"step\":%d,\"streams\":%d,\"count\":%llu,\"finite_pairs\":%llu,\"unequal\":%llu,\"nonfinite_reference\":%llu,\"nonfinite_candidate\":%llu,\"max_abs\":%.9g,\"max_reference\":%.9g,\"rms_abs\":%s,\"l2_relative\":%s}\n",
    mode,name,step,slots,m.count,m.finite_pairs,m.unequal,m.nonfinite_reference,m.nonfinite_candidate,m.max_abs,m.max_reference,rms,relative);
}
static void select(const char* mode) { if (setenv("H5_GDN_ARITH",mode,1)) { perror("setenv"); std::exit(2); } }
static float time_gdn(GdnArgs a, int slots, StateFormat format, const char* mode, int iterations, cudaStream_t stream) {
  select(mode); cudaGraph_t graph; cudaGraphExec_t exec; cudaEvent_t begin,end;
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  launch_gdn(a,slots,format,stream);
  CHECK(cudaStreamEndCapture(stream,&graph)); CHECK(cudaGraphInstantiate(&exec,graph,0));
  for (int i=0;i<5;i++) CHECK(cudaGraphLaunch(exec,stream));
  CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end)); CHECK(cudaEventRecord(begin,stream));
  for (int i=0;i<iterations;i++) CHECK(cudaGraphLaunch(exec,stream));
  CHECK(cudaEventRecord(end,stream)); CHECK(cudaEventSynchronize(end)); float ms;
  CHECK(cudaEventElapsedTime(&ms,begin,end)); CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(graph));
  CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end)); return ms*1000.f/iterations;
}
template<class T> int run(const char* mode, StateFormat format, int slots, int iterations, float scale) {
  constexpr int rows_per_slot=21, pending=8;
  const size_t state_n=((size_t)(slots-1)*NGDN+1)*MS;
  const size_t ps=(size_t)2*pending*PEND_GDN, pend_n=((size_t)(slots-1)*NGDN+1)*ps;
  const int rows=slots*rows_per_slot;
  T* m0=alloc<T>(state_n), *m1=alloc<T>(state_n), *p0=alloc<T>(pend_n), *p1=alloc<T>(pend_n);
  float* co=alloc<float>((size_t)rows*CONV_CH), *gates=alloc<float>((size_t)rows*96);
  float* y0=alloc<float>((size_t)rows*VDIM), *y1=alloc<float>((size_t)rows*VDIM);
  Seg* segs=alloc<Seg>(slots); Metric* scratch=alloc<Metric>(256);
  cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  init_state<<<256,256,0,stream>>>(m0,m1,(size_t)slots*MS,scale); CHECK(cudaGetLastError());
  std::vector<float> data((size_t)rows*CONV_CH), gs((size_t)rows*96);
  for (int r=0;r<rows;r++) {
    for (int kh=0;kh<HK;kh++) for (int qk=0;qk<2;qk++) {
      float ss=0; const int base=r*CONV_CH+qk*KDIM+kh*128;
      for (int d=0;d<128;d++) { const float v=sinf((r%rows_per_slot+1)*.19f+d*.31f+kh*.13f+qk*.7f); data[base+d]=v; ss+=v*v; }
      for (int d=0;d<128;d++) data[base+d]/=sqrtf(ss);
    }
    for (int d=0;d<VDIM;d++) data[(size_t)r*CONV_CH+2*KDIM+d]=scale*.5f*sinf(d*.11f+r*.17f);
    for (int h=0;h<HV;h++) { gs[r*96+h]=.93f+.065f*(h%7)/6.f; gs[r*96+HV+h]=.2f+.7f*(h%5)/4.f; }
  }
  CHECK(cudaMemcpyAsync(co,data.data(),data.size()*4,cudaMemcpyHostToDevice,stream));
  CHECK(cudaMemcpyAsync(gates,gs.data(),gs.size()*4,cudaMemcpyHostToDevice,stream));
  GdnArgs a{}; a.conv_out=co; a.state=m0; a.y=y0; a.segs=segs; a.pend=p0; a.gates=gates;
  a.replay_delta=getenv("H5_GDN_DELTA") && atoi(getenv("H5_GDN_DELTA"));
  GdnArgs b=a; b.state=m1; b.y=y1; b.pend=p1;
  std::vector<Seg> host(slots);
  const int steps[][4]={{8,0,1,0},{8,4,1,1},{8,8,0,0},{21,0,0,0}};
  for (int step=0;step<4;step++) {
    for (int s=0;s<slots;s++) host[s]={s,s*rows_per_slot,steps[step][0],steps[step][1],steps[step][2],steps[step][3]};
    CHECK(cudaMemcpyAsync(segs,host.data(),host.size()*sizeof(Seg),cudaMemcpyHostToDevice,stream));
    select("fp32"); launch_gdn(a,slots,format,stream); select(mode); launch_gdn(b,slots,format,stream);
    CHECK(cudaGetLastError());
    report(mode,"output",step,y0,y1,(size_t)steps[step][0]*VDIM,(size_t)rows_per_slot*VDIM,slots,scratch,stream);
    report(mode,"committed_state",step,m0,m1,MS,NGDN*MS,slots,scratch,stream);
    if (steps[step][2]) report(mode,"pending",step,p0+(size_t)steps[step][3]*pending*PEND_GDN,
      p1+(size_t)steps[step][3]*pending*PEND_GDN,(size_t)8*PEND_GDN,NGDN*ps,slots,scratch,stream);
  }
  // Frozen speculative work: keep stores the initial state; repeats cannot accumulate drift.
  for (int s=0;s<slots;s++) host[s]={s,s*rows_per_slot,8,0,1,0};
  CHECK(cudaMemcpyAsync(segs,host.data(),host.size()*sizeof(Seg),cudaMemcpyHostToDevice,stream));
  // Identical initial bytes for timing; numerical differences never change the workload.
  init_state<<<256,256,0,stream>>>(m0,m1,(size_t)slots*MS,scale); CHECK(cudaGetLastError());
  const float ordinary=time_gdn(a,slots,format,"fp32",iterations,stream);
  const float packed=time_gdn(b,slots,format,mode,iterations,stream);
  printf("{\"kind\":\"timing\",\"mode\":\"%s\",\"streams\":%d,\"rows_per_stream\":8,\"iterations\":%d,\"scale\":%.9g,\"replay_delta\":%s,\"fp32_us\":%.9g,\"packed_us\":%.9g,\"speedup\":%.9g}\n",
    mode,slots,iterations,scale,a.replay_delta?"true":"false",ordinary,packed,ordinary/packed);
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(m0)); CHECK(cudaFree(m1)); CHECK(cudaFree(p0)); CHECK(cudaFree(p1));
  CHECK(cudaFree(co)); CHECK(cudaFree(gates)); CHECK(cudaFree(y0)); CHECK(cudaFree(y1)); CHECK(cudaFree(segs)); CHECK(cudaFree(scratch));
  return 0;
}
int main(int argc,char** argv) {
  if (argc<2 || argc>5 || (strcmp(argv[1],"fp16x2") && strcmp(argv[1],"bf16x2"))) {
    fprintf(stderr,"usage: gdn-packed fp16x2|bf16x2 [streams=64] [iterations=100] [state/value scale=1]\n"); return 3;
  }
  char* end=nullptr;
  const long slots=argc>2?strtol(argv[2],&end,10):64;
  if ((argc>2 && (end==argv[2] || *end)) || slots<1 || slots>64) return 3;
  const long iterations=argc>3?strtol(argv[3],&end,10):100;
  if ((argc>3 && (end==argv[3] || *end)) || iterations<1 || iterations>1000) return 3;
  const float scale=argc>4?strtof(argv[4],&end):1.f;
  if ((argc>4 && (end==argv[4] || *end)) || !std::isfinite(scale) || scale<0) return 3;
  try {
    return !strcmp(argv[1],"fp16x2") ? run<__half>(argv[1],StateFormat::F16,slots,iterations,scale)
      : run<__nv_bfloat16>(argv[1],StateFormat::BF16,slots,iterations,scale);
  } catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 2; }
}
