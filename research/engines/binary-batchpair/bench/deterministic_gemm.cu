#include "deterministic_gemm.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

namespace h5 {
cudaError_t launch_scale_carrier(int, int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, int, float*, int, const float*, cudaStream_t);
cudaError_t launch_scale_carrier_wf0(int, int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, int, float*, int, const float*, cudaStream_t);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
}
static void ck(cudaError_t e, const char* label) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(e)); exit(1); }
}
template<class T> static T* alloc(size_t n) { T* p; ck(cudaMalloc(&p, n * sizeof(T)), "alloc"); return p; }
__global__ void fill_bytes(uint8_t* p, size_t n, unsigned seed, unsigned mod, unsigned base) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    unsigned x = unsigned(i) * 2654435761u ^ seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = base + x % mod;
  }
}
__global__ void fill_float(float* p, size_t n, float value) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) p[i] = value;
}
__global__ void fill_half(__half* p, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) p[i] = __float2half(0.001f + float(i % 13) * 0.0001f);
}
struct Shape { const char* name; int N, K, ci, split, wf; bool carrier; };
int main() {
  cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, 0), "device");
  printf("device=%s cc=%d.%d\n", prop.name, prop.major, prop.minor);
  cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
  const Shape shapes[] = {{"in_gdn",16384,5120,26,4,3,true}, {"in_att",14336,5120,26,6,3,true},
    {"out",5120,6144,26,6,3,true}, {"down",5120,17408,26,8,3,true},
    {"down17",5120,17408,26,17,3,true}, {"draft_down",5120,17408,12,8,1,false},
    {"draft_down_b3",5120,17408,12,8,4,false}, {"draft_gu_b2",34816,5120,12,4,5,false},
    {"tern_wf0",5120,6144,26,6,0,false}, {"carrier_wf0",5120,6144,26,6,0,true}};
  unsigned long long failures = 0;
  for (int R : {1,8,65}) for (const auto& s : shapes) {
    const int ldo = s.N + 32, copies = 8;
    const size_t plane = size_t(R) * s.N, count = size_t(R) * ldo;
    const size_t ab = size_t((R + 63) / 64) * 64 * s.K / 2, sb = ab / 8;
    const size_t cw = s.wf == 1 ? 2048 : s.wf == 3 ? 832 : s.wf == 4 ? 1536 : 1024;
    const size_t wb = size_t(s.N / 32) * (s.K / 128) * cw;
    const size_t ws = size_t(s.N / 32) * (s.K / 128) * (s.wf == 1 ? 256 : s.wf >= 4 ? 128 : 64);
    auto A = alloc<uint8_t>(ab), AS = alloc<uint8_t>(sb), S = alloc<uint8_t>(ws);
    const int padded_rows = (R + 63) / 64 * 64;
    auto AG = alloc<float>(padded_rows), RS = alloc<float>(s.N), out = alloc<float>(count), partials = alloc<float>(plane * s.split);
    std::vector<uint8_t*> W;
    for (int j=0;j<copies;j++) { W.push_back(alloc<uint8_t>(wb)); fill_bytes<<<512,256,0,st>>>(W.back(),wb,99+j,s.wf==3?243:256,0); }
    fill_bytes<<<512,256,0,st>>>(A,ab,1,256,0); fill_bytes<<<512,256,0,st>>>(AS,sb,2,16,0x30);
    if (s.wf == 0) fill_half<<<512,256,0,st>>>((__half*)S, ws/2); else fill_bytes<<<512,256,0,st>>>(S,ws,7,16,0x30);
    fill_float<<<64,256,0,st>>>(AG,padded_rows,0.125f); fill_float<<<512,256,0,st>>>(RS,s.N,0.001f);
    ck(cudaStreamSynchronize(st), "fill");
    auto launch = [&](int route, int split, bool add, int copy) {
      if (route == 2) return h5::launch_gemm_fp4_deterministic(s.ci,split,A,AS,AG,R,s.K,W[copy],S,s.N,out,ldo,add,st,RS,s.wf,s.carrier,partials,plane*s.split);
      if (s.carrier) {
        const auto control = s.wf == 0 ? h5::launch_scale_carrier_wf0 : h5::launch_scale_carrier;
        return control(s.ci,split,split>1?2:add?1:0,A,AS,AG,R,s.K,W[copy],s.N,out,ldo,RS,st);
      }
      h5::launch_gemm_fp4_cfg(s.ci,split,A,AS,AG,R,s.K,W[copy],S,s.N,out,ldo,add,st,RS,s.wf); return cudaGetLastError();
    };
    if (h5::launch_gemm_fp4_deterministic(s.ci,s.split,A,AS,AG,R,s.K,W[0],S,s.N,out,ldo,false,st,RS,s.wf,s.carrier,partials,plane*s.split-1) != cudaErrorInvalidValue) failures++;
    for (bool add : {false,true}) {
      std::vector<float> reference(count), got(count), atomic_ref(count), serial_ref(count);
      for (int route : {0,1,2}) {
        const int split = route == 0 ? 1 : s.split;
        cudaGraph_t graph; cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal),"begin repeat capture");
        fill_float<<<64,256,0,st>>>(out,count,add?0.375f:0.f);
        ck(launch(route,split,add,0),"launch repeat");
        ck(cudaStreamEndCapture(st,&graph),"end repeat capture"); ck(cudaGraphInstantiate(&exec,graph,0),"instantiate repeat");
        unsigned long long changed = 0;
        for (int rep=0;rep<50;rep++) {
          // Replays must overwrite every partial rather than relying on initialization.
          fill_float<<<64,256,0,st>>>(partials,plane*s.split,float(rep+1));
          ck(cudaGraphLaunch(exec,st),"repeat"); ck(cudaMemcpyAsync(got.data(),out,count*4,cudaMemcpyDeviceToHost,st),"copy"); ck(cudaStreamSynchronize(st),"join");
          for (int r=0;r<R;r++) for (int n=0;n<ldo;n++) {
            if (n>=s.N) { if (got[size_t(r)*ldo+n] != (add?0.375f:0.f)) failures++; }
            else if (!std::isfinite(got[size_t(r)*ldo+n])) failures++;
          }
          if (!rep) reference=got; else for (size_t i=0;i<count;i++) if (memcmp(&reference[i],&got[i],4)) changed++;
        }
        if (route==0) serial_ref=reference;
        if (route==1) atomic_ref=reference;
        if (route==2) {
          if (changed) failures++;
          double absmax=0, serialmax=0;
          std::vector<float> host_partials(plane*s.split);
          ck(cudaMemcpy(host_partials.data(),partials,host_partials.size()*4,cudaMemcpyDeviceToHost),"copy partials");
          for(int r=0;r<R;r++) for(int n=0;n<s.N;n++) {
            const size_t p=size_t(r)*s.N+n, i=size_t(r)*ldo+n;
            float folded=host_partials[p];
            for(int z=1;z<s.split;z++) folded=folded+host_partials[size_t(z)*plane+p];
            if(add) folded=0.375f+folded;
            if(memcmp(&folded,&reference[i],4)) failures++;
            if(fabsf(reference[i]-serial_ref[i])>0.0001f*(1.f+fabsf(serial_ref[i]))) failures++;
            absmax=std::max(absmax,double(fabsf(reference[i]-atomic_ref[i]))); serialmax=std::max(serialmax,double(fabsf(reference[i]-serial_ref[i])));
          }
          printf("repeat %s R%d add%d split%d deterministic_changed=%llu atomic_max_abs=%.9g serial_max_abs=%.9g\n",s.name,R,int(add),split,changed,absmax,serialmax);
        } else if (route==1) printf("repeat %s R%d add%d split%d atomic_changed=%llu\n",s.name,R,int(add),split,changed);
        ck(cudaGraphExecDestroy(exec),"destroy exec"); ck(cudaGraphDestroy(graph),"destroy graph");
      }
      for (int route : {0,1,2}) {
        const int split=route==0?1:s.split;
        cudaGraph_t graph; cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal),"begin timing capture");
        for(int j=0;j<copies;j++) {
          fill_float<<<64,256,0,st>>>(out,count,add?0.375f:0.f);
          ck(launch(route,split,add,j),"timing launch");
        }
        ck(cudaStreamEndCapture(st,&graph),"end timing capture"); ck(cudaGraphInstantiate(&exec,graph,0),"instantiate timing");
        for(int j=0;j<8;j++) ck(cudaGraphLaunch(exec,st),"warm");
        cudaEvent_t start,end; ck(cudaEventCreate(&start),"event"); ck(cudaEventCreate(&end),"event");
        ck(cudaEventRecord(start,st),"start");
        const int reps=100; for(int j=0;j<reps;j++) ck(cudaGraphLaunch(exec,st),"time");
        ck(cudaEventRecord(end,st),"end"); ck(cudaEventSynchronize(end),"wait event"); float ms; ck(cudaEventElapsedTime(&ms,start,end),"elapsed");
        printf("time %s R%d add%d route=%s split%d us=%.6f\n",s.name,R,int(add),route==0?"serial":route==1?"atomic":"ordered",split,ms*1000/reps/copies);
        ck(cudaEventDestroy(start),"destroy event"); ck(cudaEventDestroy(end),"destroy event"); ck(cudaGraphExecDestroy(exec),"destroy exec"); ck(cudaGraphDestroy(graph),"destroy graph");
      }
      fflush(stdout);
    }
    for(auto w:W) ck(cudaFree(w),"free W"); for(void* p : { (void*)A,(void*)AS,(void*)S,(void*)AG,(void*)RS,(void*)out,(void*)partials}) ck(cudaFree(p),"free");
  }
  ck(cudaStreamDestroy(st),"destroy stream"); printf("failures=%llu\n",failures); return failures?1:0;
}
