#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace h5 {
int fp4_ncfg();
bool fp4_cfg_wg2(int);
const char* fp4_cfg_name(int, int);
cudaError_t fp4_profile_resources(int, int, int, cudaFuncAttributes*, int*);
void launch_gemm_fp4_silu(int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, cudaStream_t, const float*, int);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
}
static void check(cudaError_t e) {
  if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(1); }
}
template<class T> static T* upload(const std::vector<T>& v) {
  T* p; check(cudaMalloc(&p, v.size() * sizeof(T)));
  check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
int main(int argc, char** argv) {
  if (argc != 8) {
    fprintf(stderr, "usage: gemm-residency R N K fused(0|1) baseline_cfg candidate_cfg splits\n"); return 2;
  }
  int x[7];
  for (int i=0; i<7; ++i) {
    char* end; long v = strtol(argv[i+1], &end, 10);
    if (!*argv[i+1] || *end || v<0 || v>1000000) return 2;
    x[i] = int(v);
  }
  const int R=x[0], N=x[1], K=x[2], fused=x[3], base=x[4], candidate=x[5], splits=x[6];
  if ((R!=1 && R!=8) || N==0 || N%32 || K==0 || K%128 || fused>1 || base>=h5::fp4_ncfg() || candidate>=h5::fp4_ncfg() || splits<1 || splits>K/128 || (fused && (splits!=1 || N%64 || !h5::fp4_cfg_wg2(base) || !h5::fp4_cfg_wg2(candidate)))) return 2;
  const int ldo=fused?N/2:N, nkb=K/128;
  std::vector<uint8_t> weights(size_t(N/32)*nkb*832);
  // All 26 labels, including the tail's unused two trits, are valid base-three words.
  uint32_t rng=17;
  for (auto& v: weights) { rng = rng*1664525u+1013904223u; v=(rng>>16)%243; }
  std::vector<__half> scales(size_t(N/32)*nkb*32);
  for (size_t i=0; i<scales.size(); ++i) scales[i]=__float2half(0.125f+float(i%11)/32.f);
  std::vector<uint8_t> acts(size_t(64)*nkb*64), as(size_t(64)*nkb*8, 0x38);
  for (auto& v: acts) { rng = rng*1664525u+1013904223u; v=uint8_t(rng>>16); }
  std::vector<float> ag(64, 1.f), zero(size_t(R)*ldo, 0.f);
  auto W=upload(weights); auto S=upload(scales); auto A=upload(acts); auto AS=upload(as); auto AG=upload(ag); auto out=upload(zero);
  cudaStream_t st; check(cudaStreamCreate(&st));
  auto launch=[&](int cfg) {
    if (fused) h5::launch_gemm_fp4_silu(cfg,A,AS,AG,R,K,W,S,N,out,ldo,st,nullptr,3);
    else h5::launch_gemm_fp4_cfg(cfg,splits,A,AS,AG,R,K,W,S,N,out,ldo,false,st,nullptr,3);
    check(cudaGetLastError());
  };
  auto run=[&](int cfg) {
    for(int i=0;i<3;i++) launch(cfg);
    check(cudaStreamSynchronize(st));
    std::vector<float> result(zero.size());
    check(cudaMemcpy(result.data(),out,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
    cudaFuncAttributes attrs; int blocks;
    check(h5::fp4_profile_resources(cfg, fused?3:splits>1?2:0, 3, &attrs, &blocks));
    cudaGraph_t graph; cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<32;i++) launch(cfg);
    check(cudaStreamEndCapture(st,&graph)); check(cudaGraphInstantiate(&exec,graph,0));
    cudaEvent_t begin,end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    check(cudaGraphLaunch(exec,st)); check(cudaStreamSynchronize(st));
    check(cudaEventRecord(begin,st));
    for(int i=0;i<10;i++) check(cudaGraphLaunch(exec,st));
    check(cudaEventRecord(end,st)); check(cudaEventSynchronize(end));
    float ms; check(cudaEventElapsedTime(&ms,begin,end));
    printf("cfg=%d %s us=%.6f registers=%d static_shared=%zu local_bytes=%zu max_dynamic_shared=%d resident_blocks=%d\n",cfg,h5::fp4_cfg_name(cfg,splits),ms*1000/320,attrs.numRegs,attrs.sharedSizeBytes,attrs.localSizeBytes,attrs.maxDynamicSharedSizeBytes,blocks);
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end)); check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
    return result;
  };
  auto reference=run(base); auto value=run(candidate);
  double abs=0, rel=0; size_t differing=0, nonfinite=0;
  for(size_t i=0;i<value.size();i++) {
    if(!std::isfinite(value[i]) || !std::isfinite(reference[i])) { nonfinite++; continue; }
    double delta=std::abs(double(value[i])-reference[i]);
    abs=std::max(abs,delta); rel=std::max(rel,delta/std::max(1e-5,std::abs(double(reference[i])))); differing+=value[i]!=reference[i];
  }
  printf("compared=%zu differing=%zu nonfinite=%zu max_abs=%.9g max_relative=%.9g\n",value.size(),differing,nonfinite,abs,rel);
  check(cudaStreamDestroy(st)); check(cudaFree(W)); check(cudaFree(S)); check(cudaFree(A)); check(cudaFree(AS)); check(cudaFree(AG)); check(cudaFree(out));
  return nonfinite ? 1 : 0;
}
