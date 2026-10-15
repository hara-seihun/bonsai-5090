#pragma once
#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace h5;
#define CHECK(x) do { const cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(2); } } while (0)
constexpr size_t MS = (size_t)HV * 128 * 128;
constexpr int CAP = 16, ROWS = 16;
constexpr size_t PS = (size_t)2 * CAP * PEND_GDN;
template<class T> T* alloc(size_t n) { T* p=nullptr; CHECK(cudaMalloc(&p,n*sizeof(T))); return p; }
template<class T> void upload(T* dst, const std::vector<T>& src, cudaStream_t st) {
  CHECK(cudaMemcpyAsync(dst,src.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice,st));
}
struct Metric { size_t unequal=0, nonfinite=0; double squared=0, ref_squared=0, max_abs=0; };
template<class T> Metric metric(const T* ref, const T* test, size_t n, cudaStream_t st) {
  std::vector<T> a(n),b(n);
  CHECK(cudaMemcpyAsync(a.data(),ref,n*sizeof(T),cudaMemcpyDeviceToHost,st));
  CHECK(cudaMemcpyAsync(b.data(),test,n*sizeof(T),cudaMemcpyDeviceToHost,st));
  CHECK(cudaStreamSynchronize(st)); Metric m;
  for (size_t i=0;i<n;++i) {
    m.unequal += memcmp(&a[i],&b[i],sizeof(T)) != 0;
    const double x=float(a[i]), y=float(b[i]);
    if (!std::isfinite(x) || !std::isfinite(y)) { ++m.nonfinite; continue; }
    const double d=x-y; m.squared+=d*d; m.ref_squared+=x*x; m.max_abs=std::max(m.max_abs,std::abs(d));
  }
  return m;
}
template<class T> bool report(const char* mode,int rows,int accept,bool delta,const char* tensor,
    const T* ref,const T* test,size_t n,cudaStream_t st,bool require_exact=true) {
  const Metric m=metric(ref,test,n,st);
  const bool exact=require_exact && strcmp(mode,"chunk");
  printf("{\"kind\":\"quality\",\"mode\":\"%s\",\"rows\":%d,\"accept\":%d,\"replay_delta\":%s,\"tensor\":\"%s\",\"bit_exact_required\":%s,\"count\":%zu,\"bit_unequal\":%zu,\"nonfinite_pairs\":%zu,\"max_abs\":%.9g,\"rms_abs\":%.9g,\"l2_relative\":%.9g}\n",
    mode,rows,accept,delta?"true":"false",tensor,exact?"true":"false",n,m.unequal,m.nonfinite,m.max_abs,
    sqrt(m.squared/n),m.ref_squared?sqrt(m.squared/m.ref_squared):0.);
  return !m.nonfinite && (!exact || !m.unequal);
}
