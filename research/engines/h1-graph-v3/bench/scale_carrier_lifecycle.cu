// Cache survives retirement of original scales/stats; residual and atomic ownership remain explicit.
#include "scale_carrier.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>static T* upload(const std::vector<T>&v){T*p;check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(){
 if(!getenv("H5_PDL")||(strcmp(getenv("H5_PDL"),"0")&&strcmp(getenv("H5_PDL"),"1")))return 2;
 constexpr int N=256,K=2048,R=2,CI=19,B=K/128;
 if(!std::holds_alternative<cudaError_t>(h5::ScaleCarrierView::make(nullptr,N,K)) ||
    !std::holds_alternative<cudaError_t>(h5::ScaleCarrierView::make((const float*)1,N,0))){fprintf(stderr,"invalid view admitted\n");return 1;}
 cudaStream_t st;check(cudaStreamCreate(&st));
 for(int varying=0;varying<2;varying++){
  std::vector<__half>s(size_t(N)*B);for(int b=0;b<B;b++)for(int j=0;j<N;j++)s[size_t(b)*N+j]=__float2half(.03125f*(varying?.5f+float((b+j)%13)/12.f:1.f));
  auto S=upload(s);auto mean=upload(std::vector<float>(N));auto mean2=upload(std::vector<float>(N));auto stats=upload(std::vector<float4>(N));
  check(h5::scale_carrier_prepare(S,mean,stats,N,K,st));check(h5::scale_carrier_prepare(S,mean2,nullptr,N,K,st));check(cudaStreamSynchronize(st));
  std::vector<float>m(N),m2(N);check(cudaMemcpy(m.data(),mean,N*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(m2.data(),mean2,N*4,cudaMemcpyDeviceToHost));
  if(memcmp(m.data(),m2.data(),N*4)){fprintf(stderr,"optional stats changed mean\n");return 1;}for(float v:m)if(!std::isfinite(v)){fprintf(stderr,"nonfinite cache\n");return 1;}
  check(cudaFree(S));check(cudaFree(stats));check(cudaFree(mean2));
  auto created=h5::ScaleCarrierView::make(mean,N,K);if(std::holds_alternative<cudaError_t>(created))return 1;const auto&view=std::get<h5::ScaleCarrierView>(created);
  uint32_t rng=17;std::vector<uint8_t>w(size_t(N/32)*B*832),a(size_t(64)*K/2);for(auto&v:w){rng=rng*1664525u+1013904223u;v=(rng>>16)%243;}for(auto&v:a){rng=rng*1664525u+1013904223u;v=rng>>16;}
  auto W=upload(w);auto A=upload(a);auto AS=upload(std::vector<uint8_t>(size_t(64)*K/16,0x38));auto AG=upload(std::vector<float>(64,1.f));auto Y=upload(std::vector<float>(R*N));
  auto run=[&](int split,int mode){check(h5::launch_scale_carrier(CI,split,mode,A,AS,AG,R,W,Y,N,view,st));check(cudaStreamSynchronize(st));std::vector<float>y(R*N);check(cudaMemcpy(y.data(),Y,y.size()*4,cudaMemcpyDeviceToHost));return y;};
  auto store=run(1,0);std::vector<float>residual(R*N);for(size_t j=0;j<residual.size();j++)residual[j]=(int(j%17)-8)*.125f;check(cudaMemcpy(Y,residual.data(),residual.size()*4,cudaMemcpyHostToDevice));auto add=run(1,1);
  double add_error=0,atomic_error=0;for(size_t j=0;j<add.size();j++)add_error=std::max(add_error,double(fabsf(add[j]-(store[j]+residual[j]))));
  // Caller initializes exactly once, then two explicit mode2 launches must double the store.
  check(cudaMemsetAsync(Y,0,R*N*4,st));run(4,2);auto twice=run(4,2);for(size_t j=0;j<twice.size();j++)atomic_error=std::max(atomic_error,double(fabsf(twice[j]-2*store[j])));
  if(h5::launch_scale_carrier(CI,4,1,A,AS,AG,R,W,Y,N,view,st)!=cudaErrorInvalidValue || h5::launch_scale_carrier(CI,4,3,A,AS,AG,R,W,Y,N,view,st)!=cudaErrorInvalidValue){fprintf(stderr,"unsupported mode admitted\n");return 1;}
  if(add_error!=0 || atomic_error>1e-4){fprintf(stderr,"ownership mismatch add=%g atomic=%g\n",add_error,atomic_error);return 1;}
  printf("{\"varying\":%d,\"null_stats_mean_byte_equal\":true,\"retired_scales_and_stats\":true,\"residual_max_abs\":%.9g,\"atomic_twice_max_abs\":%.9g}\n",varying,add_error,atomic_error);
  check(cudaFree(W));check(cudaFree(A));check(cudaFree(AS));check(cudaFree(AG));check(cudaFree(Y));check(cudaFree(mean));
 }
 check(cudaStreamDestroy(st));
}
