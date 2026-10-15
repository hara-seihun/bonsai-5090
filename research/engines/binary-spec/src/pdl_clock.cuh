#pragma once
// Diagnostic only: sample dependency wait at block zero, one lane per warp.
// Timer samples perturb register use and execution. They are not baseline timings.
#ifdef H5_PDLCLOCK
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime_api.h>
namespace h5 {
struct PdlSample { unsigned long long start, end; unsigned site, warp, gx, gy, gz, threads; };
#ifdef H5_PDLCLOCK_MODULE
static __device__ unsigned pdl_sample_on=0, pdl_sample_count=0;
static __device__ PdlSample pdl_samples[524288];
__device__ __forceinline__ void pdl_wait_impl(unsigned site) {
  unsigned tid=threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
  bool sample=pdl_sample_on && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && (tid&31)==0;
  unsigned long long begin=0,end=0;
  if(sample) asm volatile("mov.u64 %0, %%globaltimer;":"=l"(begin));
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if(sample) {
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(end));
    unsigned i=atomicAdd(&pdl_sample_count,1U);
    if(i<524288) pdl_samples[i]={begin,end,site,tid/32,gridDim.x,gridDim.y,gridDim.z,blockDim.x*blockDim.y*blockDim.z};
  }
}
static void pdl_copy_check(cudaError_t status) {
  if(status!=cudaSuccess) { fprintf(stderr,"PDL trace CUDA error: %s\n",cudaGetErrorString(status)); abort(); }
}
#define H5_JOIN0(a,b) a##b
#define H5_JOIN(a,b) H5_JOIN0(a,b)
extern "C" void H5_JOIN(h5_pdl_begin_,H5_PDLCLOCK_MODULE)() {
  unsigned z=0,o=1; pdl_copy_check(cudaMemcpyToSymbol(pdl_sample_count,&z,sizeof(z))); pdl_copy_check(cudaMemcpyToSymbol(pdl_sample_on,&o,sizeof(o)));
}
extern "C" void H5_JOIN(h5_pdl_end_,H5_PDLCLOCK_MODULE)(const char* path) {
  unsigned z=0,n=0; pdl_copy_check(cudaMemcpyToSymbol(pdl_sample_on,&z,sizeof(z))); pdl_copy_check(cudaMemcpyFromSymbol(&n,pdl_sample_count,sizeof(n)));
  FILE*f=fopen(path,"w"); if(!f){perror(path);abort();}
  fprintf(f,"start_ns,end_ns,site,warp,grid_x,grid_y,grid_z,threads\n");
  if(n>524288){fprintf(stderr,"PDL trace overflow: %u\n",n);fclose(f);abort();}
  std::vector<PdlSample> v(n); if(n) pdl_copy_check(cudaMemcpyFromSymbol(v.data(),pdl_samples,n*sizeof(PdlSample)));
  for(auto&r:v) fprintf(f,"%llu,%llu,%u,%u,%u,%u,%u,%u\n",r.start,r.end,r.site,r.warp,r.gx,r.gy,r.gz,r.threads);
  bool failed=ferror(f); int close_rc=fclose(f);
  if(failed || close_rc) { fprintf(stderr,"PDL trace write failed: %s\n",path); abort(); }
}
#endif
}
#endif
