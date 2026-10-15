// Scheduled batch token: the batch owner's run_batch_epoch bodies (include/batch.cuh),
// cut into segment kernels by include/tune.hpp and replayed as one CUDA graph.
// Grid is (x per class, active rows); every body is x-invariant by contract.
#include "batch.cuh"
#include "tune.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
extern "C" {
struct HaloTuneBatchApi {
 int version;const char*config_id;
 void*(*create)(const halo6000::DeviceModel*model,halo6000::BatchWorkspace*workspace,const int*grids,int ngrids,char*err,int errlen);
 int(*step)(void*ctx,cudaStream_t stream,char*err,int errlen);
 int(*profile)(void*ctx,cudaStream_t stream,uint64_t*stamps,int capacity,char*err,int errlen);
 int(*describe)(void*ctx,char*buf,int len);
 void(*destroy)(void*ctx);
};
}
namespace halo6000 {
namespace {
static_assert(tune::segment_count>=1&&tune::segment_count<=32,"segment classes");
struct BSegArgs {BatchArgs args;uint64_t*stamps;int first,last,total;};
__device__ __forceinline__ uint64_t bglobaltimer(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
template<int Class>
__global__ void __launch_bounds__(256,tune::segment_min_blocks[Class]) batch_segment_kernel(const __grid_constant__ BSegArgs a){
 constexpr uint32_t K=tune::segment_kinds[Class];
 __shared__ float shared[batch_shared_floats<K>()];
 for(int e=a.first;e<a.last;e++){
  if(a.stamps&&blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0)a.stamps[e]=bglobaltimer();
  run_batch_epoch<K>(a.args,a.args.epochs[e],shared);
  if(e+1<a.last)cooperative_groups::this_grid().sync();
 }
 if(a.stamps&&a.last==a.total&&blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0)a.stamps[a.total]=bglobaltimer();
}
using BKernelFn=void(*)(BSegArgs);
template<int...I>constexpr std::array<BKernelFn,sizeof...(I)>bkernel_table(std::integer_sequence<int,I...>){return{batch_segment_kernel<I>...};}
const auto BKernels=bkernel_table(std::make_integer_sequence<int,tune::segment_count>{});
struct BSegment {int first,last,cls;bool cooperative;};
struct BContext {
 const DeviceModel*model=nullptr;BatchWorkspace*w=nullptr;int rows=0;
 std::vector<Epoch>epochs;uint64_t*stamps=nullptr;
 std::vector<BSegment>segments;std::vector<int>grid,active,uses;std::vector<cudaFuncAttributes>attr;
 cudaStream_t capture=nullptr;cudaGraphExec_t exec=nullptr,profiled=nullptr;
 ~BContext(){if(exec)cudaGraphExecDestroy(exec);if(profiled)cudaGraphExecDestroy(profiled);if(capture)cudaStreamDestroy(capture);if(stamps)cudaFree(stamps);}
};
void bfail(char*err,int len,const std::string&m){if(err&&len>0)snprintf(err,size_t(len),"%s",m.c_str());}
bool bok(cudaError_t e,const char*op,char*err,int len){if(e==cudaSuccess)return true;bfail(err,len,std::string(op)+": "+cudaGetErrorString(e));return false;}
int popc(uint32_t x){return __builtin_popcount(x);}
bool bbuild(BContext&c,bool profile,cudaGraphExec_t&out,char*err,int len){
 if(!bok(cudaStreamBeginCapture(c.capture,cudaStreamCaptureModeThreadLocal),"begin capture",err,len))return false;
 bool good=bok(cudaMemsetAsync(c.w->view.error,0,4,c.capture),"clear batch error",err,len);
 for(const auto&s:c.segments){
  if(!good)break;
  BSegArgs a{{c.model->device_weights,c.model->state,c.w->view,c.w->device_epochs,c.w->epoch_count()},profile?c.stamps:nullptr,s.first,s.last,int(c.epochs.size())};
  cudaLaunchAttribute attrs[1];int na=0;
  if(s.cooperative){attrs[na].id=cudaLaunchAttributeCooperative;attrs[na].val.cooperative=1;na++;}
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(c.grid[s.cls],c.rows);cfg.blockDim=dim3(256);cfg.stream=c.capture;cfg.attrs=attrs;cfg.numAttrs=na;
  good=bok(cudaLaunchKernelEx(&cfg,BKernels[s.cls],a),"segment launch",err,len);
 }
 cudaGraph_t g=nullptr;const cudaError_t end=cudaStreamEndCapture(c.capture,&g);
 if(!good){if(g)cudaGraphDestroy(g);return false;}
 if(!bok(end,"end capture",err,len))return false;
 const cudaError_t inst=cudaGraphInstantiate(&out,g,0);cudaGraphDestroy(g);return bok(inst,"graph instantiate",err,len);
}
void*bcreate(const DeviceModel*model,BatchWorkspace*w,const int*grids,int ngrids,char*err,int len){
 auto c=std::make_unique<BContext>();c->model=model;c->w=w;c->rows=w->view.rows;
 if(c->rows<1){bfail(err,len,"configure batch rows before create");return nullptr;}
 c->epochs.resize(w->epoch_count());
 if(!bok(cudaMemcpy(c->epochs.data(),w->device_epochs,c->epochs.size()*sizeof(Epoch),cudaMemcpyDeviceToHost),"read batch epochs",err,len))return nullptr;
 for(int e=0;e<int(c->epochs.size());e++){const int k=c->epochs[e].kind;if(e==0||(tune::boundary_kinds>>k&1u))c->segments.push_back({e,e+1,-1,false});else c->segments.back().last=e+1;}
 c->uses.assign(tune::segment_count,0);
 for(auto&s:c->segments){
  uint32_t mask=0;for(int e=s.first;e<s.last;e++)mask|=1u<<c->epochs[e].kind;
  int best=-1;for(int i=0;i<tune::segment_count;i++)if((tune::segment_kinds[i]&mask)==mask&&(best<0||popc(tune::segment_kinds[i])<popc(tune::segment_kinds[best])))best=i;
  if(best<0){char m[96];snprintf(m,sizeof m,"no segment class covers kinds 0x%x",mask);bfail(err,len,m);return nullptr;}
  s.cls=best;s.cooperative=s.last-s.first>1;c->uses[best]++;
 }
 // ArgPart/ArgFinish partitions are gridDim.x-dependent: they must share one launch.
 for(size_t s=0;s<c->segments.size();s++){const auto&g=c->segments[s];if(c->epochs[g.first].kind==int(EpochKind::ArgFinish)){bfail(err,len,"arg_finish must share a segment with arg_part");return nullptr;}}
 cudaDeviceProp prop{};if(!bok(cudaGetDeviceProperties(&prop,0),"device",err,len))return nullptr;
 c->grid.assign(tune::segment_count,0);c->active.assign(tune::segment_count,0);c->attr.resize(tune::segment_count);
 for(int i=0;i<tune::segment_count;i++){
  if(!bok(cudaFuncGetAttributes(&c->attr[i],(const void*)BKernels[i]),"attributes",err,len))return nullptr;
  if(!bok(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&c->active[i],BKernels[i],256,0),"occupancy",err,len))return nullptr;
  // grids[i]>0 exact x; <0 |x| clamped to cooperative residency; 0 header default or SMs*min_blocks/rows.
  const int req=i<ngrids&&grids&&grids[i]?grids[i]:tune::segment_grids[i];
  const int cap=std::max(1,c->active[i]*prop.multiProcessorCount/c->rows);
  int want=req>0?req:req<0?-req:std::max(1,std::min(tune::segment_min_blocks[i],c->active[i])*prop.multiProcessorCount/c->rows);
  bool coop=false;for(const auto&s:c->segments)coop|=s.cls==i&&s.cooperative;
  if(coop&&(req<=0))want=std::min(want,cap);
  if(coop&&want>cap){char m[128];snprintf(m,sizeof m,"class %d x %d exceeds cooperative residency %d",i,want,cap);bfail(err,len,m);return nullptr;}
  c->grid[i]=want;
 }
 if(!bok(cudaMalloc(&c->stamps,(c->epochs.size()+1)*sizeof(uint64_t)),"stamps",err,len))return nullptr;
 if(!bok(cudaStreamCreateWithFlags(&c->capture,cudaStreamNonBlocking),"capture stream",err,len))return nullptr;
 if(!bbuild(*c,false,c->exec,err,len))return nullptr;
 return c.release();
}
int bstep(void*ctx,cudaStream_t s,char*err,int len){auto*c=static_cast<BContext*>(ctx);return bok(cudaGraphLaunch(c->exec,s),"graph launch",err,len)?0:1;}
int bprofile(void*ctx,cudaStream_t s,uint64_t*out,int capacity,char*err,int len){
 auto*c=static_cast<BContext*>(ctx);const int n=int(c->epochs.size())+1;if(capacity<n){bfail(err,len,"stamp capacity");return 1;}
 if(!c->profiled&&!bbuild(*c,true,c->profiled,err,len))return 1;
 if(!bok(cudaGraphLaunch(c->profiled,s),"profile launch",err,len)||!bok(cudaStreamSynchronize(s),"profile sync",err,len))return 1;
 return bok(cudaMemcpy(out,c->stamps,n*sizeof(uint64_t),cudaMemcpyDeviceToHost),"stamps read",err,len)?0:1;
}
int bdescribe(void*ctx,char*buf,int len){
 auto*c=static_cast<BContext*>(ctx);std::string j="{\"config\":\"";j+=tune::config_id;j+="\",\"rows\":"+std::to_string(c->rows)+",\"epochs\":"+std::to_string(c->epochs.size())+",\"segments\":"+std::to_string(c->segments.size())+",\"classes\":[";
 for(int i=0;i<tune::segment_count;i++){char b[256];snprintf(b,sizeof b,"%s{\"kinds\":%u,\"min_blocks\":%d,\"registers\":%d,\"local_bytes\":%zu,\"static_smem\":%zu,\"active_per_sm\":%d,\"grid\":%d,\"segments\":%d}",i?",":"",tune::segment_kinds[i],tune::segment_min_blocks[i],c->attr[i].numRegs,c->attr[i].localSizeBytes,c->attr[i].sharedSizeBytes,c->active[i],c->grid[i],c->uses[i]);j+=b;}
 j+="],\"epoch_kinds\":[";for(size_t e=0;e<c->epochs.size();e++)j+=(e?",":"")+std::to_string(c->epochs[e].kind);
 j+="],\"segment_first\":[";for(size_t s=0;s<c->segments.size();s++)j+=(s?",":"")+std::to_string(c->segments[s].first);
 j+="]}";if(int(j.size())+1>len)return 1;memcpy(buf,j.c_str(),j.size()+1);return 0;
}
void bdestroy(void*ctx){delete static_cast<BContext*>(ctx);}
const HaloTuneBatchApi BApi{1,tune::config_id,bcreate,bstep,bprofile,bdescribe,bdestroy};
} // namespace
} // namespace halo6000
extern "C" __attribute__((visibility("default"))) const HaloTuneBatchApi*halo_tune_batch_api(){return &halo6000::BApi;}
