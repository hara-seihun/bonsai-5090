#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
namespace cg=cooperative_groups;
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}
template<int Mode> __device__ __forceinline__ void sync_grid(unsigned*arrival,unsigned target){
 if constexpr(Mode==0){cg::this_grid().sync();}
 else{
  __syncthreads();__threadfence();
  if(threadIdx.x==0){cuda::atomic_ref<unsigned,cuda::thread_scope_device>a(*arrival);a.fetch_add(1,cuda::memory_order_acq_rel);while(a.load(cuda::memory_order_acquire)<target){if constexpr(Mode==2)__nanosleep(64);}}
  __syncthreads();
 }
}
template<int Mode,bool Verify=false> __global__ void barrier_rate(unsigned*arrival,unsigned*payload,unsigned*errors,int loops,int smem_bytes){
 extern __shared__ unsigned shared[];if(smem_bytes&&threadIdx.x==0)shared[0]=blockIdx.x;__syncthreads();
 for(int i=0;i<loops;i++){
  if constexpr(Verify)payload[blockIdx.x*blockDim.x+threadIdx.x]=i+1;
  sync_grid<Mode>(arrival,(i+1)*(Verify?2:1)*gridDim.x-(Verify?gridDim.x:0));
  if constexpr(Verify){unsigned v=((volatile unsigned*)payload)[((blockIdx.x+1)%gridDim.x)*blockDim.x+threadIdx.x];if(v!=(unsigned)i+1)atomicAdd(errors,1u);sync_grid<Mode>(arrival,(i+1)*2*gridDim.x);}
 }
 if(smem_bytes&&threadIdx.x==0)payload[blockIdx.x]=shared[0];
}
__global__ void leaf(unsigned*o){if(threadIdx.x==0)o[blockIdx.x]=blockIdx.x;}
int main(){
 cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));int coop;ck(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,0));if(!coop){fprintf(stderr,"cooperative launch unavailable\n");return 3;}
 cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));unsigned*arrival,*payload,*errors;ck(cudaMalloc(&arrival,4));ck(cudaMalloc(&payload,prop.multiProcessorCount*8*1024*4));ck(cudaMalloc(&errors,4));cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
 const int threads=256,loops=512,smem=16*1024;int occupancy[3];ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy[0],barrier_rate<0>,threads,smem));ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy[1],barrier_rate<1>,threads,smem));ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy[2],barrier_rate<2>,threads,smem));
 printf("{\"SMs\":%d,\"threads\":%d,\"shared_bytes\":%d,\"max_blocks_per_SM\":[%d,%d,%d],\"rows\":[",prop.multiProcessorCount,threads,smem,occupancy[0],occupancy[1],occupancy[2]);bool comma=false;
 for(int blocks:{1,8,32,94,188,376}){
  if(blocks>prop.multiProcessorCount*std::min({occupancy[0],occupancy[1],occupancy[2]}))continue;
  for(int mode=0;mode<3;mode++){
   ck(cudaMemset(arrival,0,4));ck(cudaMemset(errors,0,4));int verify_loops=64;void*args[]={&arrival,&payload,&errors,&verify_loops,(void*)&smem};
   void*kernel=mode==0?(void*)barrier_rate<0,true>:(mode==1?(void*)barrier_rate<1,true>:(void*)barrier_rate<2,true>);
   ck(cudaLaunchCooperativeKernel(kernel,blocks,threads,args,smem));unsigned err;ck(cudaMemcpy(&err,errors,4,cudaMemcpyDeviceToHost));if(err){fprintf(stderr,"barrier visibility failed: mode%d blocks%d errors%u\n",mode,blocks,err);return 4;}
   std::vector<float>samples;
   for(int rep=0;rep<9;rep++){
    ck(cudaMemset(arrival,0,4));void*args2[]={&arrival,&payload,&errors,(void*)&loops,(void*)&smem};
    kernel=mode==0?(void*)barrier_rate<0>:(mode==1?(void*)barrier_rate<1>:(void*)barrier_rate<2>);
    ck(cudaEventRecord(a));ck(cudaLaunchCooperativeKernel(kernel,blocks,threads,args2,smem));ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));if(rep>1)samples.push_back(ms*1000/loops);
   }
   std::sort(samples.begin(),samples.end());printf("%s{\"blocks\":%d,\"mode\":%d,\"barrier_us\":%.6f,\"visibility_errors\":0}",comma?",":"",blocks,mode,samples[3]);comma=true;
  }
  // A graph of leaf kernels has the same serial dependencies but pays fill/drain per node.
  cudaGraph_t graph;cudaGraphExec_t exec;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<loops;i++)leaf<<<blocks,threads,0,stream>>>(payload);ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));std::vector<float>samples;
  for(int rep=0;rep<9;rep++){ck(cudaEventRecord(a,stream));ck(cudaGraphLaunch(exec,stream));ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));if(rep>1)samples.push_back(ms*1000/loops);}std::sort(samples.begin(),samples.end());printf(",{\"blocks\":%d,\"mode\":3,\"graph_leaf_us\":%.6f}",blocks,samples[3]);ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
 }
 printf("],\"mode_names\":[\"cooperative_groups\",\"atomic_acqrel_spin\",\"atomic_acqrel_nanosleep\",\"graph_leaf\"],\"note\":\"Median7 after2 discarded; all barriers visibility-tested across neighboring blocks. Cost is at declared256threads/16KB SMEM, not a universal barrier constant. Timings under other GPU load are provisional.\"}\n");
}
