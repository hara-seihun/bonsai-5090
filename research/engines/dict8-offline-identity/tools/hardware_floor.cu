// Empirical streaming/read, copy and launch floors; no model code or tensor-core library.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#define CK(x) do { auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }while(0)
__global__ void read_stream(const uint4*a,size_t n,unsigned long long*out){
 unsigned long long x=0;for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){uint4 v=a[i];x+=(unsigned long long)v.x+v.y+v.z+v.w;}
 out[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=x;
}
__global__ void empty(){}
int main(){try{cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));printf("{\"gpu\":\"%s\",\"sms\":%d,\"samples\":[\n",prop.name,prop.multiProcessorCount);
 uint4*a,*b;unsigned long long*out;size_t cap=8ULL<<30;CK(cudaMalloc(&a,cap));CK(cudaMalloc(&b,cap));CK(cudaMalloc(&out,prop.multiProcessorCount*16*256*sizeof(*out)));CK(cudaMemset(a,7,cap));CK(cudaMemset(b,5,cap));
 cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));bool comma=false;
 for(size_t size:{64ULL<<20,256ULL<<20,1ULL<<30,4ULL<<30,8ULL<<30})for(int mode=0;mode<2;mode++){
  for(int w=0;w<4;w++){if(mode==0)read_stream<<<prop.multiProcessorCount*16,256>>>(a,size/16,out);else CK(cudaMemcpyAsync(b,a,size,cudaMemcpyDeviceToDevice));}CK(cudaDeviceSynchronize());
  for(int rep=0;rep<5;rep++){CK(cudaEventRecord(start));for(int i=0;i<10;i++){if(mode==0)read_stream<<<prop.multiProcessorCount*16,256>>>(a,size/16,out);else CK(cudaMemcpyAsync(b,a,size,cudaMemcpyDeviceToDevice));}CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));double s=ms/1000./10;if(comma)printf(",\n");comma=true;printf("{\"mode\":\"%s\",\"bytes\":%zu,\"rep\":%d,\"seconds\":%.9f,\"GBps\":%.3f}",mode==0?"read":"copy",size,rep,s,(double)size*(mode==0?1:2)/s/1e9);}
 }
 CK(cudaEventRecord(start));for(int i=0;i<10000;i++)empty<<<1,32>>>();CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));double eager=ms*1000/10000;cudaStream_t stream;CK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<1000;i++)empty<<<1,32,0,stream>>>();CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));CK(cudaEventRecord(start,stream));for(int i=0;i<100;i++)CK(cudaGraphLaunch(exec,stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));CK(cudaEventElapsedTime(&ms,start,end));printf("\n],\"eager_empty_us\":%.5f,\"graph_empty_us\":%.5f}\n",eager,ms*1000/100000);CK(cudaGetLastError());return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
