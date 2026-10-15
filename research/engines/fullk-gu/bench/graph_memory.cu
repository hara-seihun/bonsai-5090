#include "graph_memory.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while (0)
__global__ void pdl_identity() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

struct Edge { cudaGraphNode_t from, to; cudaGraphEdgeData data; };
static std::vector<Edge> pdl_edges(cudaGraph_t graph) {
  size_t n = 0; CHECK(cudaGraphGetEdges_v2(graph, nullptr, nullptr, nullptr, &n));
  std::vector<cudaGraphNode_t> f(n), t(n); std::vector<cudaGraphEdgeData> d(n);
  CHECK(cudaGraphGetEdges_v2(graph, f.data(), t.data(), d.data(), &n));
  std::vector<Edge> edges;
  for (size_t i = 0; i < n; i++) if (d[i].type == cudaGraphDependencyTypeProgrammatic) edges.push_back({f[i],t[i],d[i]});
  return edges;
}
int main() { try {
  unsigned char *host_in, *host_out, *src, *dst;
  cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  CHECK(cudaMallocHost(&host_in,96)); CHECK(cudaMallocHost(&host_out,144));
  CHECK(cudaMalloc(&src,96)); CHECK(cudaMalloc(&dst,144));
  cudaEvent_t start,end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  CHECK(cudaEventRecordWithFlags(start,stream,cudaEventRecordExternal));
  CHECK(cudaMemcpyAsync(src,host_in,96,cudaMemcpyHostToDevice,stream));
  cudaLaunchAttribute attr{}; attr.id=cudaLaunchAttributeProgrammaticStreamSerialization; attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t cfg{}; cfg.gridDim=dim3(1); cfg.blockDim=dim3(32); cfg.stream=stream; cfg.attrs=&attr; cfg.numAttrs=1;
  CHECK(cudaLaunchKernelEx(&cfg,pdl_identity)); CHECK(cudaLaunchKernelEx(&cfg,pdl_identity));
  CHECK(cudaStreamWaitEvent(stream,start,cudaEventWaitExternal));
  CHECK(cudaMemsetAsync(dst,0,144,stream));
  CHECK(cudaMemcpy2DAsync(dst,48,src,32,17,3,cudaMemcpyDeviceToDevice,stream));
  CHECK(cudaMemcpyAsync(host_out,dst,144,cudaMemcpyDeviceToHost,stream));
  CHECK(cudaEventRecordWithFlags(end,stream,cudaEventRecordExternal));
  cudaGraph_t native, kernel; CHECK(cudaStreamEndCapture(stream,&native)); CHECK(cudaGraphClone(&kernel,native));
  auto before=pdl_edges(kernel); if(before.empty())throw std::runtime_error("fixture has no programmatic edge");
  auto result=h5::graph_memory_rewrite(kernel);
  if(result.error!=h5::GraphMemoryError::None)throw std::runtime_error("rewrite failed: "+std::to_string((int)result.error));
  auto after=pdl_edges(kernel);
  if(before.size()!=after.size())throw std::runtime_error("PDL edge count changed");
  for(const auto& old:before){bool found=false;for(const auto& edge:after)if(old.from==edge.from && old.to==edge.to && !memcmp(&old.data,&edge.data,sizeof(cudaGraphEdgeData)))found=true;if(!found)throw std::runtime_error("PDL edge changed");}
  cudaGraphExec_t a,b; CHECK(cudaGraphInstantiate(&a,native,nullptr,nullptr,0)); CHECK(cudaGraphInstantiate(&b,kernel,nullptr,nullptr,0));
  for(int pass=0;pass<3;pass++){
    for(int i=0;i<96;i++)host_in[i]=(i*13+pass*7)&255;
    std::array<unsigned char,144> expected{};
    for(int r=0;r<3;r++)memcpy(expected.data()+r*48,host_in+r*32,17);
    CHECK(cudaGraphLaunch(a,stream));CHECK(cudaStreamSynchronize(stream));
    if(memcmp(host_out,expected.data(),144))throw std::runtime_error("native fixture failed");
    memset(host_out,0xee,144);
    CHECK(cudaGraphLaunch(b,stream));CHECK(cudaStreamSynchronize(stream));
    if(memcmp(host_out,expected.data(),144))throw std::runtime_error("mapped copy/clear mismatch");
    float ms; CHECK(cudaEventElapsedTime(&ms,start,end));
    if(!(ms>=0.f))throw std::runtime_error("timing event records were not preserved");
  }
  cudaGraph_t unsupported; CHECK(cudaGraphCreate(&unsupported,0));
  cudaMemsetParams bad{}; bad.dst=dst;bad.width=32;bad.height=1;bad.elementSize=1;bad.value=7;
  cudaGraphNode_t node; CHECK(cudaGraphAddMemsetNode(&node,unsupported,nullptr,0,&bad));
  auto rejected=h5::graph_memory_rewrite(unsupported);size_t remaining=0;CHECK(cudaGraphGetNodes(unsupported,nullptr,&remaining));
  if(rejected.error!=h5::GraphMemoryError::UnsupportedLayout || remaining!=1)throw std::runtime_error("unsupported fill was not rejected before mutation");
  CHECK(cudaGraphDestroy(unsupported));CHECK(cudaGraphExecDestroy(a));CHECK(cudaGraphExecDestroy(b));CHECK(cudaGraphDestroy(native));CHECK(cudaGraphDestroy(kernel));
  CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));
  CHECK(cudaFree(src));CHECK(cudaFree(dst));CHECK(cudaFreeHost(host_in));CHECK(cudaFreeHost(host_out));CHECK(cudaStreamDestroy(stream));
  printf("{\"pass\":true,\"pdl_edges_preserved\":%zu,\"copies\":%llu,\"clears\":%llu,\"copy_bytes\":%llu,\"clear_bytes\":%llu,\"host_read_bytes\":%llu,\"host_write_bytes\":%llu}\n",before.size(),(unsigned long long)result.copies,(unsigned long long)result.clears,(unsigned long long)result.copy_bytes,(unsigned long long)result.clear_bytes,(unsigned long long)result.host_read_bytes,(unsigned long long)result.host_write_bytes);
  return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
