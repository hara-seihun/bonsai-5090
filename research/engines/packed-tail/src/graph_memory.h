#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace h5 {
inline cudaError_t graph_edges(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to,
                               cudaGraphEdgeData* data, size_t* count) {
#if CUDART_VERSION >= 13000
  return cudaGraphGetEdges(graph, from, to, data, count);
#else
  return cudaGraphGetEdges_v2(graph, from, to, data, count);
#endif
}
inline cudaError_t graph_dependencies(cudaGraph_t graph, const cudaGraphNode_t* from,
                                      const cudaGraphNode_t* to, const cudaGraphEdgeData* data, size_t count) {
#if CUDART_VERSION >= 13000
  return cudaGraphAddDependencies(graph, from, to, data, count);
#else
  return cudaGraphAddDependencies_v2(graph, from, to, data, count);
#endif
}
enum class GraphMemoryError : int {
  None = 0, Cuda = -100, UnsupportedLayout = -101, UnmappedHost = -102,
  UnsupportedEdge = -103, UnsupportedNode = -104, SizeOverflow = -105,
  Allocation = -106
};
struct GraphMemoryResult {
  GraphMemoryError error;
  cudaError_t cuda_error;
  uint64_t copies, clears, copy_bytes, clear_bytes, host_read_bytes, host_write_bytes;
};
GraphMemoryResult graph_memory_rewrite(cudaGraph_t);
}
