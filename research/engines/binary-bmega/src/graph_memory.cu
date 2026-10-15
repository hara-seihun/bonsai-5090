#define H5_LIT_MODULE gmem
#define H5_LIT_MODULE_ID 4u
#include "smlit.cuh"
#include "graph_memory.h"
#include <vector>
#include <map>
#include <limits>
#include <new>
#include <cstring>

namespace h5 {
struct MemoryOp {
  unsigned char* dst;
  const unsigned char* src;
  size_t width, height, dst_pitch, src_pitch;
  int packet;
  bool host_dst;
};

__global__ void graph_copy(MemoryOp op) { H5_LIT(0);
  const size_t packets = op.width / op.packet;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= packets * op.height) return;
  const size_t row = i / packets, col = (i % packets) * op.packet;
  auto* dst = op.dst + row * op.dst_pitch + col;
  const auto* src = op.src + row * op.src_pitch + col;
  if (op.packet == 16) *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
  else if (op.packet == 4) *reinterpret_cast<unsigned*>(dst) = *reinterpret_cast<const unsigned*>(src);
  else *dst = *src;
  // Stream completion owns host visibility; the mapped-output producer also publishes explicitly.
  if (op.host_dst) __threadfence_system();
}

__global__ void graph_clear(MemoryOp op) { H5_LIT(0);
  const size_t packets = op.width / op.packet;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= packets * op.height) return;
  const size_t row = i / packets, col = (i % packets) * op.packet;
  auto* dst = op.dst + row * op.dst_pitch + col;
  if (op.packet == 16) *reinterpret_cast<uint4*>(dst) = uint4{0, 0, 0, 0};
  else if (op.packet == 4) *reinterpret_cast<unsigned*>(dst) = 0;
  else *dst = 0;
}

struct Planned { cudaGraphNode_t old; MemoryOp op; bool clear; };

static GraphMemoryError pointer(void* p, void** device, bool* host, cudaError_t* cuda_error) {
  cudaPointerAttributes attr{};
  *cuda_error = cudaPointerGetAttributes(&attr, p);
  if (*cuda_error != cudaSuccess) return GraphMemoryError::Cuda;
  if (attr.type == cudaMemoryTypeDevice) {
    int current;
    *cuda_error = cudaGetDevice(&current);
    if (*cuda_error != cudaSuccess) return GraphMemoryError::Cuda;
    if (attr.device != current) return GraphMemoryError::UnsupportedLayout;
    *device = p; *host = false; return GraphMemoryError::None;
  }
  if (attr.type != cudaMemoryTypeHost) return GraphMemoryError::UnsupportedLayout;
  *host = true;
  *cuda_error = cudaHostGetDevicePointer(device, p, 0);
  if (*cuda_error != cudaSuccess || !*device) return GraphMemoryError::UnmappedHost;
  return GraphMemoryError::None;
}

static bool overflow(size_t a, size_t b) {
  return b && a > std::numeric_limits<size_t>::max() / b;
}

GraphMemoryResult graph_memory_rewrite(cudaGraph_t graph) {
  GraphMemoryResult result{GraphMemoryError::None, cudaSuccess, 0, 0, 0, 0, 0, 0};
#define GM_CUDA(call) do { result.cuda_error = (call); if (result.cuda_error != cudaSuccess) { result.error = GraphMemoryError::Cuda; return result; } } while (0)
#define GM_ERROR(code) do { result.error = (code); return result; } while (0)
  try {
    size_t n = 0, ne = 0;
    GM_CUDA(cudaGraphGetNodes(graph, nullptr, &n));
    GM_CUDA(graph_edges(graph, nullptr, nullptr, nullptr, &ne));
    std::vector<cudaGraphNode_t> nodes(n), from(ne), to(ne);
    std::vector<cudaGraphEdgeData> edges(ne);
    GM_CUDA(cudaGraphGetNodes(graph, nodes.data(), &n));
    GM_CUDA(graph_edges(graph, from.data(), to.data(), edges.data(), &ne));
    std::vector<Planned> plans;
    std::map<cudaGraphNode_t, cudaGraphNode_t> replaced;
    for (auto node : nodes) {
      cudaGraphNodeType type;
      GM_CUDA(cudaGraphNodeGetType(node, &type));
      if (type == cudaGraphNodeTypeKernel) continue;
      if (type != cudaGraphNodeTypeMemcpy && type != cudaGraphNodeTypeMemset) GM_ERROR(GraphMemoryError::UnsupportedNode);
      MemoryOp op{};
      bool clear = type == cudaGraphNodeTypeMemset;
      if (clear) {
        cudaMemsetParams p{};
        GM_CUDA(cudaGraphMemsetNodeGetParams(node, &p));
        if (p.value != 0 || (p.elementSize != 1 && p.elementSize != 2 && p.elementSize != 4)) GM_ERROR(GraphMemoryError::UnsupportedLayout);
        if (overflow(p.width, p.elementSize)) GM_ERROR(GraphMemoryError::SizeOverflow);
        op.dst = static_cast<unsigned char*>(p.dst);
        op.width = p.width * p.elementSize; op.height = p.height; op.dst_pitch = p.pitch;
        void* alias; bool host;
        auto error = pointer(op.dst, &alias, &host, &result.cuda_error);
        if (error != GraphMemoryError::None) GM_ERROR(error);
        if (host) GM_ERROR(GraphMemoryError::UnsupportedLayout);
      } else {
        cudaMemcpy3DParms p{};
        GM_CUDA(cudaGraphMemcpyNodeGetParams(node, &p));
        if (p.srcArray || p.dstArray || p.extent.depth != 1 || p.srcPos.x || p.srcPos.y || p.srcPos.z || p.dstPos.x || p.dstPos.y || p.dstPos.z)
          GM_ERROR(GraphMemoryError::UnsupportedLayout);
        op.width = p.extent.width; op.height = p.extent.height;
        op.dst_pitch = p.dstPtr.pitch; op.src_pitch = p.srcPtr.pitch;
        void *src, *dst; bool host_src, host_dst;
        auto error = pointer(p.srcPtr.ptr, &src, &host_src, &result.cuda_error);
        if (error != GraphMemoryError::None) GM_ERROR(error);
        error = pointer(p.dstPtr.ptr, &dst, &host_dst, &result.cuda_error);
        if (error != GraphMemoryError::None) GM_ERROR(error);
        if (host_src && host_dst) GM_ERROR(GraphMemoryError::UnsupportedLayout);
        cudaMemcpyKind kind = host_src ? cudaMemcpyHostToDevice : host_dst ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
        if (p.kind != kind && p.kind != cudaMemcpyDefault) GM_ERROR(GraphMemoryError::UnsupportedLayout);
        op.src = static_cast<const unsigned char*>(src); op.dst = static_cast<unsigned char*>(dst); op.host_dst = host_dst;
        if (overflow(op.width, op.height)) GM_ERROR(GraphMemoryError::SizeOverflow);
        const size_t bytes = op.width * op.height;
        result.copies++; result.copy_bytes += bytes;
        if (host_src) result.host_read_bytes += bytes;
        if (host_dst) result.host_write_bytes += bytes;
      }
      if (!op.width || !op.height || (op.height > 1 && (op.dst_pitch < op.width || (!clear && op.src_pitch < op.width))))
        GM_ERROR(GraphMemoryError::UnsupportedLayout);
      if (overflow(op.width, op.height) || overflow(op.dst_pitch, op.height) || (!clear && overflow(op.src_pitch, op.height)))
        GM_ERROR(GraphMemoryError::SizeOverflow);
      if (clear) { result.clears++; result.clear_bytes += op.width * op.height; }
      auto aligned = [&](size_t unit) {
        return op.width % unit == 0 && (op.height == 1 || (op.dst_pitch % unit == 0 && (clear || op.src_pitch % unit == 0))) &&
          reinterpret_cast<uintptr_t>(op.dst) % unit == 0 && (clear || reinterpret_cast<uintptr_t>(op.src) % unit == 0);
      };
      op.packet = aligned(16) ? 16 : aligned(4) ? 4 : 1;
      if (op.width * op.height / op.packet > uint64_t(std::numeric_limits<unsigned>::max()) * 256) GM_ERROR(GraphMemoryError::SizeOverflow);
      plans.push_back(Planned{node, op, clear});
      replaced.emplace(node, nullptr);
    }
    // Only the replaced nodes' edges change endpoints. Kernel-kernel PDL edge data stays untouched.
    for (size_t i = 0; i < ne; i++) {
      if (!replaced.count(from[i]) && !replaced.count(to[i])) continue;
      const cudaGraphEdgeData ordinary{};
      if (memcmp(&edges[i], &ordinary, sizeof(ordinary))) GM_ERROR(GraphMemoryError::UnsupportedEdge);
    }
    for (auto& plan : plans) {
      void* args[] = {&plan.op};
      cudaKernelNodeParams p{};
      p.func = plan.clear ? reinterpret_cast<void*>(graph_clear) : reinterpret_cast<void*>(graph_copy);
      p.blockDim = dim3(256);
      const size_t packets = plan.op.width * plan.op.height / plan.op.packet;
      p.gridDim = dim3(static_cast<unsigned>((packets + 255) / 256));
      p.kernelParams = args;
      GM_CUDA(cudaGraphAddKernelNode(&replaced.at(plan.old), graph, nullptr, 0, &p));
    }
    for (auto& plan : plans) GM_CUDA(cudaGraphDestroyNode(plan.old));
    for (size_t i = 0; i < ne; i++) {
      auto f = replaced.find(from[i]), t = replaced.find(to[i]);
      if (f == replaced.end() && t == replaced.end()) continue;
      cudaGraphNode_t source = f == replaced.end() ? from[i] : f->second;
      cudaGraphNode_t target = t == replaced.end() ? to[i] : t->second;
      GM_CUDA(graph_dependencies(graph, &source, &target, &edges[i], 1));
    }
    return result;
  } catch (const std::bad_alloc&) {
    result.error = GraphMemoryError::Allocation;
    return result;
  }
#undef GM_CUDA
#undef GM_ERROR
}
}

H5_LIT_EXPORT
