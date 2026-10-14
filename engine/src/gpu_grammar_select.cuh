#pragma once
#include "gpu_grammar_runtime.cuh"
#include <math_constants.h>
#include <climits>

namespace h5::gpu_grammar {
constexpr int SELECT_PARTS = 128, SELECT_TOP = 16;
struct SelectWorkspace { float* values; int* ids; };
namespace selection_detail {
static __global__ void external_mask_kernel(const uint32_t* source, int words,
    uint32_t* mask, int* ids, uint32_t* count, RowResult* result) {
  if (threadIdx.x) return;
  uint32_t n = 0;
  for (int w = 0; w < words; ++w) {
    const uint32_t bits = mask[w] = source[w];
    for (uint32_t left = bits; left; left &= left - 1) {
      if (n < COMPACT_CAPACITY) ids[n] = 32*w + __ffs(left)-1;
      ++n;
    }
  }
  *count = result->allowed = n;
  result->before = result->after;
  if (!n) result->error = uint32_t(RuntimeError::NoLegalTokens);
}
__device__ inline bool better(float value, int id, float other, int other_id) {
  return value > other || (value == other && id < other_id);
}
template<int TOP> __device__ inline void insert(float value, int id, float (&values)[TOP], int (&ids)[TOP]) {
  if (!isfinite(value) || id == INT_MAX) return;
  for (int k=0;k<TOP;++k) if (better(value,id,values[k],ids[k])) {
    for (int j=TOP-1;j>k;--j) { values[j]=values[j-1]; ids[j]=ids[j-1]; }
    values[k]=value; ids[k]=id; break;
  }
}
template<int TOP> __device__ inline void reduce(float (&values)[TOP], int (&ids)[TOP],
    float* topv, int* topi, RowResult* result) {
  __shared__ float scores[256];
  __shared__ int tokens[256], safe;
  const int t=threadIdx.x;
  for (int k=0;k<TOP;++k) {
    scores[t]=values[0]; tokens[t]=ids[0]; __syncthreads();
    for (int step=128;step;step>>=1) {
      if (t<step && better(scores[t+step],tokens[t+step],scores[t],tokens[t])) {
        scores[t]=scores[t+step]; tokens[t]=tokens[t+step];
      }
      __syncthreads();
    }
    const int winner=tokens[0];
    if (!t) {
      if (!k) {
        safe=winner==INT_MAX?-1:winner;
        if (result && safe<0) result->error=uint32_t(RuntimeError::NonfiniteLogits);
      }
      topi[k]=winner==INT_MAX?(result?safe:INT_MAX):winner;
      if (topv) topv[k]=scores[0];
    }
    if (ids[0]==winner) {
      for (int j=0;j+1<TOP;++j) { values[j]=values[j+1]; ids[j]=ids[j+1]; }
      values[TOP-1]=-CUDART_INF_F; ids[TOP-1]=INT_MAX;
    }
    __syncthreads();
  }
}
__device__ inline float adjusted(float value, int id, const int2* adjustments, int count) {
  for (int j=0;j<count;++j) if (adjustments[j].x==id)
    value=__fadd_rn(value,__int_as_float(adjustments[j].y));
  return value;
}
template<int TOP> static __global__ void partial_kernel(const float* dense, int vocabulary,
    const uint32_t* mask, const uint32_t* count, const RowResult* result,
    float* partialv, int* partiali, const int2* adjustments, int adjustment_count, const int* dense_row) {
  if (*count<=COMPACT_CAPACITY || result->error) return;
  if (dense_row) dense += size_t(*dense_row)*vocabulary;
  float values[TOP]; int ids[TOP];
  for (int k=0;k<TOP;++k) { values[k]=-CUDART_INF_F; ids[k]=INT_MAX; }
  const int chunk=(vocabulary+SELECT_PARTS-1)/SELECT_PARTS;
  const int begin=blockIdx.x*chunk, end=min(begin+chunk,vocabulary);
  for (int id=begin+threadIdx.x;id<end;id+=blockDim.x) if (mask[id/32]&(1u<<(id%32)))
    insert<TOP>(adjusted(dense[id],id,adjustments,adjustment_count),id,values,ids);
  reduce<TOP>(values,ids,partialv+blockIdx.x*TOP,partiali+blockIdx.x*TOP,nullptr);
}
template<int TOP> static __global__ void finish_kernel(const float* sparse,
    const int* compact_ids, const uint32_t* count, float* topv, int* topi,
    RowResult* result, const float* partialv, const int* partiali,
    const int2* adjustments, int adjustment_count) {
  const int t=threadIdx.x; const uint32_t legal=*count;
  if (result->error || !legal) {
    if (!t) for (int k=0;k<TOP;++k) { topi[k]=-1; if(topv)topv[k]=-CUDART_INF_F; }
    return;
  }
  if (legal==1) {
    if (!t) for (int k=0;k<TOP;++k) { topi[k]=compact_ids[0]; if(topv)topv[k]=k?-CUDART_INF_F:0.f; }
    return;
  }
  float values[TOP]; int ids[TOP];
  for (int k=0;k<TOP;++k) { values[k]=-CUDART_INF_F; ids[k]=INT_MAX; }
  if (legal<=COMPACT_CAPACITY) {
    for (int i=t;i<int(legal);i+=blockDim.x) {
      const int id=compact_ids[i]; insert<TOP>(adjusted(sparse[i],id,adjustments,adjustment_count),id,values,ids);
    }
  } else {
    for (int i=t;i<SELECT_PARTS*TOP;i+=blockDim.x) insert<TOP>(partialv[i],partiali[i],values,ids);
  }
  reduce<TOP>(values,ids,topv,topi,result);
}
}
inline cudaError_t launch_external_mask(const uint32_t* source,int words,
    uint32_t* mask,int* ids,uint32_t* count,RowResult* result,cudaStream_t stream) {
  selection_detail::external_mask_kernel<<<1,1,0,stream>>>(source,words,mask,ids,count,result);
  return cudaGetLastError();
}
template<int TOP> inline cudaError_t launch_select(const float* dense,const float* sparse,int vocabulary,
    const uint32_t* mask,const int* ids,const uint32_t* count,float* topv,int* topi,RowResult* result,
    const int2* adjustments,int adjustment_count,SelectWorkspace workspace,cudaStream_t stream,const int* dense_row=nullptr) {
  static_assert(TOP==1 || TOP==SELECT_TOP,"Only argmax and exact grammar top16 are supported");
  if (!workspace.values || !workspace.ids || vocabulary<1) return cudaErrorInvalidValue;
  selection_detail::partial_kernel<TOP><<<SELECT_PARTS,256,0,stream>>>(dense,vocabulary,mask,count,result,
      workspace.values,workspace.ids,adjustments,adjustment_count,dense_row);
  cudaError_t status=cudaGetLastError(); if(status!=cudaSuccess)return status;
  selection_detail::finish_kernel<TOP><<<1,256,0,stream>>>(sparse,ids,count,topv,topi,result,
      workspace.values,workspace.ids,adjustments,adjustment_count);
  return cudaGetLastError();
}
}
