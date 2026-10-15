#pragma once
#include "gpu_grammar_runtime.cuh"
namespace h5::gpu_grammar {
static __global__ void batch_wide_rows(int rows, const DeviceRow* input, const RowResult* results,
    DeviceVocabulary vocabulary, int* sources, int* ranks, uint32_t* count) {
  if (threadIdx.x) return;
  uint32_t n=0;
  for(int r=0;r<rows;++r) {
    const auto* plan=input[r].plan;
    const auto state=results[r].after;
    const uint32_t legal=results[r].error?0:!plan?vocabulary.size:
        plan->accepting[state.id]?vocabulary.eos_count:plan->counts[state.id];
    ranks[r]=-1;
    if(legal>COMPACT_CAPACITY) { ranks[r]=int(n); sources[n++]=r; }
  }
  for(uint32_t r=n;r<uint32_t(rows);++r) sources[r]=0;
  *count=n;
}
}
