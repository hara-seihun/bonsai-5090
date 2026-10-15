#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace h5::micro {
enum Phase : unsigned { Setup, WeightPrefetch, Predecessor, EmptyWait, WeightIssue,
  ActivationIssue, FullWait, Decode, ActivationOperand, MmaScale, EmptyRelease,
  FinalPredecessor, Epilogue, PhaseCount };
inline constexpr const char* phase_names[] = {"setup", "weight_prefetch_initial",
  "predecessor_acquire", "empty_stage_wait", "weight_issue", "activation_issue",
  "ring_weight_activation_wait", "dense5_decode_scale_load", "activation_operand",
  "mma_scale_accumulate", "empty_stage_release", "final_predecessor_acquire", "epilogue"};
struct Span { uint64_t begin_ns, end_ns, begin_cycles, end_cycles; unsigned sm; };
struct Interval { uint64_t begin_ns, end_ns, cycles; unsigned visits; };
struct Config { Span* spans; Interval* intervals; unsigned ctas, samples, warps, slots; };
#ifdef H5_WARP_MICROSCOPE
static __device__ Config config;
__device__ __forceinline__ uint64_t ns() {
  uint64_t v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v) :: "memory"); return v;
}
__device__ __forceinline__ uint64_t cycles() {
  uint64_t v; asm volatile("mov.u64 %0, %%clock64;" : "=l"(v) :: "memory"); return v;
}
#endif

template<bool Enabled> struct Trace {
  __device__ __forceinline__ void begin(Phase, int = -1) {}
  __device__ __forceinline__ void end(Phase, int = -1) {}
  __device__ __forceinline__ void finish() {}
  template<class T> __device__ __forceinline__ void consume(const T&) {}
};
#ifdef H5_WARP_MICROSCOPE
template<> struct Trace<true> {
  bool active;
  unsigned index;
  uint64_t start_ns, start_cycles;
  __device__ __forceinline__ Trace() {
    unsigned cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    unsigned sample = config.samples == 1 ? 0 :
      (unsigned)(((uint64_t)cta * (config.samples - 1) + config.ctas - 2) / (config.ctas - 1));
    unsigned selected = config.samples == 1 ? config.ctas - 1 :
      (unsigned)((uint64_t)sample * (config.ctas - 1) / (config.samples - 1));
    active = (threadIdx.x & 31) == 0 && cta == selected && sample < config.samples;
    index = sample * config.warps + (threadIdx.x >> 5);
    if (active) {
      Span& s = config.spans[index];
      s.begin_ns = ns(); s.begin_cycles = cycles();
      asm volatile("mov.u32 %0, %%smid;" : "=r"(s.sm));
    }
  }
  __device__ __forceinline__ void begin(Phase, int = -1) {
    if (active) { start_ns = ns(); start_cycles = cycles(); }
  }
  __device__ __forceinline__ void end(Phase phase, int kb = -1) {
    if (active) {
      uint64_t end_cycles = cycles(), end_ns = ns();
      Interval& r = config.intervals[(index * config.slots + kb + 1) * PhaseCount + phase];
      r.begin_ns = start_ns; r.end_ns = end_ns;
      r.cycles = end_cycles - start_cycles; r.visits = 1;
    }
  }
  template<class T> __device__ __forceinline__ void consume(const T& value) {
    // Pin register-only work before the ending timestamp without adding a warp barrier.
    asm volatile("" :: "f"(value) : "memory");
  }
  __device__ __forceinline__ void consume(const unsigned& value) {
    asm volatile("" :: "r"(value) : "memory");
  }
  __device__ __forceinline__ void finish() {
    if (active) { config.spans[index].end_cycles = cycles(); config.spans[index].end_ns = ns(); }
  }
};
#endif
} // namespace h5::micro
