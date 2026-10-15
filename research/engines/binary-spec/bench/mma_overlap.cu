#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA_ERROR %s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
  std::exit(2); } } while (0)

enum class Work { Tensor, Fma, Int, Both };
enum class Style { Pure, Phases, Interleave, Warps, Ctas };
enum class Launch { Single, Serial, Streams };
struct Result { float tensor, fp32; uint32_t integer; };
struct Span { unsigned long long begin, end; unsigned sm, role; };
struct Config {
  int iters, repeats, warps, blocks_per_sm, chains, steps;
  std::string selected, trace;
};
struct Case { std::string name; Style style; Work work; Launch launch; };
struct Checksum { double tensor = 0, fp32 = 0; uint64_t integer = 0; };
struct Stats { std::vector<float> ms; Checksum sum; double overlap = 0; int regs = 0, active = 0; };

__device__ __forceinline__ void tensor_step(float (&v)[4]) {
  const unsigned a = 0x22222222u, scale = 0x38383838u;
  const uint16_t z = 0;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%0,%1,%2,%3},%5,{%6,%6},%5,{%6,%6};"
    : "+f"(v[0]), "+f"(v[1]), "+f"(v[2]), "+f"(v[3]) : "r"(a), "r"(scale), "h"(z));
}

template<Work W>
__device__ __forceinline__ void scalar_step(float& f, unsigned& i, float add) {
  if constexpr (W == Work::Fma || W == Work::Both)
    asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(f) : "f"(0.99999994f), "f"(add));
  if constexpr (W == Work::Int || W == Work::Both)
    asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;" : "+r"(i));
}

// Every chain is loop-carried; distinct chains have no dependencies on each other.
// All 32 lanes execute each warp's MMA, and all results reach unconditional stores.
template<int C, Style S, Work W>
__global__ void overlap_kernel(int iters, int scalar_steps, int tensor_warps,
                               Result* output, Span* spans) {
  const int warp = threadIdx.x / 32;
  const int logical_block = S == Style::Ctas ? blockIdx.x / 2 : blockIdx.x;
  const int logical_thread = logical_block * tensor_warps * 32 + threadIdx.x % (tensor_warps * 32);
  bool do_tensor, do_scalar;
  if constexpr (S == Style::Pure) {
    do_tensor = W == Work::Tensor; do_scalar = W != Work::Tensor;
  } else if constexpr (S == Style::Warps) {
    do_tensor = warp < tensor_warps; do_scalar = !do_tensor;
  } else if constexpr (S == Style::Ctas) {
    do_tensor = (blockIdx.x & 1) == 0; do_scalar = !do_tensor;
  } else {
    do_tensor = true; do_scalar = true;
  }
  float t[C][4] = {}, f[C]; unsigned i[C];
  const float add = 0.0001f + (logical_thread & 31) * 0.000001f;
#pragma unroll
  for (int ch = 0; ch < C; ++ch) {
    f[ch] = 0.5f + ch * 0.125f + (logical_thread & 31) * 0.001f;
    i[ch] = unsigned(logical_thread) * 2654435761u + ch + 1;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    spans[blockIdx.x] = {static_cast<unsigned long long>(clock64()), 0, sm, do_tensor ? (do_scalar ? 3u : 1u) : 2u};
  }
  if constexpr (S == Style::Phases) {
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int ch = 0; ch < C; ++ch) tensor_step(t[ch]);
    }
    for (int it = 0; it < iters; ++it) {
      for (int s = 0; s < scalar_steps; ++s) {
#pragma unroll
        for (int ch = 0; ch < C; ++ch) scalar_step<W>(f[ch], i[ch], add);
      }
    }
  } else if constexpr (S == Style::Interleave) {
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int ch = 0; ch < C; ++ch) {
        tensor_step(t[ch]);
        scalar_step<W>(f[ch], i[ch], add);
      }
      for (int s = 1; s < scalar_steps; ++s) {
#pragma unroll
        for (int ch = 0; ch < C; ++ch) scalar_step<W>(f[ch], i[ch], add);
      }
    }
  } else {
    if (do_tensor) {
      for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int ch = 0; ch < C; ++ch) tensor_step(t[ch]);
      }
    } else {
      for (int it = 0; it < iters; ++it) {
        for (int s = 0; s < scalar_steps; ++s) {
#pragma unroll
          for (int ch = 0; ch < C; ++ch) scalar_step<W>(f[ch], i[ch], add);
        }
      }
    }
  }
  Result r = {};
#pragma unroll
  for (int ch = 0; ch < C; ++ch) {
    if (do_tensor) r.tensor += (t[ch][0] + t[ch][1]) + (t[ch][2] + t[ch][3]);
    if (do_scalar) {
      if constexpr (W == Work::Fma || W == Work::Both) r.fp32 += f[ch];
      if constexpr (W == Work::Int || W == Work::Both) r.integer += i[ch];
    }
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = r;
  __syncthreads();
  if (threadIdx.x == 0) spans[blockIdx.x].end = clock64();
}

struct Buffers { Result* values; Span* spans; };
struct Runtime {
  Config c; int sms, grid; Buffers b[2];
  cudaStream_t control, tensor, scalar;
  cudaEvent_t start, end, tensor_done, scalar_done;
};

template<int C, Style S, Work W>
void launch_kernel(Runtime& rt, Buffers& b, cudaStream_t stream, Stats* stats) {
  int blocks = rt.grid * (S == Style::Ctas ? 2 : 1);
  int threads = rt.c.warps * 32 * (S == Style::Warps ? 2 : 1);
  if (stats) {
    cudaFuncAttributes attr;
    CUDA(cudaFuncGetAttributes(&attr, overlap_kernel<C, S, W>));
    stats->regs = attr.numRegs;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&stats->active, overlap_kernel<C, S, W>, threads, 0));
    if (stats->active == 0) { std::fprintf(stderr, "RESOURCE_ERROR kernel cannot reside\n"); std::exit(3); }
  }
  overlap_kernel<C, S, W><<<blocks, threads, 0, stream>>>(rt.c.iters, rt.c.steps, rt.c.warps, b.values, b.spans);
  CUDA(cudaGetLastError());
}

template<int C, Work W>
void launch_style(Runtime& rt, const Case& c, Buffers& b, cudaStream_t stream, Stats* stats) {
  switch (c.style) {
    case Style::Pure: launch_kernel<C, Style::Pure, W>(rt, b, stream, stats); break;
    case Style::Phases: launch_kernel<C, Style::Phases, W>(rt, b, stream, stats); break;
    case Style::Interleave: launch_kernel<C, Style::Interleave, W>(rt, b, stream, stats); break;
    case Style::Warps: launch_kernel<C, Style::Warps, W>(rt, b, stream, stats); break;
    case Style::Ctas: launch_kernel<C, Style::Ctas, W>(rt, b, stream, stats); break;
  }
}

template<int C>
void dispatch(Runtime& rt, const Case& c, Buffers& b, cudaStream_t stream, Stats* stats) {
  switch (c.work) {
    case Work::Tensor: launch_kernel<C, Style::Pure, Work::Tensor>(rt, b, stream, stats); break;
    case Work::Fma: launch_style<C, Work::Fma>(rt, c, b, stream, stats); break;
    case Work::Int: launch_style<C, Work::Int>(rt, c, b, stream, stats); break;
    case Work::Both: launch_style<C, Work::Both>(rt, c, b, stream, stats); break;
  }
}
void dispatch(Runtime& rt, const Case& c, Buffers& b, cudaStream_t stream, Stats* stats) {
  if (rt.c.chains == 1) dispatch<1>(rt, c, b, stream, stats);
  else dispatch<8>(rt, c, b, stream, stats);
}

float measure(Runtime& rt, const Case& c, Stats* stats) {
  Case tensor = {"tensor", Style::Pure, Work::Tensor, Launch::Single};
  Case scalar = {"scalar", Style::Pure, c.work, Launch::Single};
  CUDA(cudaEventRecord(rt.start, rt.control));
  if (c.launch == Launch::Streams) {
    CUDA(cudaStreamWaitEvent(rt.tensor, rt.start, 0));
    CUDA(cudaStreamWaitEvent(rt.scalar, rt.start, 0));
    dispatch(rt, tensor, rt.b[0], rt.tensor, stats);
    dispatch(rt, scalar, rt.b[1], rt.scalar, nullptr);
    CUDA(cudaEventRecord(rt.tensor_done, rt.tensor));
    CUDA(cudaEventRecord(rt.scalar_done, rt.scalar));
    CUDA(cudaStreamWaitEvent(rt.control, rt.tensor_done, 0));
    CUDA(cudaStreamWaitEvent(rt.control, rt.scalar_done, 0));
  } else if (c.launch == Launch::Serial) {
    dispatch(rt, tensor, rt.b[0], rt.control, stats);
    dispatch(rt, scalar, rt.b[1], rt.control, nullptr);
  } else {
    dispatch(rt, c, rt.b[0], rt.control, stats);
  }
  CUDA(cudaEventRecord(rt.end, rt.control));
  CUDA(cudaEventSynchronize(rt.end));
  float ms; CUDA(cudaEventElapsedTime(&ms, rt.start, rt.end));
  return ms;
}

// Per-SM union intervals only: clock64 is compared within an SM, never across SMs.
double overlap_fraction(const std::vector<Span>& spans) {
  struct Edge { unsigned sm; unsigned long long cycle; int t, s; };
  std::vector<Edge> edges;
  for (const auto& x : spans) {
    int t = (x.role & 1) != 0, s = (x.role & 2) != 0;
    edges.push_back({x.sm, x.begin, t, s}); edges.push_back({x.sm, x.end, -t, -s});
  }
  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
    return a.sm != b.sm ? a.sm < b.sm : a.cycle < b.cycle;
  });
  unsigned sm = 0; unsigned long long last = 0; int t = 0, s = 0;
  long double both = 0, either = 0;
  for (const auto& e : edges) {
    if (e.sm != sm) { sm = e.sm; t = s = 0; last = e.cycle; }
    auto delta = e.cycle - last;
    if (t || s) either += delta;
    if (t && s) both += delta;
    t += e.t; s += e.s; last = e.cycle;
  }
  return either == 0 ? 0 : double(both / either);
}

Checksum collect(Runtime& rt, const Case& c, std::vector<Span>& spans) {
  Checksum sum;
  const int banks = c.launch == Launch::Single ? 1 : 2;
  for (int bank = 0; bank < banks; ++bank) {
    int blocks = rt.grid * (c.launch == Launch::Single && c.style == Style::Ctas ? 2 : 1);
    int threads = rt.c.warps * 32 * (c.launch == Launch::Single && c.style == Style::Warps ? 2 : 1);
    std::vector<Result> values(size_t(blocks) * threads);
    std::vector<Span> local(blocks);
    CUDA(cudaMemcpy(values.data(), rt.b[bank].values, values.size() * sizeof(Result), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(local.data(), rt.b[bank].spans, local.size() * sizeof(Span), cudaMemcpyDeviceToHost));
    for (const auto& v : values) {
      if (!std::isfinite(v.tensor) || !std::isfinite(v.fp32)) {
        std::fprintf(stderr, "CHECKSUM_ERROR nonfinite output in %s\n", c.name.c_str()); std::exit(4);
      }
      sum.tensor += v.tensor; sum.fp32 += v.fp32; sum.integer += v.integer;
    }
    spans.insert(spans.end(), local.begin(), local.end());
  }
  return sum;
}

int number(const char* s) {
  errno = 0; char* end; long n = std::strtol(s, &end, 10);
  if (errno || end == s || *end || n < 1 || n > 100000000) {
    std::fprintf(stderr, "ARGUMENT_ERROR invalid positive integer: %s\n", s); std::exit(1);
  }
  return int(n);
}
void usage() {
  std::puts("mma_overlap --iters N --repeats N --warps N --blocks-per-sm N --chains 1|8 --scalar-steps N --case all|NAME [--trace FILE]\n"
            "N is positive; warps <=16; blocks-per-sm <=32; repeats <=1000; scalar-steps <=256.\n"
            "Cases: tensor, fma, int, fma_int, and {serial,phases,interleave,warps,ctas,streams}_{fma,int,fma_int}.\n"
            "All geometry arguments are required. --case NAME also runs its isolated references.");
}

int main(int argc, char** argv) {
  Config cfg = {}; unsigned seen = 0;
  for (int a = 1; a < argc; a += 2) {
    if (!std::strcmp(argv[a], "--help")) { usage(); return 0; }
    if (a + 1 == argc) { usage(); return 1; }
    std::string key = argv[a]; const char* value = argv[a + 1]; unsigned bit = 0;
    if (key == "--iters") { cfg.iters = number(value); bit = 1; }
    else if (key == "--repeats") { cfg.repeats = number(value); bit = 2; }
    else if (key == "--warps") { cfg.warps = number(value); bit = 4; }
    else if (key == "--blocks-per-sm") { cfg.blocks_per_sm = number(value); bit = 8; }
    else if (key == "--chains") { cfg.chains = number(value); bit = 16; }
    else if (key == "--scalar-steps") { cfg.steps = number(value); bit = 32; }
    else if (key == "--case") { cfg.selected = value; bit = 64; }
    else if (key == "--trace") { cfg.trace = value; bit = 128; }
    else { std::fprintf(stderr, "ARGUMENT_ERROR unknown option %s\n", key.c_str()); return 1; }
    if (seen & bit) { std::fprintf(stderr, "ARGUMENT_ERROR duplicate option %s\n", key.c_str()); return 1; }
    seen |= bit;
  }
  if ((seen & 127) != 127 || cfg.warps > 16 || cfg.blocks_per_sm > 32 || cfg.repeats > 1000 ||
      cfg.steps > 256 || (cfg.chains != 1 && cfg.chains != 8) || cfg.selected.empty() ||
      ((seen & 128) && cfg.trace.empty())) { usage(); return 1; }
  std::vector<Case> cases = {{"tensor", Style::Pure, Work::Tensor, Launch::Single},
    {"fma", Style::Pure, Work::Fma, Launch::Single}, {"int", Style::Pure, Work::Int, Launch::Single},
    {"fma_int", Style::Pure, Work::Both, Launch::Single}};
  for (auto w : {Work::Fma, Work::Int, Work::Both}) {
    std::string suffix = w == Work::Fma ? "fma" : w == Work::Int ? "int" : "fma_int";
    cases.push_back({"serial_" + suffix, Style::Pure, w, Launch::Serial});
    cases.push_back({"phases_" + suffix, Style::Phases, w, Launch::Single});
    cases.push_back({"interleave_" + suffix, Style::Interleave, w, Launch::Single});
    cases.push_back({"warps_" + suffix, Style::Warps, w, Launch::Single});
    cases.push_back({"ctas_" + suffix, Style::Ctas, w, Launch::Single});
    cases.push_back({"streams_" + suffix, Style::Pure, w, Launch::Streams});
  }
  if (cfg.selected != "all") {
    auto found = std::find_if(cases.begin(), cases.end(), [&](const Case& c) { return c.name == cfg.selected; });
    if (found == cases.end()) { std::fprintf(stderr, "ARGUMENT_ERROR unknown case %s\n", cfg.selected.c_str()); return 1; }
    Case selected = *found;
    cases.resize(4);
    if (selected.name != "tensor" && selected.name != "fma" && selected.name != "int" && selected.name != "fma_int")
      cases.push_back(selected);
  }
  CUDA(cudaSetDevice(0)); cudaDeviceProp prop; CUDA(cudaGetDeviceProperties(&prop, 0));
  if (prop.major != 12 || prop.minor != 0) {
    std::fprintf(stderr, "DEVICE_ERROR requires sm_120a hardware, got sm_%d%d\n", prop.major, prop.minor); return 2;
  }
  Runtime rt = {}; rt.c = cfg; rt.sms = prop.multiProcessorCount; rt.grid = rt.sms * cfg.blocks_per_sm;
  size_t capacity = size_t(rt.grid) * cfg.warps * 32 * 2;
  for (auto& b : rt.b) {
    CUDA(cudaMalloc(&b.values, capacity * sizeof(Result)));
    CUDA(cudaMalloc(&b.spans, size_t(rt.grid) * 2 * sizeof(Span)));
  }
  CUDA(cudaStreamCreateWithFlags(&rt.control, cudaStreamNonBlocking));
  CUDA(cudaStreamCreateWithFlags(&rt.tensor, cudaStreamNonBlocking));
  CUDA(cudaStreamCreateWithFlags(&rt.scalar, cudaStreamNonBlocking));
  CUDA(cudaEventCreate(&rt.start)); CUDA(cudaEventCreate(&rt.end));
  CUDA(cudaEventCreate(&rt.tensor_done)); CUDA(cudaEventCreate(&rt.scalar_done));
  std::printf("# device=%s sms=%d concurrent_kernels=%d iters=%d repeats=%d tensor_warps=%d blocks_per_sm_requested=%d chains=%d scalar_steps=%d\n",
    prop.name, rt.sms, prop.concurrentKernels, cfg.iters, cfg.repeats, cfg.warps, cfg.blocks_per_sm, cfg.chains, cfg.steps);
  std::vector<Stats> stats(cases.size());
  // Resource queries stay outside every event interval.
  for (size_t n = 0; n < cases.size(); ++n) {
    dispatch(rt, cases[n], rt.b[0], rt.control, &stats[n]);
    CUDA(cudaStreamSynchronize(rt.control));
    measure(rt, cases[n], nullptr);
  }
  FILE* trace = nullptr;
  if (!cfg.trace.empty()) {
    trace = std::fopen(cfg.trace.c_str(), "w");
    if (!trace) { std::perror("TRACE_ERROR"); return 5; }
    std::fprintf(trace, "case,sample,cta_index,sm,role,begin_cycle,end_cycle\n");
  }
  for (int r = 0; r < cfg.repeats; ++r) {
    for (size_t j = 0; j < cases.size(); ++j) {
      size_t n = (j + r) % cases.size();
      stats[n].ms.push_back(measure(rt, cases[n], nullptr));
      if (r == cfg.repeats - 1) {
        std::vector<Span> spans;
        stats[n].sum = collect(rt, cases[n], spans);
        // A mixed CTA's interval cannot identify which phase was executing.
        if (cases[n].style == Style::Ctas || cases[n].launch == Launch::Streams || cases[n].launch == Launch::Serial)
          stats[n].overlap = overlap_fraction(spans);
        else stats[n].overlap = -1;
        if (trace) for (size_t k = 0; k < spans.size(); ++k) {
          auto x = spans[k];
          std::fprintf(trace, "%s,%d,%zu,%u,%u,%llu,%llu\n", cases[n].name.c_str(), r, k, x.sm, x.role, x.begin, x.end);
        }
      }
    }
  }
  if (trace && std::fclose(trace) != 0) { std::perror("TRACE_ERROR"); return 5; }
  if (stats[3].sum.fp32 != stats[1].sum.fp32 || stats[3].sum.integer != stats[2].sum.integer) {
    std::fprintf(stderr, "CHECKSUM_ERROR combined scalar work differs from individual references\n"); return 4;
  }
  auto median = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v.size() % 2 ? double(v[v.size()/2]) : (double(v[v.size()/2-1]) + v[v.size()/2]) / 2; };
  const double units = double(rt.grid) * cfg.warps * cfg.iters * cfg.chains;
  std::puts("case,median_ms,min_ms,max_ms,mma_count,fp32_fma_count,int32_mad_count,Gwarp_MMA_s,TMAC_s,TFLOP_s,GINT_MAD_s,speedup_vs_isolated_sum,time_vs_isolated_max,cta_overlap_fraction_last_sample,registers_per_thread,occupancy_max_ctas_per_sm,checksum_tensor,checksum_fp32,checksum_int");
  for (size_t n = 0; n < cases.size(); ++n) {
    const auto& c = cases[n]; const auto& s = stats[n];
    bool tensor = c.work == Work::Tensor || c.style != Style::Pure || c.launch != Launch::Single;
    bool fma = c.work == Work::Fma || c.work == Work::Both;
    bool integer = c.work == Work::Int || c.work == Work::Both;
    size_t scalar_ref = c.work == Work::Fma ? 1 : c.work == Work::Int ? 2 : 3;
    Checksum expect;
    if (tensor) expect.tensor = stats[0].sum.tensor;
    if (fma) expect.fp32 = stats[scalar_ref].sum.fp32;
    if (integer) expect.integer = stats[scalar_ref].sum.integer;
    if (s.sum.tensor != expect.tensor || s.sum.fp32 != expect.fp32 || s.sum.integer != expect.integer) {
      std::fprintf(stderr, "CHECKSUM_ERROR %s differs from isolated work\n", c.name.c_str()); return 4;
    }
    double ms = median(s.ms), sec = ms * 1e-3;
    double mm = tensor ? units : 0, ff = fma ? units * 32 * cfg.steps : 0, ii = integer ? units * 32 * cfg.steps : 0;
    double sum_ms = (tensor ? median(stats[0].ms) : 0) + (fma || integer ? median(stats[scalar_ref].ms) : 0);
    double max_ms = std::max(tensor ? median(stats[0].ms) : 0, fma || integer ? median(stats[scalar_ref].ms) : 0);
    std::printf("%s,%.6f,%.6f,%.6f,%.0f,%.0f,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%.17g,%.17g,%llu\n",
      c.name.c_str(), ms, *std::min_element(s.ms.begin(), s.ms.end()), *std::max_element(s.ms.begin(), s.ms.end()),
      mm, ff, ii, mm/sec/1e9, mm*8192/sec/1e12, ff*2/sec/1e12, ii/sec/1e9,
      sum_ms/ms, ms/max_ms, s.overlap, s.regs, s.active, s.sum.tensor, s.sum.fp32, (unsigned long long)s.sum.integer);
  }
  for (auto& b : rt.b) { CUDA(cudaFree(b.values)); CUDA(cudaFree(b.spans)); }
  CUDA(cudaEventDestroy(rt.start)); CUDA(cudaEventDestroy(rt.end));
  CUDA(cudaEventDestroy(rt.tensor_done)); CUDA(cudaEventDestroy(rt.scalar_done));
  CUDA(cudaStreamDestroy(rt.control)); CUDA(cudaStreamDestroy(rt.tensor)); CUDA(cudaStreamDestroy(rt.scalar));
  return 0;
}
