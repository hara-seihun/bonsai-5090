#pragma once
#include <cerrno>

namespace fused_causal {
enum class Arm { Generic, Single };
enum class Failure { Usage, Shape, Format, Configuration, Device, Residency, Counter, Nonfinite };
static int fail(Failure e, const char* detail) {
  const char* tag = nullptr;
  switch (e) {
    case Failure::Usage: tag = "usage"; break;
    case Failure::Shape: tag = "shape"; break;
    case Failure::Format: tag = "format"; break;
    case Failure::Configuration: tag = "configuration"; break;
    case Failure::Device: tag = "device"; break;
    case Failure::Residency: tag = "residency"; break;
    case Failure::Counter: tag = "counter"; break;
    case Failure::Nonfinite: tag = "nonfinite"; break;
  }
  fprintf(stderr, "error=%s detail=%s\n", tag, detail);
  return e == Failure::Counter || e == Failure::Nonfinite ? 3 : 1;
}
static bool integer(const char* s, int lo, int hi, int& value) {
  char* end; errno = 0; const long v = strtol(s, &end, 10);
  if (errno || end == s || *end || v < lo || v > hi) return false;
  value = (int)v; return true;
}
static bool env_is(const char* name, const char* value, bool unset_ok) {
  const char* s = getenv(name); return s ? !strcmp(s, value) : unset_ok;
}
static const char* name(Arm arm) { return arm == Arm::Generic ? "generic-fused" : "single-fused-pad40"; }

__global__ void acquire(Seg* segs, float* ab, int* counter, int streams, int visit) {
  pdl_wait();
  const int t = threadIdx.x;
  if (t < streams) segs[t] = Seg{streams - 1 - t, t, 1, 0, 0, 0, (visit / NGDN) & 1};
  for (int i = t; i < streams * 96; i += blockDim.x)
    ab[i] = float(i % 31 - 15) * .04f + float(visit % 17 - 8) * .002f;
  if (!t) *counter = visit + 1;
  __syncthreads(); pdl_launch();
}
struct Run {
  Fixture& b; int* counter; bool counters_valid = true;
  explicit Run(Fixture& fixture) : b(fixture), counter(alloc<int>(1)) {}
  ~Run() { ck(cudaFree(counter)); }
  void reset() {
    b.reset(2, NGDN);
    ck(cudaMemsetAsync(counter, 0, sizeof(int), b.stream));
    ck(cudaStreamSynchronize(b.stream));
  }
  void visit(Arm arm, int visit) {
    pdl(acquire, dim3(1), dim3(128), 0, b.stream, b.segs, b.ab, counter, b.streams, visit);
    auto c = b.cf; auto a = b.af; c.gl = a.gl = visit % NGDN;
    switch (arm) {
      case Arm::Generic: launch_conv_gdn(c, a, b.streams, b.format, b.stream); break;
      case Arm::Single: launch_conv_gdn_single(c, a, b.streams, b.format, b.stream); break;
    }
    ck(cudaGetLastError());
  }
  cudaGraphExec_t graph(Arm arm, int sweeps, int& resident) {
    ck(cudaStreamBeginCapture(b.stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < sweeps * NGDN; ++i) visit(arm, i);
    cudaGraph_t g; ck(cudaStreamEndCapture(b.stream, &g));
    size_t count = 0; ck(cudaGraphGetNodes(g, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count); ck(cudaGraphGetNodes(g, nodes.data(), &count));
    resident = 0;
    for (auto node : nodes) {
      cudaGraphNodeType type; ck(cudaGraphNodeGetType(node, &type));
      if (type != cudaGraphNodeTypeKernel) continue;
      cudaKernelNodeParams p; ck(cudaGraphKernelNodeGetParams(node, &p));
      if (p.blockDim.x != 256) continue;
      cudaFuncAttributes attr; ck(cudaFuncGetAttributes(&attr, p.func));
      ck(cudaFuncSetAttribute(p.func, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, p.func, 256, p.sharedMemBytes));
      printf("resource arm=%s registers=%d static_shared=%zu dynamic_shared=%u local_bytes=%zu max_ctas_per_sm=%d\n",
          name(arm), attr.numRegs, attr.sharedSizeBytes, p.sharedMemBytes, attr.localSizeBytes, resident);
      break;
    }
    cudaGraphExec_t e; ck(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0)); ck(cudaGraphDestroy(g)); return e;
  }
  float sample(cudaGraphExec_t graph, int sweeps) {
    reset(); const float us = b.time(graph, sweeps);
    counters_valid &= reached(sweeps * NGDN); return us;
  }
  bool reached(int visits) { return copy(counter, 1)[0] == visits; }
};
struct Graph {
  cudaGraphExec_t exec;
  ~Graph() { ck(cudaGraphExecDestroy(exec)); }
};
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end()); size_t n = v.size();
  return (v[(n - 1) / 2] + v[n / 2]) * .5;
}
static void summary(const char* metric, const std::vector<double>& v) {
  const double mean = std::accumulate(v.begin(), v.end(), 0.) / v.size();
  double ss = 0; for (double x : v) ss += (x - mean) * (x - mean);
  std::vector<double> dev; const double med = median(v);
  for (double x : v) dev.push_back(fabs(x - med));
  printf("summary metric=%s samples=%zu mean=%.6f median=%.6f sample_sd=%.6f mad=%.6f min=%.6f max=%.6f\n",
      metric, v.size(), mean, med, sqrt(ss / (v.size() - 1)), median(dev),
      *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
}
static int main(int argc, char** argv) {
  const bool probe = !strcmp(argv[1], "--fused-probe");
  if (argc != 6) return fail(Failure::Usage,
      "--fused-causal STREAMS f16|bf16|f32 EVEN_SWEEPS PAIRS; --fused-probe STREAMS f16|bf16|f32 generic|single LAYER");
  int streams, sweeps = 2, pairs = 0, layer = 0; Arm arm = Arm::Generic;
  if (!integer(argv[2], 1, 64, streams)) return fail(Failure::Shape, "STREAMS must be 1..64");
  if (probe) {
    if (!strcmp(argv[4], "generic")) arm = Arm::Generic;
    else if (!strcmp(argv[4], "single")) arm = Arm::Single;
    else return fail(Failure::Usage, "probe arm must be generic|single");
    if (!integer(argv[5], 0, NGDN - 1, layer)) return fail(Failure::Shape, "probe LAYER must be 0..NGDN-1");
  } else if (!integer(argv[4], 2, 32, sweeps) || sweeps % 2 || !integer(argv[5], 2, 100, pairs))
    return fail(Failure::Shape, "SWEEPS must be even 2..32 and PAIRS 2..100");
  StateFormat format;
  if (!strcmp(argv[3], "f16")) format = StateFormat::F16;
  else if (!strcmp(argv[3], "bf16")) format = StateFormat::BF16;
  else if (!strcmp(argv[3], "f32")) format = StateFormat::F32;
  else return fail(Failure::Format, "fused comparison requires f16|bf16|f32; q8 is separate-only");
  if (!env_is("H5_GDN_CG", "8", false) || !env_is("H5_GDN_CG_SEGS", "0", true) ||
      !env_is("H5_CARVEOUT", "100", false) || !env_is("H5_CARVEOUT_SCOPE", "all", false) ||
      !(env_is("H5_PDL", "0", false) || env_is("H5_PDL", "1", false)) ||
      !env_is("H5_GDN_SINGLE_PAD_KIB", "40", false))
    return fail(Failure::Configuration, "requires CG=8 CG_SEGS=0(unset allowed) CARVEOUT=100 CARVEOUT_SCOPE=all PDL=0|1 SINGLE_PAD_KIB=40 (all H5_ prefix)");
  validate_gdn_single(GdnSingleMode::Fused, format, true, false, 0);
  int device; ck(cudaGetDevice(&device)); cudaDeviceProp prop; ck(cudaGetDeviceProperties(&prop, device));
  if (prop.major != 12 || prop.minor != 0 || prop.sharedMemPerMultiprocessor != 102400)
    return fail(Failure::Device, "this residency experiment requires sm120 with 100KiB shared memory per SM");
  const size_t state_bytes = (size_t)streams * NGDN * MS * (format == StateFormat::F32 ? 4 : 2);
  if (state_bytes <= 2ull * prop.l2CacheSize)
    return fail(Failure::Shape, "active recurrent state must exceed twice L2 capacity; use S64");
  Fixture b(streams, format); Run run(b);
  printf("config experiment=generic-fused-vs-single-fused-capped2 mode=%s streams=%d dtype=%s layers=%d sweeps=%d pairs=%d cg=8 carveout=100 pdl=%d state_bytes=%zu state_rw_bytes_per_sweep=%zu conv_active_bytes=%zu l2_bytes=%d same_addresses=1 reset_each_sample=1 input_counter_start=0\n",
      probe ? "probe" : "abba", streams, argv[3], NGDN, sweeps, pairs, pdl_enabled(), state_bytes,
      state_bytes * 2, (size_t)streams * NGDN * CS * 2 * sizeof(float), prop.l2CacheSize);
  int ga, sa; Graph generic{run.graph(Arm::Generic, sweeps, ga)}, single{run.graph(Arm::Single, sweeps, sa)};
  if (ga != 2 || sa != 2) return fail(Failure::Residency, "both target kernels must have an occupancy ceiling of exactly two CTAs per SM");
  if (probe) {
    // Same all-layer warmup and generic prefix in either process, outside the profiler range.
    run.sample(generic.exec, sweeps); run.reset();
    for (int i = 0; i < layer; ++i) run.visit(Arm::Generic, i);
    ck(cudaStreamSynchronize(b.stream));
    ck(cudaProfilerStart()); run.visit(arm, layer); ck(cudaStreamSynchronize(b.stream)); ck(cudaProfilerStop());
    if (!run.reached(layer + 1)) return fail(Failure::Counter, "probe did not reach expected input counter");
    printf("probe arm=%s layer=%d input_counter=%d profile_targets=1\n", name(arm), layer, layer + 1);
    return 0;
  }
  for (int i = 0; i < 2; ++i) { run.sample(generic.exec, sweeps); run.sample(single.exec, sweeps); }
  std::vector<double> refs, opts, saved, ratios;
  for (int i = 0; i < pairs; ++i) {
    const double a1 = run.sample(generic.exec, sweeps), b1 = run.sample(single.exec, sweeps);
    const double b2 = run.sample(single.exec, sweeps), a2 = run.sample(generic.exec, sweeps);
    if (!run.counters_valid) return fail(Failure::Counter, "graph did not reach expected input counter");
    const double a = (a1 + a2) * .5, c = (b1 + b2) * .5;
    refs.push_back(a); opts.push_back(c); saved.push_back(a - c); ratios.push_back(a / c);
    printf("pair index=%d unit=us_per_48_layer_sweep A1=%.6f B1=%.6f B2=%.6f A2=%.6f generic=%.6f single=%.6f saved=%.6f speedup=%.6f A_half_range=%.6f B_half_range=%.6f\n",
        i, a1, b1, b2, a2, a, c, a - c, a / c, fabs(a1 - a2) * .5, fabs(b1 - b2) * .5);
    fflush(stdout);
  }
  summary("generic_us", refs); summary("single_us", opts); summary("saved_us", saved); summary("speedup", ratios);
  // Observe fused numerical differences, without imposing bit-exactness.
  run.sample(generic.exec, sweeps); const auto reference = copy(b.y1, (size_t)streams * VDIM);
  run.sample(single.exec, sweeps); const auto candidate = copy(b.y1, (size_t)streams * VDIM);
  Error output; for (size_t i = 0; i < reference.size(); ++i) output.add(reference[i], candidate[i]);
  output.show("generic-fused-vs-single-fused-pad40", "last_layer_output", sweeps);
  return output.finite ? 0 : fail(Failure::Nonfinite, "last-layer output contains nonfinite values");
}
}
static int fused_causal_main(int argc, char** argv) {
  try { return fused_causal::main(argc, argv); }
  catch (const std::exception& e) { return fused_causal::fail(fused_causal::Failure::Configuration, e.what()); }
}
