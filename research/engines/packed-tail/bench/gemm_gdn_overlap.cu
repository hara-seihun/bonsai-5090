// Actual engine kernels; this include also exposes the compiled CG4 function for resource queries.
#include "../src/kernels.cu"
#include "ternary_operands.h"
#include <array>
#include <cerrno>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

#if defined(EXP_GDN_NOMEM) || defined(EXP_GDN_NOCOMP) || defined(EXP_NOMMA) || defined(EXP_NOWAIT) || defined(H5_PROFILE_GEMM_STREAM)
#error "Overlap benchmark requires unabbreviated production work"
#endif
#ifndef H5_PROFILE_GEMM
#error "Build with H5_PROFILE_GEMM for real GEMM resource queries"
#endif
namespace h5 {
int fp4_ncfg();
bool fp4_cfg_wg2(int);
const char* fp4_cfg_name(int, int);
cudaError_t fp4_profile_resources(int, int, int, cudaFuncAttributes*, int*);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int,
                         const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
void launch_gemm_fp4_silu(int, const uint8_t*, const uint8_t*, const float*, int, int,
                          const void*, const void*, int, float*, int, cudaStream_t, const float*, int);
}
using namespace h5;
static constexpr int SLOTS = 64, ROWS_PER_SLOT = 8, ROWS = SLOTS * ROWS_PER_SLOT;
static constexpr size_t STATE_SLOT = size_t(HV) * SS * SS;
static constexpr size_t STATE_N = STATE_SLOT * SLOTS;
static constexpr size_t STATE_STRIDED_N = (size_t(SLOTS - 1) * NGDN + 1) * STATE_SLOT;
static constexpr size_t PEND_STRIDED_N = (size_t(SLOTS - 1) * NGDN + 1) * 2 * PMAX * PEND_GDN;
static void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "CUDA_ERROR %s: %s\n", what, cudaGetErrorString(e)); exit(3); }
}
#define CUDA(x) ck((x), #x)
static void fail(const std::string& s) { fprintf(stderr, "ERROR %s\n", s.c_str()); exit(2); }
template<class T> struct Buffer {
  T* p = nullptr; size_t n;
  explicit Buffer(size_t count) : n(count) { CUDA(cudaMalloc(&p, n * sizeof(T))); }
  Buffer(const Buffer&) = delete;
  ~Buffer() { CUDA(cudaFree(p)); }
  void upload(const std::vector<T>& v) {
    if (v.size() != n) fail("upload size mismatch");
    CUDA(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> v(n); CUDA(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
  }
};
struct Stream {
  cudaStream_t p;
  Stream() { CUDA(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking)); }
  ~Stream() { CUDA(cudaStreamDestroy(p)); }
};
struct Event {
  cudaEvent_t p;
  Event() { CUDA(cudaEventCreate(&p)); }
  ~Event() { CUDA(cudaEventDestroy(p)); }
};
static uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
static float sample(size_t i, unsigned seed) { return (int(hash32(uint32_t(i) ^ seed) % 2001) - 1000) / 1000.f; }
struct Shape { const char* name; int n, k, mode; };
static const Shape SHAPES[] = {{"in_gdn", GDN_IN, D, 0}, {"in_att", ATT_IN, D, 0},
                              {"out", D, VDIM, 1}, {"gu_silu", 2 * FF, D, 3}, {"down", D, FF, 1}};
static void usage() {
  puts("gemm-gdn-overlap --wf wf2|wf6 --shape in_gdn|in_att|out|gu_silu|down|all\n"
       "  --cfg all|INDEX --repeats 1..101 --launches 1..1024 --weights hot|cold\n"
       "Fixed: 64 streams x 8 rows, unsplit GEMM, bf16 CG4 GDN, one layer per launch.\n"
       "All six options required. Cold requires launches >= printed weight-copy count.");
}
static int integer(const std::string& s, int lo, int hi) {
  char* end = nullptr; errno = 0; long v = strtol(s.c_str(), &end, 10);
  if (errno || end == s.c_str() || *end || v < lo || v > hi) fail("integer out of range: " + s);
  return int(v);
}
struct Options { int wf, cfg, repeats, launches; bool cold; std::string shape; };
static Options options(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") { usage(); exit(0); }
  std::map<std::string, std::string> m;
  for (int i = 1; i < argc; i += 2) {
    const std::string key = argv[i];
    if (i + 1 == argc || (key != "--wf" && key != "--shape" && key != "--cfg" && key != "--repeats" && key != "--launches" && key != "--weights") || !m.emplace(key, argv[i + 1]).second)
      fail("unknown, duplicate or incomplete option: " + key);
  }
  if (m.size() != 6) { usage(); fail("all six options required"); }
  if (m.at("--wf") != "wf2" && m.at("--wf") != "wf6") fail("wf must be wf2 or wf6");
  if (m.at("--weights") != "hot" && m.at("--weights") != "cold") fail("weights must be hot or cold");
  bool known = m.at("--shape") == "all";
  for (const auto& s : SHAPES) known |= m.at("--shape") == s.name;
  if (!known) fail("unknown shape");
  return {m.at("--wf") == "wf2" ? 2 : 6,
          m.at("--cfg") == "all" ? -1 : integer(m.at("--cfg"), 0, fp4_ncfg() - 1),
          integer(m.at("--repeats"), 1, 101), integer(m.at("--launches"), 1, 1024),
          m.at("--weights") == "cold", m.at("--shape")};
}
struct GdnWork {
  Buffer<float> conv{size_t(ROWS) * CONV_CH}, gates{size_t(ROWS) * 2 * HV}, y{size_t(ROWS) * VDIM};
  Buffer<Seg> segs{SLOTS};
  Buffer<__nv_bfloat16> initial{STATE_N}, state{STATE_STRIDED_N}, pending{PEND_STRIDED_N};
  GdnArgs args{};
  GdnWork() {
    std::vector<float> co(conv.n), gs(gates.n);
    for (size_t i = 0; i < co.size(); ++i) co[i] = sample(i, 19) * .1f;
    // k_conv supplies unit-normalized keys and queries to the recurrence.
    for (int r = 0; r < ROWS; ++r) for (int h = 0; h < 2 * HK; ++h) {
      float* p = co.data() + size_t(r) * CONV_CH + h * SS; double ss = 0;
      for (int j = 0; j < SS; ++j) ss += double(p[j]) * p[j];
      for (int j = 0; j < SS; ++j) p[j] *= float(1 / std::sqrt(ss + NORM_EPS));
    }
    for (size_t i = 0; i < gs.size(); ++i) gs[i] = i % (2 * HV) < HV ? .96f + .02f * (sample(i, 29) + 1) / 2 : .2f + .4f * (sample(i, 31) + 1) / 2;
    std::vector<__nv_bfloat16> seed(STATE_N);
    for (size_t i = 0; i < seed.size(); ++i) seed[i] = __float2bfloat16(sample(i, 37) * .01f);
    std::vector<Seg> sg(SLOTS);
    for (int i = 0; i < SLOTS; ++i) sg[i] = {i, i * ROWS_PER_SLOT, ROWS_PER_SLOT, 0, 0, 0};
    conv.upload(co); gates.upload(gs); initial.upload(seed); segs.upload(sg);
    args.conv_out = conv.p; args.gates = gates.p; args.state = state.p; args.pend = pending.p;
    args.gl = 0; args.y = y.p; args.segs = segs.p;
  }
  void reset(cudaStream_t st) {
    for (int i = 0; i < SLOTS; ++i) CUDA(cudaMemcpyAsync(state.p + size_t(i) * NGDN * STATE_SLOT,
        initial.p + size_t(i) * STATE_SLOT, STATE_SLOT * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, st));
    CUDA(cudaMemsetAsync(y.p, 0xff, y.n * sizeof(float), st));
  }
  void launch(cudaStream_t st) { launch_gdn(args, SLOTS, StateFormat::BF16, st); CUDA(cudaGetLastError()); }
  std::vector<__nv_bfloat16> snapshot() const {
    std::vector<__nv_bfloat16> v(STATE_N);
    for (int i = 0; i < SLOTS; ++i) CUDA(cudaMemcpy(v.data() + size_t(i) * STATE_SLOT,
        state.p + size_t(i) * NGDN * STATE_SLOT, STATE_SLOT * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    return v;
  }
};
struct GemmWork {
  Shape shape; int wf, copies, ldo;
  Buffer<uint8_t> a, as, w, scales;
  Buffer<float> ag, rs, out;
  size_t wb, sb;
  GemmWork(Shape s, const Options& o, int l2) : shape(s), wf(o.wf),
      copies(o.cold ? int((size_t(2) * l2) / (size_t(s.n / 32) * (s.k / 128) * (wf == 6 ? 2176 : 1088))) + 2 : 1),
      ldo(s.mode == 3 ? s.n / 2 : s.n), a(size_t(ROWS) * s.k / 2), as(size_t(ROWS) * s.k / 16),
      w(size_t(copies) * (s.n / 32) * (s.k / 128) * (wf == 6 ? 2048 : 1024)),
      scales(size_t(copies) * (s.n / 32) * (s.k / 128) * (wf == 6 ? 128 : 64)),
      ag(ROWS), rs(s.n), out(size_t(ROWS) * ldo), wb(w.n / copies), sb(scales.n / copies) {
    if (o.cold && o.launches < copies) fail("cold schedule requires at least " + std::to_string(copies) + " launches for " + s.name);
    std::vector<uint8_t> av(a.n), asc(as.n), weights(w.n), ws(scales.n);
    for (size_t i = 0; i < av.size(); ++i) av[i] = hash32(uint32_t(i) ^ 41);
    for (size_t i = 0; i < asc.size(); ++i) asc[i] = 0x30 + hash32(uint32_t(i) ^ 43) % 8;
    const size_t chunks = size_t(s.n / 32) * (s.k / 128);
    for (int c = 0; c < copies; ++c) for (size_t b = 0; b < chunks; ++b) {
      uint8_t packed[1024], sc[64] = {};
      for (size_t j = 0; j < 1024; ++j) packed[j] = hash32(uint32_t(b * 1024 + j) ^ (47u + c));
      // WF2 reserves 64 bytes per chunk but consumes its first 32 scale bytes.
      for (size_t j = 0; j < 32; ++j) sc[j] = 0x30 + hash32(uint32_t(b * 32 + j) ^ (53u + c)) % 8;
      if (wf == 6) {
        expand_ternary_chunk(packed, weights.data() + c * wb + b * 2048);
        expand_ternary_scales(sc, ws.data() + c * sb + b * 128);
      } else {
        std::memcpy(weights.data() + c * wb + b * 1024, packed, 1024);
        std::memcpy(ws.data() + c * sb + b * 64, sc, 64);
      }
    }
    a.upload(av); as.upload(asc); w.upload(weights); scales.upload(ws);
    ag.upload(std::vector<float>(ROWS, 1.f)); rs.upload(std::vector<float>(s.n, 1e-3f));
  }
  void reset(cudaStream_t st) {
    CUDA(cudaMemsetAsync(out.p, shape.mode == 1 ? 0 : 0xff, out.n * sizeof(float), st));
  }
  void launch(int ci, int it, cudaStream_t st) {
    const int c = it % copies;
    if (shape.mode == 3) launch_gemm_fp4_silu(ci, a.p, as.p, ag.p, ROWS, shape.k, w.p + c * wb,
        scales.p + c * sb, shape.n, out.p, ldo, st, rs.p, wf);
    else launch_gemm_fp4_cfg(ci, 1, a.p, as.p, ag.p, ROWS, shape.k, w.p + c * wb,
        scales.p + c * sb, shape.n, out.p, ldo, shape.mode == 1, st, rs.p, wf);
    CUDA(cudaGetLastError());
  }
};
enum class Case { Gemm, Gdn, SerialGemmFirst, SerialGdnFirst, ConcurrentGemmFirst, ConcurrentGdnFirst };
static constexpr std::array<Case, 6> CASES{Case::Gemm, Case::Gdn, Case::SerialGemmFirst, Case::SerialGdnFirst, Case::ConcurrentGemmFirst, Case::ConcurrentGdnFirst};
static const char* name(Case c) {
  switch (c) {
    case Case::Gemm: return "gemm"; case Case::Gdn: return "gdn";
    case Case::SerialGemmFirst: return "serial_gemm_first"; case Case::SerialGdnFirst: return "serial_gdn_first";
    case Case::ConcurrentGemmFirst: return "concurrent_gemm_first"; case Case::ConcurrentGdnFirst: return "concurrent_gdn_first";
  }
  abort();
}
struct Graph {
  cudaGraph_t graph; cudaGraphExec_t exec;
  Graph(Case c, int ci, int count, GemmWork& gm, GdnWork& gd, Stream& control, Stream& other) {
    Event fork, join;
    CUDA(cudaStreamBeginCapture(control.p, cudaStreamCaptureModeGlobal));
    if (c == Case::ConcurrentGemmFirst || c == Case::ConcurrentGdnFirst) {
      CUDA(cudaEventRecord(fork.p, control.p)); CUDA(cudaStreamWaitEvent(other.p, fork.p, 0));
      const bool gf = c == Case::ConcurrentGemmFirst;
      for (int i = 0; i < count; ++i) {
        if (gf) gm.launch(ci, i, control.p); else gd.launch(control.p);
      }
      for (int i = 0; i < count; ++i) {
        if (gf) gd.launch(other.p); else gm.launch(ci, i, other.p);
      }
      CUDA(cudaEventRecord(join.p, other.p)); CUDA(cudaStreamWaitEvent(control.p, join.p, 0));
    } else for (int i = 0; i < count; ++i) {
      switch (c) {
        case Case::Gemm: gm.launch(ci, i, control.p); break;
        case Case::Gdn: gd.launch(control.p); break;
        case Case::SerialGemmFirst: gm.launch(ci, i, control.p); gd.launch(control.p); break;
        case Case::SerialGdnFirst: gd.launch(control.p); gm.launch(ci, i, control.p); break;
        case Case::ConcurrentGemmFirst: case Case::ConcurrentGdnFirst: abort();
      }
    }
    CUDA(cudaStreamEndCapture(control.p, &graph)); CUDA(cudaGraphInstantiate(&exec, graph, 0));
  }
  ~Graph() { CUDA(cudaGraphExecDestroy(exec)); CUDA(cudaGraphDestroy(graph)); }
  float run(GemmWork& gm, GdnWork& gd, Stream& control) {
    gm.reset(control.p); gd.reset(control.p);
    CUDA(cudaStreamSynchronize(control.p));
    Event begin, end; CUDA(cudaEventRecord(begin.p, control.p));
    CUDA(cudaGraphLaunch(exec, control.p)); CUDA(cudaEventRecord(end.p, control.p));
    CUDA(cudaEventSynchronize(end.p));
    float ms; CUDA(cudaEventElapsedTime(&ms, begin.p, end.p)); return ms;
  }
};
static uint64_t digest(const void* ptr, size_t n) {
  const auto p = static_cast<const uint8_t*>(ptr); uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; } return h;
}
template<class T> static double value(T x) { return double(x); }
template<> double value(__nv_bfloat16 x) { return __bfloat162float(x); }
template<class T> static bool compare(const std::vector<T>& ref, const std::vector<T>& v, const char* what, Case c) {
  size_t unequal = 0, bad = 0; double max_abs = 0, sum = 0, squares = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    const double x = value(v[i]), r = value(ref[i]);
    if (!std::isfinite(x) || !std::isfinite(r)) ++bad;
    else { max_abs = std::max(max_abs, std::abs(x - r)); sum += x; squares += x * x; }
    unequal += std::memcmp(&v[i], &ref[i], sizeof(T)) != 0;
  }
  printf("CHECK case=%s output=%s elements=%zu unequal=%zu nonfinite=%zu max_abs=%.9g sum=%.17g sumsq=%.17g fnv1a64=%016llx\n",
      name(c), what, v.size(), unequal, bad, max_abs, sum, squares, (unsigned long long)digest(v.data(), v.size() * sizeof(T)));
  return !bad && !unequal && squares > 0;
}
static void resources(const char* label, cudaFuncAttributes at, int active, dim3 grid, int threads, size_t dynamic) {
  printf("RESOURCE kernel=%s grid=%u,%u,%u ctas=%llu threads=%d regs_per_thread=%d static_smem_B=%zu dynamic_smem_B=%zu local_B_per_thread=%zu isolated_active_ctas_per_sm=%d\n",
      label, grid.x, grid.y, grid.z, (unsigned long long)grid.x * grid.y * grid.z, threads,
      at.numRegs, at.sharedSizeBytes, dynamic, at.localSizeBytes, active);
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); const size_t i = v.size() / 2; return v.size() % 2 ? v[i] : (v[i - 1] + v[i]) / 2; }
int main(int argc, char** argv) {
  const Options o = options(argc, argv);
  // These schedules deliberately measure ordinary stream serialization, not PDL sibling overlap.
  const std::pair<const char*, const char*> settings[] = {{"H5_PDL", "0"}, {"H5_CARVEOUT", "none"}, {"H5_CARVEOUT_SCOPE", "all"},
      {"H5_GDN_CG", "4"}, {"H5_GDN_CG_SEGS", "0"}, {"H5_GDN_OCC", "2"}, {"H5_GDN_SCALED", "0"}, {"H5_GDN_PIPE", "0"}, {"H5_GDN_ARITH", "fp32"}};
  for (const auto& p : settings) if (setenv(p.first, p.second, 1)) fail("setenv failed");
  int device; CUDA(cudaGetDevice(&device)); cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop, device));
  if (prop.major != 12 || prop.minor != 0) fail("device must be sm_120 (sm_120a binary)");
  printf("DEVICE name=%s SMs=%d l2_B=%d shared_B_per_sm=%zu regs_per_sm=%d concurrent_kernels=%d\n", prop.name,
      prop.multiProcessorCount, prop.l2CacheSize, prop.sharedMemPerMultiprocessor, prop.regsPerMultiprocessor, prop.concurrentKernels);
  printf("WORK wf=%d slots=%d rows_per_slot=%d rows=%d launches=%d repeats=%d weights=%s split_k=1 CG=4 HK=%d HV=%d SS=%d NGDN=%d gl=0 keep=0 commit=0 PDL=0 arithmetic=fp32\n",
      o.wf, SLOTS, ROWS_PER_SLOT, ROWS, o.launches, o.repeats, o.cold ? "cold" : "hot", HK, HV, SS, NGDN);
  Stream control, other; GdnWork gd;
  cudaFuncAttributes gat{}; int gactive;
  CUDA(cudaFuncGetAttributes(&gat, k_gdn<__nv_bfloat16, 4>));
  CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gactive, k_gdn<__nv_bfloat16, 4>, 256, 0));
  resources("gdn_bf16_CG4", gat, gactive, dim3(2 * HV, SLOTS), 256, 0);
  const uint64_t gstate = STATE_N * sizeof(__nv_bfloat16) * 2;
  const uint64_t ginput = uint64_t(ROWS) * (CONV_CH + 2 * HV) * sizeof(float);
  const uint64_t gy = uint64_t(ROWS) * VDIM * sizeof(float);
  printf("GDN_COST per_launch_state_read_write_B=%llu unique_conv_gates_B=%llu output_B=%llu state_updates=%llu state_update_FMA=%llu state_dot_FMA=%llu new_tokens=%d state_allocation_B=%llu pending_allocation_B=%llu\n",
      (unsigned long long)gstate, (unsigned long long)ginput, (unsigned long long)gy,
      (unsigned long long)STATE_N * ROWS_PER_SLOT, (unsigned long long)STATE_N * ROWS_PER_SLOT,
      (unsigned long long)STATE_N * ROWS_PER_SLOT * 2, ROWS,
      (unsigned long long)gd.state.n * sizeof(__nv_bfloat16), (unsigned long long)gd.pending.n * sizeof(__nv_bfloat16));
  size_t completed = 0;
  for (const Shape& s : SHAPES) {
    if (o.shape != "all" && o.shape != s.name) continue;
    GemmWork gm(s, o, prop.l2CacheSize);
    printf("GEMM_COST shape=%s N=%d K=%d R=%d mode=%d weight_copies=%d weights_B_per_copy=%zu scales_B_per_copy=%zu weight_working_set_B=%zu unique_activation_B=%zu ag_B=%zu rs_B=%zu output_write_B=%zu output_read_B=%zu MAC_per_launch=%llu\n",
        s.name, s.n, s.k, ROWS, s.mode, gm.copies, gm.wb, gm.sb, gm.w.n + gm.scales.n, gm.a.n + gm.as.n,
        gm.ag.n * 4, gm.rs.n * 4, gm.out.n * 4, s.mode == 1 ? gm.out.n * 4 : 0,
        (unsigned long long)ROWS * s.n * s.k);
    for (int ci = 0; ci < fp4_ncfg(); ++ci) {
      if (o.cfg != -1 && o.cfg != ci) continue;
      if (s.mode == 3 && !fp4_cfg_wg2(ci)) { printf("SKIP shape=%s cfg=%d reason=fused_GU_requires_wg2\n", s.name, ci); continue; }
      cudaFuncAttributes at{}; int active;
      CUDA(fp4_profile_resources(ci, s.mode, o.wf, &at, &active));
      if (!active) { printf("SKIP shape=%s cfg=%d reason=unlaunchable\n", s.name, ci); continue; }
      const std::string cfg = fp4_cfg_name(ci, 1); int fw, wg, tok, ks, pipe;
      if (sscanf(cfg.c_str(), "fw%d wg%d tok%d ks%d pipe%d", &fw, &wg, &tok, &ks, &pipe) != 5) fail("unknown engine configuration name");
      printf("CONFIG shape=%s cfg=%d name='%s'\n", s.name, ci, cfg.c_str());
      resources("gemm", at, active, dim3((ROWS + tok - 1) / tok, (s.n / 32 + fw * wg - 1) / (fw * wg)),
          (fw + 1) * 32, at.maxDynamicSharedSizeBytes);
      std::array<std::vector<float>, 6> times;
      std::array<Graph*, 6> graphs{};
      for (size_t i = 0; i < CASES.size(); ++i) graphs[i] = new Graph(CASES[i], ci, o.launches, gm, gd, control, other);
      std::vector<float> gemm_ref, y_ref; std::vector<__nv_bfloat16> state_ref;
      bool valid = true;
      // Initial executions are warmups and full elementwise correctness checks, not timed samples.
      for (size_t i = 0; i < CASES.size(); ++i) {
        graphs[i]->run(gm, gd, control);
        if (CASES[i] == Case::Gemm) { gemm_ref = gm.out.download(); valid &= compare(gemm_ref, gemm_ref, "gemm", CASES[i]); }
        else if (CASES[i] == Case::Gdn) {
          y_ref = gd.y.download(); state_ref = gd.snapshot();
          valid &= compare(y_ref, y_ref, "gdn_y", CASES[i]); valid &= compare(state_ref, state_ref, "gdn_state", CASES[i]);
        } else {
          valid &= compare(gemm_ref, gm.out.download(), "gemm", CASES[i]);
          valid &= compare(y_ref, gd.y.download(), "gdn_y", CASES[i]);
          valid &= compare(state_ref, gd.snapshot(), "gdn_state", CASES[i]);
        }
      }
      if (!valid) { fprintf(stderr, "CORRECTNESS_FAILED shape=%s cfg=%d\n", s.name, ci); return 4; }
      for (int rep = 0; rep < o.repeats; ++rep) for (size_t j = 0; j < CASES.size(); ++j) {
        const size_t i = (j + size_t(rep)) % CASES.size();
        times[i].push_back(graphs[i]->run(gm, gd, control));
      }
      const double g = median(times[0]), d = median(times[1]);
      for (size_t i = 0; i < CASES.size(); ++i) {
        const double m = median(times[i]);
        printf("TIME shape=%s wf=%d cfg=%d case=%s total_median_ms=%.6f min_ms=%.6f max_ms=%.6f us_per_launch_set=%.6f gemm_launches=%d gdn_launches=%d speedup_vs_isolated_sum=%.6f time_vs_isolated_max=%.6f\n",
            s.name, o.wf, ci, name(CASES[i]), m, *std::min_element(times[i].begin(), times[i].end()), *std::max_element(times[i].begin(), times[i].end()),
            m * 1000 / o.launches, CASES[i] == Case::Gdn ? 0 : o.launches, CASES[i] == Case::Gemm ? 0 : o.launches,
            i < 2 ? 1 : (g + d) / m, i < 2 ? 1 : m / std::max(g, d));
      }
      for (size_t i = 4; i < CASES.size(); ++i) {
        const double serial = median(times[i - 2]), concurrent = median(times[i]);
        printf("OVERLAP shape=%s wf=%d cfg=%d case=%s speedup_vs_same_order_serial=%.6f saved_us_per_pair=%.6f helps=%s\n",
            s.name, o.wf, ci, name(CASES[i]), serial / concurrent, (serial - concurrent) * 1000 / o.launches, concurrent < serial ? "yes" : "no");
      }
      for (auto p : graphs) delete p;
      ++completed; fflush(stdout);
    }
  }
  if (!completed) fail("no launchable selected configuration");
  printf("PASS configurations=%zu exact_unsplit_gemm_and_GDN_output_state_comparison=1\n", completed);
}
