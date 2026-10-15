#include "quad13.cuh"
#include "compact.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace h5 {
int fp4_ncfg();
const char* fp4_cfg_name(int, int);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*,
                        int, int, const void*, const void*, int, float*, int,
                        bool, cudaStream_t, const float*, int);
}
[[noreturn]] static void fail(const char* category, const char* detail) {
  fprintf(stderr, "error[%s]: %s\n", category, detail); exit(1);
}
static void require(bool ok, const char* detail) { if (!ok) fail("contract", detail); }
static void check(cudaError_t e) { if (e != cudaSuccess) fail("cuda", cudaGetErrorString(e)); }
static uint32_t next(uint32_t& state) { return state = state * 1664525u + 1013904223u; }
static uint64_t hash_bytes(const void* data, size_t bytes) {
  auto p = static_cast<const uint8_t*>(data);
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
template<class T> static uint64_t hash(const std::vector<T>& v) {
  return hash_bytes(v.data(), v.size() * sizeof(T));
}
template<class T> static void read_exact(const char* path, std::vector<T>& v) {
  FILE* f = fopen(path, "rb"); if (!f) fail("input", path);
  size_t got = fread(v.data(), sizeof(T), v.size(), f);
  int extra = fgetc(f); bool closed = fclose(f) == 0;
  require(got == v.size() && extra == EOF && closed, "input file length/read mismatch");
}
template<class T> static void save_new(const std::string& path, const std::vector<T>& v) {
  FILE* f = fopen(path.c_str(), "wbx"); if (!f) fail("output", path.c_str());
  size_t written = fwrite(v.data(), sizeof(T), v.size(), f);
  bool closed = fclose(f) == 0;
  require(written == v.size() && closed, "output write failed");
}
static uint8_t label(const uint8_t* chunk, int lane, int b) {
  return chunk[b < 24 ? (b / 4) * 128 + lane * 4 + b % 4 : 768 + lane * 2 + b - 24];
}
static void decode_dense_chunk(const uint8_t* chunk, int8_t (&rows)[32][128]) {
  for (int lane = 0; lane < 32; ++lane) {
    int8_t stream[128];
    for (int b = 0; b < 26; ++b) {
      unsigned c = label(chunk, lane, b);
      require(c < 243, "dense5 label outside legal five-trit range");
      for (int j = 0; j < 5 && 5 * b + j < 128; ++j, c /= 3)
        stream[5 * b + j] = int(c % 3) - 1;
    }
    for (int w = 0; w < 8; ++w) {
      int m = w / 4, h = (w / 2) & 1, rh = w & 1;
      int row = 16 * m + 8 * rh + lane / 4;
      for (int j = 0; j < 8; ++j) {
        rows[row][64 * h + 8 * (lane & 3) + j] = stream[w * 16 + j];
        rows[row][64 * h + 32 + 8 * (lane & 3) + j] = stream[w * 16 + 8 + j];
      }
    }
  }
}
static void encode_dense_chunk(const int8_t (&rows)[32][128], uint8_t* chunk) {
  for (int lane = 0; lane < 32; ++lane) {
    int8_t stream[128];
    for (int w = 0; w < 8; ++w) {
      int row = 16 * (w / 4) + 8 * (w & 1) + lane / 4, h = (w / 2) & 1;
      for (int j = 0; j < 8; ++j) {
        stream[w * 16 + j] = rows[row][64 * h + 8 * (lane & 3) + j];
        stream[w * 16 + 8 + j] = rows[row][64 * h + 32 + 8 * (lane & 3) + j];
      }
    }
    dense5::pack_lane(chunk, lane, stream);
  }
}
struct Fixture {
  int R, N, K, nkb;
  std::vector<uint8_t> dense_w, a, as, w;
  std::vector<uint16_t> dense_s, s;
  std::vector<float> ag;
  Fixture(int r, int n, int k): R(r), N(n), K(k), nkb(k / 128),
    dense_w(size_t(n) * nkb * 26), a(size_t(nkb) * 64 * 64),
    as(size_t(nkb) * 64 * 8), w(dense_w.size()),
    dense_s(size_t(n) * nkb), s(dense_s.size()), ag(64) {}
};
static void synthetic(Fixture& f) {
  uint32_t rng = 17;
  for (int kb = 0; kb < f.nkb; ++kb)
    for (int g = 0; g < f.N / 32; ++g) {
      int8_t rows[32][128];
      for (auto& row : rows) for (auto& t : row) t = int((next(rng) >> 16) % 3) - 1;
      encode_dense_chunk(rows, f.dense_w.data() + (size_t(kb) * (f.N / 32) + g) * 832);
      for (int gid = 0; gid < 8; ++gid) for (int q = 0; q < 4; ++q) {
        __half h = __float2half(0.125f + float((kb + g + gid + q) % 11) / 32);
        memcpy(&f.dense_s[(size_t(kb) * (f.N / 32) + g) * 32 + gid * 4 + q], &h, 2);
      }
    }
  for (auto& b : f.a) b = uint8_t(next(rng) >> 16);
  for (auto& b : f.as) b = uint8_t(0x28 + ((next(rng) >> 16) % 33));
  for (int r = 0; r < 64; ++r) f.ag[r] = 0.25f + float(r % 7) / 8;
}
static float half_value(uint16_t bits) {
  __half h; memcpy(&h, &bits, 2); return __half2float(h);
}
static void validate_acts(const Fixture& f) {
  for (auto b : f.as) require(b <= 0x7e, "activation local scale not finite nonnegative E4M3");
  for (float g : f.ag) require(std::isfinite(g), "activation global scale not finite");
  for (auto b : f.dense_s) require(std::isfinite(half_value(b)), "weight scale not finite FP16");
}
static void repack(Fixture& f) {
  for (int kb = 0; kb < f.nkb; ++kb)
    for (int g = 0; g < f.N / 32; ++g) {
      int8_t rows[32][128];
      decode_dense_chunk(f.dense_w.data() + (size_t(kb) * (f.N / 32) + g) * 832, rows);
      for (int row = 0; row < 32; ++row) {
        uint8_t bytes[26]; quad13::encode_block(rows[row], bytes);
        int n = g * 32 + row;
        for (int b = 0; b < 26; ++b) f.w[(size_t(kb) * 26 + b) * f.N + n] = bytes[b];
        int source = (row % 8) * 4 + (row / 16) * 2 + ((row / 8) & 1);
        f.s[size_t(kb) * f.N + n] = f.dense_s[(size_t(kb) * (f.N / 32) + g) * 32 + source];
      }
    }
}
static void test_decode() {
  for (unsigned b = 0; b <= 0x7e; ++b) {
    int exponent = int(b >> 3), fraction = int(b & 7);
    double expected = exponent == 0 ? std::ldexp(double(fraction), -9) : std::ldexp(1. + fraction / 8., exponent - 7);
    require(double(quad13::local_scale(uint8_t(b))) == expected, "E4M3 finite scale dictionary differs");
  }
  size_t checks = 0;
  for (unsigned z = 0; z < 8192; ++z) {
    int8_t t[8]; bool valid = quad13::decode_pair(z, t);
    require(valid == (z < 6561), "reserved pair accepted/legal pair rejected");
    if (!valid) continue;
    require(quad13::encode_quad(t) + 81 * quad13::encode_quad(t + 4) == z,
            "pair encoding is not inverse");
    for (int basis = 0; basis < 9; ++basis) {
      int64_t x[8] = {};
      if (basis < 8) x[basis] = 1;
      else for (int j = 0; j < 8; ++j) x[j] = (j & 1 ? -1 : 1) * (j + 1) * 1234567;
      int64_t reference = 0;
      for (int j = 0; j < 8; ++j) reference += t[j] * x[j];
      unsigned c0 = z % 81, c1 = z / 81;
      int64_t mapped = quad13::rep_sign(c0) * quad13::quad_sum(40 + quad13::rep_index(c0), x) +
                       quad13::rep_sign(c1) * quad13::quad_sum(40 + quad13::rep_index(c1), x + 4);
      require(mapped == reference, "signed-table linear coefficients differ"); ++checks;
    }
    for (int p = 0; p < 16; ++p) {
      int8_t block[128] = {};
      memcpy(block + p * 8, t, 8);
      uint8_t bytes[26]; quad13::encode_block(block, bytes);
      require(quad13::pair_at(bytes, p) == z, "13-bit bitstream boundary mismatch");
      for (int other = 0; other < 16; ++other)
        if (p != other) require(quad13::pair_at(bytes, other) == 3280, "pair damaged adjacent bits");
    }
  }
  printf("cpu_pair_proof legal=6561 reserved=1631 basis_and_integer_checks=%zu bit_positions=16 PASS\n", checks);
}
static void test_layout() {
  Fixture f(8, 64, 256); synthetic(f);
  uint64_t before_w = hash(f.dense_w), before_s = hash(f.dense_s); repack(f);
  for (int kb = 0; kb < f.nkb; ++kb) for (int g = 0; g < f.N / 32; ++g) {
    int8_t rows[32][128]; const uint8_t* chunk = f.dense_w.data() + (kb * (f.N / 32) + g) * 832;
    decode_dense_chunk(chunk, rows);
    uint8_t rebuilt[832]; encode_dense_chunk(rows, rebuilt);
    require(memcmp(chunk, rebuilt, 832) == 0, "dense5 canonical byte roundtrip failed");
    for (int row = 0; row < 32; ++row) {
      int n = 32 * g + row; uint8_t bytes[26];
      for (int b = 0; b < 26; ++b) bytes[b] = f.w[(kb * 26 + b) * f.N + n];
      int8_t got[128];
      for (int p = 0; p < 16; ++p) require(quad13::decode_pair(quad13::pair_at(bytes, p), got + p * 8), "packed reserved code");
      require(memcmp(got, rows[row], 128) == 0, "row/scale boundary repack differs");
      int sc = (row % 8) * 4 + (row / 16) * 2 + ((row / 8) & 1);
      require(f.s[kb * f.N + n] == f.dense_s[(kb * (f.N / 32) + g) * 32 + sc], "scale bits changed");
      for (int r = 0; r < 8; ++r) {
        int64_t xs[128];
        for (int k = 0; k < 128; ++k) xs[k] = ((r + k + kb) % 17 - 8) * 3;
        int64_t ref = 0, mapped = 0;
        for (int k = 0; k < 128; ++k) ref += rows[row][k] * xs[k];
        for (int q = 0; q < 32; ++q) {
          unsigned z = quad13::pair_at(bytes, q / 2), c = q & 1 ? z / 81 : z % 81;
          mapped += quad13::rep_sign(c) * quad13::quad_sum(40 + quad13::rep_index(c), xs + 4 * q);
        }
        require(ref * (n + kb + 1) == mapped * (n + kb + 1), "block-scaled integer arithmetic differs");
      }
    }
  }
  for (int r = 0; r < 8; ++r) for (int k = 0; k < f.K; ++k) {
    int kb = k / 128, kk = k % 128, c = kk / 32;
    size_t base = (size_t(kb) * 64 + r);
    int location = ((c ^ ((r >> 1) & 3)) * 16 + (kk % 32) / 2);
    unsigned nib = (f.a[base * 64 + location] >> (4 * (kk & 1))) & 15;
    const double magnitudes[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    double e = (nib & 8 ? -1 : 1) * magnitudes[nib & 7];
    uint8_t sb = f.as[base * 8 + kk / 16];
    int exponent = (sb >> 3) & 15, fraction = sb & 7;
    double sv = exponent == 0 ? std::ldexp(double(fraction), -9) : std::ldexp(1. + fraction / 8., exponent - 7);
    double expected = e * sv * f.ag[r];
    require(double(quad13::activation(f.a.data(), f.as.data(), f.ag.data(), f.nkb, r, k)) == expected,
            "activation swizzle/E2M1/E4M3 reconstruction differs");
  }
  require(hash(f.dense_w) == before_w && hash(f.dense_s) == before_s, "original bytes mutated");
  printf("cpu_layout_proof rows=64 K=256 live_rows=8 scale_bits=preserved original_bytes=unchanged PASS\n");
}
static void prove_repack(const Fixture& f) {
  size_t trits = 0;
  for (int kb = 0; kb < f.nkb; ++kb) for (int g = 0; g < f.N / 32; ++g) {
    int8_t rows[32][128];
    decode_dense_chunk(f.dense_w.data() + (size_t(kb) * (f.N / 32) + g) * 832, rows);
    for (int row = 0; row < 32; ++row) {
      int n = 32 * g + row; uint8_t bytes[26];
      for (int b = 0; b < 26; ++b) bytes[b] = f.w[(size_t(kb) * 26 + b) * f.N + n];
      int8_t t[128];
      for (int p = 0; p < 16; ++p)
        require(quad13::decode_pair(quad13::pair_at(bytes, p), t + 8 * p), "repacked reserved code");
      require(memcmp(t, rows[row], 128) == 0, "full fixture ternary repack differs");
      int sc = (row % 8) * 4 + (row / 16) * 2 + ((row / 8) & 1);
      require(f.s[size_t(kb) * f.N + n] == f.dense_s[(size_t(kb) * (f.N / 32) + g) * 32 + sc], "full fixture scale bits differ");
      trits += 128;
    }
  }
  printf("cpu_fixture_proof trits=%zu label_bytes=%zu scale_bytes=%zu PASS\n", trits, f.w.size(), f.s.size() * 2);
}
static double real_activation(const Fixture& f, int r, int k) {
  int kk = k % 128; size_t base = size_t(k / 128) * 64 + r;
  int byte = kk / 2, offset = ((byte / 16) ^ ((r >> 1) & 3)) * 16 + byte % 16;
  unsigned c = (f.a[base * 64 + offset] >> (4 * (kk & 1))) & 15;
  uint8_t sb = f.as[base * 8 + kk / 16];
  int exponent = (sb >> 3) & 15, fraction = sb & 7;
  double scale = exponent == 0 ? std::ldexp(double(fraction), -9) : std::ldexp(1. + fraction / 8., exponent - 7);
  const double values[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
  return (c & 8 ? -values[c & 7] : values[c & 7]) * scale * double(f.ag[r]);
}
static std::vector<double> reference(const Fixture& f) {
  std::vector<double> out(size_t(f.R) * f.N, 0), xs(size_t(f.R) * f.K);
  for (int r = 0; r < f.R; ++r) for (int k = 0; k < f.K; ++k)
    xs[size_t(r) * f.K + k] = real_activation(f, r, k);
  for (int kb = 0; kb < f.nkb; ++kb) for (int g = 0; g < f.N / 32; ++g) {
    int8_t rows[32][128]; decode_dense_chunk(f.dense_w.data() + (size_t(kb) * (f.N / 32) + g) * 832, rows);
    for (int row = 0; row < 32; ++row) {
      int n = 32 * g + row, sc = (row % 8) * 4 + (row / 16) * 2 + ((row / 8) & 1);
      double ws = half_value(f.dense_s[(size_t(kb) * (f.N / 32) + g) * 32 + sc]);
      for (int r = 0; r < f.R; ++r) {
        double dot = 0; const double* x = xs.data() + size_t(r) * f.K + kb * 128;
        for (int k = 0; k < 128; ++k) dot += rows[row][k] * x[k];
        out[size_t(r) * f.N + n] += ws * dot;
      }
    }
  }
  return out;
}
template<class T> static T* upload(const std::vector<T>& v) {
  T* p = nullptr; check(cudaMalloc(&p, v.size() * sizeof(T)));
  check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
template<class T> static std::vector<T> download(const T* p, size_t count) {
  std::vector<T> v(count); check(cudaMemcpy(v.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
using Kernel = void(*)(const uint8_t*, const uint16_t*, int, int, const uint8_t*, const uint8_t*, const float*, const float*, int, float*);
template<int R> static Kernel select_kernel(quad13::Residency mode) {
  switch (mode) {
    case quad13::Residency::direct: return quad13::contract<R, quad13::Residency::direct>;
    case quad13::Residency::staged: return quad13::contract<R, quad13::Residency::staged>;
    case quad13::Residency::local: return quad13::contract<R, quad13::Residency::local>;
  }
  fail("arguments", "unrecognized residency");
}
static double measure(const char* name, const std::function<void()>& launch, cudaStream_t st, int iters) {
  launch(); launch(); check(cudaStreamSynchronize(st));
  cudaGraph_t graph; cudaGraphExec_t exec;
  check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < iters; ++i) launch();
  check(cudaStreamEndCapture(st, &graph)); check(cudaGraphInstantiate(&exec, graph, 0));
  check(cudaGraphLaunch(exec, st)); check(cudaStreamSynchronize(st));
  cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
  check(cudaEventRecord(begin, st));
  for (int i = 0; i < 5; ++i) check(cudaGraphLaunch(exec, st));
  check(cudaEventRecord(end, st)); check(cudaEventSynchronize(end));
  float ms; check(cudaEventElapsedTime(&ms, begin, end));
  double us = double(ms) * 1000 / (5 * iters);
  printf("timing stage=%s us=%.6f graph_instances=%d\n", name, us, 5 * iters);
  check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
  check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); return us;
}
template<class T> static void compare(const char* label, const std::vector<float>& a, const std::vector<T>& b) {
  double max_abs = 0, max_rel = 0, sq = 0; size_t changed = 0, nonfinite = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++nonfinite; continue; }
    double d = std::abs(double(a[i]) - double(b[i]));
    max_abs = std::max(max_abs, d); max_rel = std::max(max_rel, d / std::max(1e-12, std::abs(double(b[i]))));
    sq += d * d; changed += double(a[i]) != double(b[i]);
  }
  printf("numerics comparison=%s outputs=%zu changed=%zu nonfinite=%zu max_abs=%.9g max_relative=%.9g rmse=%.9g\n",
         label, a.size(), changed, nonfinite, max_abs, max_rel, std::sqrt(sq / a.size()));
  require(nonfinite == 0, "nonfinite benchmark outputs");
}
static int number(const char* p) {
  char* end; long v = strtol(p, &end, 10);
  if (!*p || *end || v < 0 || v > 1000000) fail("arguments", "invalid nonnegative integer");
  return int(v);
}
static void usage() {
  fprintf(stderr, "usage: quad13 cpu-test\n"
    "       quad13 cpu-fixture R N K synthetic\n"
    "       quad13 cpu-fixture R N K dense5 W S A AS AG\n"
    "       quad13 bench R N K direct|staged|local F SPLITS STRIDE ITERS BASE_CFG synthetic new OUTPUT_PREFIX\n"
    "       quad13 bench R N K direct|staged|local F SPLITS STRIDE ITERS BASE_CFG dense5 W S A AS AG new OUTPUT_PREFIX\n"
    "R=1|8, N%%32=0, K%%128=0; F=64|128|256; STRIDE=41|48|64; ITERS=1..256\n");
}
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "cpu-test") { test_decode(); test_layout(); return 0; }
  if ((argc == 6 || argc == 11) && std::string(argv[1]) == "cpu-fixture") {
    int R = number(argv[2]), N = number(argv[3]), K = number(argv[4]);
    require((R == 1 || R == 8) && N > 0 && N % 32 == 0 && K > 0 && K % 128 == 0 && size_t(N) * K <= 200000000,
            "unsupported/beyond-bound CPU fixture shape");
    std::string source = argv[5];
    require((source == "synthetic" && argc == 6) || (source == "dense5" && argc == 11), "CPU fixture source/arity mismatch");
    test_decode(); test_layout(); Fixture f(R, N, K);
    auto start = std::chrono::steady_clock::now();
    if (source == "synthetic") synthetic(f);
    else {
      read_exact(argv[6], f.dense_w); read_exact(argv[7], f.dense_s);
      read_exact(argv[8], f.a); read_exact(argv[9], f.as); read_exact(argv[10], f.ag);
    }
    validate_acts(f); auto original_w = hash(f.dense_w), original_s = hash(f.dense_s);
    repack(f); prove_repack(f);
    require(hash(f.dense_w) == original_w && hash(f.dense_s) == original_s, "CPU fixture mutated original bytes");
    auto ref = reference(f);
    for (double v : ref) require(std::isfinite(v), "CPU scalar reference not finite");
    printf("cpu_fixture R=%d N=%d K=%d outputs=%zu reference_fnv64=%016llx seconds=%.6f PASS\n",
           R, N, K, ref.size(), (unsigned long long)hash(ref),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return 0;
  }
  if ((argc != 14 && argc != 19) || std::string(argv[1]) != "bench") { usage(); return 2; }
  int R = number(argv[2]), N = number(argv[3]), K = number(argv[4]);
  std::string mode_name = argv[5]; quad13::Residency mode;
  if (mode_name == "direct") mode = quad13::Residency::direct;
  else if (mode_name == "staged") mode = quad13::Residency::staged;
  else if (mode_name == "local") mode = quad13::Residency::local;
  else fail("arguments", "unknown residency");
  int F = number(argv[6]), splits = number(argv[7]), stride = number(argv[8]), iters = number(argv[9]), base = number(argv[10]);
  require((R == 1 || R == 8) && N > 0 && N % 32 == 0 && K > 0 && K % 128 == 0, "unsupported matrix shape");
  require((F == 64 || F == 128 || F == 256) && (stride == 41 || stride == 48 || stride == 64), "unsupported CTA/table geometry");
  require(splits > 0 && splits <= K / 128 && splits <= 16 && iters > 0 && iters <= 256, "unbounded/invalid splits or iterations");
  require(base < h5::fp4_ncfg(), "baseline configuration outside linked fp4 object");
  int fw, wg, tokw, ks, pipe;
  require(sscanf(h5::fp4_cfg_name(base, splits), "fw%d wg%d tok%d ks%d pipe%d", &fw, &wg, &tokw, &ks, &pipe) == 5 && tokw <= 64,
          "baseline must fit the explicit 64-row activation allocation (TOK128 unsupported)");
  require(size_t(N) * K <= 200000000 && size_t(R) * N * splits <= 10000000, "shape exceeds bounded benchmark memory/work contract");
  std::string source = argv[11];
  require((source == "synthetic" && argc == 14) || (source == "dense5" && argc == 19), "input source/arity mismatch");
  const char* prefix = argv[argc - 1];
  // argv[12] is an explicit output policy: new. It prevents accidental overwrite of retained runs.
  require(std::string(argv[argc - 2]) == "new", "output policy must be new");
  test_decode(); test_layout();
  Fixture f(R, N, K);
  if (source == "synthetic") synthetic(f);
  else {
    read_exact(argv[12], f.dense_w); read_exact(argv[13], f.dense_s);
    read_exact(argv[14], f.a); read_exact(argv[15], f.as); read_exact(argv[16], f.ag);
  }
  validate_acts(f);
  uint64_t original_w = hash(f.dense_w), original_s = hash(f.dense_s);
  auto t0 = std::chrono::steady_clock::now(); repack(f); auto t1 = std::chrono::steady_clock::now();
  require(hash(f.dense_w) == original_w && hash(f.dense_s) == original_s, "repack mutated original source bytes");
  prove_repack(f); auto ref = reference(f);
  size_t table_count = size_t(R) * f.nkb * 32 * stride;
  size_t shared = mode == quad13::Residency::direct ? 0 : size_t(R) * 32 * stride * 4;
  size_t ctas = (N + F - 1) / F;
  printf("fixture source=%s R=%d N=%d K=%d mode=%s F=%d splits=%d stride=%d iters=%d baseline_cfg=%d baseline_name=\"%s\"\n",
         source.c_str(), R, N, K, mode_name.c_str(), F, splits, stride, iters, base, h5::fp4_cfg_name(base, splits));
  printf("encoding weight_label_bytes=%zu scale_bytes=%zu original_dense_fnv64=%016llx original_scale_fnv64=%016llx quad_fnv64=%016llx repack_ms=%.6f\n",
         f.w.size(), f.s.size() * 2, (unsigned long long)original_w, (unsigned long long)original_s,
         (unsigned long long)hash(f.w), std::chrono::duration<double, std::milli>(t1 - t0).count());
  printf("bill activation_source_bytes=%zu logical_full_table_bytes=%zu global_table_allocation_bytes=%zu shared_bytes_per_CTA=%zu lookup_count=%zu logical_lookup_bytes=%zu table_build_quads=%zu local_replicated_build_quads=%zu staged_table_bytes=%zu partial_bytes=%zu reduction_read_bytes=%zu output_bytes=%zu\n",
         f.a.size() + f.as.size() + f.ag.size() * 4, table_count * 4,
         mode == quad13::Residency::local ? size_t(0) : table_count * 4, shared,
         size_t(R) * N * K / 4, size_t(R) * N * K,
         mode == quad13::Residency::local ? size_t(0) : size_t(R) * K / 4,
         mode == quad13::Residency::local ? ctas * R * K / 4 : size_t(0),
         mode == quad13::Residency::staged ? ctas * table_count * 4 : size_t(0),
         splits == 1 ? size_t(0) : size_t(splits) * R * N * 4,
         splits == 1 ? size_t(0) : size_t(splits) * R * N * 4, size_t(R) * N * 4);
  Kernel kernel = R == 1 ? select_kernel<1>(mode) : select_kernel<8>(mode);
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
  cudaFuncAttributes attrs; check(cudaFuncGetAttributes(&attrs, kernel));
  int active; check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, F, shared));
  printf("resources registers=%d local_bytes=%zu static_shared=%zu dynamic_shared=%zu active_CTAs_per_SM=%d\n",
         attrs.numRegs, attrs.localSizeBytes, attrs.sharedSizeBytes, shared, active);
  auto W = upload(f.w); auto S = upload(f.s); auto DW = upload(f.dense_w); auto DS = upload(f.dense_s);
  auto A = upload(f.a); auto AS = upload(f.as); auto AG = upload(f.ag);
  float *tables = nullptr, *partial = nullptr, *out = nullptr, *native = nullptr;
  if (mode != quad13::Residency::local) check(cudaMalloc(&tables, table_count * 4));
  if (splits != 1) check(cudaMalloc(&partial, size_t(splits) * R * N * 4));
  check(cudaMalloc(&out, size_t(R) * N * 4)); check(cudaMalloc(&native, size_t(R) * N * 4));
  cudaStream_t st; check(cudaStreamCreate(&st));
  auto prep = [&]() {
    quad13::prepare_tables<<<(R * f.nkb * 32 + 127) / 128, 128, 0, st>>>(A, AS, AG, f.nkb, R, stride, tables);
    check(cudaGetLastError());
  };
  auto contraction = [&]() {
    kernel<<<dim3(unsigned(ctas), unsigned(splits)), F, shared, st>>>(W, S, N, f.nkb, A, AS, AG, tables, stride, splits == 1 ? out : partial);
    check(cudaGetLastError());
  };
  auto reduction = [&]() {
    quad13::reduce<<<(R * N + 255) / 256, 256, 0, st>>>(partial, out, R * N, splits);
    check(cudaGetLastError());
  };
  auto component = [&]() {
    if (mode != quad13::Residency::local) prep();
    contraction(); if (splits != 1) reduction();
  };
  auto baseline = [&]() {
    h5::launch_gemm_fp4_cfg(base, splits, A, AS, AG, R, K, DW, DS, N, native, N, false, st, nullptr, 3);
    check(cudaGetLastError());
  };
  double prep_us = 0, reduce_us = 0;
  if (mode != quad13::Residency::local) prep_us = measure("prepare_tables", prep, st, iters);
  double contract_us = measure("contract_only", contraction, st, iters);
  if (splits != 1) reduce_us = measure("reduce_only", reduction, st, iters);
  double total_us = measure("quantized_input_to_output", component, st, iters);
  double native_us = measure("native_dense5_quantized_input_to_output", baseline, st, iters);
  auto result = download(out, size_t(R) * N), baseline_result = download(native, size_t(R) * N);
  compare("quad13_vs_scalar_double", result, ref); compare("native_vs_scalar_double", baseline_result, ref);
  compare("quad13_vs_native", result, baseline_result);
  printf("summary total_us=%.6f native_us=%.6f native_over_quad13=%.6f separate_stage_sum_us=%.6f excluded=offline_repack,upload,RMSNorm,quantization,SiLU,residual,full_target\n",
         total_us, native_us, native_us / total_us, prep_us + contract_us + reduce_us);
  save_new(std::string(prefix) + ".quad13.f32", result);
  save_new(std::string(prefix) + ".native.f32", baseline_result);
  save_new(std::string(prefix) + ".ref.f64", ref);
  check(cudaStreamDestroy(st));
  for (void* p : {static_cast<void*>(W), static_cast<void*>(S), static_cast<void*>(DW), static_cast<void*>(DS),
                 static_cast<void*>(A), static_cast<void*>(AS), static_cast<void*>(AG), static_cast<void*>(tables),
                 static_cast<void*>(partial), static_cast<void*>(out), static_cast<void*>(native)})
    if (p) check(cudaFree(p));
  return 0;
}
