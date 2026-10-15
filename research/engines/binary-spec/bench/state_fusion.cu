#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace h5;
namespace h5 {
void launch_conv(const ConvArgs&, int, StateFormat, cudaStream_t);
void launch_gdn(const GdnArgs&, int, StateFormat, cudaStream_t);
void launch_conv_gdn(const ConvArgs&, const GdnArgs&, int, StateFormat, cudaStream_t);
}
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)
template<class T> T* alloc(size_t n) {
  T* p = nullptr;
  if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess || cudaMemset(p, 0, n * sizeof(T)) != cudaSuccess) abort();
  return p;
}
static bool equal(const float* a, const float* b, size_t n, const char* what) {
  std::vector<float> x(n), y(n);
  if (cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) abort();
  float error = 0.f;
  for (size_t i = 0; i < n; i++) {
    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { fprintf(stderr, "%s nonfinite at %zu\n", what, i); return false; }
    error = std::max(error, fabsf(x[i] - y[i]));
    if (fabsf(x[i] - y[i]) > 2e-5f * (1.f + fabsf(x[i]))) {
      fprintf(stderr, "%s mismatch at %zu: %.9g %.9g\n", what, i, x[i], y[i]); return false;
    }
  }
  printf("%s max_abs=%.9g\n", what, error);
  return true;
}
int main(int argc, char** argv) {
  const int pending = argc == 2 ? atoi(argv[1]) : PMAX;
  if (argc > 2 || (pending != 8 && pending != 17 && pending != 33 && pending != 65)) {
    fprintf(stderr, "usage: state-fusion-test [8|17|33|65 pending rows]\n"); return 3;
  }
  const int rows = std::max(21, pending + 9);
  constexpr size_t cs = 3 * CONV_CH, ms = HV * 128 * 128;
  const size_t ps = (size_t)pending * CONV_CH;
  auto proj = alloc<float>(rows * GDN_IN), w = alloc<float>(CONV_CH * 4), ab = alloc<float>(rows * 96);
  auto sa = alloc<float>(HV), dt = alloc<float>(HV);
  auto c0 = alloc<float>(NGDN * cs), c1 = alloc<float>(NGDN * 2 * cs);
  auto cp0 = alloc<float>(NGDN * ps), cp1 = alloc<float>(NGDN * 2 * ps);
  auto m0 = alloc<float>(NGDN * ms), m1 = alloc<float>(NGDN * ms);
  auto p0 = alloc<float>(NGDN * 2 * pending * PEND_GDN), p1 = alloc<float>(NGDN * 2 * pending * PEND_GDN);
  auto co = alloc<float>(rows * CONV_CH), gates = alloc<float>(rows * 96);
  auto y0 = alloc<float>(rows * VDIM), y1 = alloc<float>(rows * VDIM);
  auto seg = alloc<Seg>(1);
  std::vector<float> data(rows * GDN_IN);
  for (size_t i = 0; i < data.size(); i++) data[i] = .15f * sinf(i * .23f);
  CHECK(cudaMemcpy(proj, data.data(), data.size() * 4, cudaMemcpyHostToDevice));
  data.resize(CONV_CH * 4);
  for (size_t i = 0; i < data.size(); i++) data[i] = .1f + .02f * (i % 4);
  CHECK(cudaMemcpy(w, data.data(), data.size() * 4, cudaMemcpyHostToDevice));
  data.assign(HV, -1.f); CHECK(cudaMemcpy(sa, data.data(), HV * 4, cudaMemcpyHostToDevice));
  ConvArgs c{proj, w, c0, 0, co, seg, cp0, ab, sa, dt, gates};
  c.pending_rows = pending;
  ConvArgs cf = c; cf.state = c1; cf.pend = cp1;
  GdnArgs a{co, proj, ab, sa, dt, nullptr, m0, 0, y0, seg, p0, gates};
  a.pending_rows = pending;
  a.replay_delta = getenv("H5_GDN_DELTA") && atoi(getenv("H5_GDN_DELTA"));
  GdnArgs af = a; af.state = m1; af.y = y1; af.pend = p1; af.conv_out = nullptr; af.gates = nullptr;
  int par = 0, bank = 0;
  const int steps[][3] = {{rows,0,0}, {pending,0,1}, {pending,pending/2,1},
      {pending-3,0,1}, {rows,pending-5,0}, {pending,0,1}, {1,pending,0}};
  for (const auto& step : steps) {
    Seg sg{0,0,step[0],step[1],step[2],par,bank};
    CHECK(cudaMemcpy(seg, &sg, sizeof(sg), cudaMemcpyHostToDevice));
    launch_conv(c, 1, StateFormat::F32, nullptr);
    launch_gdn(a, 1, StateFormat::F32, nullptr);
    launch_conv_gdn(cf, af, 1, StateFormat::F32, nullptr);
    CHECK(cudaDeviceSynchronize());
    bank ^= 1;
    if (!equal(c0, c1 + bank * cs, cs, "conv committed") ||
        !equal(m0, m1, ms, "GDN committed") || !equal(y0, y1, step[0] * VDIM, "output")) return 2;
    if (sg.keep) {
      if (!equal(cp0, cp1 + par * ps, step[0] * CONV_CH, "conv pending") ||
          !equal(p0 + par * pending * PEND_GDN, p1 + par * pending * PEND_GDN, step[0] * PEND_GDN, "GDN pending")) return 2;
      par ^= 1;
    }
  }
  printf("state fusion lifecycle passed (cg=%s, delta=%d, pending=%d)\n", getenv("H5_GDN_CG") ? getenv("H5_GDN_CG") : "4", a.replay_delta, pending);
}
