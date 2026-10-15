// Standalone preparation comparison: same binary, graph work and seeded inputs.
#include "../src/kernels.cu"
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>

static void ck(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template <typename T> static T* alloc(size_t count) {
  T* p; ck(cudaMalloc(&p, count * sizeof(T))); return p;
}
static float fp8(uint8_t b) {
  const int exponent = (b >> 3) & 15, mantissa = b & 7;
  return exponent ? std::ldexp(1.f + mantissa / 8.f, exponent - 7) : mantissa / 512.f;
}
static float decoded(const std::vector<uint8_t>& q, const std::vector<uint8_t>& s,
                     const std::vector<float>& g, int row, int n, int element) {
  static const float f4[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const int kb = element / 128, c = element % 128 / 32, r = element % 32, rr = row & 63;
  const size_t base = ((size_t)(row >> 6) * (n / 128) + kb) * 64 + rr;
  const uint8_t pair = q[base * 64 + ((c ^ ((rr >> 1) & 3)) << 4) + r / 2];
  const uint8_t v = (pair >> ((r & 1) * 4)) & 15;
  return (v & 8 ? -1.f : 1.f) * f4[v & 7] * fp8(s[base * 8 + c * 2 + r / 16]) * g[row];
}
int main(int argc, char** argv) { try {
  if (argc != 2) throw std::runtime_error("usage: prep-bench FIXED_G4 (e.g. 0.015625)");
  char* end; errno = 0; const float fixed = strtof(argv[1], &end);
  if (errno || end == argv[1] || *end || !std::isfinite(fixed) || fixed <= 0.f || !std::isfinite(1.f/fixed))
    throw std::runtime_error("invalid fixed global scale");
  cudaStream_t stream; ck(cudaStreamCreate(&stream));
  std::cout << "rows,n,flags,row_us,chunk_us,f32_max_abs,row_quant_rmse,chunk_quant_rmse,clipped_elements\n";
  for (int rows : {1, 8}) for (int kind = 0; kind < 4; kind++) {
    const int n = kind == 0 ? 5120 : kind == 3 ? 17408 : 6144;
    const unsigned flags = h5::P_SIGN | h5::P_HADAMARD | h5::P_QUANT |
      (kind == 0 ? h5::P_NORM | h5::P_OUT_NORM : kind == 1 ? h5::P_PERMUTE_GDN | h5::P_GDN_NORM : 0);
    std::mt19937 rng(1709 + kind); std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> x(rows*n), gz(rows*n), norm(n), signs(n), headnorm(128);
    for (float& v : x) v = normal(rng);
    for (float& v : gz) v = normal(rng);
    for (float& v : norm) v = .8f + .4f * std::fabs(normal(rng));
    for (float& v : signs) v = rng() & 1 ? 1.f : -1.f;
    for (float& v : headnorm) v = .9f + .2f * std::fabs(normal(rng));
    std::vector<void*> owned;
    auto floats = [&](const std::vector<float>& v) {
      float* p = alloc<float>(v.size()); owned.push_back(p);
      ck(cudaMemcpy(p,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice)); return p;
    };
    h5::PrepArgs a{}; a.x = floats(x); a.ldx = n; a.n = n; a.flags = flags;
    a.gz = floats(gz); a.gz_ld = n; a.gnorm_w = floats(headnorm);
    a.norm_w = floats(norm); a.signs = floats(signs); a.ldo = n;
    a.q4 = alloc<uint8_t>(64*n/2); owned.push_back(a.q4);
    a.s4 = alloc<uint8_t>(64*n/16); owned.push_back(a.s4);
    a.g4 = alloc<float>(rows); owned.push_back(a.g4);
    a.out_norm = alloc<float>(rows*n); owned.push_back(a.out_norm);
    a.out_f32 = alloc<float>(rows*n); owned.push_back(a.out_f32);
    auto launch = [&](bool chunk) {
      if (chunk) h5::pdl(h5::k_prep_chunk<false>,dim3(rows,n/1024),dim3(32),0,stream,a,fixed);
      else h5::pdl(h5::k_prep2,dim3(rows),dim3(32*n/1024),0,stream,a);
      ck(cudaGetLastError());
    };
    float micros[2]; std::vector<float> reference(rows*n);
    double err[2] = {}; float f32_delta = 0; size_t clipped = 0;
    for (int mode = 0; mode < 2; mode++) {
      cudaGraph_t graph; cudaGraphExec_t exec;
      ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < 256; i++) launch(mode);
      ck(cudaStreamEndCapture(stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,0));
      ck(cudaGraphLaunch(exec,stream)); ck(cudaStreamSynchronize(stream));
      cudaEvent_t start,stop; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
      std::vector<float> timings;
      for (int repeat = 0; repeat < 7; repeat++) {
        ck(cudaEventRecord(start,stream)); ck(cudaGraphLaunch(exec,stream)); ck(cudaEventRecord(stop,stream));
        ck(cudaEventSynchronize(stop)); float ms; ck(cudaEventElapsedTime(&ms,start,stop));
        timings.push_back(ms*1000/256);
      }
      std::sort(timings.begin(),timings.end()); micros[mode] = timings[3];
      ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop)); ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
      a.flags |= h5::P_STORE_F32; launch(mode); ck(cudaStreamSynchronize(stream)); a.flags = flags;
      std::vector<float> prepared(rows*n),g(rows); std::vector<uint8_t> q(64*n/2),s(64*n/16);
      ck(cudaMemcpy(prepared.data(),a.out_f32,rows*n*sizeof(float),cudaMemcpyDeviceToHost));
      ck(cudaMemcpy(q.data(),a.q4,q.size(),cudaMemcpyDeviceToHost));
      ck(cudaMemcpy(s.data(),a.s4,s.size(),cudaMemcpyDeviceToHost));
      ck(cudaMemcpy(g.data(),a.g4,rows*sizeof(float),cudaMemcpyDeviceToHost));
      if (!mode) reference = prepared;
      for (int r = 0; r < rows; r++) for (int c = 0; c < n; c++) {
        const int i = r*n+c;
        if (!std::isfinite(prepared[i])) throw std::runtime_error("nonfinite prepared output");
        f32_delta = std::max(f32_delta,std::fabs(prepared[i]-reference[i]));
        const double diff = decoded(q,s,g,r,n,c)-reference[i]; err[mode] += diff*diff;
        if (mode && std::fabs(prepared[i]) > 2688*fixed) clipped++;
      }
    }
    std::cout << rows << ',' << n << ',' << flags << ',' << std::setprecision(9)
      << micros[0] << ',' << micros[1] << ',' << f32_delta << ','
      << std::sqrt(err[0]/(rows*n)) << ',' << std::sqrt(err[1]/(rows*n)) << ',' << clipped << '\n';
    for (void* p : owned) ck(cudaFree(p));
  }
  ck(cudaStreamDestroy(stream)); return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
