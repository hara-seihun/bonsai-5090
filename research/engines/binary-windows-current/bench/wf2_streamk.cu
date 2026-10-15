// Standalone selected-production versus balanced flattened tile/K128 work. No engine integration.
#define H5_WARP_MICROSCOPE_STANDALONE
#undef H5_PROFILE_SCALE_CARRIER
#include "../src/fp4.cu"
#include "wf2_streamk.cuh"
#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstring>
#include <vector>

namespace sk = h5::wf2_streamk;
static void ck(cudaError_t e, const char* op) {
  if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(e)); std::exit(1); }
}
static bool integer(const char* s, int lo, int hi, int& v) {
  const auto end = s + std::strlen(s); const auto r = std::from_chars(s,end,v);
  return r.ec == std::errc{} && r.ptr == end && v >= lo && v <= hi;
}
__global__ void fill_bytes(uint8_t* p, size_t n, unsigned seed, bool scale) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)blockDim.x * gridDim.x) {
    unsigned x = (unsigned)i * 2654435761u ^ seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = scale ? 0x30 + (x & 15) : x;
  }
}
__global__ void fill_float(float* p, size_t n, float base, float step) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)blockDim.x * gridDim.x)
    p[i] = base + float(i % 17) * step;
}
struct Check { unsigned long long bad, unequal, padding; float max_abs, max_normalized; };
__global__ void compare(const float* a, const float* b, int R, int N, int ldo, bool exact, Check* c) {
  unsigned bad = 0, unequal = 0, padding = 0;
  float max_abs = 0, max_normalized = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)R * ldo; i += (size_t)blockDim.x * gridDim.x) {
    if (i % ldo >= (size_t)N) { padding += b[i] != 17.f; continue; }
    const float ref = a[i], v = b[i], error = fabsf(ref - v), norm = error / fmaxf(1.f, fabsf(ref));
    const bool differs = __float_as_uint(ref) != __float_as_uint(v);
    unequal += differs;
    bad += !isfinite(v) || !isfinite(ref) || (exact ? differs : norm > 2e-5f);
    max_abs = fmaxf(max_abs,error); max_normalized = fmaxf(max_normalized,norm);
  }
  for (int d = 16; d; d /= 2) {
    bad += __shfl_down_sync(0xffffffffu,bad,d); unequal += __shfl_down_sync(0xffffffffu,unequal,d);
    padding += __shfl_down_sync(0xffffffffu,padding,d);
    max_abs = fmaxf(max_abs,__shfl_down_sync(0xffffffffu,max_abs,d));
    max_normalized = fmaxf(max_normalized,__shfl_down_sync(0xffffffffu,max_normalized,d));
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&c->bad,(unsigned long long)bad); atomicAdd(&c->unequal,(unsigned long long)unequal);
    atomicAdd(&c->padding,(unsigned long long)padding);
    atomicMax((unsigned*)&c->max_abs,__float_as_uint(max_abs));
    atomicMax((unsigned*)&c->max_normalized,__float_as_uint(max_normalized));
  }
}

enum class Mode { Production, Split2, Heads, Atomic };
static const char* name(Mode m) {
  switch (m) {
    case Mode::Production: return "production-unsplit";
    case Mode::Split2: return "production-split2";
    case Mode::Heads: return "streamk-heads";
    case Mode::Atomic: return "streamk-atomic";
  }
  std::abort();
}
static void clear(sk::Args p, cudaStream_t st) {
  if (p.ldo == p.N) ck(cudaMemsetAsync(p.out,0,(size_t)p.R * p.N * 4,st),"clear output");
  else ck(cudaMemset2DAsync(p.out,(size_t)p.ldo * 4,0,(size_t)p.N * 4,p.R,st),"clear output with padding");
}
static void launch(Mode m, sk::Args p, int workers, cudaStream_t st) {
  if (m == Mode::Production || m == Mode::Split2) {
    if (m == Mode::Split2) clear(p,st);
    const dim3 grid((p.R + sk::TOK - 1) / sk::TOK,(p.N + sk::COLS - 1) / sk::COLS,m == Mode::Split2 ? 2 : 1);
    if (m == Mode::Production)
      h5::pdl(h5::k_gemm_fp4<8,1,128,0,2>,grid,dim3(sk::THREADS),h5::WsGeom<8,1,128,2>::SMEM,st,
              p.A,p.AS,p.AG,p.R,p.K,p.W,p.S,p.N,p.out,p.ldo,p.RS);
    else
      h5::pdl(h5::k_gemm_fp4<8,1,128,2,2>,grid,dim3(sk::THREADS),h5::WsGeom<8,1,128,2>::SMEM,st,
              p.A,p.AS,p.AG,p.R,p.K,p.W,p.S,p.N,p.out,p.ldo,p.RS);
  } else if (m == Mode::Heads) {
    h5::pdl(sk::gemm<false>,dim3(workers),dim3(sk::THREADS),sk::SMEM,st,p);
    // Ordinary stream serialization is deliberate: every out owner must finish before reduction.
    const int64_t tiles = (int64_t)((p.R + sk::TOK - 1) / sk::TOK) * ((p.N + sk::COLS - 1) / sk::COLS);
    if (workers <= tiles) sk::merge_heads<true><<<workers,256,0,st>>>(p,workers);
    else sk::merge_heads<false><<<workers,256,0,st>>>(p,workers);
  } else {
    clear(p,st);
    h5::pdl(sk::gemm<true>,dim3(workers),dim3(sk::THREADS),sk::SMEM,st,p);
  }
  ck(cudaGetLastError(),name(m));
}

template <typename F>
static void resources(F f, const char* label, int bytes) {
  ck(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes),label);
  cudaFuncAttributes a{}; int occ;
  ck(cudaFuncGetAttributes(&a,f),"attributes");
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,f,sk::THREADS,bytes),"occupancy");
  std::printf("resources name=%s regs=%d local_bytes=%zu smem=%d resident_blocks_per_sm=%d\n",label,a.numRegs,a.localSizeBytes,bytes,occ);
}

int main(int argc, char** argv) {
  int R,N,K,workers,copies,reps,rounds;
  if (argc != 8 || !integer(argv[1],1,4096,R) || !integer(argv[2],32,262144,N) || N % 32
      || !integer(argv[3],128,32768,K) || K % 128 || !integer(argv[4],1,4096,workers)
      || !integer(argv[5],2,64,copies) || !integer(argv[6],1,10000,reps) || !integer(argv[7],1,100,rounds)) {
    std::fprintf(stderr,"usage: wf2-streamk ROWS N K WORKERS WEIGHT_COPIES REPS ROUNDS\n"); return 2;
  }
  const int64_t tiles = (int64_t)((R + sk::TOK - 1) / sk::TOK) * ((N + sk::COLS - 1) / sk::COLS);
  const int nkb = K / 128; const int64_t total = tiles * nkb;
  if (workers > total) { std::fprintf(stderr,"WORKERS exceeds K128 work units\n"); return 2; }
  cudaDeviceProp prop; ck(cudaGetDeviceProperties(&prop,0),"device");
  const size_t wb = (size_t)(N / 32) * nkb * 1024, sb = (size_t)(N / 32) * nkb * 64;
  if ((wb + sb) * (copies - 1) <= (size_t)prop.l2CacheSize) {
    std::fprintf(stderr,"WEIGHT_COPIES too small for cold rotation: intervening weight bytes <= L2 (%d B)\n",prop.l2CacheSize); return 2;
  }
  int heads = 0, segments = 0, minwork = INT_MAX, maxwork = 0;
  for (int i = 0; i < workers; ++i) {
    const int64_t a = sk::begin(total,i,workers), b = sk::begin(total,i + 1,workers);
    heads += a % nkb != 0; segments += int((b - 1) / nkb - a / nkb + 1);
    minwork = std::min(minwork,int(b - a)); maxwork = std::max(maxwork,int(b - a));
  }
  std::printf("schedule device=%s sms=%d R=%d N=%d K=%d tiles=%lld units=%lld workers=%d units_per_worker=%d..%d segments=%d continuation_heads=%d scratch_bytes=%zu rotated_weight_bytes=%zu l2_bytes=%d\n",
    prop.name,prop.multiProcessorCount,R,N,K,(long long)tiles,(long long)total,workers,minwork,maxwork,segments,heads,
    (size_t)workers * sk::TOK * sk::COLS * 4,(wb + sb) * copies,prop.l2CacheSize);
  resources(h5::k_gemm_fp4<8,1,128,0,2>,"production-unsplit",h5::WsGeom<8,1,128,2>::SMEM);
  resources(h5::k_gemm_fp4<8,1,128,2,2>,"production-split2",h5::WsGeom<8,1,128,2>::SMEM);
  resources(sk::gemm<false>,"streamk-heads",sk::SMEM);
  resources(sk::gemm<true>,"streamk-atomic",sk::SMEM);
  static_assert(sk::SMEM == h5::WsGeom<8,1,128,2>::SMEM,"identical ring geometry");
  const int Rp = (R + sk::TOK - 1) / sk::TOK * sk::TOK, ldo = N + 16;
  uint8_t *A,*AS; float *AG,*RS,*out,*ref,*scratch; Check *check;
  ck(cudaMalloc(&A,(size_t)Rp * K / 2),"A"); ck(cudaMalloc(&AS,(size_t)Rp * K / 16),"AS");
  ck(cudaMalloc(&AG,(size_t)Rp * 4),"AG"); ck(cudaMalloc(&RS,(size_t)N * 4),"RS");
  ck(cudaMalloc(&out,(size_t)R * ldo * 4),"out"); ck(cudaMalloc(&ref,(size_t)R * ldo * 4),"reference");
  ck(cudaMalloc(&scratch,(size_t)workers * sk::TOK * sk::COLS * 4),"scratch"); ck(cudaMalloc(&check,sizeof(Check)),"check");
  std::vector<uint8_t*> W(copies),S(copies);
  fill_bytes<<<512,256>>>(A,(size_t)Rp * K / 2,1,false); fill_bytes<<<512,256>>>(AS,(size_t)Rp * K / 16,2,true);
  fill_float<<<64,256>>>(AG,Rp,0.8f,0.025f); fill_float<<<64,256>>>(RS,N,0.0008f,0.000025f);
  for (int i = 0; i < copies; ++i) {
    ck(cudaMalloc(&W[i],wb),"W"); ck(cudaMalloc(&S[i],sb),"S");
    fill_bytes<<<512,256>>>(W[i],wb,10 + i,false); fill_bytes<<<512,256>>>(S[i],sb,70 + i,true);
  }
  ck(cudaGetLastError(),"initialization launch"); ck(cudaDeviceSynchronize(),"initialization");
  cudaStream_t st; ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");
  sk::Args p{A,AS,W[0],S[0],AG,RS,out,scratch,R,K,N,ldo};
  for (int copy = 0; copy < copies; ++copy) {
    p.W = W[copy]; p.S = S[copy]; p.out = ref;
    fill_float<<<512,256,0,st>>>(ref,(size_t)R * ldo,17.f,0.f);
    launch(Mode::Production,p,workers,st);
    for (int variant = 0; variant < 4; ++variant) {
      const Mode m = variant == 0 ? Mode::Heads : variant == 1 ? Mode::Split2 : variant == 2 ? Mode::Heads : Mode::Atomic;
      const int nw = variant == 0 ? 1 : workers;
      p.out = out; fill_float<<<512,256,0,st>>>(out,(size_t)R * ldo,17.f,0.f);
      launch(m,p,nw,st); ck(cudaMemsetAsync(check,0,sizeof(Check),st),"reset check");
      compare<<<512,256,0,st>>>(ref,out,R,N,ldo,variant == 0,check);
      Check c{}; ck(cudaMemcpyAsync(&c,check,sizeof(c),cudaMemcpyDeviceToHost,st),"copy check"); ck(cudaStreamSynchronize(st),"correctness sync");
      std::printf("correctness name=%s workers=%d copy=%d bad=%llu padding_bad=%llu unequal=%llu max_abs=%.9g max_normalized=%.9g\n",
          name(m),nw,copy,c.bad,c.padding,c.unequal,c.max_abs,c.max_normalized);
      if (c.bad || c.padding) return 1;
    }
  }
  p.out = out; p.ldo = N;
  const Mode modes[] = {Mode::Production,Mode::Split2,Mode::Heads,Mode::Atomic};
  std::vector<float> samples[4]; cudaEvent_t e0,e1;
  ck(cudaEventCreate(&e0),"event"); ck(cudaEventCreate(&e1),"event");
  // Whole rotations before every sample remove cross-mode warm-weight advantage.
  for (int round = 0; round < rounds; ++round) for (int j = 0; j < 4; ++j) {
    const int mi = (j + round) % 4; const Mode m = modes[mi];
    for (int i = 0; i < copies; ++i) { const int wi = (i + round) % copies; p.W = W[wi]; p.S = S[wi]; launch(m,p,workers,st); }
    ck(cudaEventRecord(e0,st),"start timing");
    for (int i = 0; i < reps; ++i) { const int wi = (i + round) % copies; p.W = W[wi]; p.S = S[wi]; launch(m,p,workers,st); }
    ck(cudaEventRecord(e1,st),"stop timing"); ck(cudaEventSynchronize(e1),"timing sync");
    float ms; ck(cudaEventElapsedTime(&ms,e0,e1),"elapsed time"); samples[mi].push_back(ms * 1000 / reps);
    std::printf("sample name=%s round=%d inclusive_us=%.6f\n",name(m),round,samples[mi].back()); std::fflush(stdout);
  }
  float baseline = 0;
  for (int i = 0; i < 4; ++i) {
    auto& v = samples[i]; std::sort(v.begin(),v.end()); const float us = v[v.size() / 2]; if (!i) baseline = us;
    std::printf("result name=%s median_inclusive_us=%.6f min_us=%.6f max_us=%.6f speedup=%.6f TMAC_s=%.3f\n",
        name(modes[i]),us,v.front(),v.back(),baseline / us,(double)R * N * K / us / 1e6);
  }
  ck(cudaEventDestroy(e0),"event destroy"); ck(cudaEventDestroy(e1),"event destroy"); ck(cudaStreamDestroy(st),"stream destroy");
  for (auto w : W) ck(cudaFree(w),"free W"); for (auto s : S) ck(cudaFree(s),"free S");
  ck(cudaFree(A),"free A"); ck(cudaFree(AS),"free AS"); ck(cudaFree(AG),"free AG"); ck(cudaFree(RS),"free RS");
  ck(cudaFree(out),"free out"); ck(cudaFree(ref),"free ref"); ck(cudaFree(scratch),"free scratch"); ck(cudaFree(check),"free check");
  return 0;
}
