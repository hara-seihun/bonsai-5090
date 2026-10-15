#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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

enum class Op { F32Fma, H2Fma, H2Add, H2Mul, B2Fma, B2Add, B2Mul,
                Dp4a, Dp2aLo, Dp2aHi, Lop3, Prmt, Imad, Iadd, Isub, SwarAdd, SwarSub };
enum class Schedule { Tensor, Isolated, Phases, Interleave };
struct Spec { Op op; const char* name; int target_insns, ptx_insns, elements, arithmetic; };
const Spec specs[] = {
  {Op::F32Fma, "ffma", 1, 1, 1, 2}, {Op::H2Fma, "f16x2_fma", 1, 1, 2, 4},
  {Op::H2Add, "f16x2_add", 2, 2, 4, 4}, {Op::H2Mul, "f16x2_mul", 1, 1, 2, 2},
  {Op::B2Fma, "bf16x2_fma", 1, 1, 2, 4}, {Op::B2Add, "bf16x2_add", 2, 2, 4, 4},
  {Op::B2Mul, "bf16x2_mul", 1, 1, 2, 2}, {Op::Dp4a, "dp4a", 1, 1, 4, 8},
  {Op::Dp2aLo, "dp2a_lo", 1, 1, 2, 4}, {Op::Dp2aHi, "dp2a_hi", 1, 1, 2, 4},
  {Op::Lop3, "lop3", 1, 1, 1, 0}, {Op::Prmt, "prmt", 1, 1, 1, 0},
  {Op::Imad, "imad", 1, 1, 1, 2}, {Op::Iadd, "iadd", 1, 1, 1, 1},
  {Op::Isub, "isub", 1, 1, 1, 1}, {Op::SwarAdd, "swar15x2_add", 1, 2, 2, 2},
  {Op::SwarSub, "swar15x2_sub", 1, 3, 2, 2}
};
struct Config { int iters, repeats, warps, blocks_per_sm, chains, steps; std::string op; };
struct Value { uint32_t tensor[4], scalar; };
struct Stats { std::vector<float> ms; int regs = 0, active = 0; uint64_t tensor_hash = 0, scalar_hash = 0; };
struct Runtime { Config c; int blocks, threads; Value* output; cudaStream_t stream; cudaEvent_t start, stop; };

__device__ __forceinline__ void tensor_step(float (&v)[4]) {
  const unsigned a = 0x22222222u, scale = 0x38383838u;
  const uint16_t z = 0;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%0,%1,%2,%3},%5,{%6,%6},%5,{%6,%6};"
    : "+f"(v[0]), "+f"(v[1]), "+f"(v[2]), "+f"(v[3]) : "r"(a), "r"(scale), "h"(z));
}

template<Op O>
__device__ __forceinline__ uint32_t initial(int lane, int chain) {
  const float lo = 0.5f + chain * 0.125f + lane * 0.0625f, hi = lo + 0.125f;
  if constexpr (O == Op::F32Fma) return __float_as_uint(lo);
  else if constexpr (O == Op::H2Fma || O == Op::H2Add || O == Op::H2Mul) {
    const __half2_raw r = __floats2half2_rn(lo, hi);
    return uint32_t(r.x) | (uint32_t(r.y) << 16);
  } else if constexpr (O == Op::B2Fma || O == Op::B2Add || O == Op::B2Mul) {
    const __nv_bfloat162_raw r = __floats2bfloat162_rn(lo, hi);
    return uint32_t(r.x) | (uint32_t(r.y) << 16);
  } else {
    const uint32_t seed = (uint32_t(lane) * 2654435761u + chain * 2246822519u + 1u);
    return (O == Op::SwarAdd || O == Op::SwarSub) ? seed & 0x7fff7fffu : seed;
  }
}

template<Op O>
__device__ __forceinline__ void scalar_step(uint32_t& u, uint32_t multiplier) {
#if defined(PACKED_OPS_HOIST) && PACKED_OPS_HOIST
  const uint32_t half_neg = multiplier, bfloat_neg = multiplier;
#else
  const uint32_t half_neg = 0xbc00bc00u, bfloat_neg = 0xbf80bf80u;
#endif
  if constexpr (O == Op::F32Fma) {
    float f = __uint_as_float(u);
    asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(f) : "f"(-1.f), "f"(0.25f));
    u = __float_as_uint(f);
  } else if constexpr (O == Op::H2Fma) {
    asm volatile("fma.rn.f16x2 %0,%0,%1,%2;" : "+r"(u) : "r"(half_neg), "r"(0x36003400u));
  } else if constexpr (O == Op::H2Add) {
    // A balanced pair keeps every add active, without growth into rounding stagnation.
    asm volatile("add.rn.f16x2 %0,%0,%1; add.rn.f16x2 %0,%0,%2;" : "+r"(u) : "r"(0x2c003000u), "r"(0xac00b000u));
  } else if constexpr (O == Op::H2Mul) {
    asm volatile("mul.rn.f16x2 %0,%0,%1;" : "+r"(u) : "r"(half_neg));
  } else if constexpr (O == Op::B2Fma) {
    asm volatile("fma.rn.bf16x2 %0,%0,%1,%2;" : "+r"(u) : "r"(bfloat_neg), "r"(0x3ec03e80u));
  } else if constexpr (O == Op::B2Add) {
    asm volatile("add.rn.bf16x2 %0,%0,%1; add.rn.bf16x2 %0,%0,%2;" : "+r"(u) : "r"(0x3d803e00u), "r"(0xbd80be00u));
  } else if constexpr (O == Op::B2Mul) {
    asm volatile("mul.rn.bf16x2 %0,%0,%1;" : "+r"(u) : "r"(bfloat_neg));
  } else if constexpr (O == Op::Dp4a) {
    asm volatile("dp4a.s32.s32 %0,%1,%2,%0;" : "+r"(u) : "r"(0x01020304u), "r"(0x05060708u));
  } else if constexpr (O == Op::Dp2aLo) {
    asm volatile("dp2a.lo.s32.s32 %0,%1,%2,%0;" : "+r"(u) : "r"(0x00020003u), "r"(0x04030201u));
  } else if constexpr (O == Op::Dp2aHi) {
    asm volatile("dp2a.hi.s32.s32 %0,%1,%2,%0;" : "+r"(u) : "r"(0x00020003u), "r"(0x04030201u));
  } else if constexpr (O == Op::Lop3) {
    asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(u) : "r"(0x5a5a5a5au), "r"(0xa5a5a5a5u));
  } else if constexpr (O == Op::Prmt) {
    asm volatile("prmt.b32 %0,%0,%0,0x1032;" : "+r"(u));
  } else if constexpr (O == Op::Imad) {
    asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;" : "+r"(u));
  } else if constexpr (O == Op::Iadd) {
    asm volatile("add.u32 %0,%0,65537;" : "+r"(u));
  } else if constexpr (O == Op::Isub) {
    asm volatile("sub.u32 %0,%0,65537;" : "+r"(u));
  } else if constexpr (O == Op::SwarAdd) {
    asm volatile("add.u32 %0,%0,65537; and.b32 %0,%0,0x7fff7fff;" : "+r"(u));
  } else if constexpr (O == Op::SwarSub) {
    asm volatile("or.b32 %0,%0,0x80008000; sub.u32 %0,%0,65537; and.b32 %0,%0,0x7fff7fff;" : "+r"(u));
  }
}

// Chains share constants but no mutable state. Scalar and tensor chains are also disjoint.
template<Op O, int C, Schedule S>
__global__ void packed_kernel(int iters, int steps, Value* output, uint32_t multiplier) {
  float t[C][4] = {}; uint32_t u[C];
#pragma unroll
  for (int ch = 0; ch < C; ++ch) u[ch] = initial<O>(threadIdx.x & 31, ch);
  if constexpr (S == Schedule::Tensor || S == Schedule::Phases) {
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int ch = 0; ch < C; ++ch) tensor_step(t[ch]);
    }
  }
  if constexpr (S != Schedule::Tensor) {
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
      if constexpr (S == Schedule::Interleave) {
#pragma unroll
        for (int ch = 0; ch < C; ++ch) { tensor_step(t[ch]); scalar_step<O>(u[ch], multiplier); }
      }
#pragma unroll 1
      for (int s = S == Schedule::Interleave ? 1 : 0; s < steps; ++s) {
#pragma unroll
        for (int ch = 0; ch < C; ++ch) scalar_step<O>(u[ch], multiplier);
      }
    }
  }
  const size_t base = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * C;
#pragma unroll
  for (int ch = 0; ch < C; ++ch) {
    Value v = {};
    if constexpr (S != Schedule::Isolated) {
#pragma unroll
      for (int k = 0; k < 4; ++k) v.tensor[k] = __float_as_uint(t[ch][k]);
    }
    if constexpr (S != Schedule::Tensor) v.scalar = u[ch];
    output[base + ch] = v;
  }
}

template<Op O, int C, Schedule S>
void launch(Runtime& rt, Stats* stats) {
  if (stats) {
    cudaFuncAttributes a; CUDA(cudaFuncGetAttributes(&a, packed_kernel<O,C,S>));
    stats->regs = a.numRegs;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&stats->active, packed_kernel<O,C,S>, rt.threads, 0));
    if (!stats->active) { std::fprintf(stderr, "RESOURCE_ERROR no resident CTA\n"); std::exit(3); }
  }
  constexpr uint32_t multiplier = O == Op::H2Fma || O == Op::H2Mul ? 0xbc00bc00u : O == Op::B2Fma || O == Op::B2Mul ? 0xbf80bf80u : 0u;
  packed_kernel<O,C,S><<<rt.blocks,rt.threads,0,rt.stream>>>(rt.c.iters, rt.c.steps, rt.output, multiplier);
  CUDA(cudaGetLastError());
}
template<Op O, int C>
void schedule(Runtime& rt, Schedule s, Stats* stats) {
  switch (s) {
    case Schedule::Tensor: launch<Op::F32Fma,C,Schedule::Tensor>(rt,stats); break;
    case Schedule::Isolated: launch<O,C,Schedule::Isolated>(rt,stats); break;
    case Schedule::Phases: launch<O,C,Schedule::Phases>(rt,stats); break;
    case Schedule::Interleave: launch<O,C,Schedule::Interleave>(rt,stats); break;
  }
}
template<Op O>
void chains(Runtime& rt, Schedule s, Stats* stats) {
  if (rt.c.chains == 1) schedule<O,1>(rt,s,stats);
  else schedule<O,8>(rt,s,stats);
}
void dispatch(Runtime& rt, Op o, Schedule s, Stats* stats) {
#define DISPATCH(name) case Op::name: chains<Op::name>(rt,s,stats); break
  switch (o) {
    DISPATCH(F32Fma); DISPATCH(H2Fma); DISPATCH(H2Add); DISPATCH(H2Mul);
    DISPATCH(B2Fma); DISPATCH(B2Add); DISPATCH(B2Mul); DISPATCH(Dp4a);
    DISPATCH(Dp2aLo); DISPATCH(Dp2aHi); DISPATCH(Lop3); DISPATCH(Prmt);
    DISPATCH(Imad); DISPATCH(Iadd); DISPATCH(Isub); DISPATCH(SwarAdd); DISPATCH(SwarSub);
  }
#undef DISPATCH
}
float measure(Runtime& rt, Op o, Schedule s) {
  CUDA(cudaEventRecord(rt.start,rt.stream)); dispatch(rt,o,s,nullptr);
  CUDA(cudaEventRecord(rt.stop,rt.stream)); CUDA(cudaEventSynchronize(rt.stop));
  float ms; CUDA(cudaEventElapsedTime(&ms,rt.start,rt.stop));
  if (!(ms > 0) || !std::isfinite(ms)) { std::fprintf(stderr,"TIMING_ERROR invalid event interval\n"); std::exit(5); }
  return ms;
}
std::vector<Value> collect(Runtime& rt, Stats& stats) {
  std::vector<Value> v(size_t(rt.blocks) * rt.threads * rt.c.chains);
  CUDA(cudaMemcpy(v.data(),rt.output,v.size()*sizeof(Value),cudaMemcpyDeviceToHost));
  uint64_t ht = 14695981039346656037ull, hs = ht;
  for (const auto& x : v) {
    for (auto t : x.tensor) { ht = (ht ^ t) * 1099511628211ull; }
    hs = (hs ^ x.scalar) * 1099511628211ull;
  }
  stats.tensor_hash = ht; stats.scalar_hash = hs; return v;
}
float bits_float(uint32_t bits) { float f; std::memcpy(&f,&bits,4); return f; }
uint32_t float_bits(float f) { uint32_t bits; std::memcpy(&bits,&f,4); return bits; }
uint32_t pack_half(float lo, float hi) {
  const __half2_raw r = __floats2half2_rn(lo,hi); return uint32_t(r.x) | (uint32_t(r.y)<<16);
}
uint32_t pack_bfloat(float lo, float hi) {
  const __nv_bfloat162_raw r = __floats2bfloat162_rn(lo,hi); return uint32_t(r.x) | (uint32_t(r.y)<<16);
}
uint32_t expected_scalar(Op op, int lane, int chain, uint32_t updates) {
  const float lo=0.5f+chain*0.125f+lane*0.0625f, hi=lo+0.125f;
  const bool odd=updates&1u;
  const uint32_t seed=uint32_t(lane)*2654435761u+uint32_t(chain)*2246822519u+1u;
  switch (op) {
    case Op::F32Fma: return float_bits(odd?0.25f-lo:lo);
    case Op::H2Fma: return pack_half(odd?0.25f-lo:lo,odd?0.375f-hi:hi);
    case Op::B2Fma: return pack_bfloat(odd?0.25f-lo:lo,odd?0.375f-hi:hi);
    case Op::H2Add: return pack_half(lo,hi);
    case Op::B2Add: return pack_bfloat(lo,hi);
    case Op::H2Mul: return pack_half(odd?-lo:lo,odd?-hi:hi);
    case Op::B2Mul: return pack_bfloat(odd?-lo:lo,odd?-hi:hi);
    case Op::Dp4a: return seed+updates*70u;
    case Op::Dp2aLo: return seed+updates*7u;
    case Op::Dp2aHi: return seed+updates*17u;
    case Op::Lop3: return odd?~seed:seed;
    case Op::Prmt: return odd?(seed<<16)|(seed>>16):seed;
    case Op::Imad: {
      uint32_t x=seed, a=1664525u, b=1013904223u;
      for (uint32_t n=updates;n;n>>=1) { if (n&1u) x=a*x+b; b=a*b+b; a=a*a; }
      return x;
    }
    case Op::Iadd: return seed+updates*65537u;
    case Op::Isub: return seed-updates*65537u;
    case Op::SwarAdd: return ((seed+updates)&0x7fffu) | ((((seed>>16)+updates)&0x7fffu)<<16);
    case Op::SwarSub: return ((seed-updates)&0x7fffu) | ((((seed>>16)-updates)&0x7fffu)<<16);
  }
  std::fprintf(stderr,"INVARIANT_ERROR invalid internal operation\n"); std::abort();
}
void validate_scalar(const std::vector<Value>& v, Op op, const Config& c) {
  uint32_t expected[32][8];
  for (int lane=0;lane<32;++lane) for (int ch=0;ch<c.chains;++ch)
    expected[lane][ch]=expected_scalar(op,lane,ch,uint32_t(c.iters)*c.steps);
  for (size_t i=0;i<v.size();++i) {
    const auto& x=v[i];
    if (x.scalar!=expected[(i/c.chains)%32][i%c.chains]) {
      std::fprintf(stderr,"CHECKSUM_ERROR isolated scalar chain %zu differs from exact recurrence\n",i); std::exit(4);
    }
    bool good = true;
    if (op == Op::F32Fma) good = std::isfinite(bits_float(x.scalar)) && std::fabs(bits_float(x.scalar)) < 4.f;
    else if (op == Op::H2Fma || op == Op::H2Add || op == Op::H2Mul) {
      for (int k=0;k<2;++k) { __half_raw r; r.x = uint16_t(x.scalar >> (16*k)); const float f = __half2float(__half(r)); good &= std::isfinite(f) && std::fabs(f) < 4.f; }
    } else if (op == Op::B2Fma || op == Op::B2Add || op == Op::B2Mul) {
      for (int k=0;k<2;++k) { const float f = bits_float((uint32_t(uint16_t(x.scalar >> (16*k)))) << 16); good &= std::isfinite(f) && std::fabs(f) < 4.f; }
    } else if (op == Op::SwarAdd || op == Op::SwarSub) good = !(x.scalar & 0x80008000u);
    if (!good) { std::fprintf(stderr,"CHECKSUM_ERROR scalar outside declared domain\n"); std::exit(4); }
  }
}
double median(std::vector<float> v) { std::sort(v.begin(),v.end()); return v.size()%2 ? v[v.size()/2] : (double(v[v.size()/2-1])+v[v.size()/2])/2; }
int number(const char* s) {
  errno=0; char* end; long n=std::strtol(s,&end,10);
  if (errno || end==s || *end || n<1 || n>65536) { std::fprintf(stderr,"ARGUMENT_ERROR invalid positive integer: %s\n",s); std::exit(1); }
  return int(n);
}
void usage() {
  std::puts("packed_ops --iters N --repeats N --warps N --blocks-per-sm N --chains 1|8 --scalar-steps N --op all|NAME\n"
    "All arguments required. iters<=65536; repeats<=1000; warps<=32; blocks-per-sm<=32; scalar-steps<=64.\n"
    "Each op runs isolated, phases and interleave, plus one common tensor reference.\n"
    "Ops: ffma f16x2_fma f16x2_add f16x2_mul bf16x2_fma bf16x2_add bf16x2_mul\n"
    "     dp4a dp2a_lo dp2a_hi lop3 prmt imad iadd isub swar15x2_add swar15x2_sub");
}
int main(int argc,char** argv) {
  Config c={}; unsigned seen=0;
  for (int a=1;a<argc;a+=2) {
    if (!std::strcmp(argv[a],"--help")) { usage(); return 0; }
    if (a+1==argc) { usage(); return 1; }
    std::string key=argv[a]; const char* val=argv[a+1]; unsigned bit=0;
    if (key=="--iters") { c.iters=number(val); bit=1; }
    else if (key=="--repeats") { c.repeats=number(val); bit=2; }
    else if (key=="--warps") { c.warps=number(val); bit=4; }
    else if (key=="--blocks-per-sm") { c.blocks_per_sm=number(val); bit=8; }
    else if (key=="--chains") { c.chains=number(val); bit=16; }
    else if (key=="--scalar-steps") { c.steps=number(val); bit=32; }
    else if (key=="--op") { c.op=val; bit=64; }
    else { std::fprintf(stderr,"ARGUMENT_ERROR unknown option %s\n",key.c_str()); return 1; }
    if (seen&bit) { std::fprintf(stderr,"ARGUMENT_ERROR duplicate option %s\n",key.c_str()); return 1; }
    seen|=bit;
  }
  if (seen!=127 || c.repeats>1000 || c.warps>32 || c.blocks_per_sm>32 || c.steps>64 || (c.chains!=1 && c.chains!=8)) { usage(); return 1; }
  std::vector<Spec> selected;
  for (auto s:specs) if (c.op=="all" || c.op==s.name) selected.push_back(s);
  if (selected.empty()) { std::fprintf(stderr,"ARGUMENT_ERROR unknown op %s\n",c.op.c_str()); return 1; }
  CUDA(cudaSetDevice(0)); cudaDeviceProp p; CUDA(cudaGetDeviceProperties(&p,0));
  if (p.major!=12 || p.minor!=0) { std::fprintf(stderr,"DEVICE_ERROR requires sm_120, got sm_%d%d\n",p.major,p.minor); return 2; }
  Runtime rt={}; rt.c=c; rt.blocks=p.multiProcessorCount*c.blocks_per_sm; rt.threads=32*c.warps;
  CUDA(cudaMalloc(&rt.output,size_t(rt.blocks)*rt.threads*c.chains*sizeof(Value)));
  CUDA(cudaStreamCreateWithFlags(&rt.stream,cudaStreamNonBlocking)); CUDA(cudaEventCreate(&rt.start)); CUDA(cudaEventCreate(&rt.stop));
  struct Case { Op op; Schedule s; std::string name; const Spec* spec; };
  std::vector<Case> cases={{Op::F32Fma,Schedule::Tensor,"tensor",nullptr}};
  for (const auto& s:selected) {
    cases.push_back({s.op,Schedule::Isolated,std::string("isolated_")+s.name,&s});
    cases.push_back({s.op,Schedule::Phases,std::string("phases_")+s.name,&s});
    cases.push_back({s.op,Schedule::Interleave,std::string("interleave_")+s.name,&s});
  }
  std::vector<Stats> stats(cases.size());
  for (size_t n=0;n<cases.size();++n) {
    dispatch(rt,cases[n].op,cases[n].s,&stats[n]); CUDA(cudaStreamSynchronize(rt.stream));
    measure(rt,cases[n].op,cases[n].s);
  }
  for (int r=0;r<c.repeats;++r) for (size_t j=0;j<cases.size();++j) {
    const size_t n=(j+r)%cases.size(); stats[n].ms.push_back(measure(rt,cases[n].op,cases[n].s));
  }
  // Untimed fresh launches validate every chain, not merely a potentially colliding reduction.
  dispatch(rt,Op::F32Fma,Schedule::Tensor,nullptr); CUDA(cudaStreamSynchronize(rt.stream));
  const auto tensor=collect(rt,stats[0]);
  for (const auto& v:tensor) for (auto bits:v.tensor) if (bits_float(bits)!=64.f*c.iters) {
    std::fprintf(stderr,"CHECKSUM_ERROR tensor reference differs from +64 per iteration\n"); return 4;
  }
  std::vector<Value> scalar;
  for (size_t n=1;n<cases.size();++n) {
    const auto& a=cases[n]; dispatch(rt,a.op,a.s,nullptr); CUDA(cudaStreamSynchronize(rt.stream));
    auto v=collect(rt,stats[n]);
    if (a.s==Schedule::Isolated) { scalar=v; validate_scalar(scalar,a.op,c); }
    else for (size_t i=0;i<v.size();++i) if (v[i].scalar!=scalar[i].scalar || std::memcmp(v[i].tensor,tensor[i].tensor,sizeof(v[i].tensor))) {
      std::fprintf(stderr,"CHECKSUM_ERROR %s chain %zu differs from isolated work\n",a.name.c_str(),i); return 4;
    }
  }
#if defined(PACKED_OPS_HOIST) && PACKED_OPS_HOIST
  const char* constants = "kernel-parameter";
#else
  const char* constants = "literal";
#endif
  std::printf("# device=%s sms=%d iters=%d repeats=%d warps=%d blocks_per_sm_requested=%d chains=%d scalar_steps=%d constants=%s\n",p.name,p.multiProcessorCount,c.iters,c.repeats,c.warps,c.blocks_per_sm,c.chains,c.steps,constants);
  std::puts("case,median_ms,min_ms,max_ms,mma_warp_instructions,target_lane_instructions,scalar_ptx_lane_instructions,element_updates,arithmetic_ops,Gwarp_MMA_s,FP4_TMAC_s,Gtarget_lane_instructions_s,Gscalar_ptx_lane_instructions_s,Gelement_updates_s,Garithmetic_ops_s,speedup_vs_isolated_sum,time_vs_isolated_max,registers_per_thread,occupancy_max_ctas_per_sm,checksum_tensor_bits,checksum_scalar_bits");
  const double units=double(rt.blocks)*c.warps*c.iters*c.chains;
  for (size_t n=0;n<cases.size();++n) {
    const auto& a=cases[n]; const auto& s=stats[n]; const bool t=a.s!=Schedule::Isolated;
    const double scalar_steps=a.spec ? units*32*c.steps : 0;
    const double mm=t?units:0, ti=a.spec?scalar_steps*a.spec->target_insns:0, pi=a.spec?scalar_steps*a.spec->ptx_insns:0;
    const double el=a.spec?scalar_steps*a.spec->elements:0, ar=a.spec?scalar_steps*a.spec->arithmetic:0;
    const double ms=median(s.ms), sec=ms*1e-3;
    const size_t ref=n?1+((n-1)/3)*3:0;
    const double tm=t?median(stats[0].ms):0, sm=a.spec?median(stats[ref].ms):0;
    std::printf("%s,%.6f,%.6f,%.6f,%.0f,%.0f,%.0f,%.0f,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%016llx,%016llx\n",
      a.name.c_str(),ms,*std::min_element(s.ms.begin(),s.ms.end()),*std::max_element(s.ms.begin(),s.ms.end()),mm,ti,pi,el,ar,
      mm/sec/1e9,mm*8192/sec/1e12,ti/sec/1e9,pi/sec/1e9,el/sec/1e9,ar/sec/1e9,(tm+sm)/ms,ms/std::max(tm,sm),s.regs,s.active,
      (unsigned long long)s.tensor_hash,(unsigned long long)s.scalar_hash);
  }
  CUDA(cudaFree(rt.output)); CUDA(cudaEventDestroy(rt.start)); CUDA(cudaEventDestroy(rt.stop)); CUDA(cudaStreamDestroy(rt.stream));
  return 0;
}
