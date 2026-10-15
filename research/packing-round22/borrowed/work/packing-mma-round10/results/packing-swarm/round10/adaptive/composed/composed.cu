#include "~/work/packing-mma-round9/results/packing-swarm/round6/source/kernels.cu"
#include "~/work/packing-mma-round10/method/packing-swarm/tools/round7/grammar/packet.cuh"
namespace h5 {
template <int Window>
__device__ __forceinline__ r7grammar::Error cw_body(const GdnArgs& a, const Seg sg, int bx, uint8_t* smraw, const uint32_t* packet, unsigned packet_bytes, r7grammar::Tag input_tag, uint32_t* destination, unsigned capacity, unsigned base, r7grammar::Scratch& scratch, unsigned& saved_bytes, r7grammar::Tag& saved_tag) {
  using ST=__half; constexpr int CG=4;
  float (*sk)[128] = (float (*)[128])smraw;
  float (*sq)[128] = (float (*)[128])(smraw + Window * 128 * 4);
  float (*sv)[64] = (float (*)[64])(smraw + Window * 256 * 4);
  float* sg_ = (float*)(smraw + Window * 320 * 4);
  float* sb_ = sg_ + Window;
  constexpr int JR = 256 / CG, MC = 128 / CG, CH = 64 + JR / 4;
  const int h = bx / (CG / 2), half = bx % (CG / 2), kh = h % HK;
  const int tid = threadIdx.x, lane = tid & 31;
  const int jl = tid / CG, j = JR * half + jl, cg = lane & (CG - 1);
  ST* pbase = (ST*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * 2 * a.pending_rows * PEND_GDN;
  const ST* prd = pbase + (size_t)(sg.par ^ 1) * a.pending_rows * PEND_GDN;
  ST* pwr = pbase + (size_t)sg.par * a.pending_rows * PEND_GDN;
  float m[MC];
  auto error=r7grammar::load_chunk(packet,packet_bytes,input_tag,m,scratch);
  if(error!=r7grammar::Error::ok) return error;
  auto save = [&]() {
    __syncthreads();
    const auto result=r7grammar::save_chunk(m,destination,capacity,packet,packet_bytes,base,scratch);
    saved_tag=result.tag;
    saved_bytes=result.bytes;
    return result.error;
  };
  int done = 0, replayed = 0;
  bool first = true;
  if (sg.keep && sg.commit == 0) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
  while (first || replayed < sg.commit || done < sg.len) {
    auto chunk = replay_chunk(sg.commit, sg.len, replayed, done, Window);
    if (sg.keep && chunk.replay_rows) chunk.new_rows=0;
    const int nrep = chunk.replay_rows, nnew = chunk.new_rows;
    const int T = nrep + nnew;
    __syncthreads();
    for (int i = tid; i < T * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      float4 v4; bool have = true;
      if (t < nrep) {
        const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN;
        if (c < 32) have = false;
        else v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      } else {
        const float* co = a.conv_out + (size_t)(sg.start + done + t - nrep) * CONV_CH;
        v4 = *(const float4*)(c < 32 ? co + kh * 128 + c * 4 : c < 64 ? co + KDIM + kh * 128 + (c - 32) * 4 : co + 2 * KDIM + h * 128 + JR * half + (c - 64) * 4);
      }
      if (have) {
        float* dst = c < 32 ? &sq[t][c * 4] : c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4];
        *(float4*)dst = v4;
      }
    }
    if (tid < T) {
      const int t = tid;
      if (t < nrep) { const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN; sg_[t] = st_ld(pr + KDIM + VDIM + h); sb_[t] = st_ld(pr + KDIM + VDIM + HV + h); }
      else { const float* gr = a.gates + (size_t)(sg.start + done + t - nrep) * 96; sg_[t] = gr[h]; sb_[t] = gr[HV + h]; }
    }
    __syncthreads();
    if (sg.keep) {   // new speculative rows -> pending[par] (disjoint from the replay buffer)
      for (int i = tid; i < nnew * 128; i += 256) {
        const int t = nrep + (i >> 7), d = i & 127;
        ST* pw = pwr + (size_t)(done + t - nrep) * PEND_GDN;
        if (h < HK && half == 0) st_st(pw + kh * 128 + d, sk[t][d]);
        if (d < JR && !a.replay_delta) st_st(pw + KDIM + h * 128 + JR * half + d, sv[t][d]);
        if (d == 0 && half == 0) { st_st(pw + KDIM + VDIM + h, sg_[t]); st_st(pw + KDIM + VDIM + HV + h, sb_[t]); }
      }
    }
#ifdef EXP_GDN_NOCOMP
    if (T > 1000)
#endif
    for (int t = 0; t < T; t++) {
      const float g = sg_[t], beta = sb_[t];
      const float* kp = &sk[t][4 * cg];
      float kr[MC];
#pragma unroll
      for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
      float delta;
      if (a.replay_delta && t < nrep) delta = sv[t][jl];
      else {
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
        float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
        delta = beta * (sv[t][jl] - g * s);
        if (a.replay_delta && sg.keep && cg == 0)
          st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
      }
#pragma unroll
      for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
      if (t >= nrep) {
        const float* qp = &sq[t][4 * cg];
        float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
        float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
        #pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
        if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
      }
    }
    if (sg.keep && nrep && replayed + nrep == sg.commit) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
    replayed += nrep; done += nnew; first = false;
  }
  if (!sg.keep) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
  return r7grammar::Error::ok;
}

struct CWPacketIO {
 const uint32_t* const* input; const unsigned* input_bytes; const r7grammar::Tag* input_tags;
 uint32_t* const* output; const unsigned* output_capacity;
 unsigned* output_bytes; r7grammar::Tag* output_tags; r7grammar::Error* status;
 unsigned common_window_base;
};
template<int Window> union alignas(16) CWPhaseArena {
 uint8_t tokens[Window*322*4];
 r7grammar::Scratch codec;
};
template<int Window>
__global__ void __launch_bounds__(256,2) cw_kernel(GdnArgs a,CWPacketIO io) {
 static_assert(Window==1 || Window==8);
 __shared__ CWPhaseArena<Window> arena;
 const Seg sg=a.segs[blockIdx.y];
 const size_t chunk=((size_t(sg.slot)*NGDN+a.gl)*HV*2)+blockIdx.x;
 unsigned saved_bytes=0; r7grammar::Tag saved_tag=r7grammar::Tag::raw16384;
 auto error=cw_body<Window>(a,sg,blockIdx.x,arena.tokens,io.input[chunk],io.input_bytes[chunk],io.input_tags[chunk],
       io.output[chunk],io.output_capacity[chunk],io.common_window_base,arena.codec,saved_bytes,saved_tag);
 if(threadIdx.x==0) {
   if(error==r7grammar::Error::ok) { io.output_bytes[chunk]=saved_bytes;io.output_tags[chunk]=saved_tag; }
   io.status[chunk]=error;
 }
}
template __global__ void cw_kernel<1>(GdnArgs,CWPacketIO);
template __global__ void cw_kernel<8>(GdnArgs,CWPacketIO);
}

#include "~/work/packing-mma-round10/method/packing-swarm/tools/round8/directory/packet.cuh"
namespace h5 {
template <int Window>
__device__ __forceinline__ r7grammar::Error parallel_cw_body(const GdnArgs& a, const Seg sg, int bx, uint8_t* smraw, const uint32_t* packet, unsigned packet_bytes, r7grammar::Tag input_tag, uint32_t* destination, unsigned capacity, unsigned base, r8directory::Scratch& scratch, unsigned& saved_bytes, r7grammar::Tag& saved_tag) {
  using ST=__half; constexpr int CG=4;
  float (*sk)[128] = (float (*)[128])smraw;
  float (*sq)[128] = (float (*)[128])(smraw + Window * 128 * 4);
  float (*sv)[64] = (float (*)[64])(smraw + Window * 256 * 4);
  float* sg_ = (float*)(smraw + Window * 320 * 4);
  float* sb_ = sg_ + Window;
  constexpr int JR = 256 / CG, MC = 128 / CG, CH = 64 + JR / 4;
  const int h = bx / (CG / 2), half = bx % (CG / 2), kh = h % HK;
  const int tid = threadIdx.x, lane = tid & 31;
  const int jl = tid / CG, j = JR * half + jl, cg = lane & (CG - 1);
  ST* pbase = (ST*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * 2 * a.pending_rows * PEND_GDN;
  const ST* prd = pbase + (size_t)(sg.par ^ 1) * a.pending_rows * PEND_GDN;
  ST* pwr = pbase + (size_t)sg.par * a.pending_rows * PEND_GDN;
  float m[MC];
  auto error=r8directory::load_chunk(packet,packet_bytes,input_tag,m,scratch);
  if(error!=r7grammar::Error::ok) return error;
  auto save = [&]() {
    __syncthreads();
    const auto result=r8directory::save_chunk(m,destination,capacity,packet,packet_bytes,base,scratch);
    saved_tag=result.tag;
    saved_bytes=result.bytes;
    return result.error;
  };
  int done = 0, replayed = 0;
  bool first = true;
  if (sg.keep && sg.commit == 0) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
  while (first || replayed < sg.commit || done < sg.len) {
    auto chunk = replay_chunk(sg.commit, sg.len, replayed, done, Window);
    if (sg.keep && chunk.replay_rows) chunk.new_rows=0;
    const int nrep = chunk.replay_rows, nnew = chunk.new_rows;
    const int T = nrep + nnew;
    __syncthreads();
    for (int i = tid; i < T * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      float4 v4; bool have = true;
      if (t < nrep) {
        const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN;
        if (c < 32) have = false;
        else v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      } else {
        const float* co = a.conv_out + (size_t)(sg.start + done + t - nrep) * CONV_CH;
        v4 = *(const float4*)(c < 32 ? co + kh * 128 + c * 4 : c < 64 ? co + KDIM + kh * 128 + (c - 32) * 4 : co + 2 * KDIM + h * 128 + JR * half + (c - 64) * 4);
      }
      if (have) {
        float* dst = c < 32 ? &sq[t][c * 4] : c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4];
        *(float4*)dst = v4;
      }
    }
    if (tid < T) {
      const int t = tid;
      if (t < nrep) { const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN; sg_[t] = st_ld(pr + KDIM + VDIM + h); sb_[t] = st_ld(pr + KDIM + VDIM + HV + h); }
      else { const float* gr = a.gates + (size_t)(sg.start + done + t - nrep) * 96; sg_[t] = gr[h]; sb_[t] = gr[HV + h]; }
    }
    __syncthreads();
    if (sg.keep) {   // new speculative rows -> pending[par] (disjoint from the replay buffer)
      for (int i = tid; i < nnew * 128; i += 256) {
        const int t = nrep + (i >> 7), d = i & 127;
        ST* pw = pwr + (size_t)(done + t - nrep) * PEND_GDN;
        if (h < HK && half == 0) st_st(pw + kh * 128 + d, sk[t][d]);
        if (d < JR && !a.replay_delta) st_st(pw + KDIM + h * 128 + JR * half + d, sv[t][d]);
        if (d == 0 && half == 0) { st_st(pw + KDIM + VDIM + h, sg_[t]); st_st(pw + KDIM + VDIM + HV + h, sb_[t]); }
      }
    }
#ifdef EXP_GDN_NOCOMP
    if (T > 1000)
#endif
    for (int t = 0; t < T; t++) {
      const float g = sg_[t], beta = sb_[t];
      const float* kp = &sk[t][4 * cg];
      float kr[MC];
#pragma unroll
      for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
      float delta;
      if (a.replay_delta && t < nrep) delta = sv[t][jl];
      else {
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
        float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
        delta = beta * (sv[t][jl] - g * s);
        if (a.replay_delta && sg.keep && cg == 0)
          st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
      }
#pragma unroll
      for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
      if (t >= nrep) {
        const float* qp = &sq[t][4 * cg];
        float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
        float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
        #pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
        if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
      }
    }
    if (sg.keep && nrep && replayed + nrep == sg.commit) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
    replayed += nrep; done += nnew; first = false;
  }
  if (!sg.keep) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
  return r7grammar::Error::ok;
}

struct ParallelCWPacketIO {
 const uint32_t* const* input; const unsigned* input_bytes; const r7grammar::Tag* input_tags;
 uint32_t* const* output; const unsigned* output_capacity;
 unsigned* output_bytes; r7grammar::Tag* output_tags; r7grammar::Error* status;
 unsigned common_window_base;
};
template<int Window> union alignas(16) ParallelCWPhaseArena {
 uint8_t tokens[Window*322*4];
 r8directory::Scratch codec;
};
template<int Window>
__global__ void __launch_bounds__(256,2) parallel_cw_kernel(GdnArgs a,ParallelCWPacketIO io) {
 static_assert(Window==1 || Window==8);
 __shared__ ParallelCWPhaseArena<Window> arena;
 const Seg sg=a.segs[blockIdx.y];
 const size_t chunk=((size_t(sg.slot)*NGDN+a.gl)*HV*2)+blockIdx.x;
 unsigned saved_bytes=0; r7grammar::Tag saved_tag=r7grammar::Tag::raw16384;
 auto error=parallel_cw_body<Window>(a,sg,blockIdx.x,arena.tokens,io.input[chunk],io.input_bytes[chunk],io.input_tags[chunk],
       io.output[chunk],io.output_capacity[chunk],io.common_window_base,arena.codec,saved_bytes,saved_tag);
 if(threadIdx.x==0) {
   if(error==r7grammar::Error::ok) { io.output_bytes[chunk]=saved_bytes;io.output_tags[chunk]=saved_tag; }
   io.status[chunk]=error;
 }
}
template __global__ void parallel_cw_kernel<1>(GdnArgs,ParallelCWPacketIO);
template __global__ void parallel_cw_kernel<8>(GdnArgs,ParallelCWPacketIO);
}

#include "~/work/packing-mma-round10/method/packing-swarm/tools/round9/adaptive/packet.cuh"
#include "~/work/packing-mma-round10/method/packing-swarm/tools/round6/decoder/decoder.cuh"
#include "~/work/packing-mma-round10/method/packing-swarm/tools/round6/encoder/save.cuh"
namespace h5 {
template <r6enc::Mode mode, int Window>
__device__ __forceinline__ r6wire::Error window_body(const GdnArgs& a, const Seg sg, int bx, uint8_t* smraw, const uint32_t* packet, unsigned packet_bytes, uint32_t* destination, unsigned capacity, r6enc::SaveScratch& scratch, unsigned& refusal, unsigned& saved_bytes) {
  using ST=__half; constexpr int CG=4;
  float (*sk)[128] = (float (*)[128])smraw;
  float (*sq)[128] = (float (*)[128])(smraw + Window * 128 * 4);
  float (*sv)[64] = (float (*)[64])(smraw + Window * 256 * 4);
  float* sg_ = (float*)(smraw + Window * 320 * 4);
  float* sb_ = sg_ + Window;
  constexpr int JR = 256 / CG, MC = 128 / CG, CH = 64 + JR / 4;
  const int h = bx / (CG / 2), half = bx % (CG / 2), kh = h % HK;
  const int tid = threadIdx.x, lane = tid & 31;
  const int jl = tid / CG, j = JR * half + jl, cg = lane & (CG - 1);
  ST* pbase = (ST*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * 2 * a.pending_rows * PEND_GDN;
  const ST* prd = pbase + (size_t)(sg.par ^ 1) * a.pending_rows * PEND_GDN;
  ST* pwr = pbase + (size_t)sg.par * a.pending_rows * PEND_GDN;
  float m[MC];
  auto error=r6decode::load_chunk(packet,packet_bytes,m,refusal);
  if(error!=r6wire::Error::ok) return error;
  auto save = [&]() {
    __syncthreads();
    const auto result=r6enc::save_chunk<mode>(m,destination,capacity,packet,packet_bytes,scratch,
          mode==r6enc::Mode::fixed_window?3:0,mode==r6enc::Mode::fixed_window?10:0);
    saved_bytes=result.bytes;
    return result.error;
  };
  int done = 0, replayed = 0;
  bool first = true;
  if (sg.keep && sg.commit == 0) { auto e=save(); if(e!=r6wire::Error::ok) return e; }
  while (first || replayed < sg.commit || done < sg.len) {
    auto chunk = replay_chunk(sg.commit, sg.len, replayed, done, Window);
    if (sg.keep && chunk.replay_rows) chunk.new_rows=0;
    const int nrep = chunk.replay_rows, nnew = chunk.new_rows;
    const int T = nrep + nnew;
    __syncthreads();
    for (int i = tid; i < T * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      float4 v4; bool have = true;
      if (t < nrep) {
        const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN;
        if (c < 32) have = false;
        else v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      } else {
        const float* co = a.conv_out + (size_t)(sg.start + done + t - nrep) * CONV_CH;
        v4 = *(const float4*)(c < 32 ? co + kh * 128 + c * 4 : c < 64 ? co + KDIM + kh * 128 + (c - 32) * 4 : co + 2 * KDIM + h * 128 + JR * half + (c - 64) * 4);
      }
      if (have) {
        float* dst = c < 32 ? &sq[t][c * 4] : c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4];
        *(float4*)dst = v4;
      }
    }
    if (tid < T) {
      const int t = tid;
      if (t < nrep) { const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN; sg_[t] = st_ld(pr + KDIM + VDIM + h); sb_[t] = st_ld(pr + KDIM + VDIM + HV + h); }
      else { const float* gr = a.gates + (size_t)(sg.start + done + t - nrep) * 96; sg_[t] = gr[h]; sb_[t] = gr[HV + h]; }
    }
    __syncthreads();
    if (sg.keep) {   // new speculative rows -> pending[par] (disjoint from the replay buffer)
      for (int i = tid; i < nnew * 128; i += 256) {
        const int t = nrep + (i >> 7), d = i & 127;
        ST* pw = pwr + (size_t)(done + t - nrep) * PEND_GDN;
        if (h < HK && half == 0) st_st(pw + kh * 128 + d, sk[t][d]);
        if (d < JR && !a.replay_delta) st_st(pw + KDIM + h * 128 + JR * half + d, sv[t][d]);
        if (d == 0 && half == 0) { st_st(pw + KDIM + VDIM + h, sg_[t]); st_st(pw + KDIM + VDIM + HV + h, sb_[t]); }
      }
    }
#ifdef EXP_GDN_NOCOMP
    if (T > 1000)
#endif
    for (int t = 0; t < T; t++) {
      const float g = sg_[t], beta = sb_[t];
      const float* kp = &sk[t][4 * cg];
      float kr[MC];
#pragma unroll
      for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
      float delta;
      if (a.replay_delta && t < nrep) delta = sv[t][jl];
      else {
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
        float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
        delta = beta * (sv[t][jl] - g * s);
        if (a.replay_delta && sg.keep && cg == 0)
          st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
      }
#pragma unroll
      for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
      if (t >= nrep) {
        const float* qp = &sq[t][4 * cg];
        float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
        float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
        #pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
        if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
      }
    }
    if (sg.keep && nrep && replayed + nrep == sg.commit) { auto e=save(); if(e!=r6wire::Error::ok) return e; }
    replayed += nrep; done += nnew; first = false;
  }
  if (!sg.keep) { auto e=save(); if(e!=r6wire::Error::ok) return e; }
  return r6wire::Error::ok;
}

struct PacketIO {
 const uint32_t* const* input; const unsigned* input_bytes;
 uint32_t* const* output; const unsigned* output_capacity;
 unsigned* output_bytes; r6wire::Error* status;
};
template<int Window> union alignas(16) PhaseArena {
 uint8_t tokens[Window*322*4];
 r6enc::SaveScratch codec;
};
template<r6enc::Mode mode,int Window>
__global__ void __launch_bounds__(256,2) window_kernel(GdnArgs a,PacketIO io) {
 static_assert(Window==1 || Window==8);
 __shared__ PhaseArena<Window> arena;
 __shared__ unsigned refusal;
 const Seg sg=a.segs[blockIdx.y];
 const size_t chunk=((size_t(sg.slot)*NGDN+a.gl)*HV*2)+blockIdx.x;
 unsigned saved_bytes=0;
 auto error=window_body<mode,Window>(a,sg,blockIdx.x,arena.tokens,io.input[chunk],io.input_bytes[chunk],
       io.output[chunk],io.output_capacity[chunk],arena.codec,refusal,saved_bytes);
 if(threadIdx.x==0) {
   if(error==r6wire::Error::ok) io.output_bytes[chunk]=saved_bytes;
   io.status[chunk]=error;
 }
}
template __global__ void window_kernel<r6enc::Mode::adaptive,1>(GdnArgs,PacketIO);
template __global__ void window_kernel<r6enc::Mode::raw,1>(GdnArgs,PacketIO);
template __global__ void window_kernel<r6enc::Mode::fixed_window,1>(GdnArgs,PacketIO);
template __global__ void window_kernel<r6enc::Mode::adaptive,8>(GdnArgs,PacketIO);
template __global__ void window_kernel<r6enc::Mode::raw,8>(GdnArgs,PacketIO);
template __global__ void window_kernel<r6enc::Mode::fixed_window,8>(GdnArgs,PacketIO);
template <int Window>
__device__ __forceinline__ r7grammar::Error adaptive_cw_body(const GdnArgs& a, const Seg sg, int bx, uint8_t* smraw, const uint32_t* packet, unsigned packet_bytes, r7grammar::Tag input_tag, uint32_t* destination, unsigned capacity, r9adaptive::Scratch& scratch, unsigned& saved_bytes, r7grammar::Tag& saved_tag) {
  using ST=__half; constexpr int CG=4;
  float (*sk)[128] = (float (*)[128])smraw;
  float (*sq)[128] = (float (*)[128])(smraw + Window * 128 * 4);
  float (*sv)[64] = (float (*)[64])(smraw + Window * 256 * 4);
  float* sg_ = (float*)(smraw + Window * 320 * 4);
  float* sb_ = sg_ + Window;
  constexpr int JR = 256 / CG, MC = 128 / CG, CH = 64 + JR / 4;
  const int h = bx / (CG / 2), half = bx % (CG / 2), kh = h % HK;
  const int tid = threadIdx.x, lane = tid & 31;
  const int jl = tid / CG, j = JR * half + jl, cg = lane & (CG - 1);
  ST* pbase = (ST*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * 2 * a.pending_rows * PEND_GDN;
  const ST* prd = pbase + (size_t)(sg.par ^ 1) * a.pending_rows * PEND_GDN;
  ST* pwr = pbase + (size_t)sg.par * a.pending_rows * PEND_GDN;
  float m[MC];
  auto error=r9adaptive::load_chunk(packet,packet_bytes,input_tag,m,scratch);
  if(error!=r7grammar::Error::ok) return error;
  auto save = [&]() {
    __syncthreads();
    const auto result=r9adaptive::save_chunk(m,destination,capacity,packet,packet_bytes,scratch);
    saved_tag=result.tag;
    saved_bytes=result.bytes;
    return result.error;
  };
  int done = 0, replayed = 0;
  bool first = true;
  if (sg.keep && sg.commit == 0) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
  while (first || replayed < sg.commit || done < sg.len) {
    auto chunk = replay_chunk(sg.commit, sg.len, replayed, done, Window);
    if (sg.keep && chunk.replay_rows) chunk.new_rows=0;
    const int nrep = chunk.replay_rows, nnew = chunk.new_rows;
    const int T = nrep + nnew;
    __syncthreads();
    for (int i = tid; i < T * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      float4 v4; bool have = true;
      if (t < nrep) {
        const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN;
        if (c < 32) have = false;
        else v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      } else {
        const float* co = a.conv_out + (size_t)(sg.start + done + t - nrep) * CONV_CH;
        v4 = *(const float4*)(c < 32 ? co + kh * 128 + c * 4 : c < 64 ? co + KDIM + kh * 128 + (c - 32) * 4 : co + 2 * KDIM + h * 128 + JR * half + (c - 64) * 4);
      }
      if (have) {
        float* dst = c < 32 ? &sq[t][c * 4] : c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4];
        *(float4*)dst = v4;
      }
    }
    if (tid < T) {
      const int t = tid;
      if (t < nrep) { const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN; sg_[t] = st_ld(pr + KDIM + VDIM + h); sb_[t] = st_ld(pr + KDIM + VDIM + HV + h); }
      else { const float* gr = a.gates + (size_t)(sg.start + done + t - nrep) * 96; sg_[t] = gr[h]; sb_[t] = gr[HV + h]; }
    }
    __syncthreads();
    if (sg.keep) {   // new speculative rows -> pending[par] (disjoint from the replay buffer)
      for (int i = tid; i < nnew * 128; i += 256) {
        const int t = nrep + (i >> 7), d = i & 127;
        ST* pw = pwr + (size_t)(done + t - nrep) * PEND_GDN;
        if (h < HK && half == 0) st_st(pw + kh * 128 + d, sk[t][d]);
        if (d < JR && !a.replay_delta) st_st(pw + KDIM + h * 128 + JR * half + d, sv[t][d]);
        if (d == 0 && half == 0) { st_st(pw + KDIM + VDIM + h, sg_[t]); st_st(pw + KDIM + VDIM + HV + h, sb_[t]); }
      }
    }
#ifdef EXP_GDN_NOCOMP
    if (T > 1000)
#endif
    for (int t = 0; t < T; t++) {
      const float g = sg_[t], beta = sb_[t];
      const float* kp = &sk[t][4 * cg];
      float kr[MC];
#pragma unroll
      for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
      float delta;
      if (a.replay_delta && t < nrep) delta = sv[t][jl];
      else {
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
        float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
        delta = beta * (sv[t][jl] - g * s);
        if (a.replay_delta && sg.keep && cg == 0)
          st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
      }
#pragma unroll
      for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
      if (t >= nrep) {
        const float* qp = &sq[t][4 * cg];
        float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
        float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
        #pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
        if (cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
      }
    }
    if (sg.keep && nrep && replayed + nrep == sg.commit) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
    replayed += nrep; done += nnew; first = false;
  }
  if (!sg.keep) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
  return r7grammar::Error::ok;
}

struct AdaptiveCWPacketIO {
 const uint32_t* const* input; const unsigned* input_bytes; const r7grammar::Tag* input_tags;
 uint32_t* const* output; const unsigned* output_capacity;
 unsigned* output_bytes; r7grammar::Tag* output_tags; r7grammar::Error* status;
};
template<int Window> union alignas(16) AdaptiveCWPhaseArena {
 uint8_t tokens[Window*322*4];
 r9adaptive::Scratch codec;
};
template<int Window>
__global__ void __launch_bounds__(256,2) adaptive_cw_kernel(GdnArgs a,AdaptiveCWPacketIO io) {
 static_assert(Window==1 || Window==8);
 __shared__ AdaptiveCWPhaseArena<Window> arena;
 const Seg sg=a.segs[blockIdx.y];
 const size_t chunk=((size_t(sg.slot)*NGDN+a.gl)*HV*2)+blockIdx.x;
 unsigned saved_bytes=0; r7grammar::Tag saved_tag=r7grammar::Tag::raw16384;
 auto error=adaptive_cw_body<Window>(a,sg,blockIdx.x,arena.tokens,io.input[chunk],io.input_bytes[chunk],io.input_tags[chunk],
       io.output[chunk],io.output_capacity[chunk],arena.codec,saved_bytes,saved_tag);
 if(threadIdx.x==0) {
   if(error==r7grammar::Error::ok) { io.output_bytes[chunk]=saved_bytes;io.output_tags[chunk]=saved_tag; }
   io.status[chunk]=error;
 }
}
template __global__ void adaptive_cw_kernel<1>(GdnArgs,AdaptiveCWPacketIO);
template __global__ void adaptive_cw_kernel<8>(GdnArgs,AdaptiveCWPacketIO);
}

#include "~/work/packing-mma-round10/method/packing-swarm/tools/round10/adaptive/packet.cuh"
namespace h5 {
template <int Window,bool Discover,bool LayerLocal>
__device__ __forceinline__ r7grammar::Error direct_adaptive_body(const GdnArgs& a, const Seg sg, int bx, uint8_t* smraw, const uint32_t* packet, unsigned packet_bytes, r7grammar::Tag input_tag, uint32_t* destination, unsigned capacity, unsigned expected_base, r7grammar::Tag expected_tag, r10adaptive::Scratch& scratch, unsigned& saved_bytes, r7grammar::Tag& saved_tag, unsigned& saved_base, const __half* pending_read0,const __half* pending_read1,__half* pending_write0,__half* pending_write1) {
  using ST=__half; constexpr int CG=4;
  float (*sk)[128] = (float (*)[128])smraw;
  float (*sq)[128] = (float (*)[128])(smraw + Window * 128 * 4);
  float (*sv)[64] = (float (*)[64])(smraw + Window * 256 * 4);
  float* sg_ = (float*)(smraw + Window * 320 * 4);
  float* sb_ = sg_ + Window;
  constexpr int JR = 256 / CG, MC = 128 / CG, CH = 64 + JR / 4;
  const int h = bx / (CG / 2), half = bx % (CG / 2), kh = h % HK;
  const int tid = threadIdx.x, lane = tid & 31;
  const int jl = tid / CG, j = JR * half + jl, cg = lane & (CG - 1);
  const size_t pending_offset=(LayerLocal?size_t(sg.slot):(size_t(sg.slot)*NGDN+a.gl))*a.pending_rows*PEND_GDN;
  const ST* prd=(sg.par?pending_read0:pending_read1)+pending_offset;
  ST* pwr=nullptr;
  if constexpr(!Discover) pwr=(sg.par?pending_write1:pending_write0)+pending_offset;
  float m[MC];
  auto error=r10adaptive::load_chunk(packet,packet_bytes,input_tag,m,scratch);
  if(error!=r7grammar::Error::ok) return error;
  auto save = [&]() {
    __syncthreads();
    r7grammar::Result result;
    result=r10adaptive::save_chunk<Discover>(m,destination,capacity,packet,packet_bytes,scratch,expected_base,capacity,expected_tag);
    saved_tag=result.tag;
    if(result.error==r7grammar::Error::ok) saved_base=scratch.base;
    saved_bytes=result.bytes;
    return result.error;
  };
  int done = 0, replayed = 0;
  bool first = true;
  if (sg.keep && sg.commit == 0) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
  while (first || replayed < sg.commit || done < sg.len) {
    auto chunk = replay_chunk(sg.commit, sg.len, replayed, done, Window);
    if (sg.keep && chunk.replay_rows) chunk.new_rows=0;
    const int nrep = chunk.replay_rows, nnew = chunk.new_rows;
    const int T = nrep + nnew;
    __syncthreads();
    for (int i = tid; i < T * CH; i += 256) {
      const int t = i / CH, c = i % CH;
      float4 v4; bool have = true;
      if (t < nrep) {
        const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN;
        if (c < 32) have = false;
        else v4 = ld4(c < 64 ? pr + kh * 128 + (c - 32) * 4 : pr + KDIM + h * 128 + JR * half + (c - 64) * 4);
      } else {
        const float* co = a.conv_out + (size_t)(sg.start + done + t - nrep) * CONV_CH;
        v4 = *(const float4*)(c < 32 ? co + kh * 128 + c * 4 : c < 64 ? co + KDIM + kh * 128 + (c - 32) * 4 : co + 2 * KDIM + h * 128 + JR * half + (c - 64) * 4);
      }
      if (have) {
        float* dst = c < 32 ? &sq[t][c * 4] : c < 64 ? &sk[t][(c - 32) * 4] : &sv[t][(c - 64) * 4];
        *(float4*)dst = v4;
      }
    }
    if (tid < T) {
      const int t = tid;
      if (t < nrep) { const ST* pr = prd + (size_t)(replayed + t) * PEND_GDN; sg_[t] = st_ld(pr + KDIM + VDIM + h); sb_[t] = st_ld(pr + KDIM + VDIM + HV + h); }
      else { const float* gr = a.gates + (size_t)(sg.start + done + t - nrep) * 96; sg_[t] = gr[h]; sb_[t] = gr[HV + h]; }
    }
    __syncthreads();
    if (!Discover && sg.keep) {
      for (int i = tid; i < nnew * 128; i += 256) {
        const int t = nrep + (i >> 7), d = i & 127;
        ST* pw = pwr + (size_t)(done + t - nrep) * PEND_GDN;
        if (h < HK && half == 0) st_st(pw + kh * 128 + d, sk[t][d]);
        if (d < JR && !a.replay_delta) st_st(pw + KDIM + h * 128 + JR * half + d, sv[t][d]);
        if (d == 0 && half == 0) { st_st(pw + KDIM + VDIM + h, sg_[t]); st_st(pw + KDIM + VDIM + HV + h, sb_[t]); }
      }
    }
#ifdef EXP_GDN_NOCOMP
    if (T > 1000)
#endif
    for (int t = 0; t < T; t++) {
      const float g = sg_[t], beta = sb_[t];
      const float* kp = &sk[t][4 * cg];
      float kr[MC];
#pragma unroll
      for (int i = 0; i < MC / 4; i++) { const float4 k4 = *(const float4*)(kp + CG * 4 * i); kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w; }
      float delta;
      if (a.replay_delta && t < nrep) delta = sv[t][jl];
      else {
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC; i++) s4[i & 3] = fmaf(m[i], kr[i], s4[i & 3]);
        float s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
        delta = beta * (sv[t][jl] - g * s);
        if (!Discover && a.replay_delta && sg.keep && cg == 0)
          st_st(pwr + (size_t)(done + t - nrep) * PEND_GDN + KDIM + h * 128 + j, delta);
      }
#pragma unroll
      for (int i = 0; i < MC; i++) m[i] = fmaf(g, m[i], delta * kr[i]);
      if (t >= nrep) {
        const float* qp = &sq[t][4 * cg];
        float o4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < MC / 4; i++) { const float4 q4 = *(const float4*)(qp + CG * 4 * i); o4[0] = fmaf(m[4 * i], q4.x, o4[0]); o4[1] = fmaf(m[4 * i + 1], q4.y, o4[1]); o4[2] = fmaf(m[4 * i + 2], q4.z, o4[2]); o4[3] = fmaf(m[4 * i + 3], q4.w, o4[3]); }
        float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
        #pragma unroll
        for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
        if (!Discover && cg == 0) a.y[(size_t)(sg.start + done + t - nrep) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
      }
    }
    if (sg.keep && nrep && replayed + nrep == sg.commit) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
    replayed += nrep; done += nnew; first = false;
  }
  if (!sg.keep) { auto e=save(); if(e!=r7grammar::Error::ok) return e; }
  return r7grammar::Error::ok;
}

struct DirectAdaptiveIO {
 const uint32_t* const* input; const unsigned* input_bytes; const r7grammar::Tag* input_tags;
 uint32_t* const* output; const unsigned* capacity;
 const unsigned* expected_bases; const r7grammar::Tag* expected_tags;
 unsigned* bytes; r7grammar::Tag* tags; unsigned* bases; r7grammar::Error* status;
 const __half* pending_read[2]; __half* pending_write[2];
};
template<int Window> union alignas(16) DirectAdaptiveArena { uint8_t tokens[Window*322*4];r10adaptive::Scratch codec; };
template<int Window,bool Discover>
__global__ void __launch_bounds__(256,2) direct_adaptive_kernel(GdnArgs a,DirectAdaptiveIO io) {
 static_assert(Window==1||Window==8);
 __shared__ DirectAdaptiveArena<Window> arena;
 const Seg sg=a.segs[blockIdx.y];
 const size_t chunk=((size_t(sg.slot)*NGDN+a.gl)*HV*2)+blockIdx.x;
 unsigned bytes=0,base=0,expected_base=0,capacity=0;
 r7grammar::Tag tag=r7grammar::Tag::raw16384,expected_tag=r7grammar::Tag::raw16384;
 uint32_t* output=nullptr;
 if constexpr(!Discover) {output=io.output[chunk];capacity=io.capacity[chunk];expected_base=io.expected_bases[chunk];expected_tag=io.expected_tags[chunk];}
 auto error=direct_adaptive_body<Window,Discover,false>(a,sg,blockIdx.x,arena.tokens,io.input[chunk],io.input_bytes[chunk],io.input_tags[chunk],output,capacity,expected_base,expected_tag,arena.codec,bytes,tag,base,io.pending_read[0],io.pending_read[1],io.pending_write[0],io.pending_write[1]);
 if(threadIdx.x==0) {if(error==r7grammar::Error::ok){io.bytes[chunk]=bytes;io.tags[chunk]=tag;io.bases[chunk]=base;}io.status[chunk]=error;}
}
template __global__ void direct_adaptive_kernel<1,true>(GdnArgs,DirectAdaptiveIO);
template __global__ void direct_adaptive_kernel<8,true>(GdnArgs,DirectAdaptiveIO);
template __global__ void direct_adaptive_kernel<1,false>(GdnArgs,DirectAdaptiveIO);
template __global__ void direct_adaptive_kernel<8,false>(GdnArgs,DirectAdaptiveIO);
template<int Window,bool Discover>
__global__ void __launch_bounds__(256,2) direct_adaptive_local_kernel(GdnArgs a,DirectAdaptiveIO io) {
 static_assert(Window==1||Window==8);
 __shared__ DirectAdaptiveArena<Window> arena;
 const Seg sg=a.segs[blockIdx.y];
 const size_t chunk=size_t(sg.slot)*HV*2+blockIdx.x;
 unsigned bytes=0,base=0,expected_base=0,capacity=0;
 r7grammar::Tag tag=r7grammar::Tag::raw16384,expected_tag=r7grammar::Tag::raw16384;
 uint32_t* output=nullptr;
 if constexpr(!Discover) {output=io.output[chunk];capacity=io.capacity[chunk];expected_base=io.expected_bases[chunk];expected_tag=io.expected_tags[chunk];}
 auto error=direct_adaptive_body<Window,Discover,true>(a,sg,blockIdx.x,arena.tokens,io.input[chunk],io.input_bytes[chunk],io.input_tags[chunk],output,capacity,expected_base,expected_tag,arena.codec,bytes,tag,base,io.pending_read[0],io.pending_read[1],io.pending_write[0],io.pending_write[1]);
 if(threadIdx.x==0) {if(error==r7grammar::Error::ok){io.bytes[chunk]=bytes;io.tags[chunk]=tag;io.bases[chunk]=base;}io.status[chunk]=error;}
}
template __global__ void direct_adaptive_local_kernel<1,true>(GdnArgs,DirectAdaptiveIO);
template __global__ void direct_adaptive_local_kernel<1,false>(GdnArgs,DirectAdaptiveIO);
template __global__ void direct_adaptive_local_kernel<8,true>(GdnArgs,DirectAdaptiveIO);
template __global__ void direct_adaptive_local_kernel<8,false>(GdnArgs,DirectAdaptiveIO);
}
