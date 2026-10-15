// Included in kernels.cu after its storage and launch-selection helpers.
template <int R, int CG, bool CHUNK> struct alignas(16) GdnVerifyShared {
  static constexpr int JR = 256 / CG;
  float q[R][128], k[16][128], v[16][JR], g[16], beta[16];
};
template <int R, int CG> struct alignas(16) GdnVerifyChunkShared : GdnVerifyShared<R, CG, true> {
  float kk[R][R], kq[R][R];
};
template <int CG>
__device__ __forceinline__ float verify_dot(const float* m, const float* row, int cg) {
  float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 128 / CG / 4; ++i) {
    const float4 x = *(const float4*)(row + 4 * cg + CG * 4 * i);
    s[0] = fmaf(m[4*i], x.x, s[0]); s[1] = fmaf(m[4*i+1], x.y, s[1]);
    s[2] = fmaf(m[4*i+2], x.z, s[2]); s[3] = fmaf(m[4*i+3], x.w, s[3]);
  }
  float out = (s[0] + s[1]) + (s[2] + s[3]);
#pragma unroll
  for (int sh = 1; sh < CG; sh <<= 1) out += __shfl_xor_sync(0xffffffff, out, sh);
  return out;
}
template <int CG>
__device__ __forceinline__ void verify_update(float* m, const float* k, float g, float delta, int cg) {
#pragma unroll
  for (int i = 0; i < 128 / CG / 4; ++i) {
    const float4 x = *(const float4*)(k + 4 * cg + CG * 4 * i);
    m[4*i] = fmaf(g, m[4*i], delta * x.x); m[4*i+1] = fmaf(g, m[4*i+1], delta * x.y);
    m[4*i+2] = fmaf(g, m[4*i+2], delta * x.z); m[4*i+3] = fmaf(g, m[4*i+3], delta * x.w);
  }
}
template <int CG>
__device__ __forceinline__ void verify_save(__half* dst, const float* m) {
#pragma unroll
  for (int i = 0; i < 128 / CG / 4; ++i) {
    uint2 u;
    ((__half2*)&u)[0] = __floats2half2_rn(m[4*i], m[4*i+1]);
    ((__half2*)&u)[1] = __floats2half2_rn(m[4*i+2], m[4*i+3]);
    *(uint2*)(dst + CG * 4 * i) = u;
  }
}

template <int R, int CG, bool CHUNK>
__global__ void __launch_bounds__(256, 2) k_gdn_verify(GdnArgs a) { H5_LIT_SITE(GdnVerify, 0);
  constexpr int JR = 256 / CG, MC = 128 / CG, PARTS = CG / 2;
  using Shared = std::conditional_t<CHUNK, GdnVerifyChunkShared<R, CG>, GdnVerifyShared<R, CG, false>>;
  __shared__ Shared sm;
#ifndef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  const Seg sg = a.segs[0];
  if (sg.len != R || sg.keep != 1 || sg.commit < 0 || sg.commit > a.pending_rows ||
      sg.slot < 0 || sg.start < 0 || (sg.par != 0 && sg.par != 1)) { asm volatile("trap;"); return; }
  const int tid = threadIdx.x, lane = tid & 31, cg = tid & (CG - 1), jl = tid / CG;
  const int h = blockIdx.x / PARTS, part = blockIdx.x % PARTS, kh = h % HK, j = part * JR + jl;
  __half* M = (__half*)a.state + (((size_t)sg.slot * NGDN + a.gl) * HV + h) * 128 * 128 + j * 128 + 4 * cg;
  __half* pbase = (__half*)a.pend + ((size_t)sg.slot * NGDN + a.gl) * 2 * a.pending_rows * PEND_GDN;
  const __half* prd = pbase + (size_t)(sg.par ^ 1) * a.pending_rows * PEND_GDN;
  __half* pwr = pbase + (size_t)sg.par * a.pending_rows * PEND_GDN;
  float m[MC];
#pragma unroll
  for (int i = 0; i < MC / 4; ++i) {
    const float4 x = ld4(M + CG * 4 * i);
    m[4*i] = x.x; m[4*i+1] = x.y; m[4*i+2] = x.z; m[4*i+3] = x.w;
  }
#ifdef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  H5_LIT_MARKX(0);
  const bool combined = sg.commit + R <= 16;
  const int base = combined ? sg.commit : 0;
  auto stage_new = [&]() {
    constexpr int PIECES = 64 + JR / 4;
    for (int i = tid; i < R * PIECES; i += 256) {
      const int t = i / PIECES, d = i % PIECES;
      const float* src = a.conv_out + (size_t)(sg.start + t) * CONV_CH;
      if (d < 32) *(float4*)(sm.q[t] + d * 4) = *(const float4*)(src + kh * 128 + d * 4);
      else if (d < 64) *(float4*)(sm.k[base + t] + (d - 32) * 4) = *(const float4*)(src + KDIM + kh * 128 + (d - 32) * 4);
      else *(float4*)(sm.v[base + t] + (d - 64) * 4) = *(const float4*)(src + 2 * KDIM + h * 128 + part * JR + (d - 64) * 4);
    }
    if (tid < R) {
      const float* src = a.gates + (size_t)(sg.start + tid) * 96;
      sm.g[base + tid] = src[h]; sm.beta[base + tid] = src[HV + h];
    }
  };
  constexpr int REP_PIECES = 32 + JR / 4;
  for (int i = tid; i < sg.commit * REP_PIECES; i += 256) {
    const int t = i / REP_PIECES, d = i % REP_PIECES;
    const __half* src = prd + (size_t)t * PEND_GDN;
    if (d < 32) *(float4*)(sm.k[t] + d * 4) = ld4(src + kh * 128 + d * 4);
    else *(float4*)(sm.v[t] + (d - 32) * 4) = ld4(src + KDIM + h * 128 + part * JR + (d - 32) * 4);
  }
  if (tid < sg.commit) {
    const __half* src = prd + (size_t)tid * PEND_GDN;
    sm.g[tid] = st_ld(src + KDIM + VDIM + h);
    sm.beta[tid] = st_ld(src + KDIM + VDIM + HV + h);
  }
  if (combined) stage_new();
  __syncthreads();
  // New rows share the staging barrier when they fit beside replay rows. The checkpoint is F16,
  // but new rows continue from the unrounded replay registers, as in the generic path.
  for (int t = 0; t < sg.commit; ++t) {
    const float delta = a.replay_delta ? sm.v[t][jl] :
      sm.beta[t] * (sm.v[t][jl] - sm.g[t] * verify_dot<CG>(m, sm.k[t], cg));
    verify_update<CG>(m, sm.k[t], sm.g[t], delta, cg);
  }
  if (sg.commit) verify_save<CG>(M, m);
  if (!combined) {
    __syncthreads();
    stage_new();
    __syncthreads();
  }
  H5_LIT_MARKX(1);
  for (int i = tid; i < R * 128; i += 256) {
    const int t = i / 128, d = i % 128;
    __half* dst = pwr + (size_t)t * PEND_GDN;
    if (h < HK && part == 0) st_st(dst + kh * 128 + d, sm.k[base + t][d]);
    if (d < JR && !a.replay_delta) st_st(dst + KDIM + h * 128 + part * JR + d, sm.v[base + t][d]);
    if (d == 0 && part == 0) {
      st_st(dst + KDIM + VDIM + h, sm.g[base + t]); st_st(dst + KDIM + VDIM + HV + h, sm.beta[base + t]);
    }
  }
  if constexpr (!CHUNK) {
#pragma unroll 1
    for (int t = 0; t < R; ++t) {
      const float g = sm.g[base + t], delta = sm.beta[base + t] * (sm.v[base + t][jl] - g * verify_dot<CG>(m, sm.k[base + t], cg));
      if (cg == 0 && a.replay_delta) st_st(pwr + (size_t)t * PEND_GDN + KDIM + h * 128 + j, delta);
      verify_update<CG>(m, sm.k[base + t], g, delta, cg);
      const float o = verify_dot<CG>(m, sm.q[t], cg);
      if (cg == 0) a.y[(size_t)(sg.start + t) * VDIM + h * 128 + j] = o * 0.08838834764831845f;
    }
  } else {
    const int warp = tid >> 5;
    for (int pair = warp; pair < 2 * R * R; pair += 8) {
      const bool isq = pair >= R * R;
      const int u = (pair % (R * R)) / R, t = pair % R;
      if (isq ? u > t : u >= t) continue;
      const float4 x = *(const float4*)(sm.k[base + u] + lane * 4);
      const float4 y = *(const float4*)((isq ? sm.q[t] : sm.k[base + t]) + lane * 4);
      float d = fmaf(x.x, y.x, fmaf(x.y, y.y, fmaf(x.z, y.z, x.w * y.w)));
#pragma unroll
      for (int sh = 16; sh; sh >>= 1) d += __shfl_xor_sync(0xffffffff, d, sh);
      if (lane == 0) (isq ? sm.kq : sm.kk)[u][t] = d;
    }
    float mk[R], mq[R];
#pragma unroll
    for (int t = 0; t < R; ++t) {
      mk[t] = verify_dot<CG>(m, sm.k[base + t], cg);
      mq[t] = verify_dot<CG>(m, sm.q[t], cg);
    }
    __syncthreads();
    // No new state is observed when keep=1. Only this scalar recurrence, y and pending innovations
    // survive; materializing speculative M_t (the old chunk's expensive O(R^2*MC) tail) is unnecessary.
    if (cg == 0) {
      float w[R], decay = 1.f;
#pragma unroll
      for (int t = 0; t < R; ++t) {
        float s0 = decay * mk[t], s1 = 0.f;
#pragma unroll
        for (int u = 0; u < t; ++u) {
          if (u & 1) s1 = fmaf(w[u], sm.kk[u][t], s1);
          else s0 = fmaf(w[u], sm.kk[u][t], s0);
        }
        const float g = sm.g[base + t], delta = sm.beta[base + t] * (sm.v[base + t][jl] - g * (s0 + s1));
        if (a.replay_delta) st_st(pwr + (size_t)t * PEND_GDN + KDIM + h * 128 + j, delta);
#pragma unroll
        for (int u = 0; u < t; ++u) w[u] *= g;
        decay *= g; w[t] = delta;
        float o0 = decay * mq[t], o1 = 0.f;
#pragma unroll
        for (int u = 0; u <= t; ++u) {
          if (u & 1) o1 = fmaf(w[u], sm.kq[u][t], o1);
          else o0 = fmaf(w[u], sm.kq[u][t], o0);
        }
        a.y[(size_t)(sg.start + t) * VDIM + h * 128 + j] = (o0 + o1) * 0.08838834764831845f;
      }
    }
  }
  H5_LIT_MARKX(2);
  prep_signal(a.sig);
}

GdnVerifyMode gdn_verify_mode_from_env() {
  const char* value = getenv("H5_GDN_VERIFY");
  if (!value || !strcmp(value, "off")) return GdnVerifyMode::Off;
  if (!strcmp(value, "serial")) return GdnVerifyMode::Serial;
  if (!strcmp(value, "chunk")) return GdnVerifyMode::Chunk;
  if (!strcmp(value, "fused")) return GdnVerifyMode::Fused;
  throw std::runtime_error("H5_GDN_VERIFY requires off|serial|chunk|fused");
}
template <int R, bool CHUNK>
void gdn_verify_dispatch(const GdnArgs& a, cudaStream_t st) {
  const int cg = gdn_cg(1);
  if (cg == 4) pdl(k_gdn_verify<R,4,CHUNK>, dim3(2 * HV), dim3(256), 0, st, a);
  else if (cg == 8) pdl(k_gdn_verify<R,8,CHUNK>, dim3(4 * HV), dim3(256), 0, st, a);
  else pdl(k_gdn_verify<R,16,CHUNK>, dim3(8 * HV), dim3(256), 0, st, a);
}
void launch_gdn_verify(const GdnArgs& a, int nseg, int rows, GdnVerifyMode mode, StateFormat f, cudaStream_t st) {
  if (mode == GdnVerifyMode::Off) { launch_gdn(a, nseg, f, st); return; }
  if (mode == GdnVerifyMode::Fused)
    throw std::runtime_error("fused GDN verify requires launch_conv_gdn_verify and two-bank convolution state");
  if (mode != GdnVerifyMode::Serial && mode != GdnVerifyMode::Chunk)
    throw std::runtime_error("unknown GDN verify mode");
  if (nseg != 1 || (rows != 8 && rows != 16) || f != StateFormat::F16 || !a.state || !a.segs ||
      !a.pend || !a.y || !a.conv_out || !a.gates || a.gl < 0 || a.gl >= NGDN ||
      (a.pending_rows != 8 && a.pending_rows != 16) || a.pending_rows < rows)
    throw std::runtime_error("GDN verify requires one F16 speculative segment, 8/16 rows, pending capacity 8/16 and all operands");
  if (gdn_packed(f) || gdn_env("H5_GDN_SCALED", 0, {0, 1}) ||
      gdn_env("H5_GDN_PIPE", 0, {0, 2, 4, 8, 16, 24, 32, 48, 64}) ||
      gdn_env("H5_GDN_OCC", 2, {2, 3}) != 2 || (gdn_host_variant & 96u))
    throw std::runtime_error("GDN verify requires fp32 arithmetic, SCALED=0, PIPE=0, OCC=2 and no 0x20/0x40 variants");
#if defined(EXP_GDN_NOMEM) || defined(EXP_GDN_NOCOMP)
  throw std::runtime_error("GDN verify requires state memory and recurrence computation");
#endif
  if (rows == 8) {
    if (mode == GdnVerifyMode::Chunk) gdn_verify_dispatch<8,true>(a, st);
    else gdn_verify_dispatch<8,false>(a, st);
  } else {
    if (mode == GdnVerifyMode::Chunk) gdn_verify_dispatch<16,true>(a, st);
    else gdn_verify_dispatch<16,false>(a, st);
  }
}
