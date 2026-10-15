// Included in kernels.cu after the shared storage and launch-selection helpers.
template <int CG> struct alignas(16) GdnSingleShared {
  float q[128], k[128], v[256 / CG], gates[2], red[8];
};

template <int CG>
__device__ __forceinline__ void single_conv_stage(const ConvArgs& c, Seg sg, int h, int part,
    GdnSingleShared<CG>& sm, CgCarry carry) {
  constexpr int JR = 256 / CG;
  const int t = threadIdx.x, d = t & 127, kh = h % HK;
  const int channel = (t < 128 ? 0 : KDIM) + kh * 128 + d;
  const size_t layer = (size_t)sg.slot * NGDN + c.gl;
  float* dst = c.state + (2 * layer + (sg.conv_par ^ 1)) * 3 * CONV_CH;
  const float x = c.proj[(size_t)sg.start * GDN_IN + channel];
  const float4 w = carry.wk;
  float v = silu(w.x * carry.s0 + w.y * carry.s1 + w.z * carry.s2 + w.w * x);
  float ss = v * v;
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  if ((t & 31) == 0) sm.red[t >> 5] = ss;
  if (h < HK && part == 0) {
    dst[channel] = carry.s1; dst[CONV_CH + channel] = carry.s2; dst[2 * CONV_CH + channel] = x;
  }
  if (t < JR) {
    const int vc = 2 * KDIM + h * 128 + JR * part + t;
    const float vx = c.proj[(size_t)sg.start * GDN_IN + vc];
    const float4 vw = carry.vw;
    sm.v[t] = silu(vw.x * carry.v0 + vw.y * carry.v1 + vw.z * carry.v2 + vw.w * vx);
    dst[vc] = carry.v1; dst[CONV_CH + vc] = carry.v2; dst[2 * CONV_CH + vc] = vx;
  }
  if (t == 0) {
    const float g = c.ab[(size_t)sg.start * 96 + h] + c.dt_bias[h];
    const float sp = g > 20.f ? g : log1pf(__expf(g));
    sm.gates[0] = __expf(c.ssm_a[h] * sp);
    sm.gates[1] = sigm(c.ab[(size_t)sg.start * 96 + HV + h]);
  }
  __syncthreads();
  const int b = t < 128 ? 0 : 4;
  ss = sm.red[b] + sm.red[b + 1] + sm.red[b + 2] + sm.red[b + 3];
  (t < 128 ? sm.q : sm.k)[d] = v * (1.f / fmaxf(sqrtf(ss), NORM_EPS));
}

template <typename ST, int CG, bool FUSED, bool Q8 = false>
__global__ void __launch_bounds__(256, 2) k_gdn_single(GdnArgs a, ConvArgs c) { H5_LIT_SITE(GdnSingle, 0);
  constexpr int JR = 256 / CG, MC = 128 / CG;
  __shared__ GdnSingleShared<CG> sm;
#ifndef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  const Seg sg = a.segs[blockIdx.y];
  if (sg.len != 1 || sg.commit || sg.keep) { asm volatile("trap;"); return; }
  const int h = blockIdx.x / (CG / 2), part = blockIdx.x % (CG / 2), kh = h % HK;
  const int t = threadIdx.x, cg = t & (CG - 1), jl = t / CG, j = part * JR + jl;
  static_assert(!Q8 || !FUSED);
  ST* M = Q8 ? nullptr : (ST*)a.state + (((size_t)sg.slot * NGDN + a.gl) * HV + h) * 128 * 128 + j * 128 + 4 * cg;
  GdnQ8Layer* q8 = Q8 ? (GdnQ8Layer*)a.state + (size_t)sg.slot * NGDN + a.gl : nullptr;
  float m[MC];
  if constexpr (Q8) gdn_load_q8_row<CG>(q8, h, j, cg, m);
  else {
#pragma unroll
    for (int i = 0; i < MC / 4; ++i) {
      const float4 v = ld4(M + CG * 4 * i);
      m[4*i] = v.x; m[4*i+1] = v.y; m[4*i+2] = v.z; m[4*i+3] = v.w;
    }
  }
  CgCarry carry;
  if constexpr (FUSED) {
    const int channel = (t < 128 ? 0 : KDIM) + kh * 128 + (t & 127);
    const float* src = c.state + (2 * ((size_t)sg.slot * NGDN + c.gl) + sg.conv_par) * 3 * CONV_CH;
    carry.s0 = src[channel]; carry.s1 = src[CONV_CH + channel]; carry.s2 = src[2 * CONV_CH + channel];
    carry.wk = *(const float4*)(c.w + channel * 4);
    if (t < JR) {
      const int vc = 2 * KDIM + h * 128 + part * JR + t;
      carry.v0 = src[vc]; carry.v1 = src[CONV_CH + vc]; carry.v2 = src[2 * CONV_CH + vc];
      carry.vw = *(const float4*)(c.w + vc * 4);
    }
  }
#ifdef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  H5_LIT_MARKX(0);
  if constexpr (FUSED) {
    if (c.pf_placement == h5::PfChainPlacement::Prologue) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
    single_conv_stage<CG>(c, sg, h, part, sm, carry);
  } else {
    const float* co = a.conv_out + (size_t)sg.start * CONV_CH;
    if (t < 32) {
      *(float4*)(sm.q + t * 4) = *(const float4*)(co + kh * 128 + t * 4);
      *(float4*)(sm.k + t * 4) = *(const float4*)(co + KDIM + kh * 128 + t * 4);
    }
    if (t < JR / 4) *(float4*)(sm.v + t * 4) = *(const float4*)(co + 2 * KDIM + h * 128 + part * JR + t * 4);
    if (t == 0) {
      sm.gates[0] = a.gates[(size_t)sg.start * 96 + h];
      sm.gates[1] = a.gates[(size_t)sg.start * 96 + HV + h];
    }
  }
  __syncthreads();
  H5_LIT_MARKX(1);
  float kr[MC], dot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < MC / 4; ++i) {
    const float4 k = *(const float4*)(sm.k + CG * 4 * i + 4 * cg);
    kr[4*i] = k.x; kr[4*i+1] = k.y; kr[4*i+2] = k.z; kr[4*i+3] = k.w;
  }
#pragma unroll
  for (int i = 0; i < MC; ++i) dot[i & 3] = fmaf(m[i], kr[i], dot[i & 3]);
  float s = (dot[0] + dot[1]) + (dot[2] + dot[3]);
#pragma unroll
  for (int sh = 1; sh < CG; sh <<= 1) s += __shfl_xor_sync(0xffffffff, s, sh);
  const float g = sm.gates[0], delta = sm.gates[1] * (sm.v[jl] - g * s);
#pragma unroll
  for (int i = 0; i < MC; ++i) m[i] = fmaf(g, m[i], delta * kr[i]);
  float out[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < MC / 4; ++i) {
    const float4 q = *(const float4*)(sm.q + CG * 4 * i + 4 * cg);
    out[0] = fmaf(m[4*i], q.x, out[0]); out[1] = fmaf(m[4*i+1], q.y, out[1]);
    out[2] = fmaf(m[4*i+2], q.z, out[2]); out[3] = fmaf(m[4*i+3], q.w, out[3]);
  }
  float o = (out[0] + out[1]) + (out[2] + out[3]);
#pragma unroll
  for (int sh = 1; sh < CG; sh <<= 1) o += __shfl_xor_sync(0xffffffff, o, sh);
  if (cg == 0) a.y[(size_t)sg.start * VDIM + h * 128 + j] = o * 0.08838834764831845f;
  if (a.sig) prep_signal(a.sig);
  H5_LIT_MARKX(2);
  if constexpr (Q8) gdn_store_row<ST, CG, true>(M, q8, h, j, cg, m, a.q8_scale);
  else {
#pragma unroll
    for (int i = 0; i < MC / 4; ++i) {
      if constexpr (sizeof(ST) == 2) {
        uint2 v;
#pragma unroll
        for (int k = 0; k < 4; ++k) st_st((ST*)&v + k, m[4*i+k]);
        *(uint2*)(M + CG * 4 * i) = v;
      } else *(float4*)(M + CG * 4 * i) = make_float4(m[4*i], m[4*i+1], m[4*i+2], m[4*i+3]);
    }
  }
  if constexpr (FUSED) {
    H5_LIT_MARKX(3);
    if (c.pf_placement == h5::PfChainPlacement::Tail) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
  }
}

size_t gdn_single_pad_bytes() {
  const int kib = gdn_env("H5_GDN_SINGLE_PAD_KIB", 0, {0, 40});
  if (kib && launch_carveout() != 100)
    throw std::runtime_error("H5_GDN_SINGLE_PAD_KIB=40 requires H5_CARVEOUT=100 with all-kernel scope");
  return (size_t)kib * 1024;
}
void validate_gdn_single(GdnSingleMode mode, StateFormat format, bool fusion, bool mega, unsigned tail) {
  if (mode == GdnSingleMode::Off) return;
  (void)gdn_single_pad_bytes();
  if (format != StateFormat::F16 && format != StateFormat::BF16 && format != StateFormat::F32 && format != StateFormat::Q8)
    throw std::runtime_error("H5_GDN_SINGLE supports f16, bf16, f32 and q8 state");
  if (format == StateFormat::Q8 && mode != GdnSingleMode::Separate)
    throw std::runtime_error("Q8 single GDN requires the separate route");
  if ((mode == GdnSingleMode::Fused) != fusion)
    throw std::runtime_error("H5_GDN_SINGLE=fused requires H5_STATE_FUSION=1; separate requires H5_STATE_FUSION=0");
  if (mega || (tail & 1)) throw std::runtime_error("H5_GDN_SINGLE requires normal graph route and H5_TAIL_PREP bit 1 unset");
  if (gdn_packed(format) || gdn_env("H5_GDN_SCALED", 0, {0, 1}) ||
      gdn_env("H5_GDN_PIPE", 0, {0, 2, 4, 8, 16, 24, 32, 48, 64}) ||
      gdn_env("H5_GDN_OCC", 2, {2, 3}) != 2 || (gdn_host_variant & 96u))
    throw std::runtime_error("H5_GDN_SINGLE requires fp32 arithmetic, SCALED=0, PIPE=0, OCC=2 and no 0x20/0x40 variants");
#if defined(EXP_GDN_NOMEM) || defined(EXP_GDN_NOCOMP)
  throw std::runtime_error("H5_GDN_SINGLE requires state memory and recurrence computation");
#endif
}

template <typename ST, bool FUSED, bool Q8 = false>
void gdn_single_dispatch(const ConvArgs& c, const GdnArgs& a, int nseg, cudaStream_t st) {
  const int cg = gdn_cg(nseg);
  const size_t padding = gdn_single_pad_bytes();
  if (cg == 4) pdl(k_gdn_single<ST,4,FUSED,Q8>, dim3(2 * HV, nseg), dim3(256), padding, st, a, c);
  else if (cg == 8) pdl(k_gdn_single<ST,8,FUSED,Q8>, dim3(4 * HV, nseg), dim3(256), padding, st, a, c);
  else pdl(k_gdn_single<ST,16,FUSED,Q8>, dim3(8 * HV, nseg), dim3(256), padding, st, a, c);
}
template <bool FUSED>
void gdn_single_launch(const ConvArgs& c, const GdnArgs& a, int nseg, StateFormat f, cudaStream_t st) {
  validate_q8_scale_mode(a.q8_scale);
  validate_gdn_single(FUSED ? GdnSingleMode::Fused : GdnSingleMode::Separate, f, FUSED, false, 0);
  if (nseg < 1 || !a.state || !a.segs || !a.y || a.gl < 0 || a.gl >= NGDN)
    throw std::runtime_error("single GDN requires state, segments, output and valid layer");
  if constexpr (FUSED) {
    if (c.ab_w || !c.state || !c.proj || !c.w || !c.ab || !c.ssm_a || !c.dt_bias || c.segs != a.segs || c.gl != a.gl)
      throw std::runtime_error("single fused GDN requires matching conv operands and separate alpha/beta projection");
  } else if (!a.conv_out || !a.gates) throw std::runtime_error("single GDN requires conv output and gates");
  switch (f) {
    case StateFormat::F16: gdn_single_dispatch<__half,FUSED>(c, a, nseg, st); break;
    case StateFormat::BF16: gdn_single_dispatch<__nv_bfloat16,FUSED>(c, a, nseg, st); break;
    case StateFormat::F32: gdn_single_dispatch<float,FUSED>(c, a, nseg, st); break;
    case StateFormat::Q8:
      if constexpr (!FUSED) gdn_single_dispatch<__half,false,true>(c, a, nseg, st);
      else throw std::runtime_error("Q8 single GDN requires the separate route");
      break;
    default: throw std::runtime_error("unsupported single GDN state format");
  }
}
void launch_gdn_single(const GdnArgs& a, int nseg, StateFormat f, cudaStream_t st) {
  gdn_single_launch<false>({}, a, nseg, f, st);
}
void launch_conv_gdn_single(const ConvArgs& c, const GdnArgs& a, int nseg, StateFormat f, cudaStream_t st) {
  gdn_single_launch<true>(c, a, nseg, f, st);
}
