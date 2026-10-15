// Included after gdn_verify.cuh. Convolution state and pending use the existing two-bank layout.
template <int R> struct alignas(16) GdnVerifyFusedShared {
  float q[R][128], k[16][128], v[16][64], g[16], beta[16];
};
template <int R, int CG>
__global__ void __launch_bounds__(256, 2) k_conv_gdn_verify(ConvArgs c, GdnArgs a) { H5_LIT_SITE(ConvGdnVerify, 0);
  constexpr int JR = 256 / CG, MC = 128 / CG, PARTS = CG / 2;
  __shared__ GdnVerifyFusedShared<R> sm;
#ifndef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  const Seg sg = a.segs[0];
  if (sg.len != R || sg.keep != 1 || sg.commit < 0 || sg.commit > a.pending_rows ||
      sg.slot < 0 || sg.start < 0 || (sg.par != 0 && sg.par != 1) ||
      (sg.conv_par != 0 && sg.conv_par != 1)) { asm volatile("trap;"); return; }
  const int tid = threadIdx.x, cg = tid & (CG - 1), jl = tid / CG;
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
  CgCarry carry;
  conv_gdn_pre<__half,CG>(c, sg, h, part, carry);
#ifdef H5_PDL_GDN_STATE_OVERLAP
  pdl_enter_e(a.early);
#endif
  if (c.pf_placement == PfChainPlacement::Prologue) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
  H5_LIT_MARKX(0);
  if (sg.commit) {
    constexpr int PIECES = 32 + JR / 4;
    for (int i = tid; i < sg.commit * PIECES; i += 256) {
      const int t = i / PIECES, d = i % PIECES;
      const __half* src = prd + (size_t)t * PEND_GDN;
      if (d < 32) *(float4*)(sm.k[t] + d * 4) = ld4(src + kh * 128 + d * 4);
      else *(float4*)(sm.v[t] + (d - 32) * 4) = ld4(src + KDIM + h * 128 + part * JR + (d - 32) * 4);
    }
    if (tid < sg.commit) {
      const __half* src = prd + (size_t)tid * PEND_GDN;
      sm.g[tid] = st_ld(src + KDIM + VDIM + h);
      sm.beta[tid] = st_ld(src + KDIM + VDIM + HV + h);
    }
    __syncthreads();
    for (int t = 0; t < sg.commit; ++t) {
      const float delta = a.replay_delta ? sm.v[t][jl] :
        sm.beta[t] * (sm.v[t][jl] - sm.g[t] * verify_dot<CG>(m, sm.k[t], cg));
      verify_update<CG>(m, sm.k[t], sm.g[t], delta, cg);
    }
    verify_save<CG>(M, m);
    __syncthreads();
  }
  conv_gdn_stage_b<__half,CG>(c, sg, h, part, true, 0, 0, R,
      sm.q, sm.k, sm.v, sm.g, sm.beta, carry);
  __syncthreads();
  H5_LIT_MARKX(1);
  for (int i = tid; i < R * 128; i += 256) {
    const int t = i / 128, d = i % 128;
    __half* dst = pwr + (size_t)t * PEND_GDN;
    if (h < HK && part == 0) st_st(dst + kh * 128 + d, sm.k[t][d]);
    if (d < JR && !a.replay_delta) st_st(dst + KDIM + h * 128 + part * JR + d, sm.v[t][d]);
    if (d == 0 && part == 0) {
      st_st(dst + KDIM + VDIM + h, sm.g[t]); st_st(dst + KDIM + VDIM + HV + h, sm.beta[t]);
    }
  }
#pragma unroll 1
  for (int t = 0; t < R; ++t) {
    const float g = sm.g[t], delta = sm.beta[t] * (sm.v[t][jl] - g * verify_dot<CG>(m, sm.k[t], cg));
    if (cg == 0 && a.replay_delta) st_st(pwr + (size_t)t * PEND_GDN + KDIM + h * 128 + j, delta);
    verify_update<CG>(m, sm.k[t], g, delta, cg);
    const float out = verify_dot<CG>(m, sm.q[t], cg);
    if (cg == 0) a.y[(size_t)(sg.start + t) * VDIM + h * 128 + j] = out * 0.08838834764831845f;
  }
  H5_LIT_MARKX(2);
  if (c.pf_placement == PfChainPlacement::Tail) chain_l2pf(c.pf, c.pf_bytes, c.pf_granularity);
  prep_signal(a.sig);
}
template <int R>
void gdn_verify_fused_dispatch(const ConvArgs& c, const GdnArgs& a, cudaStream_t st) {
  const int cg = gdn_cg(1);
  if (cg == 4) pdl(k_conv_gdn_verify<R,4>, dim3(2 * HV), dim3(256), 0, st, c, a);
  else if (cg == 8) pdl(k_conv_gdn_verify<R,8>, dim3(4 * HV), dim3(256), 0, st, c, a);
  else pdl(k_conv_gdn_verify<R,16>, dim3(8 * HV), dim3(256), 0, st, c, a);
}
void launch_conv_gdn_verify(const ConvArgs& c, const GdnArgs& a, int nseg, int rows, StateFormat f, cudaStream_t st) {
  if (nseg != 1 || (rows != 8 && rows != 16) || f != StateFormat::F16 || !a.state || !a.segs ||
      !a.pend || !a.y || a.gl < 0 || a.gl >= NGDN || (a.pending_rows != 8 && a.pending_rows != 16) || a.pending_rows < rows ||
      c.segs != a.segs || c.gl != a.gl || c.pending_rows != a.pending_rows || !c.state || !c.pend ||
      !c.proj || !c.w || !c.ab || !c.ssm_a || !c.dt_bias || c.ab_w)
    throw std::runtime_error("fused GDN verify requires one F16 keep1 R8/R16 visit, explicit two-bank conv state/pending and matching operands");
  if (gdn_packed(f) || gdn_env("H5_GDN_SCALED", 0, {0, 1}) ||
      gdn_env("H5_GDN_PIPE", 0, {0, 2, 4, 8, 16, 24, 32, 48, 64}) ||
      gdn_env("H5_GDN_OCC", 2, {2, 3}) != 2 || (gdn_host_variant & 96u))
    throw std::runtime_error("fused GDN verify requires fp32 arithmetic, SCALED=0, PIPE=0, OCC=2 and no 0x20/0x40 variants");
#if defined(EXP_GDN_NOMEM) || defined(EXP_GDN_NOCOMP)
  throw std::runtime_error("fused GDN verify requires state memory and recurrence computation");
#endif
  if (rows == 8) gdn_verify_fused_dispatch<8>(c, a, st);
  else gdn_verify_fused_dispatch<16>(c, a, st);
}
