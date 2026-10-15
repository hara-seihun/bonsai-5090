// Conv input is immutable for a complete grid. Each owned channel writes the opposite state
// bank and pending parity, so duplicated Q/K work cannot race replay or committed-state reads.
template <typename ST, int CG>
__device__ __forceinline__ void conv_gdn_stage(const ConvArgs& a, const Seg sg, int h, int half,
    int done, int nrep, int nnew, float (*sq)[128], float (*sk)[128], float (*sv)[64], float* gs, float* bs) {
  constexpr int JR = 256 / CG;
  const int tid = threadIdx.x, d = tid & 127, kh = h % HK;
  const int c = (tid < 128 ? 0 : KDIM) + kh * 128 + d;
  const bool owner = h < HK && half == 0;
  const size_t layer = (size_t)sg.slot * NGDN + a.gl;
  const float* src = a.state + (layer * 2 + sg.conv_par) * 3 * CONV_CH;
  float* dst = a.state + (layer * 2 + (sg.conv_par ^ 1)) * 3 * CONV_CH;
  const ST* pr = (const ST*)a.pend + (layer * 2 + (sg.par ^ 1)) * a.pending_rows * CONV_CH;
  ST* pw = (ST*)a.pend + (layer * 2 + sg.par) * a.pending_rows * CONV_CH;
  float s0 = src[c], s1 = src[CONV_CH + c], s2 = src[2 * CONV_CH + c];
  for (int i = 0; i < sg.commit; i++) { s0 = s1; s1 = s2; s2 = st_ld(pr + (size_t)i * CONV_CH + c); }
  if (owner && sg.keep) { dst[c] = s0; dst[CONV_CH + c] = s1; dst[2 * CONV_CH + c] = s2; }
  for (int i = 0; i < done; i++) { s0 = s1; s1 = s2; s2 = a.proj[(size_t)(sg.start + i) * GDN_IN + c]; }
  const float4 wk = *(const float4*)(a.w + c * 4);
  __shared__ float red[8];
  for (int i = 0; i < nnew; i++) {
    const int row = sg.start + done + i;
    const float xc = a.proj[(size_t)row * GDN_IN + c];
    float v = silu(wk.x * s0 + wk.y * s1 + wk.z * s2 + wk.w * xc);
    float ss = v * v;
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    const int b = tid < 128 ? 0 : 4;
    ss = red[b] + red[b + 1] + red[b + 2] + red[b + 3];
    v *= 1.f / fmaxf(sqrtf(ss), NORM_EPS);
    (tid < 128 ? sq : sk)[nrep + i][d] = v;
    if (owner && sg.keep) st_st(pw + (size_t)(done + i) * CONV_CH + c, xc);
    s0 = s1; s1 = s2; s2 = xc;
    __syncthreads();
  }
  if (owner && !sg.keep && done + nnew == sg.len) {
    dst[c] = s0; dst[CONV_CH + c] = s1; dst[2 * CONV_CH + c] = s2;
  }
  if (tid < JR) {
    const int vc = 2 * KDIM + h * 128 + JR * half + tid;
    float v0 = src[vc], v1 = src[CONV_CH + vc], v2 = src[2 * CONV_CH + vc];
    for (int i = 0; i < sg.commit; i++) { v0 = v1; v1 = v2; v2 = st_ld(pr + (size_t)i * CONV_CH + vc); }
    if (sg.keep) { dst[vc] = v0; dst[CONV_CH + vc] = v1; dst[2 * CONV_CH + vc] = v2; }
    for (int i = 0; i < done; i++) { v0 = v1; v1 = v2; v2 = a.proj[(size_t)(sg.start + i) * GDN_IN + vc]; }
    const float4 vw = *(const float4*)(a.w + vc * 4);
    for (int i = 0; i < nnew; i++) {
      const float xc = a.proj[(size_t)(sg.start + done + i) * GDN_IN + vc];
      sv[nrep + i][tid] = silu(vw.x * v0 + vw.y * v1 + vw.z * v2 + vw.w * xc);
      if (sg.keep) st_st(pw + (size_t)(done + i) * CONV_CH + vc, xc);
      v0 = v1; v1 = v2; v2 = xc;
    }
    if (!sg.keep && done + nnew == sg.len) { dst[vc] = v0; dst[CONV_CH + vc] = v1; dst[2 * CONV_CH + vc] = v2; }
  }
  if (tid < nnew) {
    const int row = sg.start + done + tid;
    const float xg = a.ab[(size_t)row * 96 + h] + a.dt_bias[h];
    const float sp = xg > 20.f ? xg : log1pf(__expf(xg));
    gs[nrep + tid] = __expf(a.ssm_a[h] * sp);
    bs[nrep + tid] = sigm(a.ab[(size_t)row * 96 + HV + h]);
  }
}

// Batched fused conv (H5_STATE_FUSION=1): the conv carry (state after the replayed commit rows, weights) is
// loaded before the PDL wait (all of it was written by the previous step) and carried across chunks; the
// new rows of a chunk are loaded together and their q/k norms reduced behind one barrier per eight rows.
// Arithmetic is that of conv_gdn_stage and k_conv.
struct CgCarry { float s0, s1, s2, v0, v1, v2; float4 wk, vw; };
template <typename ST, int CG>
__device__ __forceinline__ void conv_gdn_pre(const ConvArgs& a, const Seg sg, int h, int half, CgCarry& r) {
  constexpr int JR = 256 / CG;
  const int tid = threadIdx.x, d = tid & 127, kh = h % HK;
  const int c = (tid < 128 ? 0 : KDIM) + kh * 128 + d;
  const size_t layer = (size_t)sg.slot * NGDN + a.gl;
  const float* src = a.state + (layer * 2 + sg.conv_par) * 3 * CONV_CH;
  const ST* pr = (const ST*)a.pend + (layer * 2 + (sg.par ^ 1)) * a.pending_rows * CONV_CH;
  r.s0 = src[c]; r.s1 = src[CONV_CH + c]; r.s2 = src[2 * CONV_CH + c];
  for (int i = 0; i < sg.commit; i++) { r.s0 = r.s1; r.s1 = r.s2; r.s2 = st_ld(pr + (size_t)i * CONV_CH + c); }
  r.wk = *(const float4*)(a.w + c * 4);
  if (tid < JR) {
    const int vc = 2 * KDIM + h * 128 + JR * half + tid;
    r.v0 = src[vc]; r.v1 = src[CONV_CH + vc]; r.v2 = src[2 * CONV_CH + vc];
    for (int i = 0; i < sg.commit; i++) { r.v0 = r.v1; r.v1 = r.v2; r.v2 = st_ld(pr + (size_t)i * CONV_CH + vc); }
    r.vw = *(const float4*)(a.w + vc * 4);
  }
}
template <typename ST, int CG>
__device__ __forceinline__ void conv_gdn_stage_b(const ConvArgs& a, const Seg sg, int h, int half, bool first_chunk,
    int done, int nrep, int nnew, float (*sq)[128], float (*sk)[128], float (*sv)[64], float* gs, float* bs, CgCarry& r) {
  constexpr int JR = 256 / CG;
  const int tid = threadIdx.x, d = tid & 127, kh = h % HK, w = tid >> 5;
  const int c = (tid < 128 ? 0 : KDIM) + kh * 128 + d, vc = 2 * KDIM + h * 128 + JR * half + tid;
  const bool owner = h < HK && half == 0, vown = tid < JR;
  const size_t layer = (size_t)sg.slot * NGDN + a.gl;
  float* dst = a.state + (layer * 2 + (sg.conv_par ^ 1)) * 3 * CONV_CH;
  ST* pw = (ST*)a.pend + (layer * 2 + sg.par) * a.pending_rows * CONV_CH;
  if (first_chunk && sg.keep) {
    if (owner) { dst[c] = r.s0; dst[CONV_CH + c] = r.s1; dst[2 * CONV_CH + c] = r.s2; }
    if (vown) { dst[vc] = r.v0; dst[CONV_CH + vc] = r.v1; dst[2 * CONV_CH + vc] = r.v2; }
  }
  __shared__ float red[8][8];
  for (int i0 = 0; i0 < nnew; i0 += 8) {
    const int nb = min(8, nnew - i0);
    const size_t r0 = (size_t)(sg.start + done + i0);
    float xc[8], xv[8], v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) if (j < nb) xc[j] = a.proj[(r0 + j) * GDN_IN + c];
    if (vown) {
#pragma unroll
      for (int j = 0; j < 8; j++) if (j < nb) xv[j] = a.proj[(r0 + j) * GDN_IN + vc];
    }
#pragma unroll
    for (int j = 0; j < 8; j++) if (j < nb) {
      v[j] = silu(r.wk.x * r.s0 + r.wk.y * r.s1 + r.wk.z * r.s2 + r.wk.w * xc[j]);
      r.s0 = r.s1; r.s1 = r.s2; r.s2 = xc[j];
      float ss = v[j] * v[j];
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      if ((tid & 31) == 0) red[j][w] = ss;
    }
    __syncthreads();
    const int b = tid < 128 ? 0 : 4;
#pragma unroll
    for (int j = 0; j < 8; j++) if (j < nb) {
      const float ss = red[j][b] + red[j][b + 1] + red[j][b + 2] + red[j][b + 3];
      (tid < 128 ? sq : sk)[nrep + i0 + j][d] = v[j] * (1.f / fmaxf(sqrtf(ss), NORM_EPS));
      if (owner && sg.keep) st_st(pw + (size_t)(done + i0 + j) * CONV_CH + c, xc[j]);
    }
    if (vown) {
#pragma unroll
      for (int j = 0; j < 8; j++) if (j < nb) {
        sv[nrep + i0 + j][tid] = silu(r.vw.x * r.v0 + r.vw.y * r.v1 + r.vw.z * r.v2 + r.vw.w * xv[j]);
        if (sg.keep) st_st(pw + (size_t)(done + i0 + j) * CONV_CH + vc, xv[j]);
        r.v0 = r.v1; r.v1 = r.v2; r.v2 = xv[j];
      }
    }
    if (i0 + 8 < nnew) __syncthreads();
  }
  if (!sg.keep && done + nnew == sg.len) {
    if (owner) { dst[c] = r.s0; dst[CONV_CH + c] = r.s1; dst[2 * CONV_CH + c] = r.s2; }
    if (vown) { dst[vc] = r.v0; dst[CONV_CH + vc] = r.v1; dst[2 * CONV_CH + vc] = r.v2; }
  }
  if (tid < nnew) {
    const int row = sg.start + done + tid;
    const float xg = a.ab[(size_t)row * 96 + h] + a.dt_bias[h];
    const float sp = xg > 20.f ? xg : log1pf(__expf(xg));
    gs[nrep + tid] = __expf(a.ssm_a[h] * sp);
    bs[nrep + tid] = sigm(a.ab[(size_t)row * 96 + HV + h]);
  }
}
