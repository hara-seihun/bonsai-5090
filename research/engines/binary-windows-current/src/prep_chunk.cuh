// Included inside namespace h5 after the row preparation kernel.
// One warp owns 1024 features. Registers carry bits 0..1,7..9 of the feature
// index; lanes carry bits 2..6. This coalesces float4 loads without changing
// Hadamard butterfly order. Fixed NVFP4 G eliminates the post-Hadamard row max.
template <bool RMS_PREPASS = false, bool ROW_NORM = false, unsigned STATIC_FLAGS = 0>
__global__ void __launch_bounds__(ROW_NORM ? (STATIC_FLAGS ? 192 : 544) : 32) k_prep_chunk(PrepArgs a, float fixed_g) { H5_LIT(0);
  __shared__ float rms_partials[ROW_NORM ? 17 : 1];
  if constexpr (STATIC_FLAGS != 0) a.flags = STATIC_FLAGS;
  const int lane = threadIdx.x & 31, chunk = ROW_NORM ? threadIdx.x >> 5 : blockIdx.y, row = blockIdx.x;
  const int base = chunk * 1024, off = base + lane * 4;
  float4 cn[8], cs[8];
  if (!ROW_NORM && (a.flags & P_NORM)) {
#pragma unroll
    for (int i = 0; i < 8; i++) cn[i] = *(const float4*)(a.norm_w + off + i * 128);
  }
  if (!ROW_NORM && (a.flags & (P_SIGN | P_SIGN_AFTER))) {
#pragma unroll
    for (int i = 0; i < 8; i++) cs[i] = *(const float4*)(a.signs + off + i * 128);
  }
  // GDN gate z is the input projection's output, four grids upstream: every intervening grid
  // (ab GEMM, conv, gdn) triggers dependents only after its own wait, so z is complete and
  // visible before this grid's wait. Issue all head loads together instead of one per head.
  float4 zf[8], gw = make_float4(0.f, 0.f, 0.f, 0.f);
  auto load_z = [&](int zr) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int q = (base >> 7) + i, head = (a.flags & P_PERMUTE_GDN) ? q / 3 + 16 * (q % 3) : q;
      zf[i] = *(const float4*)(a.gz + (size_t)zr * a.gz_ld + head * 128 + lane * 4);
    }
  };
  const bool hoist = g_variant & 1u;
  if (!ROW_NORM && (a.flags & P_GDN_NORM)) {
    gw = *(const float4*)(a.gnorm_w + lane * 4);
    if (hoist && !a.rowmap) load_z(row);
  }
  struct ExitWait { bool on; __device__ ~ExitWait() { if (on) pdl_wait(); } } exit_wait{!ROW_NORM && a.sig != nullptr};
  if (!ROW_NORM && a.sig) {
    if (threadIdx.x == 0) while (tail_ld_acquire(a.sig) < a.sig_expect) {}
    __syncthreads();
  } else { if (a.early || (g_variant & 2u)) pdl_launch(); pdl_wait(); }
  if constexpr (!ROW_NORM) {
    // Output clearing for the next split-K GEMM (drafter precleared path): the gridDim.y chunk CTAs
    // of a row split that row's declared outputs, before dependents may launch.
    if (a.clear_a) {
      const int n4 = a.clear_a_stride / 4, per = (n4 + gridDim.y - 1) / gridDim.y, b0 = blockIdx.y * per, b1 = min(n4, b0 + per);
      float4* d = reinterpret_cast<float4*>(a.clear_a + (size_t)row * a.clear_a_stride);
      for (int i = b0 + lane; i < b1; i += 32) d[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (a.clear_b) {
      const int n4 = a.clear_b_stride / 4, per = (n4 + gridDim.y - 1) / gridDim.y, b0 = blockIdx.y * per, b1 = min(n4, b0 + per);
      float4* d = reinterpret_cast<float4*>(a.clear_b + (size_t)row * a.clear_b_stride);
      for (int i = b0 + lane; i < b1; i += 32) d[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  pdl_launch();
  const int irow = a.rowmap ? a.rowmap[row] : row;
  if (!ROW_NORM && (a.flags & P_GDN_NORM) && (!hoist || a.rowmap)) load_z(irow);
  const float* x = a.x + (size_t)irow * a.ldx;
  const float* x2 = a.x2 ? a.x2 + (size_t)irow * a.ldx : nullptr;
  float v[32];
  if (!ROW_NORM && (a.flags & P_SILU_IL)) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int gi = off + i * 128, gate = 64 * (gi >> 5) + (gi & 31);
      const float4 g = *(const float4*)(x + gate), u = *(const float4*)(x + gate + 32);
      if (a.flags & P_CONSUME_CLEAR) {
        float* w = const_cast<float*>(x);
        *(float4*)(w + gate) = make_float4(0.f, 0.f, 0.f, 0.f); *(float4*)(w + gate + 32) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      v[4*i] = silu(g.x) * u.x; v[4*i+1] = silu(g.y) * u.y; v[4*i+2] = silu(g.z) * u.z; v[4*i+3] = silu(g.w) * u.w;
    }
  } else if (!ROW_NORM && (a.flags & P_DCONV)) {
    const bool first = (row % a.dc_block_rows) == 0;
    const float* dyn = a.dc_dyn + (size_t)row * DF_KPROJ + a.dc_which * 640;
    const float* bw = a.dc_base + (size_t)a.dc_which * 2 * D;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int gi = off + i * 128, g = gi >> 4;
      const float4 x0 = *(const float4*)(x + gi);
      const float4 x1 = first ? make_float4(0.f, 0.f, 0.f, 0.f) : *(const float4*)(x - a.ldx + gi);
      const float d0 = dyn[g], d1 = dyn[320 + g];
      const float4 b0 = *(const float4*)(bw + gi), b1 = *(const float4*)(bw + D + gi);
      v[4*i] = dconv_elem(b0.x, d0, x0.x, b1.x, d1, x1.x); v[4*i+1] = dconv_elem(b0.y, d0, x0.y, b1.y, d1, x1.y);
      v[4*i+2] = dconv_elem(b0.z, d0, x0.z, b1.z, d1, x1.z); v[4*i+3] = dconv_elem(b0.w, d0, x0.w, b1.w, d1, x1.w);
    }
  } else {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int gi = off + i * 128, q = gi >> 7;
    const int in = (a.flags & P_PERMUTE_GDN) ? (gi & 127) + 128 * (q / 3 + 16 * (q % 3)) : gi;
    const float4 t = *(const float4*)(x + in);
    v[4*i] = t.x; v[4*i+1] = t.y; v[4*i+2] = t.z; v[4*i+3] = t.w;
  }
  }
  H5_LIT_MARK(0);
  if (a.flags & P_GDN_NORM) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float ss = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) ss = fmaf(v[4*i+j],v[4*i+j],ss);
      ss = warp_sum(ss);
      const float inv = rsqrtf(ss / 128.f + NORM_EPS);
      const float4 z = zf[i];
      v[4*i] = v[4*i] * inv * gw.x * silu(z.x); v[4*i+1] = v[4*i+1] * inv * gw.y * silu(z.y);
      v[4*i+2] = v[4*i+2] * inv * gw.z * silu(z.z); v[4*i+3] = v[4*i+3] * inv * gw.w * silu(z.w);
    }
  }
  if (a.flags & P_NORM) {
    float inv;
    if constexpr (ROW_NORM) {
      float ss = 0.f;
#pragma unroll
      for (int r = 0; r < 32; r++) ss = fmaf(v[r], v[r], ss);
      ss = warp_sum(ss);
      if (!lane) rms_partials[chunk] = ss;
      __syncthreads();
      ss = 0.f;
      for (int c = 0; c < a.n / 1024; c++) ss += rms_partials[c];
      inv = rsqrtf(ss / (float)a.n + NORM_EPS);
    } else if constexpr (RMS_PREPASS) {
      inv = a.rms_inv[row];
    } else {
      float ss = 0.f;
      for (int c = 0; c < a.n / 1024; c++) {
        float partial = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const float4 t = *(const float4*)(x + c * 1024 + lane * 32 + i * 4);
          partial = fmaf(t.x,t.x,partial); partial = fmaf(t.y,t.y,partial);
          partial = fmaf(t.z,t.z,partial); partial = fmaf(t.w,t.w,partial);
        }
        ss += warp_sum(partial);
      }
      inv = rsqrtf(ss / (float)a.n + NORM_EPS);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (ROW_NORM) cn[i] = *(const float4*)(a.norm_w + off + i * 128);
      v[4*i] *= inv * cn[i].x; v[4*i+1] *= inv * cn[i].y;
      v[4*i+2] *= inv * cn[i].z; v[4*i+3] *= inv * cn[i].w;
      if (a.flags & P_OUT_NORM)
        *(float4*)(a.out_norm + (size_t)row*a.ldo + off + i*128) = make_float4(v[4*i],v[4*i+1],v[4*i+2],v[4*i+3]);
    }
  }
  H5_LIT_MARK(1);
  if (a.flags & P_SILU_MUL) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 u = *(const float4*)(x2 + off + i*128);
      v[4*i] = silu(v[4*i])*u.x; v[4*i+1] = silu(v[4*i+1])*u.y;
      v[4*i+2] = silu(v[4*i+2])*u.z; v[4*i+3] = silu(v[4*i+3])*u.w;
    }
  }
  if (a.flags & P_SIGN) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (ROW_NORM) cs[i] = *(const float4*)(a.signs + off + i * 128);
      v[4*i] *= cs[i].x; v[4*i+1] *= cs[i].y; v[4*i+2] *= cs[i].z; v[4*i+3] *= cs[i].w;
    }
  }
  if (a.flags & P_HADAMARD) {
#pragma unroll
    for (int len = 1; len < 4; len <<= 1)
#pragma unroll
      for (int r = 0; r < 32; r++) if (!(r & len)) {
        const float p = v[r], q = v[r+len]; v[r] = p+q; v[r+len] = p-q;
      }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
#pragma unroll
      for (int r = 0; r < 32; r++) {
        const float q = __shfl_xor_sync(0xffffffff,v[r],o); v[r] = (lane & o) ? q-v[r] : v[r]+q;
      }
#pragma unroll
    for (int len = 4; len < 32; len <<= 1)
#pragma unroll
      for (int r = 0; r < 32; r++) if (!(r & len)) {
        const float p = v[r], q = v[r+len]; v[r] = p+q; v[r+len] = p-q;
      }
#pragma unroll
    for (int r = 0; r < 32; r++) v[r] *= .03125f;
  }
  H5_LIT_MARK(2);
  if (a.flags & P_SIGN_AFTER) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (ROW_NORM) cs[i] = *(const float4*)(a.signs + off + i * 128);
      v[4*i] *= cs[i].x; v[4*i+1] *= cs[i].y; v[4*i+2] *= cs[i].z; v[4*i+3] *= cs[i].w;
    }
  }
  if (a.flags & P_STORE_F32) {
#pragma unroll
    for (int i = 0; i < 8; i++)
      *(float4*)(a.out_f32 + (size_t)row*a.ldo + off + i*128) = make_float4(v[4*i],v[4*i+1],v[4*i+2],v[4*i+3]);
  }
  H5_LIT_MARK(3);
  if (!(a.flags & P_QUANT)) return;
  if (a.flags & P_FP4_RESIDUAL) {
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
      for (int j = 0; j < 4; j++) v[4 * i + j] -= prep_ref_fp4(a, row, off + i * 128 + j);
  }
  if (a.q4) {
    const float ig = 1.f/fixed_g;
    if (!lane && !chunk) a.g4[row] = fixed_g;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float bm = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) bm = fmaxf(bm,fabsf(v[4*i+j]));
      bm = fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,1));
      bm = fmaxf(bm,__shfl_xor_sync(0xffffffff,bm,2));
      const float target = bm/6.f*ig;
      uint8_t sb = __nv_cvt_float_to_fp8(target,__NV_SATFINITE,__NV_E4M3);
      float sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3)));
      if (sv < target && sb < 0x7e) { sb++; sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3))); }
      const float rs = sv > 0.f ? ig/sv : 0.f;
      const uint16_t word = (uint16_t)__nv_cvt_float2_to_fp4x2(make_float2(v[4*i]*rs,v[4*i+1]*rs),__NV_E2M1,cudaRoundNearest) |
        ((uint16_t)__nv_cvt_float2_to_fp4x2(make_float2(v[4*i+2]*rs,v[4*i+3]*rs),__NV_E2M1,cudaRoundNearest) << 8);
      const int rr = row & 63, c = lane >> 3, kb = chunk*8+i;
      const size_t dst = ((size_t)(row >> 6)*(a.n/128)+kb)*64+rr;
      *(uint16_t*)(a.q4 + dst*64 + ((c ^ ((rr >> 1)&3)) << 4) + (lane&7)*2) = word;
      if (!(lane&3)) a.s4[dst*8+(lane >> 2)] = sb;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float bm = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) bm = fmaxf(bm,fabsf(v[4*i+j]));
      bm = warp_max(bm);
      const float is = bm > 0.f ? 127.f/bm : 0.f;
      uint32_t packed = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) packed |= (uint32_t)(__float2int_rn(v[4*i+j]*is)&0xff) << (8*j);
      *(uint32_t*)(a.xq + (size_t)row*a.n + off + i*128) = packed;
      if (!lane) a.xs[(size_t)row*(a.n/128)+chunk*8+i] = bm/127.f;
    }
  }
}

// Wide chunk preparation (H5_PREP_CHUNK_WIDE=1): four warps per (row, 1024-feature chunk), eight features
// per lane as in k_prep2w (feature bits 0..2 in registers, 3..7 in lanes, 8..9 in the warp), so each lane's
// dependent chain is a quarter of k_prep_chunk's. Fixed NVFP4 g. Supports the non-norm chunk flags of the
// verifier: P_PERMUTE_GDN | P_GDN_NORM (gated per-head RMSNorm), P_SIGN, P_HADAMARD, P_QUANT, and the
// flag-started producer wait.
__global__ void __launch_bounds__(128) k_prep_chunkw(PrepArgs a, float fixed_g) { H5_LIT(0);
  __shared__ float xch[4][256];
  const int t = threadIdx.x, lane = t & 31, wl = t >> 5, row = blockIdx.x, c = blockIdx.y;
  const int e0 = c * 1024 + wl * 256 + lane * 8;   // first of this lane's eight features
  const int q = e0 >> 7, head = (a.flags & P_PERMUTE_GDN) ? q / 3 + 16 * (q % 3) : q;
  const int in0 = (a.flags & P_PERMUTE_GDN) ? (e0 & 127) + 128 * head : e0;
  float4 cs[2], gw[2], zf[2];
  if (a.flags & P_SIGN) { cs[0] = *(const float4*)(a.signs + e0); cs[1] = *(const float4*)(a.signs + e0 + 4); }
  if (a.flags & P_GDN_NORM) { gw[0] = *(const float4*)(a.gnorm_w + (e0 & 127)); gw[1] = *(const float4*)(a.gnorm_w + (e0 & 127) + 4); }
  struct ExitWait { bool on; __device__ ~ExitWait() { if (on) pdl_wait(); } } exit_wait{a.sig != nullptr};
  if (a.sig) {
    if (t == 0) while (tail_ld_acquire(a.sig) < a.sig_expect) {}
    __syncthreads();
  } else { if (a.early || (g_variant & 2u)) pdl_launch(); pdl_wait(); }
  pdl_launch();
  const float* x = a.x + (size_t)row * a.ldx;
  float v[8];
  { const float4 p = __ldcg((const float4*)(x + in0)), r = __ldcg((const float4*)(x + in0 + 4));
    v[0] = p.x; v[1] = p.y; v[2] = p.z; v[3] = p.w; v[4] = r.x; v[5] = r.y; v[6] = r.z; v[7] = r.w; }
  if (a.flags & P_GDN_NORM) {
    const float* zp = a.gz + (size_t)row * a.gz_ld + head * 128 + (e0 & 127);
    zf[0] = *(const float4*)zp; zf[1] = *(const float4*)(zp + 4);
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < 8; r++) ss = fmaf(v[r], v[r], ss);
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);   // one head = 16 lanes
    const float inv = rsqrtf(ss / 128.f + NORM_EPS);
    const float z[8] = {zf[0].x, zf[0].y, zf[0].z, zf[0].w, zf[1].x, zf[1].y, zf[1].z, zf[1].w};
    const float w[8] = {gw[0].x, gw[0].y, gw[0].z, gw[0].w, gw[1].x, gw[1].y, gw[1].z, gw[1].w};
#pragma unroll
    for (int r = 0; r < 8; r++) v[r] = v[r] * inv * w[r] * silu(z[r]);
  }
  if (a.flags & P_SIGN) {
    v[0] *= cs[0].x; v[1] *= cs[0].y; v[2] *= cs[0].z; v[3] *= cs[0].w;
    v[4] *= cs[1].x; v[5] *= cs[1].y; v[6] *= cs[1].z; v[7] *= cs[1].w;
  }
#pragma unroll
  for (int len = 1; len < 8; len <<= 1)
#pragma unroll
    for (int r = 0; r < 8; r++)
      if (!(r & len)) { const float p = v[r], s = v[r + len]; v[r] = p + s; v[r + len] = p - s; }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1)
#pragma unroll
    for (int r = 0; r < 8; r++) { const float s = __shfl_xor_sync(0xffffffff, v[r], o); v[r] = (lane & o) ? s - v[r] : v[r] + s; }
#pragma unroll
  for (int r = 0; r < 8; r++) xch[wl][lane * 8 + r] = v[r];
  __syncthreads();
  {
    const int b8 = wl & 1, b9 = wl >> 1;
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const float A = xch[0][lane * 8 + r], B = xch[1][lane * 8 + r], C = xch[2][lane * 8 + r], Dd = xch[3][lane * 8 + r];
      const float t0 = b8 ? A - B : A + B, t1 = b8 ? C - Dd : C + Dd;
      v[r] = (b9 ? t0 - t1 : t0 + t1) * 0.03125f;
    }
  }
  if (a.flags & P_FP4_RESIDUAL) {
#pragma unroll
    for (int r = 0; r < 8; r++) v[r] -= prep_ref_fp4(a, row, e0 + r);
  }
  float bm = 0.f;
#pragma unroll
  for (int r = 0; r < 8; r++) bm = fmaxf(bm, fabsf(v[r]));
  bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, 1));   // 16-feature block = lane pair
  const float ig = 1.f / fixed_g;
  if (t == 0 && c == 0) a.g4[row] = fixed_g;
  const float target = bm / 6.f * ig;
  __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
  float sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
  if (sv < target && sb < 0x7e) { sb++; sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3))); }
  const float rs = sv > 0.f ? ig / sv : 0.f;
  uint32_t wd = 0;
#pragma unroll
  for (int k = 0; k < 4; k++)
    wd |= (uint32_t)__nv_cvt_float2_to_fp4x2(make_float2(v[2 * k] * rs, v[2 * k + 1] * rs), __NV_E2M1, cudaRoundNearest) << (8 * k);
  const int tile = row >> 6, rr = row & 63, kb = e0 >> 7, o = e0 & 127, seg = o >> 5;
  const size_t base = ((size_t)tile * (a.n / 128) + kb) * 64 + rr;
  *(uint32_t*)(a.q4 + base * 64 + ((seg ^ ((rr >> 1) & 3)) << 4) + ((o & 31) >> 1)) = wd;
  if (!(lane & 1)) a.s4[base * 8 + (o >> 4)] = sb;
  prep_signal(a.done_sig);
}

// Chunk-parallel row-norm preparation (H5_PREP_NORM_CHUNK=1): grid (row, 1024-chunk), four warps, eight
// features per lane. Every CTA reads its whole row for the RMS (L2 hits), so no CTA waits for another.
// The NVFP4 row scale comes from a bound instead of the post-Hadamard row max: with y = x * w * inv, each
// output of the orthonormal 1024-point Hadamard satisfies |H y|_i <= ||y_chunk||_2, so
// g = max_chunk ||y_chunk||_2 / (6 * 448) never saturates and needs no second reduction across chunks.
// D-wide rows (5 chunks) with P_NORM | P_SIGN | P_HADAMARD | P_QUANT, no clears, no f32 outputs.
__global__ void __launch_bounds__(128) k_prep_normc(PrepArgs a) { H5_LIT(0);
  constexpr int NCH = D / 1024;
  __shared__ float xch[4][256];
  __shared__ float red[4][NCH + 1];
  const int t = threadIdx.x, lane = t & 31, wl = t >> 5, row = blockIdx.x, c = blockIdx.y;
  const int o0 = wl * 256 + lane * 8;   // offset of this lane's eight features inside a chunk
  float4 wv[NCH][2], cs[2];
#pragma unroll
  for (int k = 0; k < NCH; k++) { wv[k][0] = *(const float4*)(a.norm_w + k * 1024 + o0); wv[k][1] = *(const float4*)(a.norm_w + k * 1024 + o0 + 4); }
  cs[0] = *(const float4*)(a.signs + c * 1024 + o0); cs[1] = *(const float4*)(a.signs + c * 1024 + o0 + 4);
  if (a.early || (g_variant & 2u)) pdl_launch(); pdl_wait();
  pdl_launch();
  const float* x = a.x + (size_t)row * a.ldx;
  float4 xv[NCH][2];
#pragma unroll
  for (int k = 0; k < NCH; k++) { xv[k][0] = __ldcg((const float4*)(x + k * 1024 + o0)); xv[k][1] = __ldcg((const float4*)(x + k * 1024 + o0 + 4)); }
  float ss = 0.f, e[NCH];
#pragma unroll
  for (int k = 0; k < NCH; k++) {
    const float xs[8] = {xv[k][0].x, xv[k][0].y, xv[k][0].z, xv[k][0].w, xv[k][1].x, xv[k][1].y, xv[k][1].z, xv[k][1].w};
    const float ws[8] = {wv[k][0].x, wv[k][0].y, wv[k][0].z, wv[k][0].w, wv[k][1].x, wv[k][1].y, wv[k][1].z, wv[k][1].w};
    e[k] = 0.f;
#pragma unroll
    for (int r = 0; r < 8; r++) { ss = fmaf(xs[r], xs[r], ss); const float y = xs[r] * ws[r]; e[k] = fmaf(y, y, e[k]); }
  }
  ss = warp_sum(ss);
#pragma unroll
  for (int k = 0; k < NCH; k++) e[k] = warp_sum(e[k]);
  if (lane == 0) { red[wl][NCH] = ss; for (int k = 0; k < NCH; k++) red[wl][k] = e[k]; }
  __syncthreads();
  ss = red[0][NCH] + red[1][NCH] + red[2][NCH] + red[3][NCH];
  float emax = 0.f;
#pragma unroll
  for (int k = 0; k < NCH; k++) emax = fmaxf(emax, red[0][k] + red[1][k] + red[2][k] + red[3][k]);
  const float inv = rsqrtf(ss / (float)a.n + NORM_EPS);
  const float g = fmaxf(sqrtf(emax) * inv / (6.f * 448.f), 1e-30f), ig = 1.f / g;
  if (t == 0 && c == 0) a.g4[row] = g;
  float v[8];
  {
    // own chunk c (static register index through a select chain)
    float4 p = xv[0][0], q = xv[0][1], w0 = wv[0][0], w1 = wv[0][1];
#pragma unroll
    for (int k = 1; k < NCH; k++) if (k == c) { p = xv[k][0]; q = xv[k][1]; w0 = wv[k][0]; w1 = wv[k][1]; }
    v[0] = p.x * inv * w0.x * cs[0].x; v[1] = p.y * inv * w0.y * cs[0].y; v[2] = p.z * inv * w0.z * cs[0].z; v[3] = p.w * inv * w0.w * cs[0].w;
    v[4] = q.x * inv * w1.x * cs[1].x; v[5] = q.y * inv * w1.y * cs[1].y; v[6] = q.z * inv * w1.z * cs[1].z; v[7] = q.w * inv * w1.w * cs[1].w;
  }
#pragma unroll
  for (int len = 1; len < 8; len <<= 1)
#pragma unroll
    for (int r = 0; r < 8; r++)
      if (!(r & len)) { const float p = v[r], s = v[r + len]; v[r] = p + s; v[r + len] = p - s; }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1)
#pragma unroll
    for (int r = 0; r < 8; r++) { const float s = __shfl_xor_sync(0xffffffff, v[r], o); v[r] = (lane & o) ? s - v[r] : v[r] + s; }
#pragma unroll
  for (int r = 0; r < 8; r++) xch[wl][lane * 8 + r] = v[r];
  __syncthreads();
  {
    const int b8 = wl & 1, b9 = wl >> 1;
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const float A = xch[0][lane * 8 + r], B = xch[1][lane * 8 + r], C = xch[2][lane * 8 + r], Dd = xch[3][lane * 8 + r];
      const float t0 = b8 ? A - B : A + B, t1 = b8 ? C - Dd : C + Dd;
      v[r] = (b9 ? t0 - t1 : t0 + t1) * 0.03125f;
    }
  }
  float bm = 0.f;
#pragma unroll
  for (int r = 0; r < 8; r++) bm = fmaxf(bm, fabsf(v[r]));
  bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, 1));
  const float target = bm / 6.f * ig;
  __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
  float sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
  if (sv < target && sb < 0x7e) { sb++; sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3))); }
  const float rs = sv > 0.f ? ig / sv : 0.f;
  uint32_t wd = 0;
#pragma unroll
  for (int k = 0; k < 4; k++)
    wd |= (uint32_t)__nv_cvt_float2_to_fp4x2(make_float2(v[2 * k] * rs, v[2 * k + 1] * rs), __NV_E2M1, cudaRoundNearest) << (8 * k);
  const int e0 = c * 1024 + o0, tile = row >> 6, rr = row & 63, kb = e0 >> 7, o = e0 & 127, seg = o >> 5;
  const size_t base = ((size_t)tile * (a.n / 128) + kb) * 64 + rr;
  *(uint32_t*)(a.q4 + base * 64 + ((seg ^ ((rr >> 1) & 3)) << 4) + ((o & 31) >> 1)) = wd;
  if (!(lane & 1)) a.s4[base * 8 + (o >> 4)] = sb;
}
