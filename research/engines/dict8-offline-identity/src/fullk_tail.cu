#include "fullk_gu.cuh"
#include "kernels.cuh"
#include <cuda_fp8.h>
#include <cuda_fp4.h>
namespace h5 {
__device__ __forceinline__ float fullk_tail_silu(float x){return x/(1.f+__expf(-x));}
template<bool MIXED,bool H1=false> __global__ void __launch_bounds__(128) k_fullk_tail(PrepArgs a, float fixed_g, const float* reference, uint32_t* stats) { H5_LIT(0);
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
  #pragma unroll
  for(int k=0;k<(H1?4:8);k++) {
    const uint2 state=reinterpret_cast<const uint2*>(a.x)[(size_t)row*(a.ldx/(H1?2:1))+in0/(H1?2:1)+k];
    unsigned gb=state.x,ub=state.y;
    if constexpr(MIXED) {
      unsigned x,y;
      asm volatile("{ .reg .b32 t; shf.l.wrap.b32 t,%2,%2,7; xor.b32 %1,%3,t; xor.b32 %0,%2,%1; }":"=&r"(x),"=&r"(y):"r"(gb),"r"(ub));
      gb=x;ub=y;
    }
    const float g=__uint_as_float(gb),u=__uint_as_float(ub);
    if constexpr(H1){v[2*k]=g;v[2*k+1]=u;}
    else v[k]=g/(1.f+__expf(-g))*u;
  }
  if(reference) {
    #pragma unroll
    for(int r=0;r<8;r++) {
      float expected;
      if constexpr(H1){const int i=e0+(r&~1);const float a0=reference[(size_t)row*a.n+i]*a.signs[i],a1=reference[(size_t)row*a.n+i+1]*a.signs[i+1];expected=(r&1)?a0-a1:a0+a1;}
      else expected=reference[(size_t)row*a.n+e0+r];
      atomicAdd(stats,1u);
      if(__float_as_uint(v[r])!=__float_as_uint(expected)) {
        atomicAdd(stats+1,1u);atomicMax(stats+2,__float_as_uint(fabsf(v[r]-expected)));
        if(!isfinite(v[r])||!isfinite(expected))atomicAdd(stats+3,1u);
      }
    }
  }
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
    for (int r = 0; r < 8; r++) v[r] = v[r] * inv * w[r] * fullk_tail_silu(z[r]);
  }
  if (!H1 && (a.flags & P_SIGN)) {
    v[0] *= cs[0].x; v[1] *= cs[0].y; v[2] *= cs[0].z; v[3] *= cs[0].w;
    v[4] *= cs[1].x; v[5] *= cs[1].y; v[6] *= cs[1].z; v[7] *= cs[1].w;
  }
#pragma unroll
  for (int len = H1?2:1; len < 8; len <<= 1)
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


cudaError_t launch_fullk_tail(FullKGuMode mode,const uint2* carrier,const float* signs,int rows,int features,uint8_t* q,uint8_t* s,float* g,float fixed_g,const float* reference,uint32_t* stats,cudaStream_t stream) {
  if(!carrier||!signs||!q||!s||!g||rows!=64||features!=FF||!(fixed_g>0.f)||(!stats&&reference))return cudaErrorInvalidValue;
  PrepArgs a{};a.x=reinterpret_cast<const float*>(carrier);a.ldx=features;a.n=features;a.signs=signs;a.flags=P_SIGN|P_HADAMARD|P_QUANT;a.q4=q;a.s4=s;a.g4=g;a.early=4;
  if(mode==FullKGuMode::H1Mixed)pdl(k_fullk_tail<true,true>,dim3(rows,features/1024),dim3(128),0,stream,a,fixed_g,reference,stats);
  else if(mode==FullKGuMode::H1Plain)pdl(k_fullk_tail<false,true>,dim3(rows,features/1024),dim3(128),0,stream,a,fixed_g,reference,stats);
  else if(mode==FullKGuMode::TailMixed)pdl(k_fullk_tail<true>,dim3(rows,features/1024),dim3(128),0,stream,a,fixed_g,reference,stats);
  else if(mode==FullKGuMode::TailPlain)pdl(k_fullk_tail<false>,dim3(rows,features/1024),dim3(128),0,stream,a,fixed_g,reference,stats);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}
__global__ void k_compare_tail(const uint8_t*q,const uint8_t*qr,const uint8_t*s,const uint8_t*sr,const float*g,const float*gr,int rows,int features,uint32_t*stats){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<rows*features/2){atomicAdd(stats,1u);if(q[i]!=qr[i])atomicAdd(stats+1,1u);}
  if(i<rows*features/16){atomicAdd(stats+2,1u);if(s[i]!=sr[i])atomicAdd(stats+3,1u);}
  if(i<rows){atomicAdd(stats+4,1u);if(__float_as_uint(g[i])!=__float_as_uint(gr[i]))atomicAdd(stats+5,1u);}
}
cudaError_t compare_fullk_tail(const uint8_t*q,const uint8_t*qr,const uint8_t*s,const uint8_t*sr,const float*g,const float*gr,int rows,int features,uint32_t*stats,cudaStream_t stream){
  if(!q||!qr||!s||!sr||!g||!gr||!stats||rows!=64||features!=FF)return cudaErrorInvalidValue;
  k_compare_tail<<<(rows*features/2+255)/256,256,0,stream>>>(q,qr,s,sr,g,gr,rows,features,stats);return cudaGetLastError();
}
}
