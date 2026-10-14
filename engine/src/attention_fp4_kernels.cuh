#pragma once
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdint.h>
namespace p12kv4 {
// Layout: bytes [slot][head][position][128]; scales [slot][head][position][16].
// 144 B/head/position; no global scale and no decoded cache allocation.
__device__ __forceinline__ size_t offset(int64_t logical,int h,int ctx) {
 return ((logical/ctx)*4+h)*ctx+logical%ctx;
}
__device__ __forceinline__ void store(float x,int d,size_t at,uint8_t*data,uint8_t*scales) {
 float mx=fabsf(x);for(int o=8;o;o>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o));
 // E4M3 minimum subnormal prevents divide by zero for zero/tiny groups.
 uint8_t sc=__nv_cvt_float_to_fp8(fmaxf(mx/6.f,1.f/512.f),__NV_SATFINITE,__NV_E4M3);
 float scale=__half2float(__half(__nv_cvt_fp8_to_halfraw(sc,__NV_E4M3)));
 uint32_t q=__nv_cvt_float_to_fp4(x/scale,__NV_E2M1,cudaRoundNearest);
 uint32_t other=__shfl_xor_sync(0xffffffff,q,1);
 if(!(d&1))data[at*128+d/2]=q|(other<<4);
 if(!(d&15))scales[at*16+d/16]=sc;
}
__global__ void append(const float*x,size_t stride,const int64_t*ids,int rows,int ctx,uint8_t*data,uint8_t*scales) {
 const int r=blockIdx.x,h=blockIdx.y,d=threadIdx.x;
 store(*(const float*)((const char*)x+(size_t)r*stride+4*(h*256+d)),d,offset(ids[r],h,ctx),data,scales);
}
__device__ __forceinline__ float load(const uint8_t*data,const uint8_t*scales,size_t at,int d) {
 uint8_t q=(data[at*128+d/2]>>((d&1)*4))&15;
 return __half2float(__half(__nv_cvt_fp4_to_halfraw(q,__NV_E2M1)))*__half2float(__half(__nv_cvt_fp8_to_halfraw(scales[at*16+d/16],__NV_E4M3)));
}
__global__ void materialize(const uint8_t*data,const uint8_t*scales,const int64_t*map,__half*out,int cap,int ctx) {
 int p=blockIdx.x,h=blockIdx.y,d=threadIdx.x;int64_t id=map[p];if(id<0)return;
 out[(size_t)p*1024+h*256+d]=__float2half(load(data,scales,offset(id,h,ctx),d));
}
}
