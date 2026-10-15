#include "fullk_gu.cuh"
#include "kernels.cuh"
namespace h5 {
namespace fullk_gu {
constexpr int K=5120;
struct GuArgs { const uint8_t *weights,*q4,*s4; const float *ag,*means; const uint32_t*book; float*out; };
__device__ __forceinline__ void fm(float*d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned sb){const uint16_t z=0;const unsigned sa=0x38383838;__syncwarp();asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb),"h"(z));__syncwarp();}
__device__ __forceinline__ uint64_t state_label(unsigned gb,unsigned ub){unsigned lo,hi;asm volatile("{ .reg .b32 l,h,t; xor.b32 l,%2,%3; shf.l.wrap.b32 t,l,l,7; xor.b32 h,%3,t; mov.b32 %0,l; mov.b32 %1,h; }":"=r"(lo),"=r"(hi):"r"(gb),"r"(ub));return (uint64_t(hi)<<32)|lo;}
__device__ __forceinline__ void state_read(uint64_t state,unsigned&gb,unsigned&ub){unsigned lo=state,hi=state>>32;asm volatile("{ .reg .b32 g,u,t; shf.l.wrap.b32 t,%2,%2,7; xor.b32 u,%3,t; xor.b32 g,%2,u; mov.b32 %0,g; mov.b32 %1,u; }":"=r"(gb),"=r"(ub):"r"(lo),"r"(hi));}
__device__ __forceinline__ unsigned gather8(unsigned l,unsigned h,unsigned top,int nib,int group){int owner=nib/20,ofs=(nib%20)*4;unsigned ll=__shfl_sync(0xffffffff,l,group*4+owner),hh=__shfl_sync(0xffffffff,h,group*4+owner),tt=__shfl_sync(0xffffffff,top,group*4+owner),next=__shfl_sync(0xffffffff,l,group*4+((owner+1)&3));unsigned out=ofs<32?__funnelshift_r(ll,hh,ofs):ofs<64?__funnelshift_r(hh,tt,ofs-32):tt>>(ofs-64);if(ofs>48)out|=next<<(80-ofs);return out;}
// Coalesced64-K slabs: sixteen bytes/row, thirteen dense5 codes plus zero-trit padding.
// Four adjacent lanes each load four codes; every eight-feature warp reads128 contiguous bytes.
__device__ __forceinline__ void decode64(GuArgs a,int proj,int row,int k0,unsigned&first,unsigned&second){
  int lane=threadIdx.x&31,t=lane&3,g=lane>>2;
  const uint8_t*p=a.weights+(((size_t)(k0/64)*2+proj)*FF+row)*16;
  unsigned raw=*(const unsigned*)(p+4*t);
  unsigned v0=a.book[raw&255],v1=a.book[(raw>>8)&255],v2=a.book[(raw>>16)&255],v3=a.book[raw>>24];
  uint64_t lo=uint64_t(v0)|(uint64_t(v1)<<20)|(uint64_t(v2)<<40)|(uint64_t(v3)<<60);
  first=gather8(unsigned(lo),unsigned(lo>>32),v3>>4,8*t,g);
  second=gather8(unsigned(lo),unsigned(lo>>32),v3>>4,8*t+32,g);
}

template<bool MIXED> __global__ __launch_bounds__(512,3) void kernel(GuArgs a) {
  pdl_wait();
  const int warp=threadIdx.x/32,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int base=(blockIdx.x/8)*128+warp*8, nt=(blockIdx.x%8)*8;
  uint64_t state[2]={}; float accum[4]={};
  for(int kb=0;kb<K/128;kb++) {
    if constexpr(MIXED) {
      #pragma unroll
      for(int c=0;c<2;c++){unsigned gb,ub;state_read(state[c],gb,ub);accum[c]=__uint_as_float(gb);accum[c+2]=__uint_as_float(ub);}
    }
    #pragma unroll
    for(int h=0;h<2;h++) {
      const int k0=kb*128+h*64;
      unsigned a0,a1,a2,a3;decode64(a,0,base+g,k0,a0,a2);decode64(a,1,base+g,k0,a1,a3);
      const int row=nt+g,sw=(row>>1)&3;
      const size_t dst=(size_t)kb*64+row;
      const unsigned b0=*(const unsigned*)(a.q4+dst*64+((2*h)^sw)*16+4*t);
      const unsigned b1=*(const unsigned*)(a.q4+dst*64+((2*h+1)^sw)*16+4*t);
      const unsigned sb=*(const unsigned*)(a.s4+dst*8+h*4);
      fm(accum,a0,a1,a2,a3,b0,b1,sb);
    }
    if constexpr(MIXED) {
      #pragma unroll
      for(int c=0;c<2;c++)state[c]=state_label(__float_as_uint(accum[c]),__float_as_uint(accum[c+2]));
    }
  }
  pdl_launch();
  #pragma unroll
  for(int c=0;c<2;c++) {
    const int f=base+g,row=nt+2*t+c,phys=64*(f/32)+(f%32);
    float gg,uu;
    if constexpr(MIXED){unsigned gb,ub;state_read(state[c],gb,ub);gg=__uint_as_float(gb);uu=__uint_as_float(ub);}
    else{gg=accum[c];uu=accum[c+2];}
    const float ag=*reinterpret_cast<const volatile float*>(a.ag+row);
    gg*=ag;uu*=ag;gg*=a.means[phys];uu*=a.means[phys+32];
    a.out[(size_t)row*FF+f]=(gg/(1.f+__expf(-gg)))*uu;
  }
}
}
cudaError_t launch_fullk_gu(FullKGuMode mode,const uint8_t*weights,const uint8_t*q4,const uint8_t*s4,const float*ag,const float*means,const uint32_t*book,int rows,int features,int contraction,float*out,cudaStream_t stream) {
  if(!weights||!q4||!s4||!ag||!means||!book||!out||rows!=64||features!=FF||contraction!=D)return cudaErrorInvalidValue;
  fullk_gu::GuArgs a{weights,q4,s4,ag,means,book,out};
  switch(mode){
    case FullKGuMode::Mixed:pdl(fullk_gu::kernel<true>,dim3((FF/128)*8),dim3(512),0,stream,a);break;
    case FullKGuMode::Plain:pdl(fullk_gu::kernel<false>,dim3((FF/128)*8),dim3(512),0,stream,a);break;
    case FullKGuMode::RingMixed:case FullKGuMode::RingPlain:case FullKGuMode::Serving:return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
__global__ void k_fullk_compare(const float*a,const float*b,int n,uint32_t*stats){pdl_wait();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned x=__float_as_uint(a[i]),y=__float_as_uint(b[i]);if(x!=y){atomicAdd(stats+1,1u);float d=fabsf(a[i]-b[i]);if(isfinite(d))atomicMax(stats+2,__float_as_uint(d));else atomicAdd(stats+3,1u);}}if(threadIdx.x==0)atomicAdd(stats,unsigned(min(256,n-int(blockIdx.x*256))));pdl_launch();}
cudaError_t compare_fullk_gu(const float*a,const float*b,int n,uint32_t*stats,cudaStream_t st){if(!a||!b||!stats||n<=0)return cudaErrorInvalidValue;pdl(k_fullk_compare,dim3((n+255)/256),dim3(256),0,st,a,b,n,stats);return cudaGetLastError();}

}
