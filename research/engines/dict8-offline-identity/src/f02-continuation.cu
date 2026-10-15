#include "kernels.cuh"
#include <cuda_fp8.h>
#include <cuda_fp4.h>
namespace f02 {
using h5::pdl;
__global__ void __launch_bounds__(544) finish_h32(h5::PrepArgs a) {
  __shared__ float red[17];
  const int lane=threadIdx.x&31,w=threadIdx.x>>5,nw=blockDim.x>>5,row=blockIdx.x;
  const int e0=w*1024+lane*32;
  h5::pdl_wait();h5::pdl_launch();
  float v[32];
#pragma unroll
  for(int i=0;i<8;i++) {
    const float4 z=*(const float4*)(a.x+(size_t)row*a.ldx+e0+4*i);
    v[4*i]=z.x;v[4*i+1]=z.y;v[4*i+2]=z.z;v[4*i+3]=z.w;
  }
  // Bits 0..4 and signs are already paid by the GU producer. Preserve baseline stage order.
#pragma unroll
  for(int o=1;o<32;o<<=1)
#pragma unroll
    for(int r=0;r<32;r++) {
      const float q=__shfl_xor_sync(0xffffffff,v[r],o);
      v[r]=(lane&o)?q-v[r]:v[r]+q;
    }
#pragma unroll
  for(int r=0;r<32;r++)v[r]*=0.03125f;
  float bm0=0.f,bm1=0.f;
#pragma unroll
  for(int r=0;r<16;r++) {bm0=fmaxf(bm0,fabsf(v[r]));bm1=fmaxf(bm1,fabsf(v[r+16]));}
  float am=fmaxf(bm0,bm1);
  for(int o=16;o;o>>=1)am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,o));
  if(lane==0)red[w]=am;
  __syncthreads();
  am=0.f;for(int i=0;i<nw;i++)am=fmaxf(am,red[i]);
  const float g=fmaxf(am/(6.f*448.f),1e-30f),ig=1.f/g;
  if(threadIdx.x==0)a.g4[row]=g;
  uint32_t words[4];uint8_t sbs[2];
#pragma unroll
  for(int b=0;b<2;b++) {
    const float target=(b?bm1:bm0)/6.f*ig;
    __nv_fp8_storage_t sb=__nv_cvt_float_to_fp8(target,__NV_SATFINITE,__NV_E4M3);
    float sv=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3)));
    if(sv<target&&sb<0x7e){sb++;sv=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3)));}
    const float rs=sv>0.f?ig/sv:0.f;sbs[b]=sb;
#pragma unroll
    for(int h=0;h<2;h++) {
      uint32_t wd=0;
#pragma unroll
      for(int q=0;q<4;q++) {
        int r=16*b+8*h+2*q;
        wd|=(uint32_t)__nv_cvt_float2_to_fp4x2(make_float2(v[r]*rs,v[r+1]*rs),__NV_E2M1,cudaRoundNearest)<<(8*q);
      }
      words[2*b+h]=wd;
    }
  }
  const int tile=row>>6,rr=row&63,kb=e0>>7,c=lane&3;
  const size_t base=((size_t)tile*(a.n/128)+kb)*64+rr;
  *(uint4*)(a.q4+base*64+((c^((rr>>1)&3))<<4))=make_uint4(words[0],words[1],words[2],words[3]);
  *(uint16_t*)(a.s4+base*8+2*c)=(uint16_t)sbs[0]|((uint16_t)sbs[1]<<8);
}
void launch_finish(const h5::PrepArgs& a,int rows,cudaStream_t st) {
  pdl(finish_h32,dim3(rows),dim3(32*(a.n/1024)),0,st,a);
}
}
