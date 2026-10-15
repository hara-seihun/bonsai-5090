#include "f1_epilogue.cuh"
namespace f1 {
__global__ void k_prep(const float* X,const float* S,int K,int R,float fixed,uint8_t* Q,uint8_t* QS,float* QG) {
  __shared__ float maxima[17];
  int lane=threadIdx.x&31,w=threadIdx.x>>5,row=blockIdx.x,off=w*1024+lane*32;
  float v[32];
  h5::pdl_wait();h5::pdl_launch();
  #pragma unroll
  for(int i=0;i<8;i++) {
    float4 x=*(const float4*)(X+(size_t)row*K+off+4*i);
    float4 s=*(const float4*)(S+off+4*i);
    v[4*i]=x.x*s.x;v[4*i+1]=x.y*s.y;v[4*i+2]=x.z*s.z;v[4*i+3]=x.w*s.w;
  }
  #pragma unroll
  for(int b=1;b<32;b*=2)
    #pragma unroll
    for(int i=0;i<32;i++)if(!(i&b)){float a=v[i],c=v[i+b];v[i]=a+c;v[i+b]=a-c;}
  #pragma unroll
  for(int b=1;b<32;b*=2)
    #pragma unroll
    for(int i=0;i<32;i++){float a=__shfl_xor_sync(0xffffffff,v[i],b);v[i]=(lane&b)?a-v[i]:v[i]+a;}
  float bm[2]={0,0};
  #pragma unroll
  for(int i=0;i<32;i++){v[i]*=.03125f;bm[i/16]=fmaxf(bm[i/16],fabsf(v[i]));}
  float g=fixed;
  if(fixed==0) {
    float a=fmaxf(bm[0],bm[1]);
    for(int b=16;b;b/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,b));
    if(lane==0)maxima[w]=a;
    __syncthreads();a=0;
    for(int i=0;i<K/1024;i++)a=fmaxf(a,maxima[i]);
    g=fmaxf(a/(6.f*448.f),1e-30f);
  }
  if(threadIdx.x==0)QG[row]=g;
  unsigned wd[4];uint8_t sb[2];
  float ig=1.f/g;
  #pragma unroll
  for(int b=0;b<2;b++) {
    float target=bm[b]/6.f*ig;
    sb[b]=__nv_cvt_float_to_fp8(target,__NV_SATFINITE,__NV_E4M3);
    float sv=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb[b],__NV_E4M3)));
    if(sv<target && sb[b]<126){sb[b]++;sv=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb[b],__NV_E4M3)));}
    float rs=sv>0?ig/sv:0;
    #pragma unroll
    for(int h=0;h<2;h++) {
      unsigned z=0;
      #pragma unroll
      for(int j=0;j<4;j++) {
        int i=b*16+h*8+j*2;
        z|=(unsigned)__nv_cvt_float2_to_fp4x2(make_float2(v[i]*rs,v[i+1]*rs),__NV_E2M1,cudaRoundNearest)<<(8*j);
      }
      wd[b*2+h]=z;
    }
  }
  int rr=row&63,c=lane&3;
  size_t base=((size_t)(row>>6)*(K/128)+(off>>7))*64+rr;
  *(uint4*)(Q+base*64+((c^((rr>>1)&3))<<4))=make_uint4(wd[0],wd[1],wd[2],wd[3]);
  *(uint16_t*)(QS+base*8+c*2)=(unsigned)sb[0]|((unsigned)sb[1]<<8);
}
void prep(const float* X,const float* S,int K,int R,float fixed,uint8_t* Q,uint8_t* QS,float* QG,cudaStream_t st) {
  h5::pdl(k_prep,dim3(R),dim3(K/1024*32),0,st,X,S,K,R,fixed,Q,QS,QG);
}
}
