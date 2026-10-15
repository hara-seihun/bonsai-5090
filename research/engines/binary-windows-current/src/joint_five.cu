// Programme 12 joint-five coordinate. Pays its factorized encoder and native consumer.
#include "joint_five.cuh"
#include "fp4_ops.cuh"
#include <cuda_fp16.h>
#include <array>
namespace p12_joint_five {
__constant__ uint16_t xm[256],ym[256],pbook[289],qbook[817];
__constant__ int8_t value[15];
__constant__ uint8_t native_code[15];
cudaError_t initialize() {
  constexpr std::array<int8_t,15> d={-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12};
  constexpr std::array<uint8_t,15> code={15,14,13,12,11,10,9,0,1,2,3,4,5,6,7};
  std::array<uint16_t,289> p;std::array<uint16_t,817> q; p.fill(256);q.fill(256);
  std::array<uint16_t,256> x={},y={};
  for(int i=0;i<15;i++)for(int j=0;j<15;j++) {
    int a=d[i]+11*d[j]+144,b=15*d[i]+19*d[j]+408;
    if(p[a]==256)p[a]=code[i]|(uint16_t(code[j])<<4);
    if(q[b]==256)q[b]=code[i]|(uint16_t(code[j])<<4);
  }
  for(int a=-128;a<=127;a++)for(int j=0;j<15;j++) {
    int r=a-11*d[j]+144,s=a-d[j]+408;
    if(r>=0&&r<289&&p[r]!=256)x[a+128]|=uint16_t(1u<<j);
    if(s>=0&&s<817&&q[s]!=256)y[a+128]|=uint16_t(1u<<j);
  }
  for(auto a:x)for(auto b:y)if(!(a&b))return cudaErrorInvalidValue;
  cudaError_t status;
#define COPY(name,host) if((status=cudaMemcpyToSymbol(name,host.data(),sizeof(host)))!=cudaSuccess)return status
  COPY(xm,x);COPY(ym,y);COPY(pbook,p);COPY(qbook,q);COPY(value,d);COPY(native_code,code);
#undef COPY
  return cudaSuccess;
}
size_t bytes(int k,int r){return k>0&&k%128==0&&r>0?size_t(5)*((r+1)/2)*(k/2):0;}
__device__ __forceinline__ uint32_t pair(int x,int y){
  int j=__ffs(unsigned(xm[x+128]&ym[y+128]))-1;
  int d=value[j];return unsigned(pbook[x-11*d+144])|(unsigned(qbook[y-d+408])<<8)|(unsigned(native_code[j])<<16);
}
__global__ void producer(const int8_t* q,int k,int rows,uint8_t* out) {
  size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t((rows+1)/2)*(k/2);
  if(index>=total)return;
  int t=int(index/(k/2))*2,j=int(index%(k/2))*2;
  uint32_t a=pair(q[size_t(t)*k+j],t+1<rows?q[size_t(t+1)*k+j]:0);
  uint32_t b=pair(q[size_t(t)*k+j+1],t+1<rows?q[size_t(t+1)*k+j+1]:0);
#pragma unroll
  for(int p=0;p<5;p++)out[size_t(p)*total+index]=uint8_t(((a>>(4*p))&15)|(((b>>(4*p))&15)<<4));
}
cudaError_t encode(bitserial::Carrier c,int k,int rows,uint8_t* out,size_t capacity,cudaStream_t st){
  size_t need=bytes(k,rows);if(!need||rows>32||!c.q||!out||capacity<need)return cudaErrorInvalidValue;
  size_t n=size_t((rows+1)/2)*(k/2);producer<<<unsigned((n+255)/256),256,0,st>>>(c.q,k,rows,out);return cudaGetLastError();
}
__device__ __forceinline__ unsigned expand(unsigned x){x&=255;x=(x|(x<<12))&0x000f000f;x=(x|(x<<6))&0x03030303;x=(x|(x<<3))&0x11111111;return 0x22222222u|(x<<3);}
__global__ void consumer(const uint32_t* w,const __half* scales,bitserial::Carrier c,const uint8_t* digits,int n,int k,int rows,float* out,int stride,float alpha,float beta){
  int lane=threadIdx.x,g=lane>>2,t=lane&3,row=int(blockIdx.x)*16+g,token0=int(blockIdx.y)*16;
  size_t plane=size_t((rows+1)/2)*(k/2);float acc[8]={};const float zero[4]={};
  for(int kb=0;kb<k/128;kb++) {
    float dots[5][4]={};
#pragma unroll
    for(int h=0;h<2;h++) {
      unsigned a[4];
#pragma unroll
      for(int r=0;r<4;r++) {
        int feature=row+((r&1)?8:0),strip=h*2+(r>>1);
        unsigned bits=w[(size_t(kb)*4+strip)*n+feature];
        a[r]=expand(~(bits>>(8*t)));
      }
#pragma unroll
      for(int p=0;p<5;p++) {
        unsigned b0=0,b1=0;int pairid=token0/2+g;
        if(pairid<(rows+1)/2) {
          const uint8_t* b=digits+size_t(p)*plane+size_t(pairid)*(k/2)+kb*64+h*32+t*4;
#pragma unroll
          for(int j=0;j<4;j++){b0|=unsigned(b[j])<<(8*j);b1|=unsigned(b[16+j])<<(8*j);}
        }
        h5::mma_fp4(dots[p],a,b0,b1,0x38383838u,h?dots[p]:zero);
      }
    }
#pragma unroll
    for(int i=0;i<4;i++) {
      int feature=row+(i>>1)*8,pairid=token0/2+2*t+(i&1);
      float ws=__half2float(scales[size_t(kb)*n+feature]);
      float d0=__fmaf_rn(22.f,dots[4][i],__fmaf_rn(22.f,dots[1][i],__fmul_rn(2.f,dots[0][i])));
      float d1=__fmaf_rn(2.f,dots[4][i],__fmaf_rn(38.f,dots[3][i],__fmul_rn(30.f,dots[2][i])));
#pragma unroll
      for(int which=0;which<2;which++) {
        int token=2*pairid+which;if(token>=rows)continue;
        size_t pos=size_t(token)*(k/128)+kb;
        float scale=__fmul_rn(ws,c.scales[pos]);
        float dot=which?d1:d0;
        acc[2*i+which]=c.status[pos]!=bitserial::InputStatus::Valid?__int_as_float(0x7fc00000):__fmaf_rn(dot,scale,acc[2*i+which]);
      }
    }
  }
#pragma unroll
  for(int i=0;i<4;i++)for(int which=0;which<2;which++) {
    int feature=row+(i>>1)*8,token=token0+4*t+2*(i&1)+which;
    if(token<rows){size_t at=size_t(token)*stride+feature;float v=acc[2*i+which];out[at]=beta==0?__fmul_rn(alpha,v):__fmaf_rn(alpha,v,__fmul_rn(beta,out[at]));}
  }
}
cudaError_t project(const uint8_t* w,const void* s,bitserial::Carrier c,const uint8_t* d,int n,int k,int r,float* out,int ld,float a,float b,cudaStream_t st){
 if(!w||!s||!c.q||!c.scales||!c.status||!d||!out||n<=0||n%32||!bytes(k,r)||r>32||ld<n)return cudaErrorInvalidValue;
 consumer<<<dim3(n/16,(r+15)/16),32,0,st>>>((const uint32_t*)w,(const __half*)s,c,d,n,k,r,out,ld,a,b);return cudaGetLastError();
}
}
