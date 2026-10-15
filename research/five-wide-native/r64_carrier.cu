// Native R64 GU-geometry component: identical mxf4nvf4 MMA work fed from three weight carriers.
//   ARM 0 E4 : pre-expanded e2m1 operand words (4 bit/trit), no decode
//   ARM 1 WF2: WF0-style two operands interleaved per word (2 bit/trit), decode = mask / shift+mask
//   ARM 2 D5 : five trits per byte (1.6 bit/trit), decode = 256-entry shared LUT + funnel assembly
// All arms must produce bitwise-identical accumulators. Microbenchmark, not the engine kernel.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#define CK(x) do{auto ck_status=(x);if(ck_status!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(ck_status),__LINE__);exit(1);}}while(0)
constexpr int M=34816, K=5120, N=64, KS=K/64, MT=M/16, NT=N/8, WPB=2, MPW=2; // 2 m-tiles per warp
__device__ __forceinline__ void mma(float*d,const unsigned*a,unsigned b0,unsigned b1,unsigned sb){
  const uint16_t z=0; const unsigned sa=0x38383838u;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};"
   :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1),"r"(sa),"r"(sb),"h"(z));
}
template<int ARM> __global__ void __launch_bounds__(32*WPB) gemm(const uint8_t* __restrict__ W,const uint2* __restrict__ B,const unsigned* __restrict__ SB,const uint32_t* __restrict__ lut,float* out){
  __shared__ uint32_t book[256];
  if(ARM==2){for(int i=threadIdx.x;i<256;i+=blockDim.x)book[i]=lut[i];__syncthreads();}
  const int lane=threadIdx.x&31, warp=blockIdx.x*WPB+(threadIdx.x>>5);
  float acc[MPW][NT][4]; for(int a=0;a<MPW;a++)for(int n=0;n<NT;n++)for(int q=0;q<4;q++)acc[a][n][q]=0.f;
  for(int ks0=0;ks0<KS;ks0+=5){
    unsigned A[MPW][5][4];
#pragma unroll
    for(int mm=0;mm<MPW;mm++){const int mt=warp*MPW+mm;
      if(ARM==0){const uint4* p=(const uint4*)W+((size_t)mt*KS+ks0)*32+lane;
#pragma unroll
        for(int s=0;s<5;s++){uint4 v=__ldg(p+s*32);A[mm][s][0]=v.x;A[mm][s][1]=v.y;A[mm][s][2]=v.z;A[mm][s][3]=v.w;}}
      else if(ARM==1){const uint2* p=(const uint2*)W+((size_t)mt*KS+ks0)*32+lane;
#pragma unroll
        for(int s=0;s<5;s++){uint2 v=__ldg(p+s*32);A[mm][s][0]=v.x&0xAAAAAAAAu;A[mm][s][1]=(v.x<<1)&0xAAAAAAAAu;A[mm][s][2]=v.y&0xAAAAAAAAu;A[mm][s][3]=(v.y<<1)&0xAAAAAAAAu;}}
      else {const uint4* p=(const uint4*)W+((size_t)mt*(KS/5)+ks0/5)*64+lane*2; uint4 u0=__ldg(p),u1=__ldg(p+1);
        uint32_t L[8]={u0.x,u0.y,u0.z,u0.w,u1.x,u1.y,u1.z,u1.w}; uint32_t ex[33];
#pragma unroll
        for(int b=0;b<32;b++)ex[b]=book[(L[b>>2]>>(8*(b&3)))&255]; ex[32]=0;
#pragma unroll
        for(int w=0;w<20;w++){const int b=(8*w)/5,sh=((8*w)%5)*4;uint32_t v=(ex[b]>>sh)|(sh?(ex[b+1]<<(20-sh)):(ex[b+1]<<20));if(sh>=12)v|=ex[b+2]<<(40-sh);A[mm][w>>2][w&3]=v;}}
    }
#pragma unroll
    for(int s=0;s<5;s++){const int ks=ks0+s;
#pragma unroll
      for(int nt=0;nt<NT;nt++){uint2 b=__ldg(B+((size_t)ks*NT+nt)*32+lane);unsigned sb=__ldg(SB+((size_t)ks*NT+nt)*32+lane);
#pragma unroll
        for(int mm=0;mm<MPW;mm++)mma(acc[mm][nt],A[mm][s],b.x,b.y,sb);}}
  }
  const int g=lane>>2,t=lane&3;
  for(int mm=0;mm<MPW;mm++){const int r0=(warp*MPW+mm)*16+g;for(int nt=0;nt<NT;nt++){const int c=nt*8+2*t;
    out[(size_t)r0*N+c]=acc[mm][nt][0];out[(size_t)r0*N+c+1]=acc[mm][nt][1];out[(size_t)(r0+8)*N+c]=acc[mm][nt][2];out[(size_t)(r0+8)*N+c+1]=acc[mm][nt][3];}}
}
static const float E2[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
int main(int argc,char**argv){
  const int reps=argc>1?atoi(argv[1]):20;
  std::mt19937 rng(20261014);
  // ternary weights, P(0)=1/3
  std::vector<int8_t> Wt((size_t)M*K); for(auto&w:Wt)w=int8_t(int(rng()%3)-1);
  auto nib=[](int t){return t>0?0x2u:t<0?0xAu:0x0u;};
  // fragment coordinate of (mt,ks,lane,reg,i)
  auto rowk=[](int mt,int ks,int lane,int reg,int i,int&row,int&k){int g=lane>>2,t=lane&3;row=mt*16+g+(reg&1)*8;k=ks*64+(reg>>1)*32+t*8+i;};
  std::vector<uint32_t> E4((size_t)MT*KS*32*4), WF((size_t)MT*KS*32*2);
  std::vector<uint8_t> D5((size_t)MT*(KS/5)*32*32);
  for(int mt=0;mt<MT;mt++)for(int ks=0;ks<KS;ks++)for(int l=0;l<32;l++){uint32_t r[4]={0,0,0,0};
    for(int reg=0;reg<4;reg++)for(int i=0;i<8;i++){int row,k;rowk(mt,ks,l,reg,i,row,k);r[reg]|=nib(Wt[(size_t)row*K+k])<<(4*i);}
    size_t e=((size_t)mt*KS+ks)*32+l; for(int q=0;q<4;q++)E4[e*4+q]=r[q];
    // WF: word0 holds reg0 at nibble bits1,3 and reg1 at bits0,2; word1 regs2,3
    for(int h=0;h<2;h++){uint32_t a=r[2*h],b=r[2*h+1];WF[e*2+h]=(a&0xAAAAAAAAu)|((b>>1)&0x55555555u);}
    // D5: lane stream s=(ks%5)*32+reg*8+i, byte s/5 digit s%5
    for(int reg=0;reg<4;reg++)for(int i=0;i<8;i++){int row,k;rowk(mt,ks,l,reg,i,row,k);int s=(ks%5)*32+reg*8+i;int p=1;for(int d=0;d<s%5;d++)p*=3;
      size_t byte=(((size_t)mt*(KS/5)+ks/5)*32+l)*32+s/5;D5[byte]+=uint8_t((Wt[(size_t)row*K+k]+1)*p);}}
  std::vector<uint32_t> lut(256,0); for(int v=0;v<243;v++){int x=v;uint32_t o=0;for(int d=0;d<5;d++){o|=nib(x%3-1)<<(4*d);x/=3;}lut[v]=o;}
  // activations: e2m1 codes, all 16 codes
  std::vector<uint8_t> X((size_t)K*N); for(auto&x:X)x=rng()&15;
  std::vector<uint2> Bf((size_t)KS*NT*32); std::vector<unsigned> S1((size_t)KS*NT*32,0x38383838u),SR((size_t)KS*NT*32);
  for(int ks=0;ks<KS;ks++)for(int nt=0;nt<NT;nt++)for(int l=0;l<32;l++){int g=l>>2,t=l&3;uint32_t b0=0,b1=0;
    for(int i=0;i<8;i++){b0|=uint32_t(X[(size_t)(ks*64+t*8+i)*N+nt*8+g])<<(4*i);b1|=uint32_t(X[(size_t)(ks*64+32+t*8+i)*N+nt*8+g])<<(4*i);}
    size_t e=((size_t)ks*NT+nt)*32+l;Bf[e]={b0,b1};uint32_t s=0;for(int q=0;q<4;q++)s|=uint32_t(0x38+(rng()&7))<<(8*q);SR[e]=s;}
  // copies to defeat L2 (96 MiB): total per arm > 128 MiB
  const int cE=2,cW=3,cD=4; size_t bE=E4.size()*4,bW=WF.size()*4,bD=D5.size();
  uint8_t *dE,*dW,*dD;CK(cudaMalloc(&dE,bE*cE));CK(cudaMalloc(&dW,bW*cW));CK(cudaMalloc(&dD,bD*cD));
  for(int c=0;c<cE;c++)CK(cudaMemcpy(dE+c*bE,E4.data(),bE,cudaMemcpyHostToDevice));
  for(int c=0;c<cW;c++)CK(cudaMemcpy(dW+c*bW,WF.data(),bW,cudaMemcpyHostToDevice));
  for(int c=0;c<cD;c++)CK(cudaMemcpy(dD+c*bD,D5.data(),bD,cudaMemcpyHostToDevice));
  uint2*dB;unsigned*dS1,*dSR;uint32_t*dL;float*dO[3];
  CK(cudaMalloc(&dB,Bf.size()*8));CK(cudaMalloc(&dS1,S1.size()*4));CK(cudaMalloc(&dSR,SR.size()*4));CK(cudaMalloc(&dL,1024));
  for(int a=0;a<3;a++)CK(cudaMalloc(&dO[a],(size_t)M*N*4));
  CK(cudaMemcpy(dB,Bf.data(),Bf.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dS1,S1.data(),S1.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSR,SR.data(),SR.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dL,lut.data(),1024,cudaMemcpyHostToDevice));
  dim3 grid(MT/MPW/WPB),blk(32*WPB);
  auto launch=[&](int arm,int c,const unsigned*sb){
    if(arm==0)gemm<0><<<grid,blk>>>(dE+(c%cE)*bE,dB,sb,dL,dO[0]);
    else if(arm==1)gemm<1><<<grid,blk>>>(dW+(c%cW)*bW,dB,sb,dL,dO[1]);
    else gemm<2><<<grid,blk>>>(dD+(c%cD)*bD,dB,sb,dL,dO[2]);};
  // 1) exactness against CPU with unit scales (all products/sums exact in FP32)
  std::vector<float> h((size_t)M*N); long bad[3]={0,0,0};
  std::vector<int> rows; for(int r=0;r<M;r+=97)rows.push_back(r); rows.push_back(M-1);
  for(int arm=0;arm<3;arm++){launch(arm,0,dS1);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(h.data(),dO[arm],h.size()*4,cudaMemcpyDeviceToHost));
    for(int r:rows)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=Wt[(size_t)r*K+k]*E2[X[(size_t)k*N+n]];if((float)s!=h[(size_t)r*N+n])bad[arm]++;}}
  // 2) random real-format scales: all arms bitwise identical over the complete M x N output
  std::vector<float> ref((size_t)M*N); long cross[3]={0,0,0};
  for(int arm=0;arm<3;arm++){launch(arm,1,dSR);CK(cudaDeviceSynchronize());CK(cudaMemcpy(h.data(),dO[arm],h.size()*4,cudaMemcpyDeviceToHost));
    if(arm==0)ref=h; else for(size_t i=0;i<h.size();i++)cross[arm]+=memcmp(&h[i],&ref[i],4)!=0;}
  // 3) timing: ABC interleaved rounds, rotating copies
  cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));float tms[3][64];
  for(int c=0;c<6;c++)for(int arm=0;arm<3;arm++)launch(arm,c,dSR);
  for(int r=0;r<reps;r++)for(int arm=0;arm<3;arm++){CK(cudaEventRecord(e0));launch(arm,r+7,dSR);CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));CK(cudaEventElapsedTime(&tms[arm][r],e0,e1));}
  cudaFuncAttributes at[3];CK(cudaFuncGetAttributes(&at[0],gemm<0>));CK(cudaFuncGetAttributes(&at[1],gemm<1>));CK(cudaFuncGetAttributes(&at[2],gemm<2>));
  cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
  const char* nm[3]={"E4","WF2","D5"};size_t bytes[3]={bE,bW,bD};
  printf("{\"device\":\"%s\",\"M\":%d,\"K\":%d,\"N\":%d,\"cpu_rows\":%zu,\"reps\":%d,\"arms\":[",p.name,M,K,N,rows.size(),reps);
  for(int a=0;a<3;a++){std::vector<float> v(tms[a],tms[a]+reps);std::sort(v.begin(),v.end());
    printf("%s{\"arm\":\"%s\",\"weight_bytes\":%zu,\"cpu_mismatch\":%ld,\"bitwise_vs_E4\":%ld,\"median_us\":%.3f,\"min_us\":%.3f,\"weight_GBps\":%.1f,\"regs\":%d,\"local\":%zu}",a?",":"",nm[a],bytes[a],bad[a],cross[a],v[reps/2]*1e3,v[0]*1e3,bytes[a]/(v[reps/2]*1e-3)/1e9,at[a].numRegs,at[a].localSizeBytes);}
  printf("],\"scope\":\"standalone microkernel, R64 GU geometry, identical MMA work; not the engine GEMM; relative carrier sensitivity only\"}\n");
  return (bad[0]||bad[1]||bad[2]||cross[1]||cross[2])?2:0;
}
