#include "digits.cuh"
#include "../../src/kernels.cuh"
#include "../../src/fp4_ops.cuh"
#include <cmath>
namespace a8_fp4 {
using namespace h5;
size_t digit_bytes(int k,int r){return k>0&&!(k%128)&&r>0?size_t((r+63)/64)*64*k*2:0;}
size_t partial_floats(const Plan& p){return p.splits>1?size_t(p.splits)*p.rows*p.n:0;}
cudaError_t validate(const Plan& p){
  if(p.n<=0||p.n%32||p.k<=0||p.k%128||p.rows<1||p.rows>65535||p.splits<1||p.splits>p.k/128||
     (p.layout!=WeightLayout::WF8&&p.layout!=WeightLayout::Packed1))return cudaErrorInvalidValue;
  switch(p.schedule){case Schedule::F4G2T8:case Schedule::F4G2T16:case Schedule::F4G2T32:case Schedule::F8G1T32:case Schedule::F8G2T32:return cudaSuccess;}
  return cudaErrorInvalidValue;
}
__device__ __forceinline__ unsigned code(unsigned q,int p){
  unsigned d=(q>>(p*2))&3;
  if(p==3){return d==2?12u:d==3?10u:d*2;}
  return d==3?5u:d==2?4u:d*2;
}
__global__ void pack(const int8_t* q,uint8_t* out,int k,int rows){
  int kb=blockIdx.x,t=blockIdx.y*64+threadIdx.x/4,c=threadIdx.x%4;
  size_t base=(size_t(blockIdx.y)*(k/128)+kb)*64+threadIdx.x/4;
  size_t plane=size_t((rows+63)/64)*64*k/2;
  uint32_t words[4][4]={};
#pragma unroll
  for(int j=0;j<32;j++){
    unsigned v=t<rows?uint8_t(q[size_t(t)*k+kb*128+c*32+j]):0;
#pragma unroll
    for(int p=0;p<4;p++)words[p][j/8]|=code(v,p)<<(4*(j%8));
  }
#pragma unroll
  for(int p=0;p<4;p++)*(uint4*)(out+size_t(p)*plane+base*64+((c^(((threadIdx.x/4)>>1)&3))<<4))=make_uint4(words[p][0],words[p][1],words[p][2],words[p][3]);
}
cudaError_t encode(bitserial::Carrier c,int k,int r,Digits d,cudaStream_t st){
  size_t need=digit_bytes(k,r);
  if(!need||r>65535||!c.q||!c.scales||!c.status||!d.data||d.capacity<need||uintptr_t(d.data)%16)return cudaErrorInvalidValue;
  pack<<<dim3(k/128,(r+63)/64),256,0,st>>>(c.q,d.data,k,r);return cudaGetLastError();
}
__device__ __forceinline__ unsigned expand(unsigned x){x&=255;x=(x|(x<<12))&0x000f000f;x=(x|(x<<6))&0x03030303;x=(x|(x<<3))&0x11111111;return 0x22222222u|(x<<3);}
template<int FW,int WG,int TW>struct G {static constexpr int NG=FW*WG,SW=NG*512,SS=NG*64,AQ=TW*64,STAGE=SW+SS+4*AQ,STAGES=4,SMEM=STAGE*STAGES+64;};
template<int FW,int WG,int TW>
__global__ void __launch_bounds__((FW+1)*32) consume(const uint8_t* D,const uint8_t* W,const uint8_t* S,bitserial::Carrier carrier,
  int N,int K,int R,float* out,int ld,float alpha,float beta,WeightLayout layout){
  using Geo=G<FW,WG,TW>;constexpr int NT=TW/8,MT=WG*2;
  extern __shared__ __align__(128) uint8_t sm[];
  auto* full=(uint64_t*)(sm+Geo::STAGE*Geo::STAGES);auto* empty=full+Geo::STAGES;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,groups=N/32,g0=blockIdx.y*Geo::NG,t0=blockIdx.x*TW;
  int all=K/128,kb0=all*blockIdx.z/gridDim.z,nkb=all*(blockIdx.z+1)/gridDim.z-kb0;
  size_t plane=size_t((R+63)/64)*64*K/2;
  if(tid==0){for(int s=0;s<Geo::STAGES;s++){mbar_init(full+s,1);mbar_init(empty+s,FW);}asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
  __syncthreads();
  if(warp==FW){if(lane==0){
    int ng=min(Geo::NG,groups-g0),pre=min(nkb,Geo::STAGES);unsigned tx=ng*576+4*Geo::AQ;
    uint64_t policy=weight_tma_policy();
    auto stage_weights=[&](int kb,int slot){
      auto* s=sm+slot*Geo::STAGE;size_t c=size_t(kb0+kb)*groups+g0;
      mbar_expect(full+slot,tx);
      if(layout==WeightLayout::WF8)bulk_weight_g2s(s,W+c*512,ng*512,full+slot,policy);
      else {
#pragma unroll
        for(int strip=0;strip<4;strip++)bulk_weight_g2s(s+strip*Geo::NG*128,W+((size_t(kb0+kb)*4+strip)*N+g0*32)*4,ng*128,full+slot,policy);
      }
      bulk_g2s(s+Geo::SW,S+c*64,ng*64,full+slot);
    };
    for(int kb=0;kb<pre;kb++)stage_weights(kb,kb);
    pdl_wait();
    for(int kb=0;kb<nkb;kb++){int slot=kb%Geo::STAGES;auto* s=sm+slot*Geo::STAGE;
      if(kb>=pre){mbar_wait(empty+slot,((kb/Geo::STAGES)&1)^1);stage_weights(kb,slot);}
      size_t rb=(size_t(t0/64)*all+kb0+kb)*64+t0%64;
#pragma unroll
      for(int p=0;p<4;p++)bulk_g2s(s+Geo::SW+Geo::SS+p*Geo::AQ,D+size_t(p)*plane+rb*64,Geo::AQ,full+slot);
    }
  }return;}
  int gid=lane>>2,tq=lane&3,gw=warp*WG;float acc[MT][NT][4]={};const float zero[4]={};
  pdl_wait();
  for(int kb=0;kb<nkb;kb++){
    int slot=kb%Geo::STAGES;mbar_wait(full+slot,(kb/Geo::STAGES)&1);auto* s=sm+slot*Geo::STAGE;
    unsigned a[MT][2][4];float ws[MT][2];
#pragma unroll
    for(int q=0;q<WG;q++){
      if(layout==WeightLayout::WF8){
        uint4 packed=*(const uint4*)(s+(gw+q)*512+lane*16);unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
#pragma unroll
        for(int m=0;m<2;m++){
#pragma unroll
          for(int h=0;h<2;h++){unsigned w=words[m*2+h];auto* dst=a[q*2+m][h];dst[0]=expand(w);dst[2]=expand(w>>8);dst[1]=expand(w>>16);dst[3]=expand(w>>24);}
        }
        uint2 scale=*(const uint2*)(s+Geo::SW+(gw+q)*64+gid*8);auto* sh=(const __half2*)&scale;float2 x=__half22float2(sh[0]),y=__half22float2(sh[1]);ws[q*2][0]=x.x;ws[q*2][1]=x.y;ws[q*2+1][0]=y.x;ws[q*2+1][1]=y.y;
      }else{
#pragma unroll
        for(int m=0;m<2;m++){
          int row=16*m+gid;
#pragma unroll
          for(int h=0;h<2;h++){
            const uint32_t* lo=(const uint32_t*)(s+h*2*Geo::NG*128+(gw+q)*128);
            const uint32_t* hi=lo+Geo::NG*32;
            auto* dst=a[q*2+m][h];
            dst[0]=expand(~(lo[row]>>(8*tq)));dst[2]=expand(~(hi[row]>>(8*tq)));
            dst[1]=expand(~(lo[row+8]>>(8*tq)));dst[3]=expand(~(hi[row+8]>>(8*tq)));
          }
          const __half* scale=(const __half*)(s+Geo::SW+(gw+q)*64);
          ws[q*2+m][0]=__half2float(scale[row]);ws[q*2+m][1]=__half2float(scale[row+8]);
        }
      }
    }
#pragma unroll
    for(int n=0;n<NT;n++){
      int row=n*8+(lane&7),col=lane>>3;unsigned b[4][4];
#pragma unroll
      for(int p=0;p<4;p++)ldsm4(b[p][0],b[p][1],b[p][2],b[p][3],s+Geo::SW+Geo::SS+p*Geo::AQ+row*64+((col^((row>>1)&3))<<4));
      float sx[2];bool bad[2];
#pragma unroll
      for(int j=0;j<2;j++){int t=t0+n*8+tq*2+j;size_t pos=size_t(t)*all+kb0+kb;sx[j]=t<R?activation_scale(carrier.scales+pos):0;bad[j]=t<R?(*(volatile bitserial::InputStatus*)(carrier.status+pos)!=bitserial::InputStatus::Valid):false;}
#pragma unroll
      for(int m=0;m<MT;m++){
        float d[4][4];
#pragma unroll
        for(int p=0;p<4;p++){mma_fp4(d[p],a[m][0],b[p][0],b[p][1],0x38383838u,zero);mma_fp4(d[p],a[m][1],b[p][2],b[p][3],0x38383838u,d[p]);}
#pragma unroll
        for(int i=0;i<4;i++){
          int dot=__float2int_rn(d[0][i])+4*__float2int_rn(d[1][i])+16*__float2int_rn(d[2][i])+64*__float2int_rn(d[3][i]);
          float scale=__fmul_rn(ws[m][i>>1],sx[i&1]);
          acc[m][n][i]=bad[i&1]?__int_as_float(0x7fc00000):__fmaf_rn(float(dot),scale,acc[m][n][i]);
        }
      }
    }
    ring_release(empty+slot,lane==0);
  }
  pdl_wait();pdl_launch();if(g0+gw>=groups)return;
  if(gridDim.z>1)out+=size_t(blockIdx.z)*R*ld;
#pragma unroll
  for(int n=0;n<NT;n++){
#pragma unroll
    for(int j=0;j<2;j++){int t=t0+n*8+tq*2+j;if(t>=R)continue;
#pragma unroll
      for(int m=0;m<MT;m++){if(g0+gw+m/2>=groups)continue;
#pragma unroll
        for(int rr=0;rr<2;rr++){size_t at=size_t(t)*ld+(g0+gw)*32+gid+m*16+rr*8;float v=acc[m][n][rr*2+j];out[at]=gridDim.z>1?v:beta==0?__fmul_rn(alpha,v):__fmaf_rn(alpha,v,__fmul_rn(beta,out[at]));}
      }
    }
  }
}
__global__ void fold(const float* p,float* y,int N,int R,int ld,int splits,float alpha,float beta){pdl_wait();pdl_launch();size_t plane=size_t(N)*R,i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=plane)return;float v=0;for(int s=0;s<splits;s++)v=__fadd_rn(v,p[size_t(s)*plane+i]);size_t at=(i/N)*ld+i%N;y[at]=beta==0?__fmul_rn(alpha,v):__fmaf_rn(alpha,v,__fmul_rn(beta,y[at]));}
template<int FW,int WG,int TW>cudaError_t launch(const Plan& p,const uint8_t* d,const uint8_t* w,const uint8_t* s,bitserial::Carrier c,float* y,int ld,float a,float b,cudaStream_t st){
  using Geo=G<FW,WG,TW>;auto f=consume<FW,WG,TW>;
  pdl(f,dim3((p.rows+TW-1)/TW,(p.n/32+Geo::NG-1)/Geo::NG,p.splits),dim3((FW+1)*32),Geo::SMEM,st,d,w,s,c,p.n,p.k,p.rows,y,ld,a,b,p.layout);return cudaGetLastError();
}
cudaError_t configure(Schedule schedule){
  switch(schedule){
    case Schedule::F4G2T8:return cudaFuncSetAttribute(consume<4,2,8>,cudaFuncAttributeMaxDynamicSharedMemorySize,G<4,2,8>::SMEM);
    case Schedule::F4G2T16:return cudaFuncSetAttribute(consume<4,2,16>,cudaFuncAttributeMaxDynamicSharedMemorySize,G<4,2,16>::SMEM);
    case Schedule::F4G2T32:return cudaFuncSetAttribute(consume<4,2,32>,cudaFuncAttributeMaxDynamicSharedMemorySize,G<4,2,32>::SMEM);
    case Schedule::F8G1T32:return cudaFuncSetAttribute(consume<8,1,32>,cudaFuncAttributeMaxDynamicSharedMemorySize,G<8,1,32>::SMEM);
    case Schedule::F8G2T32:return cudaFuncSetAttribute(consume<8,2,32>,cudaFuncAttributeMaxDynamicSharedMemorySize,G<8,2,32>::SMEM);
  }
  return cudaErrorInvalidValue;
}
cudaError_t project(const Plan& p,const void* w,const void* s,bitserial::Carrier c,Digits d,float* y,int ld,float* partial,size_t capacity,float a,float b,cudaStream_t st){
  auto e=validate(p);if(e!=cudaSuccess)return e;size_t need=partial_floats(p);
  if(!w||!s||!c.scales||!c.status||!d.data||d.capacity<digit_bytes(p.k,p.rows)||!y||ld<p.n||!std::isfinite(a)||!std::isfinite(b)||
    ((uintptr_t(w)|uintptr_t(s)|uintptr_t(d.data))&15)||((uintptr_t(y)|uintptr_t(c.scales)|uintptr_t(c.status))&3)||
    (need&&(!partial||capacity<need||partial==y)))return cudaErrorInvalidValue;
  float* out=need?partial:y;int stride=need?p.n:ld;
  switch(p.schedule){
    case Schedule::F4G2T8:e=launch<4,2,8>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st);break;
    case Schedule::F4G2T16:e=launch<4,2,16>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st);break;
    case Schedule::F4G2T32:e=launch<4,2,32>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st);break;
    case Schedule::F8G1T32:e=launch<8,1,32>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st);break;
    case Schedule::F8G2T32:e=launch<8,2,32>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st);break;
  }
  if(e!=cudaSuccess||!need)return e;pdl(fold,dim3((size_t(p.rows)*p.n+255)/256),dim3(256),0,st,(const float*)partial,y,p.n,p.rows,ld,p.splits,a,b);return cudaGetLastError();
}
}
