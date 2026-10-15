#include "whole_ffn.cuh"
#include "../../src/kernels.cuh"
#include "../../src/fp4_ops.cuh"
namespace whole_ffn {
using namespace h5;
__device__ __forceinline__ unsigned exact_codes(int q){
  unsigned a=unsigned(q)&3,b=(unsigned(q)>>2)&3;int c=q>>4;
  unsigned cm=c<0?-c:c,bcode=b;
  if(cm==5||cm==7){int delta=(b&1)?1:-1;c+=delta;bcode=(0x95b4u>>(4*b))&15;}
  cm=c<0?-c:c;unsigned ccode=(cm<=4?cm:cm/2+2)|(c<0?8u:0u);
  return a|(bcode<<4)|(ccode<<8);
}
__device__ __forceinline__ unsigned expand(unsigned x){x&=255;x=(x|(x<<12))&0x000f000f;x=(x|(x<<6))&0x03030303;x=(x|(x<<3))&0x11111111;return 0x22222222u|(x<<3);}
template<int FW,int WG,int TW>struct G {static constexpr int NG=FW*WG,SW=NG*512,SS=NG*64,AQ=TW*64,STAGE=SW+SS+3*AQ,STAGES=4,SMEM=STAGE*STAGES+64,FUSED_SMEM=SMEM+TW*256*4;};
template<int FW,int WG,int TW>
__global__ void __launch_bounds__((FW+1)*32) compound(const uint8_t* D,const uint8_t* W,const uint8_t* S,bitserial::Carrier carrier,
  int N,int K,int R,bitserial::Carrier next,uint8_t* next_digits){
  using Geo=G<FW,WG,TW>;constexpr int NT=TW/8,MT=WG*2;
  extern __shared__ __align__(128) uint8_t sm[];
  auto* values=(float*)(sm+Geo::SMEM);
  auto* full=(uint64_t*)(sm+Geo::STAGE*Geo::STAGES);auto* empty=full+Geo::STAGES;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,groups=N/32,g0=blockIdx.y*Geo::NG,t0=blockIdx.x*TW;
  int all=K/128,kb0=all*blockIdx.z/gridDim.z,nkb=all*(blockIdx.z+1)/gridDim.z-kb0;
  size_t plane=size_t((R+63)/64)*64*K/2;
  if(tid==0){for(int s=0;s<Geo::STAGES;s++){mbar_init(full+s,1);mbar_init(empty+s,FW);}asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
  __syncthreads();
  if(warp==FW){if(lane==0){
    int ng=min(Geo::NG,groups-g0),pre=min(nkb,Geo::STAGES);unsigned tx=ng*576+3*Geo::AQ;
    uint64_t policy=weight_tma_policy();
    auto stage_weights=[&](int kb,int slot){
      auto* s=sm+slot*Geo::STAGE;size_t c=size_t(kb0+kb)*groups+g0;
      mbar_expect(full+slot,tx);
      {
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
      for(int p=0;p<3;p++)bulk_g2s(s+Geo::SW+Geo::SS+p*Geo::AQ,D+size_t(p)*plane+rb*64,Geo::AQ,full+slot);
    }
  }}
  if(warp<FW){
  int gid=lane>>2,tq=lane&3,gw=warp*WG;float acc[MT][NT][4]={};const float zero[4]={};
  pdl_wait();
  for(int kb=0;kb<nkb;kb++){
    int slot=kb%Geo::STAGES;mbar_wait(full+slot,(kb/Geo::STAGES)&1);auto* s=sm+slot*Geo::STAGE;
    unsigned a[MT][2][4];float ws[MT][2];
#pragma unroll
    for(int q=0;q<WG;q++){
      {
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
      int row=n*8+(lane&7),col=lane>>3;unsigned b[3][4];
#pragma unroll
      for(int p=0;p<3;p++)ldsm4(b[p][0],b[p][1],b[p][2],b[p][3],s+Geo::SW+Geo::SS+p*Geo::AQ+row*64+((col^((row>>1)&3))<<4));
      float sx[2];bool bad[2];
#pragma unroll
      for(int j=0;j<2;j++){int t=t0+n*8+tq*2+j;size_t pos=size_t(t)*all+kb0+kb;sx[j]=t<R?activation_scale(carrier.scales+pos):0;bad[j]=t<R?(*(volatile bitserial::InputStatus*)(carrier.status+pos)!=bitserial::InputStatus::Valid):false;}
#pragma unroll
      for(int m=0;m<MT;m++){
        float d[3][4];
#pragma unroll
        for(int p=0;p<3;p++){mma_fp4(d[p],a[m][0],b[p][0],b[p][1],0x38383838u,zero);mma_fp4(d[p],a[m][1],b[p][2],b[p][3],0x38383838u,d[p]);}
#pragma unroll
        for(int i=0;i<4;i++){
          float factor=2.f;
          int dot=__float2int_rn(factor*d[0][i])+4*__float2int_rn(factor*d[1][i])+16*__float2int_rn(factor*d[2][i]);
          float scale=__fmul_rn(ws[m][i>>1],sx[i&1]);
          acc[m][n][i]=bad[i&1]?__int_as_float(0x7fc00000):__fmaf_rn(float(dot),scale,acc[m][n][i]);
        }
      }
    }
    ring_release(empty+slot,lane==0);
  }
  pdl_wait();
#pragma unroll
  for(int n=0;n<NT;n++){
#pragma unroll
    for(int j=0;j<2;j++){
#pragma unroll
      for(int m=0;m<MT;m++){
#pragma unroll
        for(int rr=0;rr<2;rr++) values[(n*8+tq*2+j)*256+gw*32+gid+m*16+rr*8]=acc[m][n][rr*2+j];
      }
    }
  }
  }
  __syncthreads();
  int hidden=N/2,hg=g0/8,ngroups=hidden/128;
  if(tid<128){
    int lane=tid&31,warp=tid>>5;
    __shared__ float maxima[4],inverse;
    __shared__ int sums[4];
    __shared__ unsigned status;
    for(int token=0;token<TW;token++){
      int t=t0+token;float gate=values[token*256+tid],up=values[token*256+128+tid];
      float v=gate/(1.f+__expf(-gate))*up;
      float a=isfinite(v)?fabsf(v):0;
      for(int d=16;d;d/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
      if(lane==0)maxima[warp]=a;
      // The fifth producer warp waits outside this fixed 128-thread named barrier.
      unsigned bad=__ballot_sync(0xffffffff,!isfinite(v));
      if(lane==0)sums[warp]=bad!=0;
      asm volatile("bar.sync 1, 128;":::"memory");
      size_t pos=size_t(t)*ngroups+hg;
      if(tid==0){
        float m=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
        status=(sums[0]|sums[1]|sums[2]|sums[3])?1:m!=0&&(m<1e-30f||m>1e30f)?2:0;
        inverse=status||m==0?0:127.f/m;
        if(t<R){next.scales[pos]=status?0:m/127.f;next.status[pos]=bitserial::InputStatus(status);
          if(status&&next.error)atomicMax((unsigned*)next.error,status);}
      }
      asm volatile("bar.sync 1, 128;":::"memory");
      int q=t>=R||status?0:__float2int_rn(v*inverse);
      if(t<R&&next.q)next.q[size_t(t)*hidden+hg*128+tid]=int8_t(q);
      unsigned codes=exact_codes(q);
#pragma unroll
      for(int p=0;p<3;p++){
        unsigned word=(codes>>(p*4))&15;
#pragma unroll
        for(int d=1;d<8;d*=2)word|=__shfl_down_sync(0xffffffff,word,d)<<(4*d);
        if((lane&7)==0&&t<R){
          size_t rb=(size_t(t/64)*ngroups+hg)*64+t%64;
          size_t plane=size_t((R+63)/64)*64*hidden/2;
          *(uint32_t*)(next_digits+size_t(p)*plane+rb*64+((warp^((t%64>>1)&3))<<4)+(lane/8)*4)=word;
        }
      }
      for(int d=16;d;d/=2)q+=__shfl_xor_sync(0xffffffff,q,d);
      if(lane==0)sums[warp]=q;
      asm volatile("bar.sync 1, 128;":::"memory");
      if(tid==0&&t<R)next.dc[pos]=sums[0]+sums[1]+sums[2]+sums[3];
      asm volatile("bar.sync 1, 128;":::"memory");
    }
  }
  // The unobserved rows in the last 64-row digit tile are explicitly zero.
  if(blockIdx.x==gridDim.x-1){
    int end=(R+63)/64*64;
    for(int j=tid;j<(end-R)*64;j+=blockDim.x){int t=R+j/64,byte=j%64;
      size_t rb=(size_t(t/64)*ngroups+hg)*64+t%64;
      size_t plane=size_t((R+63)/64)*64*hidden/2;
#pragma unroll
      for(int p=0;p<3;p++) next_digits[size_t(p)*plane+rb*64+byte]=0;
    }
  }
  __syncthreads();pdl_launch();
}

__global__ void relabel(const uint8_t* raw,uint32_t* signs,__half* scales,int hidden,int groups){
  int n=2*hidden;size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(n)*groups)return;
  int row=i%n,g=i/n,chunk=row/256,j=row%256;
  int original=chunk*128+j%128+(j>=128?hidden:0);
  const uint8_t* b=raw+(size_t(original)*groups+g)*18;
  scales[i]=*reinterpret_cast<const __half*>(b);
#pragma unroll
  for(int w=0;w<4;w++){const uint8_t* q=b+2+w*4;signs[(size_t(g)*4+w)*n+row]=uint32_t(q[0])|(uint32_t(q[1])<<8)|(uint32_t(q[2])<<16)|(uint32_t(q[3])<<24);}
}
cudaError_t prepare(const uint8_t* raw,int hidden,int k,uint32_t* signs,__half* scales,cudaStream_t st){
  if(!raw||!signs||!scales||hidden<=0||hidden%128||k<=0||k%128)return cudaErrorInvalidValue;
  relabel<<<(size_t(hidden)*2*(k/128)+255)/256,256,0,st>>>(raw,signs,scales,hidden,k/128);return cudaGetLastError();
}
cudaError_t validate(const Plan& p){
  if(p.hidden<=0||p.hidden%128||p.k<=0||p.k%128||p.rows<1||p.rows>65535)return cudaErrorInvalidValue;
  switch(p.schedule){case signed_fp4::Schedule::F4G2T8:case signed_fp4::Schedule::F4G2T16:case signed_fp4::Schedule::F4G2T32:return cudaSuccess;default:return cudaErrorInvalidValue;}
}
cudaError_t configure(signed_fp4::Schedule s){
  switch(s){
    case signed_fp4::Schedule::F4G2T8:return cudaFuncSetAttribute(compound<4,2,8>,cudaFuncAttributeMaxDynamicSharedMemorySize,G<4,2,8>::FUSED_SMEM);
    case signed_fp4::Schedule::F4G2T16:return cudaFuncSetAttribute(compound<4,2,16>,cudaFuncAttributeMaxDynamicSharedMemorySize,G<4,2,16>::FUSED_SMEM);
    case signed_fp4::Schedule::F4G2T32:return cudaFuncSetAttribute(compound<4,2,32>,cudaFuncAttributeMaxDynamicSharedMemorySize,G<4,2,32>::FUSED_SMEM);
    default:return cudaErrorInvalidValue;
  }
}
template<int TW>cudaError_t launch(const Plan& p,const uint32_t* w,const __half* s,bitserial::Carrier c,signed_fp4::Digits d,bitserial::Carrier next,signed_fp4::Digits nd,cudaStream_t st){
  pdl(compound<4,2,TW>,dim3((p.rows+TW-1)/TW,p.hidden/128),dim3(160),G<4,2,TW>::FUSED_SMEM,st,
      (const uint8_t*)d.data,(const uint8_t*)w,(const uint8_t*)s,c,2*p.hidden,p.k,p.rows,next,nd.data);
  return cudaGetLastError();
}
cudaError_t gate_up(const Plan& p,const uint32_t* w,const __half* s,bitserial::Carrier c,signed_fp4::Digits d,bitserial::Carrier next,signed_fp4::Digits nd,cudaStream_t st){
  auto e=validate(p);if(e!=cudaSuccess)return e;
  if(!w||!s||!c.scales||!c.status||!next.scales||!next.dc||!next.status||!d.data||!nd.data||
     d.encoding!=signed_fp4::Encoding::ExactA8||nd.encoding!=signed_fp4::Encoding::ExactA8||
     d.capacity<signed_fp4::digit_bytes(p.k,p.rows)||nd.capacity<signed_fp4::digit_bytes(p.hidden,p.rows)||
     ((uintptr_t(w)|uintptr_t(s)|uintptr_t(d.data)|uintptr_t(nd.data))&15)||
     ((uintptr_t(c.scales)|uintptr_t(c.status)|uintptr_t(next.scales)|uintptr_t(next.status)|uintptr_t(next.dc))&3))return cudaErrorInvalidValue;
  switch(p.schedule){
    case signed_fp4::Schedule::F4G2T8:return launch<8>(p,w,s,c,d,next,nd,st);
    case signed_fp4::Schedule::F4G2T16:return launch<16>(p,w,s,c,d,next,nd,st);
    case signed_fp4::Schedule::F4G2T32:return launch<32>(p,w,s,c,d,next,nd,st);
    default:return cudaErrorInvalidValue;
  }
}
}
