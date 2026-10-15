#include "sparse_digits.cuh"
#include "../../src/kernels.cuh"
#include "../../src/fp4_ops.cuh"
#include <cmath>
namespace sparse_fp4 {
using namespace h5;
size_t digit_bytes(int k,int r){return k>0&&!(k%128)&&r>0?size_t((r+63)/64)*64*k:0;}
size_t count_bytes(int k,int r){return digit_bytes(k,r)/128;}
size_t entry_words(int k,int r){return count_bytes(k,r)*ListCapacity;}
size_t partial_floats(const Plan& p){return p.splits>1?size_t(p.splits)*p.rows*p.n:0;}
cudaError_t validate(const Plan& p){
  if(p.n<=0||p.n%32||p.k<=0||p.k%128||p.rows<1||p.rows>65535||p.splits<1||p.splits>p.k/128||
     p.layout!=WeightLayout::Packed1)return cudaErrorInvalidValue;
  switch(p.schedule){case Schedule::F4G2T8:case Schedule::F4G2T16:case Schedule::F4G2T32:case Schedule::F8G1T32:case Schedule::F8G2T32:return cudaSuccess;}
  return cudaErrorInvalidValue;
}
__device__ const uint16_t labels[256]={64224,64225,64226,64227,64767,64219,64229,64217,64766,64209,64765,64211,64764,64763,64762,64761,64752,64753,64754,64755,65023,64177,64757,64179,65022,64161,65021,64163,65020,65019,65018,65017,65008,65009,65010,65011,65279,64731,65013,64729,65278,64721,65277,64723,65276,65275,65274,65273,65264,65265,65266,65267,65535,64987,65269,64985,65534,64977,65533,64979,65532,65531,65530,65529,65520,65521,65522,65523,255,65243,65525,65241,254,65233,253,65235,252,251,250,249,240,241,242,243,244,65499,245,65497,246,65489,237,65491,247,235,234,233,224,225,226,227,228,219,229,217,208,209,210,211,212,203,213,201,192,193,194,195,176,177,178,179,160,161,162,163,144,145,146,147,0,1,2,3,16,17,18,19,32,33,34,35,48,49,50,51,64,65,66,67,68,91,69,89,80,81,82,83,84,107,85,105,96,97,98,99,100,347,101,345,102,337,125,339,103,123,122,121,112,113,114,115,116,603,117,601,118,593,381,595,119,379,378,377,368,369,370,371,372,859,373,857,374,849,637,851,375,635,634,633,624,625,626,627,628,1115,629,1113,630,1105,893,1107,631,891,890,889,880,881,882,883,884,1553,885,1555,886,1569,1149,1571,887,1147,1146,1145,1136,1137,1138,1139,1140,1627,1141,1625,1142,1617,1618,1619,1143,1643,1621,1641};

__global__ void pack(bitserial::Carrier carrier,Digits out,int k,int rows){
  int kb=blockIdx.x,t=blockIdx.y*64+threadIdx.x/4,c=threadIdx.x%4;
  size_t base=(size_t(blockIdx.y)*(k/128)+kb)*64+threadIdx.x/4;
  size_t plane=size_t((rows+63)/64)*64*k/2;
  uint32_t words[2][4]={};unsigned mask=0;
#pragma unroll
  for(int j=0;j<32;j++){
    int v=t<rows?int(carrier.q[size_t(t)*k+kb*128+c*32+j]):0;
    unsigned label=labels[v+128];mask|=unsigned((label>>8)!=0)<<j;
#pragma unroll
    for(int p=0;p<2;p++)words[p][j/8]|=((label>>(p*4))&15)<<(4*(j%8));
  }
#pragma unroll
  for(int p=0;p<2;p++)*(uint4*)(out.data+size_t(p)*plane+base*64+((c^(((threadIdx.x/4)>>1)&3))<<4))=make_uint4(words[p][0],words[p][1],words[p][2],words[p][3]);
  int prefix=__popc(mask);
  int x=__shfl_up_sync(0xffffffff,prefix,1);if(c>=1)prefix+=x;
  x=__shfl_up_sync(0xffffffff,prefix,2);if(c>=2)prefix+=x;
  int total=__shfl_sync(0xffffffff,prefix,(threadIdx.x&31)/4*4+3);
  if(c==0)out.counts[base]=uint8_t(total);
  if(t<rows&&total<=ListCapacity){
    int dst=prefix-__popc(mask);
    while(mask){int j=__ffs(mask)-1;mask&=mask-1;
      unsigned label=labels[int(carrier.q[size_t(t)*k+kb*128+c*32+j])+128];
      out.entries[base*ListCapacity+dst++]=uint16_t((label&0xff00)|(c*32+j));
    }
  }
}
cudaError_t encode(bitserial::Carrier c,int k,int r,Digits d,cudaStream_t st){
  size_t need=digit_bytes(k,r),counts=count_bytes(k,r),entries=entry_words(k,r);
  if(!need||r>65535||!c.q||!c.scales||!c.status||!d.data||d.capacity<need||uintptr_t(d.data)%16||
     !d.counts||d.count_capacity<counts||!d.entries||d.entry_capacity<entries||uintptr_t(d.entries)%2)return cudaErrorInvalidValue;
  pack<<<dim3(k/128,(r+63)/64),256,0,st>>>(c,d,k,r);return cudaGetLastError();
}
__device__ __forceinline__ unsigned expand(unsigned x){x&=255;x=(x|(x<<12))&0x000f000f;x=(x|(x<<6))&0x03030303;x=(x|(x<<3))&0x11111111;return 0x22222222u|(x<<3);}
template<int FW,int WG,int TW>struct G {static constexpr int NG=FW*WG,SW=NG*512,SS=NG*64,AQ=TW*64,STAGE=SW+SS+2*AQ,STAGES=4,SMEM=STAGE*STAGES+64;};
template<int FW,int WG,int TW>
__global__ void __launch_bounds__((FW+1)*32) consume(const uint8_t* D,const uint8_t* W,const uint8_t* S,bitserial::Carrier carrier,
  int N,int K,int R,float* out,int ld,float alpha,float beta,WeightLayout layout,Digits sparse){
  using Geo=G<FW,WG,TW>;constexpr int NT=TW/8,MT=WG*2;
  extern __shared__ __align__(128) uint8_t sm[];
  auto* full=(uint64_t*)(sm+Geo::STAGE*Geo::STAGES);auto* empty=full+Geo::STAGES;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,groups=N/32,g0=blockIdx.y*Geo::NG,t0=blockIdx.x*TW;
  int all=K/128,kb0=all*blockIdx.z/gridDim.z,nkb=all*(blockIdx.z+1)/gridDim.z-kb0;
  size_t plane=size_t((R+63)/64)*64*K/2;
  if(tid==0){for(int s=0;s<Geo::STAGES;s++){mbar_init(full+s,1);mbar_init(empty+s,FW);}asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
  __syncthreads();
  if(warp==FW){if(lane==0){
    int ng=min(Geo::NG,groups-g0),pre=min(nkb,Geo::STAGES);unsigned tx=ng*576+2*Geo::AQ;
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
      for(int p=0;p<2;p++)bulk_g2s(s+Geo::SW+Geo::SS+p*Geo::AQ,D+size_t(p)*plane+rb*64,Geo::AQ,full+slot);
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
      int row=n*8+(lane&7),col=lane>>3;unsigned b[2][4];
#pragma unroll
      for(int p=0;p<2;p++)ldsm4(b[p][0],b[p][1],b[p][2],b[p][3],s+Geo::SW+Geo::SS+p*Geo::AQ+row*64+((col^((row>>1)&3))<<4));
      float sx[2];bool bad[2];
#pragma unroll
      for(int j=0;j<2;j++){int t=t0+n*8+tq*2+j;size_t pos=size_t(t)*all+kb0+kb;sx[j]=t<R?activation_scale(carrier.scales+pos):0;bad[j]=t<R?(*(volatile bitserial::InputStatus*)(carrier.status+pos)!=bitserial::InputStatus::Valid):false;}
#pragma unroll
      for(int m=0;m<MT;m++){
        float d[2][4];
#pragma unroll
        for(int p=0;p<2;p++){mma_fp4(d[p],a[m][0],b[p][0],b[p][1],0x38383838u,zero);mma_fp4(d[p],a[m][1],b[p][2],b[p][3],0x38383838u,d[p]);}
#pragma unroll
        for(int i=0;i<4;i++){
          int t=t0+n*8+tq*2+(i&1),wrow=16*(m%2)+gid+(i>>1)*8;
          unsigned bits[4];
#pragma unroll
          for(int strip=0;strip<4;strip++)bits[strip]=*(const uint32_t*)(s+strip*Geo::NG*128+(gw+m/2)*128+wrow*4);
          size_t packet=(size_t(t/64)*all+kb0+kb)*64+t%64;
          unsigned count=t<R?sparse.counts[packet]:0;int correction=0,dot=0;
          if(count<=ListCapacity){
            for(int j=0;j<count;j++){
              unsigned e=sparse.entries[packet*ListCapacity+j];int pos=e&127,value=int(int8_t(e>>8));
              correction+=((bits[pos>>5]>>(pos&31))&1)?value:-value;
            }
            dot=__float2int_rn(2.f*d[0][i])+4*__float2int_rn(2.f*d[1][i])+16*correction;
          }else{
#pragma unroll 4
            for(int pos=0;pos<128;pos++){
              int value=int(carrier.q[(size_t(t)*all+kb0+kb)*128+pos]);
              dot+=((bits[pos>>5]>>(pos&31))&1)?value:-value;
            }
          }
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
template<int FW,int WG,int TW>cudaError_t launch(const Plan& p,const uint8_t* d,const uint8_t* w,const uint8_t* s,bitserial::Carrier c,float* y,int ld,float a,float b,cudaStream_t st,Digits sparse){
  using Geo=G<FW,WG,TW>;auto f=consume<FW,WG,TW>;
  pdl(f,dim3((p.rows+TW-1)/TW,(p.n/32+Geo::NG-1)/Geo::NG,p.splits),dim3((FW+1)*32),Geo::SMEM,st,d,w,s,c,p.n,p.k,p.rows,y,ld,a,b,p.layout,sparse);return cudaGetLastError();
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
  auto e=sparse_fp4::validate(p);if(e!=cudaSuccess)return e;size_t need=sparse_fp4::partial_floats(p);
  if(!w||!s||!c.q||!c.scales||!c.status||!d.data||d.capacity<digit_bytes(p.k,p.rows)||!y||ld<p.n||!std::isfinite(a)||!std::isfinite(b)||
    (!d.counts||d.count_capacity<count_bytes(p.k,p.rows)||!d.entries||d.entry_capacity<entry_words(p.k,p.rows)||uintptr_t(d.entries)%2)||
    ((uintptr_t(w)|uintptr_t(s)|uintptr_t(d.data))&15)||((uintptr_t(y)|uintptr_t(c.scales)|uintptr_t(c.status))&3)||
    (need&&(!partial||capacity<need||partial==y)))return cudaErrorInvalidValue;
  float* out=need?partial:y;int stride=need?p.n:ld;
  switch(p.schedule){
    case Schedule::F4G2T8:e=launch<4,2,8>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st,d);break;
    case Schedule::F4G2T16:e=launch<4,2,16>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st,d);break;
    case Schedule::F4G2T32:e=launch<4,2,32>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st,d);break;
    case Schedule::F8G1T32:e=launch<8,1,32>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st,d);break;
    case Schedule::F8G2T32:e=launch<8,2,32>(p,d.data,(const uint8_t*)w,(const uint8_t*)s,c,out,stride,a,b,st,d);break;
  }
  if(e!=cudaSuccess||!need)return e;pdl(fold,dim3((size_t(p.rows)*p.n+255)/256),dim3(256),0,st,(const float*)partial,y,p.n,p.rows,ld,p.splits,a,b);return cudaGetLastError();
}
}
