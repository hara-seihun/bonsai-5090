#include "native.cuh"
#include "../../src/kernels.cuh"
#include "../../src/fp4_ops.cuh"
#include <cuda_fp8.h>
#include <cmath>
namespace binary_native_sf {
using namespace h5;
__device__ __forceinline__ unsigned expand(unsigned x) {
  x &= 255; x = (x | (x << 12)) & 0x000f000fu;
  x = (x | (x << 6)) & 0x03030303u;
  x = (x | (x << 3)) & 0x11111111u;
  return 0x22222222u | (x << 3);
}
__device__ float sf_value(uint8_t b) { return __half2float(__half(__nv_cvt_fp8_to_halfraw(b,__NV_E4M3))); }
__global__ void prepare_kernel(const uint16_t* s,uint8_t* sf,float* g,ScaleStatus* status,int n,int blocks,int wp) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
  int packed=(row/32)*32+(row%8)*4+((row%32)/8);
  float maximum=0;bool valid=true;
  for(int kb=0;kb<blocks;kb++){
    float v=__half2float(*reinterpret_cast<const __half*>(s+size_t(kb)*n+packed));
    valid=valid&&isfinite(v);maximum=fmaxf(maximum,fabsf(v));
  }
  status[row]=valid?ScaleStatus::Valid:ScaleStatus::NonFinite;
  if(!valid){
    g[row]=__int_as_float(0x7fc00000);
    for(int kb=0;kb<blocks;kb++)for(int z=0;z<wp;z++)sf[(size_t(kb)*(n/32)+row/32)*32*wp+32*z+packed%32]=0;
    return;
  }
  float factor=maximum==0?1.f:exp2f(ceilf(log2f(maximum/256.f)));g[row]=factor;
  for(int kb=0;kb<blocks;kb++) {
    float v=__half2float(*reinterpret_cast<const __half*>(s+size_t(kb)*n+packed))/factor;
    uint8_t q=__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3);
    size_t chunk=size_t(kb)*(n/32)+row/32;
    sf[chunk*32*wp+packed%32]=q;
    if(wp==2)sf[chunk*64+32+packed%32]=__nv_cvt_float_to_fp8(v-sf_value(q),__NV_SATFINITE,__NV_E4M3);
  }
}
size_t scale_bytes(int n,int k,int wp) { return n>0&&k>0&&n%32==0&&k%128==0&&(wp==1||wp==2)?size_t(n)*(k/128)*wp:0; }
cudaError_t prepare(const uint16_t* s,uint8_t* sf,float* g,ScaleStatus* status,int n,int k,int wp,cudaStream_t st) {
  if(!s||!sf||!g||!status||!scale_bytes(n,k,wp)||((uintptr_t(s)|uintptr_t(sf))&15)||((uintptr_t(g)|uintptr_t(status))&3))return cudaErrorInvalidValue;
  prepare_kernel<<<(n+255)/256,256,0,st>>>(s,sf,g,status,n,k/128,wp);return cudaGetLastError();
}
template<int FW,int WG,int TW,int WP> struct Geometry {
  static constexpr int NG=FW*WG,SW=NG*512,SS=NG*32*WP,AQ=TW*64,AS=TW*8;
  static constexpr int STAGE=SW+SS+2*(AQ+AS),STAGES=8,SMEM=STAGE*STAGES+STAGES*16;
};
template<int FW,int WG,int TW,int WP>
__global__ void __launch_bounds__((FW+1)*32) joint(Plane p,Plane e,const uint8_t* W,const uint8_t* S,const float* RS,int N,int K,int R,float* out,int ld,bool add) {
  using G=Geometry<FW,WG,TW,WP>;constexpr int NT=TW/8,MT=WG*2;
  extern __shared__ __align__(128) uint8_t sm[];
  auto* full=reinterpret_cast<uint64_t*>(sm+G::STAGE*G::STAGES);auto* empty=full+G::STAGES;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,ngall=N/32,g0=blockIdx.y*G::NG,t0=blockIdx.x*TW;
  int all=K/128,kb0=all*blockIdx.z/gridDim.z,nkb=all*(blockIdx.z+1)/gridDim.z-kb0;
  if(tid==0){for(int s=0;s<G::STAGES;s++){mbar_init(full+s,1);mbar_init(empty+s,FW);}asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}__syncthreads();
  if(warp==FW){if(lane==0){
    int ng=min(G::NG,ngall-g0),npre=min(nkb,G::STAGES);unsigned tx=ng*(512+32*WP)+2*(G::AQ+G::AS);uint64_t policy=weight_tma_policy();
    for(int kb=0;kb<npre;kb++){auto* s=sm+kb*G::STAGE;size_t c=size_t(kb0+kb)*ngall+g0;mbar_expect(full+kb,tx);bulk_weight_g2s(s,W+c*512,ng*512,full+kb,policy);bulk_g2s(s+G::SW,S+c*32*WP,ng*32*WP,full+kb);}
    pdl_wait();
    for(int kb=0;kb<nkb;kb++){int slot=kb%G::STAGES;auto* s=sm+slot*G::STAGE;size_t c=size_t(kb0+kb)*ngall+g0;
      if(kb>=npre){mbar_wait(empty+slot,((kb/G::STAGES)&1)^1);mbar_expect(full+slot,tx);bulk_weight_g2s(s,W+c*512,ng*512,full+slot,policy);bulk_g2s(s+G::SW,S+c*32*WP,ng*32*WP,full+slot);}
      size_t rb=(size_t(t0/64)*all+kb0+kb)*64+t0%64;auto* a=s+G::SW+G::SS;
      bulk_g2s(a,p.q+rb*64,G::AQ,full+slot);bulk_g2s(a+G::AQ,p.sf+rb*8,G::AS,full+slot);
      bulk_g2s(a+G::AQ+G::AS,e.q+rb*64,G::AQ,full+slot);bulk_g2s(a+2*G::AQ+G::AS,e.sf+rb*8,G::AS,full+slot);
    }
  }return;}
  int gid=lane>>2,tq=lane&3,gw=warp*WG;
  float acc[2][MT][NT][4]={};
  pdl_wait();
  for(int kb=0;kb<nkb;kb++){
    int slot=kb%G::STAGES;mbar_wait(full+slot,(kb/G::STAGES)&1);auto* s=sm+slot*G::STAGE;
    unsigned a[MT][2][4],sc[WP][MT],flip[WP][MT][2];
#pragma unroll
    for(int q=0;q<WG;q++){
      uint4 packed=*(const uint4*)(s+(gw+q)*512+lane*16);unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
#pragma unroll
      for(int m=0;m<2;m++){
#pragma unroll
        for(int h=0;h<2;h++){unsigned word=words[m*2+h];unsigned* dst=a[q*2+m][h];dst[0]=expand(word);dst[2]=expand(word>>8);dst[1]=expand(word>>16);dst[3]=expand(word>>24);}
#pragma unroll
        for(int z=0;z<WP;z++){
          unsigned sb=*(const unsigned*)(s+G::SW+(gw+q)*32*WP+z*32+gid*4);
          sc[z][q*2+m]=((sb>>(8*(2*m+(tq&1))))&127)*0x01010101u;
          flip[z][q*2+m][0]=((sb>>(8*(2*m)))&128)?0x88888888u:0;
          flip[z][q*2+m][1]=((sb>>(8*(2*m+1)))&128)?0x88888888u:0;
        }
      }
    }
#pragma unroll
    for(int n=0;n<NT;n++){
      int row=n*8+(lane&7),c=lane>>3,off=row*64+((c^((row>>1)&3))<<4);
      const auto* ap=s+G::SW+G::SS;const auto* ae=ap+G::AQ+G::AS;unsigned b[2][4];
      ldsm4(b[0][0],b[0][1],b[0][2],b[0][3],ap+off);ldsm4(b[1][0],b[1][1],b[1][2],b[1][3],ae+off);
      uint2 sb[2]={*(const uint2*)(ap+G::AQ+(n*8+gid)*8),*(const uint2*)(ae+G::AQ+(n*8+gid)*8)};
#pragma unroll
      for(int m=0;m<MT;m++){
#pragma unroll
        for(int z=0;z<WP;z++){
#pragma unroll
          for(int h=0;h<2;h++){
            unsigned av[4];
#pragma unroll
            for(int i=0;i<4;i++)av[i]=a[m][h][i]^flip[z][m][i&1];
#pragma unroll
            for(int plane=0;plane<2;plane++)mma_fp4(acc[plane][m][n],av,b[plane][h*2],b[plane][h*2+1],h?sb[plane].y:sb[plane].x,acc[plane][m][n],sc[z][m]);
          }
        }
      }
    }
    ring_release(empty+slot,lane==0);
  }
  pdl_wait();pdl_launch();if(g0+gw>=ngall)return;out+=size_t(blockIdx.z)*R*ld;
#pragma unroll
  for(int n=0;n<NT;n++){
#pragma unroll
    for(int j=0;j<2;j++){
      int t=t0+n*8+tq*2+j;if(t>=R)continue;float gp=activation_scale(p.g+t),ge=activation_scale(e.g+t);
#pragma unroll
      for(int m=0;m<MT;m++){
        if(g0+gw+m/2>=ngall)continue;
#pragma unroll
        for(int r=0;r<2;r++){
          int col=(g0+gw)*32+gid+m*16+r*8;size_t at=size_t(t)*ld+col;
          float v=__fmaf_rn(acc[1][m][n][r*2+j],ge,__fmul_rn(acc[0][m][n][r*2+j],gp))*RS[col];out[at]=add?__fadd_rn(out[at],v):v;
        }
      }
    }
  }
}
__global__ void fold(const float* p,float* y,int N,int R,int ld,int splits,bool add){pdl_wait();pdl_launch();size_t plane=size_t(R)*N,i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=plane)return;float v=p[i];for(int s=1;s<splits;s++)v=__fadd_rn(v,p[size_t(s)*plane+i]);size_t at=(i/N)*ld+i%N;y[at]=add?__fadd_rn(y[at],v):v;}
size_t partial_floats(const Plan& p){return p.splits>1&&p.n>0&&p.r>0?size_t(p.splits)*p.n*p.r:0;}
template<int FW,int WG,int TW,int WP>
cudaError_t launch(const Plan& p,Plane a,Plane e,const uint8_t* w,const uint8_t* s,const float* rs,float* y,int ld,cudaStream_t st,bool add){using G=Geometry<FW,WG,TW,WP>;auto f=joint<FW,WG,TW,WP>;pdl(f,dim3((p.r+TW-1)/TW,(p.n/32+G::NG-1)/G::NG,p.splits),dim3((FW+1)*32),G::SMEM,st,a,e,w,s,rs,p.n,p.k,p.r,y,ld,add);return cudaGetLastError();}
template<int FW,int WG,int TW,int WP>
cudaError_t configure_kernel(){return cudaFuncSetAttribute(joint<FW,WG,TW,WP>,cudaFuncAttributeMaxDynamicSharedMemorySize,Geometry<FW,WG,TW,WP>::SMEM);}
template<int WP>
cudaError_t configure_weight(Schedule schedule){
#define CASE(ID,FW,WG,TW) case Schedule::ID:return configure_kernel<FW,WG,TW,WP>()
  switch(schedule){CASE(F2G2T8,2,2,8);CASE(F4G2T8,4,2,8);CASE(F8G1T8,8,1,8);CASE(F4G2T16,4,2,16);CASE(F8G1T32,8,1,32);CASE(F4G2T32,4,2,32);CASE(F8G2T16,8,2,16);default:return cudaErrorInvalidValue;}
#undef CASE
}
cudaError_t configure(Schedule schedule,int wp){if(wp==1)return configure_weight<1>(schedule);if(wp==2)return configure_weight<2>(schedule);return cudaErrorInvalidValue;}
template<int WP>
cudaError_t dispatch(const Plan& p,Plane a,Plane e,const uint8_t* w,const uint8_t* s,const float* rs,float* y,int ld,cudaStream_t st,bool add){
#define CASE(ID,FW,WG,TW) case Schedule::ID:return launch<FW,WG,TW,WP>(p,a,e,w,s,rs,y,ld,st,add)
  switch(p.schedule){CASE(F2G2T8,2,2,8);CASE(F4G2T8,4,2,8);CASE(F8G1T8,8,1,8);CASE(F4G2T16,4,2,16);CASE(F8G1T32,8,1,32);CASE(F4G2T32,4,2,32);CASE(F8G2T16,8,2,16);default:return cudaErrorInvalidValue;}
#undef CASE
}
cudaError_t project(const Plan& p,Plane a,Plane e,const void* w,const void* s,const float* rs,float* y,int ld,float* partials,size_t capacity,cudaStream_t st,bool add){
  if(!a.q||!a.sf||!a.g||!e.q||!e.sf||!e.g||!w||!s||!rs||!y||p.r<1||p.r>32||!scale_bytes(p.n,p.k,p.weight_planes)||ld<p.n||p.splits<1||p.splits>p.k/128)return cudaErrorInvalidValue;
  if((uintptr_t(a.q)|uintptr_t(a.sf)|uintptr_t(e.q)|uintptr_t(e.sf)|uintptr_t(w)|uintptr_t(s))&15u)return cudaErrorInvalidValue;
  if((uintptr_t(a.g)|uintptr_t(e.g)|uintptr_t(rs)|uintptr_t(y))&3u)return cudaErrorInvalidValue;
  size_t need=partial_floats(p);if(need&&(!partials||capacity<need||uintptr_t(partials)%4))return cudaErrorInvalidValue;
  if(need){size_t bytes=need*4,output_bytes=size_t(p.r)*ld*4;uintptr_t pp=uintptr_t(partials),py=uintptr_t(y);if(pp<=py?py-pp<bytes:pp-py<output_bytes)return cudaErrorInvalidValue;}
  float* target=need?partials:y;int stride=need?p.n:ld;cudaError_t error=p.weight_planes==1?dispatch<1>(p,a,e,(const uint8_t*)w,(const uint8_t*)s,rs,target,stride,st,need?false:add):dispatch<2>(p,a,e,(const uint8_t*)w,(const uint8_t*)s,rs,target,stride,st,need?false:add);
  if(error!=cudaSuccess||!need)return error;pdl(fold,dim3((size_t(p.r)*p.n+255)/256),dim3(256),0,st,(const float*)partials,y,p.n,p.r,ld,p.splits,add);return cudaGetLastError();
}
}
