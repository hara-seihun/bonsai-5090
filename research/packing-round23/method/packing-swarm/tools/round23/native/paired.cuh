#pragma once
namespace paired23 {
namespace rn = residual11::native;
struct Geometry {
 static constexpr unsigned NG=8,TOKW=32,CW=512,CS=64;
 static constexpr unsigned SW=NG*CW,SWS=NG*CS,SA=TOKW*64,SAS=TOKW*8;
 static constexpr unsigned PASS=SA+SAS,STAGE=SW+SWS+2*PASS,STAGES=8,RING=STAGES*STAGE+16*STAGES,GLOBALS=2*TOKW*4,SMEM=RING+GLOBALS;
};
__device__ __forceinline__ float ordered_fold(float d,float scale,float acc) {
 float out;
 asm volatile("fma.rn.f32 %0,%1,%2,%3;" : "=f"(out) : "f"(d),"f"(scale),"f"(acc));
 return out;
}
struct Args {
 const uint8_t *q[2],*s[2];const float *g[2];
 const uint8_t *w,*ws;unsigned rows,K,F,stride;
 float *out,*primary;unsigned *owners[2];
};

__global__ void __launch_bounds__(160) candidate(Args p) {
 using G=Geometry;
 constexpr int FW=4,WG=2,MT=2*WG,NT=4;
 extern __shared__ __align__(128) uint8_t smem[];
 uint64_t *full=(uint64_t*)(smem+G::STAGES*G::STAGE),*empty=full+G::STAGES;
 const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 const unsigned blocks=p.K/128,all=p.F/32,g0=blockIdx.y*8,t0=blockIdx.x*32;
 if(tid==0){for(unsigned s=0;s<8;s++){rn::mbar_init(&full[s],1);rn::mbar_init(&empty[s],FW);}asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");}
 float*globals=(float*)(smem+G::RING);
 if(tid<32){unsigned t=t0+tid;globals[tid]=t<p.rows?p.g[0][t]:0.f;globals[32+tid]=t<p.rows?p.g[1][t]:0.f;}
 __syncthreads();
 if(warp==FW){
  if(lane==0){
   const unsigned ng=min(8u,all-g0),tile=t0/64,toff=t0%64;
   const unsigned tx=ng*(G::CW+G::CS)+2*G::PASS,npre=min(blocks,8u);
   if(rn::l2pf_enabled())for(unsigned k=8;k<blocks;k++)asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p.w+(size_t(k)*all+g0)*G::CW),"r"(ng*G::CW) : "memory");
   for(unsigned k=0;k<npre;k++){
    uint8_t*st=smem+k*G::STAGE;rn::mbar_expect(&full[k],tx);
    rn::bulk_g2s(st,p.w+(size_t(k)*all+g0)*G::CW,ng*G::CW,&full[k]);
    rn::bulk_g2s(st+G::SW,p.ws+(size_t(k)*all+g0)*G::CS,ng*G::CS,&full[k]);
   }
   rn::pdl_wait();
   for(unsigned k=0;k<blocks;k++){
    const unsigned s=k%8;uint8_t*st=smem+s*G::STAGE;
    if(k>=npre){rn::mbar_wait(&empty[s],((k/8)&1)^1);rn::mbar_expect(&full[s],tx);
     rn::bulk_g2s(st,p.w+(size_t(k)*all+g0)*G::CW,ng*G::CW,&full[s]);
     rn::bulk_g2s(st+G::SW,p.ws+(size_t(k)*all+g0)*G::CS,ng*G::CS,&full[s]);}
    const size_t rb=(size_t(tile)*blocks+k)*64+toff;
    for(unsigned pass=0;pass<2;pass++){
     rn::bulk_g2s(st+G::SW+G::SWS+pass*G::PASS,p.q[pass]+rb*64,G::SA,&full[s]);
     rn::bulk_g2s(st+G::SW+G::SWS+pass*G::PASS+G::SA,p.s[pass]+rb*8,G::SAS,&full[s]);
    }
   }
  }
  return;
 }
 const unsigned gid=lane>>2,tq=lane&3,gw=warp*WG;
 float acc[2][MT][NT][4];
#pragma unroll
 for(unsigned pass=0;pass<2;pass++)
#pragma unroll
  for(unsigned m=0;m<MT;m++)
#pragma unroll
   for(unsigned n=0;n<NT;n++)
#pragma unroll
    for(unsigned i=0;i<4;i++)acc[pass][m][n][i]=0.f;
 const float zero[4]={0.f,0.f,0.f,0.f};
 for(unsigned k=0;k<blocks;k++){
  const unsigned s=k%8;rn::mbar_wait(&full[s],(k/8)&1);const uint8_t*st=smem+s*G::STAGE;
#pragma unroll
  for(unsigned q=0;q<WG;q++){
   unsigned a[2][2][4];float wsc[2][2];
   const uint4 packed=*(const uint4*)(st+(gw+q)*G::CW+lane*16);
   const unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
#pragma unroll
   for(unsigned mm=0;mm<2;mm++)
#pragma unroll
    for(unsigned h=0;h<2;h++){
     unsigned pair=words[mm*2+h];unsigned*o=a[mm][h];
     o[0]=rn::binary_e2m1(pair);o[2]=rn::binary_e2m1(pair>>8);
     o[1]=rn::binary_e2m1(pair>>16);o[3]=rn::binary_e2m1(pair>>24);
    }
   const uint2 sc=*(const uint2*)(st+G::SW+(gw+q)*G::CS+gid*8);
   const __half2*sh=(const __half2*)&sc;const float2 s01=__half22float2(sh[0]),s23=__half22float2(sh[1]);
   wsc[0][0]=s01.x;wsc[0][1]=s01.y;wsc[1][0]=s23.x;wsc[1][1]=s23.y;
#pragma unroll
  for(unsigned pass=0;pass<2;pass++){
   const uint8_t*as=st+G::SW+G::SWS+pass*G::PASS,*ssc=as+G::SA;
#pragma unroll
   for(unsigned n=0;n<NT;n++){
    const unsigned row=n*8+(lane&7),c=lane>>3;unsigned b0,b1,b2,b3;
    rn::ldsm4(b0,b1,b2,b3,as+row*64+((c^((row>>1)&3))<<4));
    const uint2 bs=*(const uint2*)(ssc+(n*8+gid)*8);
#pragma unroll
    for(unsigned mm=0;mm<2;mm++){
     const unsigned m=q*2+mm;
     float d[4];rn::mma_fp4(d,a[mm][0],b0,b1,bs.x,zero);rn::mma_fp4(d,a[mm][1],b2,b3,bs.y,d);
     acc[pass][m][n][0]=ordered_fold(d[0],wsc[mm][0],acc[pass][m][n][0]);
     acc[pass][m][n][1]=ordered_fold(d[1],wsc[mm][0],acc[pass][m][n][1]);
     acc[pass][m][n][2]=ordered_fold(d[2],wsc[mm][1],acc[pass][m][n][2]);
     acc[pass][m][n][3]=ordered_fold(d[3],wsc[mm][1],acc[pass][m][n][3]);
    }
   }
  }
  }
  __syncwarp();if(lane==0)rn::mbar_arrive(&empty[s]);
 }
 rn::pdl_wait();rn::pdl_launch();
#pragma unroll
 for(unsigned n=0;n<NT;n++)
#pragma unroll
  for(unsigned j=0;j<2;j++){
   unsigned t=t0+n*8+tq*2+j;if(t>=p.rows)continue;
   const float gt0=globals[t-t0],gt1=globals[32+t-t0];
#pragma unroll
   for(unsigned m=0;m<MT;m++)
#pragma unroll
    for(unsigned r=0;r<2;r++){
     unsigned group=g0+gw+(m>>1);if(group>=all)continue;
     unsigned col=(g0+gw)*32+gid+m*16+r*8;size_t at=size_t(t)*p.stride+col;
     float v0=acc[0][m][n][r*2+j]*gt0;v0+=p.out[at];
     p.primary[at]=v0;p.owners[0][size_t(t)*p.F+col]=0;
     float v1=acc[1][m][n][r*2+j]*gt1;v1+=v0;
     p.out[at]=v1;p.owners[1][size_t(t)*p.F+col]=0;
    }
  }
}

__global__ void __launch_bounds__(288) strongest_fw8(Args p) {
 using G=Geometry;
 constexpr int FW=8,WG=1,MT=2*WG,NT=4;
 extern __shared__ __align__(128) uint8_t smem[];
 uint64_t *full=(uint64_t*)(smem+G::STAGES*G::STAGE),*empty=full+G::STAGES;
 const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 const unsigned blocks=p.K/128,all=p.F/32,g0=blockIdx.y*8,t0=blockIdx.x*32;
 if(tid==0){for(unsigned s=0;s<8;s++){rn::mbar_init(&full[s],1);rn::mbar_init(&empty[s],FW);}asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");}
 float*globals=(float*)(smem+G::RING);
 if(tid<32){unsigned t=t0+tid;globals[tid]=t<p.rows?p.g[0][t]:0.f;globals[32+tid]=t<p.rows?p.g[1][t]:0.f;}
 __syncthreads();
 if(warp==FW){
  if(lane==0){
   const unsigned ng=min(8u,all-g0),tile=t0/64,toff=t0%64;
   const unsigned tx=ng*(G::CW+G::CS)+2*G::PASS,npre=min(blocks,8u);
   if(rn::l2pf_enabled())for(unsigned k=8;k<blocks;k++)asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p.w+(size_t(k)*all+g0)*G::CW),"r"(ng*G::CW) : "memory");
   for(unsigned k=0;k<npre;k++){
    uint8_t*st=smem+k*G::STAGE;rn::mbar_expect(&full[k],tx);
    rn::bulk_g2s(st,p.w+(size_t(k)*all+g0)*G::CW,ng*G::CW,&full[k]);
    rn::bulk_g2s(st+G::SW,p.ws+(size_t(k)*all+g0)*G::CS,ng*G::CS,&full[k]);
   }
   rn::pdl_wait();
   for(unsigned k=0;k<blocks;k++){
    const unsigned s=k%8;uint8_t*st=smem+s*G::STAGE;
    if(k>=npre){rn::mbar_wait(&empty[s],((k/8)&1)^1);rn::mbar_expect(&full[s],tx);
     rn::bulk_g2s(st,p.w+(size_t(k)*all+g0)*G::CW,ng*G::CW,&full[s]);
     rn::bulk_g2s(st+G::SW,p.ws+(size_t(k)*all+g0)*G::CS,ng*G::CS,&full[s]);}
    const size_t rb=(size_t(tile)*blocks+k)*64+toff;
    for(unsigned pass=0;pass<2;pass++){
     rn::bulk_g2s(st+G::SW+G::SWS+pass*G::PASS,p.q[pass]+rb*64,G::SA,&full[s]);
     rn::bulk_g2s(st+G::SW+G::SWS+pass*G::PASS+G::SA,p.s[pass]+rb*8,G::SAS,&full[s]);
    }
   }
  }
  return;
 }
 const unsigned gid=lane>>2,tq=lane&3,gw=warp*WG;
 float acc[2][MT][NT][4];
#pragma unroll
 for(unsigned pass=0;pass<2;pass++)
#pragma unroll
  for(unsigned m=0;m<MT;m++)
#pragma unroll
   for(unsigned n=0;n<NT;n++)
#pragma unroll
    for(unsigned i=0;i<4;i++)acc[pass][m][n][i]=0.f;
 const float zero[4]={0.f,0.f,0.f,0.f};
 for(unsigned k=0;k<blocks;k++){
  const unsigned s=k%8;rn::mbar_wait(&full[s],(k/8)&1);const uint8_t*st=smem+s*G::STAGE;
#pragma unroll
  for(unsigned q=0;q<WG;q++){
   unsigned a[2][2][4];float wsc[2][2];
   const uint4 packed=*(const uint4*)(st+(gw+q)*G::CW+lane*16);
   const unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
#pragma unroll
   for(unsigned mm=0;mm<2;mm++)
#pragma unroll
    for(unsigned h=0;h<2;h++){
     unsigned pair=words[mm*2+h];unsigned*o=a[mm][h];
     o[0]=rn::binary_e2m1(pair);o[2]=rn::binary_e2m1(pair>>8);
     o[1]=rn::binary_e2m1(pair>>16);o[3]=rn::binary_e2m1(pair>>24);
    }
   const uint2 sc=*(const uint2*)(st+G::SW+(gw+q)*G::CS+gid*8);
   const __half2*sh=(const __half2*)&sc;const float2 s01=__half22float2(sh[0]),s23=__half22float2(sh[1]);
   wsc[0][0]=s01.x;wsc[0][1]=s01.y;wsc[1][0]=s23.x;wsc[1][1]=s23.y;
#pragma unroll
  for(unsigned pass=0;pass<2;pass++){
   const uint8_t*as=st+G::SW+G::SWS+pass*G::PASS,*ssc=as+G::SA;
#pragma unroll
   for(unsigned n=0;n<NT;n++){
    const unsigned row=n*8+(lane&7),c=lane>>3;unsigned b0,b1,b2,b3;
    rn::ldsm4(b0,b1,b2,b3,as+row*64+((c^((row>>1)&3))<<4));
    const uint2 bs=*(const uint2*)(ssc+(n*8+gid)*8);
#pragma unroll
    for(unsigned mm=0;mm<2;mm++){
     const unsigned m=q*2+mm;
     float d[4];rn::mma_fp4(d,a[mm][0],b0,b1,bs.x,zero);rn::mma_fp4(d,a[mm][1],b2,b3,bs.y,d);
     acc[pass][m][n][0]=ordered_fold(d[0],wsc[mm][0],acc[pass][m][n][0]);
     acc[pass][m][n][1]=ordered_fold(d[1],wsc[mm][0],acc[pass][m][n][1]);
     acc[pass][m][n][2]=ordered_fold(d[2],wsc[mm][1],acc[pass][m][n][2]);
     acc[pass][m][n][3]=ordered_fold(d[3],wsc[mm][1],acc[pass][m][n][3]);
    }
   }
  }
  }
  __syncwarp();if(lane==0)rn::mbar_arrive(&empty[s]);
 }
 rn::pdl_wait();rn::pdl_launch();
#pragma unroll
 for(unsigned n=0;n<NT;n++)
#pragma unroll
  for(unsigned j=0;j<2;j++){
   unsigned t=t0+n*8+tq*2+j;if(t>=p.rows)continue;
   const float gt0=globals[t-t0],gt1=globals[32+t-t0];
#pragma unroll
   for(unsigned m=0;m<MT;m++)
#pragma unroll
    for(unsigned r=0;r<2;r++){
     unsigned group=g0+gw+(m>>1);if(group>=all)continue;
     unsigned col=(g0+gw)*32+gid+m*16+r*8;size_t at=size_t(t)*p.stride+col;
     float v0=acc[0][m][n][r*2+j]*gt0;v0+=p.out[at];
     p.primary[at]=v0;p.owners[0][size_t(t)*p.F+col]=0;
     float v1=acc[1][m][n][r*2+j]*gt1;v1+=v0;
     p.out[at]=v1;p.owners[1][size_t(t)*p.F+col]=0;
    }
  }
}
}
