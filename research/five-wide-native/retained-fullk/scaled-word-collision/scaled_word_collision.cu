#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#define CK(x) do{auto cuda_status=(x);if(cuda_status!=cudaSuccess){fprintf(stderr,"cuda:%s\n",cudaGetErrorString(cuda_status));exit(2);}}while(0)
__device__ __forceinline__ void mm(float*d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned sa,unsigned sb){const uint16_t z=0;asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb),"h"(z));}
struct Witness{unsigned carrier[2];float gate[2],up[2],y[2],qscale[2];int q[2];};
__global__ void probe(Witness*out){int lane=threadIdx.x&31,g=lane>>2,t=lane&3;
 #pragma unroll
 for(int variant=0;variant<2;variant++){float c[4]={},r[4]={};unsigned cu0=(g==0&&t>=2)?0x44444444:0,cg1=(g==0&&t==0&&variant)?2:0;unsigned b0=0x77777777,b1=(t==0||t==2)?1:0;mm(c,cu0,0,cg1,0,b0,b1,0x78387838,0x01017e7e);
 unsigned rg=(g==0&&t==2&&variant)?2:0,ru=(g==0&&t<2)?0x22222222:0,rb0=t<2?0x77777777:t==2?1:0;mm(r,rg,ru,0,0,rb0,0,0x38383838,0x0000017e);
 if(lane==0){float y=(r[0]/(1.f+__expf(-r[0])))*r[2];out->carrier[variant]=__float_as_uint(c[0]);out->gate[variant]=r[0];out->up[variant]=r[2];out->y[variant]=y;out->qscale[variant]=fabsf(y)/127.f;out->q[variant]=y==0.f?0:__float2int_rn(y*(127.f/fabsf(y)));}}}
int main(){Witness*d;CK(cudaMalloc(&d,sizeof(Witness)));probe<<<1,32>>>(d);CK(cudaGetLastError());CK(cudaDeviceSynchronize());Witness h;CK(cudaMemcpy(&h,d,sizeof h,cudaMemcpyDeviceToHost));FILE*f=fopen("scaled-word-collision.bin","wb");fwrite(&h,sizeof h,1,f);fclose(f);bool collision=h.carrier[0]==h.carrier[1]&&h.q[0]!=h.q[1]&&h.qscale[0]!=h.qscale[1];printf("{\"state\":\"%s\",\"carrier_bits\":[%u,%u],\"gate\":[%.9g,%.9g],\"up\":[%.9g,%.9g],\"y\":[%.9g,%.9g],\"q8_scale\":[%.9g,%.9g],\"q8\":[%d,%d],\"activation_scale_codes\":[126,1],\"weight_scale\":1,\"scope\":\"single scalar FP32 g+512u carrier across two independently scaled16K blocks; rest of K5120 and127 block features zero\"}\n",collision?"native_target_collision":"unexpected",h.carrier[0],h.carrier[1],h.gate[0],h.gate[1],h.up[0],h.up[1],h.y[0],h.y[1],h.qscale[0],h.qscale[1],h.q[0],h.q[1]);return collision?0:1;}
