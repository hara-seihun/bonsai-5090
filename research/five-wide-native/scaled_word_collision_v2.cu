#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#define CK(x) do{auto cuda_status=(x);if(cuda_status!=cudaSuccess){fprintf(stderr,"cuda:%s\n",cudaGetErrorString(cuda_status));exit(2);}}while(0)
__device__ __forceinline__ void mm(float*d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned sa,unsigned sb){const uint16_t z=0;asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb),"h"(z));}
struct Witness{unsigned carrier[2];float gate[2],up[2],y[2],qscale[2];int q[2];};
struct Lane{float c[4],r[4];};
__global__ void probe(Lane*out,int variant){int lane=threadIdx.x&31,g=lane>>2,t=lane&3;float c[4]={},r[4]={};unsigned cu0=(g==0&&t>=2)?0x44444444:0,cg1=(g==0&&t==0&&variant)?2:0;unsigned b0=0x77777777,b1=(t==0||t==2)?1:0;mm(c,cu0,0,cg1,0,b0,b1,0x78387838,0x01017e7e);unsigned rg=(g==0&&t==2&&variant)?2:0,ru=(g==0&&t<2)?0x22222222:0,rb0=t<2?0x77777777:t==2?1:0;mm(r,rg,ru,0,0,rb0,0,0x38383838,0x0000017e);
 #pragma unroll
 for(int i=0;i<4;i++){out[lane].c[i]=c[i];out[lane].r[i]=r[i];}}
int main(){Lane*d;CK(cudaMalloc(&d,sizeof(Lane)*64));probe<<<1,32>>>(d,0);probe<<<1,32>>>(d+32,1);CK(cudaGetLastError());CK(cudaDeviceSynchronize());Lane h[64];CK(cudaMemcpy(h,d,sizeof h,cudaMemcpyDeviceToHost));FILE*f=fopen("scaled-word-all-lanes.bin","wb");fwrite(h,sizeof h,1,f);fclose(f);unsigned bits[2];for(int i=0;i<2;i++)memcpy(bits+i,&h[i*32].c[0],4);bool collision=bits[0]==bits[1]&&h[0].r[0]==0.f&&h[32].r[0]>0.f;printf("{\"state\":\"%s\",\"carrier_bits\":[%u,%u],\"carrier_values\":[%.9g,%.9g],\"gate\":[%.9g,%.9g],\"up\":[%.9g,%.9g],\"activation_scale_codes\":[126,1],\"weight_scale\":1,\"padding\":\"rest of K5120 and127 features zero\"}\n",collision?"native_target_collision":"unexpected",bits[0],bits[1],h[0].c[0],h[32].c[0],h[0].r[0],h[32].r[0],h[0].r[2],h[32].r[2]);return collision?0:1;}
