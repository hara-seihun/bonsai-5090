#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <string>
#define CK(x) do{auto cuda_status=(x);if(cuda_status!=cudaSuccess){fprintf(stderr,"cuda:%s\n",cudaGetErrorString(cuda_status));exit(2);}}while(0)
constexpr int K=5120,N=64,B16=K/16,B128=K/128,DW=K/5,OFFSET=98496;
struct Args{const uint8_t *w,*x,*as;const half *gs,*us;const float *ag;const uint64_t*wt;const uint32_t *qt,*fp;int m;};
__device__ __forceinline__ float scale(unsigned x){__nv_fp8_e4m3 a;a.__x=(__nv_fp8_storage_t)x;return float(a);}
__device__ __forceinline__ unsigned w4(Args a,int proj,int row,int k){int pos=k/5,r=k%5;const uint8_t*p=a.w+((size_t)proj*a.m+row)*DW+pos;uint64_t v=a.wt[p[0]]>>(8*r);if(r>1)v|=a.wt[p[1]]<<(8*(5-r));return unsigned(v);}
__device__ __forceinline__ unsigned u64(unsigned x){return (x&0x80808080u)|((x&0x01010101u)<<6);}
__device__ __forceinline__ unsigned x4(Args a,int token,int k){return a.qt[*(const uint16_t*)(a.x+(size_t)token*(K/2)+k/2)];}
__device__ __forceinline__ unsigned wp8(Args a,int proj,int row,int k){int pos=k/5,r=k%5;const uint8_t*p=a.w+((size_t)proj*a.m+row)*DW+pos;uint64_t v=(uint64_t(a.fp[p[0]])>>(4*r))|(uint64_t(a.fp[p[1]])<<(4*(5-r)));if(r>2)v|=uint64_t(a.fp[p[2]])<<(4*(10-r));return unsigned(v);}
__device__ __forceinline__ void im(int*d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));}
__device__ __forceinline__ void fm(float*d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned sb){const uint16_t z=0;const unsigned sa=0x38383838;asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb),"h"(z));}
__device__ __forceinline__ void fragments(Args a,int base,int nt,int kb,unsigned&A0,unsigned&A1,unsigned&A2,unsigned&A3,unsigned&B){int g=(threadIdx.x&31)>>2,t=threadIdx.x&3,k=16*kb+4*t;A0=w4(a,0,base+g,k);A1=w4(a,0,base+g+8,k);A2=u64(w4(a,1,base+g,k));A3=u64(w4(a,1,base+g+8,k));B=x4(a,nt+g,k);}
__device__ __forceinline__ void simblock(int*d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b){int lane=threadIdx.x&31,g=lane>>2,t=lane&3;d[0]=d[1]=d[2]=d[3]=0;
 #pragma unroll
 for(int h=0;h<4;h++){unsigned aa[4]={__shfl_sync(0xffffffff,a0,g*4+h),__shfl_sync(0xffffffff,a1,g*4+h),__shfl_sync(0xffffffff,a2,g*4+h),__shfl_sync(0xffffffff,a3,g*4+h)};
 #pragma unroll
 for(int c=0;c<2;c++){unsigned x=__shfl_sync(0xffffffff,b,(2*t+c)*4+h),xx=(x<<3)&0xf8f8f8f8u;d[c]=__dp4a(int(aa[0]),int(x),d[c]);d[c]=__dp4a(int(aa[2]),int(xx),d[c]);d[2+c]=__dp4a(int(aa[1]),int(x),d[2+c]);d[2+c]=__dp4a(int(aa[3]),int(xx),d[2+c]);}}
}
template<bool SIMD> __global__ void producer(Args a,int*trace){int warp=threadIdx.x/32,base=(blockIdx.x/(N/8))*128+warp*16,nt=(blockIdx.x%(N/8))*8,g=(threadIdx.x&31)>>2,t=threadIdx.x&3;size_t tile=((size_t)(base/16)*(N/8)+nt/8);
 for(int kb=0;kb<B16;kb++){unsigned a0,a1,a2,a3,b;fragments(a,base,nt,kb,a0,a1,a2,a3,b);int d[4]={};if(SIMD)simblock(d,a0,a1,a2,a3,b);else im(d,a0,a1,a2,a3,b,(b<<3)&0xf8f8f8f8u);
 #pragma unroll
 for(int j=0;j<4;j++)trace[(tile*B16+kb)*128+(g+j/2*8)*8+2*t+j%2]=d[j];}}
__device__ __forceinline__ float term(int word,float s,bool up){unsigned c=word+OFFSET;int z=up?int(c>>9)-192:int(c&511)-192;return (z*.5f)*s;}
__device__ __forceinline__ float reduce8(const float*v,int mode){if(mode==0){float r=0;for(int j=0;j<8;j++)r+=v[j];return r;}if(mode==1){double p=double(v[0])+v[1]+v[2]+v[3];float r=float(p);return float(double(r)+v[4]+v[5]+v[6]+v[7]);}float r=(v[0]+v[1])+(v[2]+v[3]);return r+((v[4]+v[5])+(v[6]+v[7]));}
__device__ __forceinline__ float silu(float g,float u,float ag){g*=ag;u*=ag;return (g/(1.f+__expf(-g)))*u;}
__global__ void finish(Args a,const int*trace,float*out,int mode){int base=(blockIdx.x/(N/8))*128+(threadIdx.x/32)*16,nt=(blockIdx.x%(N/8))*8,g=(threadIdx.x&31)>>2,t=threadIdx.x&3;size_t tile=((size_t)(base/16)*(N/8)+nt/8);float ga[4]={},ua[4]={};
 for(int kb=0;kb<B128;kb++){float v[4][8],u[4][8];
 #pragma unroll
 for(int j=0;j<8;j++){
 #pragma unroll
 for(int c=0;c<4;c++){int token=nt+2*t+c%2;int w=trace[(tile*B16+8*kb+j)*128+(g+c/2*8)*8+2*t+c%2];float s=scale(a.as[token*B16+8*kb+j]);v[c][j]=term(w,s,false);u[c][j]=term(w,s,true);}}
 #pragma unroll
 for(int c=0;c<4;c++){int f=base+g+c/2*8;ga[c]=fmaf(reduce8(v[c],mode),__half2float(a.gs[kb*a.m+f]),ga[c]);ua[c]=fmaf(reduce8(u[c],mode),__half2float(a.us[kb*a.m+f]),ua[c]);}}
 #pragma unroll
 for(int c=0;c<4;c++){int f=base+g+c/2*8,token=nt+2*t+c%2;out[(size_t)token*a.m+f]=silu(ga[c],ua[c],a.ag[token]);}}
__global__ void folded_fused(Args a,float*out,int mode){int base=(blockIdx.x/(N/8))*128+(threadIdx.x/32)*16,nt=(blockIdx.x%(N/8))*8,g=(threadIdx.x&31)>>2,t=threadIdx.x&3;float ga[4]={},ua[4]={};
 for(int kb=0;kb<B128;kb++){float v[4][8],u[4][8];
 #pragma unroll
 for(int j=0;j<8;j++){unsigned a0,a1,a2,a3,b;fragments(a,base,nt,8*kb+j,a0,a1,a2,a3,b);int d[4];simblock(d,a0,a1,a2,a3,b);
 #pragma unroll
 for(int c=0;c<4;c++){float s=scale(a.as[(nt+2*t+c%2)*B16+8*kb+j]);v[c][j]=term(d[c],s,false);u[c][j]=term(d[c],s,true);}}
 #pragma unroll
 for(int c=0;c<4;c++){int f=base+g+c/2*8;ga[c]=fmaf(reduce8(v[c],mode),__half2float(a.gs[kb*a.m+f]),ga[c]);ua[c]=fmaf(reduce8(u[c],mode),__half2float(a.us[kb*a.m+f]),ua[c]);}}
 #pragma unroll
 for(int c=0;c<4;c++){int f=base+g+c/2*8,token=nt+2*t+c%2;out[(size_t)token*a.m+f]=silu(ga[c],ua[c],a.ag[token]);}}
__global__ void dp4a_fused(Args a,float*out,int mode,unsigned long long*err,const int*trace){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=a.m*N)return;int token=i/a.m,f=i%a.m;float ga=0,ua=0;size_t tile=((size_t)(f/16)*(N/8)+token/8);
 for(int kb=0;kb<B128;kb++){float v[8],u[8];
 #pragma unroll
 for(int j=0;j<8;j++){int g=0,h=0;
 #pragma unroll
 for(int r=0;r<4;r++){int k=kb*128+j*16+r*4;unsigned x=x4(a,token,k);g=__dp4a(int(w4(a,0,f,k)),int(x),g);h=__dp4a(int(w4(a,1,f,k)),int(x),h);}if(err){int actual=trace[(tile*B16+kb*8+j)*128+(f%16)*8+token%8];if(actual!=g+512*h)atomicAdd(err,1ULL);}float s=scale(a.as[token*B16+8*kb+j]);v[j]=(g*.5f)*s;u[j]=(h*.5f)*s;}ga=fmaf(reduce8(v,mode),__half2float(a.gs[kb*a.m+f]),ga);ua=fmaf(reduce8(u,mode),__half2float(a.us[kb*a.m+f]),ua);}out[i]=silu(ga,ua,a.ag[token]);}
__global__ void fp4_ref(Args a,float*out,float*diag=nullptr){int warp=threadIdx.x/32,base=(blockIdx.x/(N/8))*64+warp*8,nt=(blockIdx.x%(N/8))*8,g=(threadIdx.x&31)>>2,t=threadIdx.x&3;float ga[2]={},ua[2]={};
 for(int kb=0;kb<B128;kb++){float d[4]={};
 #pragma unroll
 for(int h=0;h<2;h++){int k=kb*128+h*64+8*t;unsigned a0=wp8(a,0,base+g,k),a1=wp8(a,1,base+g,k),a2=wp8(a,0,base+g,k+32),a3=wp8(a,1,base+g,k+32);const uint8_t*x=a.x+(nt+g)*(K/2)+k/2;unsigned b0=*(const unsigned*)x,b1=*(const unsigned*)(x+16);unsigned sb=*(const unsigned*)(a.as+(nt+g)*B16+kb*8+h*4);fm(d,a0,a1,a2,a3,b0,b1,sb);if(diag){int fs[3]={407,452,481},ts[3]={6,53,63};for(int z=0;z<3;z++)if(base+g==fs[z])for(int c=0;c<2;c++)if(nt+2*t+c==ts[z]){diag[z*160+(kb*2+h)*2]=d[c];diag[z*160+(kb*2+h)*2+1]=d[c+2];}}}float wsg=__half2float(a.gs[kb*a.m+base+g]),wsu=__half2float(a.us[kb*a.m+base+g]);
 #pragma unroll
 for(int c=0;c<2;c++){ga[c]=fmaf(d[c],wsg,ga[c]);ua[c]=fmaf(d[c+2],wsu,ua[c]);}}
 #pragma unroll
 for(int c=0;c<2;c++){int tok=nt+2*t+c;out[(size_t)tok*a.m+base+g]=silu(ga[c],ua[c],a.ag[tok]);}}
__global__ void normalize(const float*y,signed char*q,float*sc,int m){int token=blockIdx.x%N,fg=blockIdx.x/N,t=threadIdx.x,i=token*m+fg*128+t;float v=y[i],mx=fabsf(v);for(int k=16;k;k>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,k));__shared__ float am[4];if(!(t&31))am[t/32]=mx;__syncthreads();float bm=fmaxf(fmaxf(am[0],am[1]),fmaxf(am[2],am[3])),iv=bm>0.f?127.f/bm:0.f;int z=__float2int_rn(v*iv);q[i]=max(-127,min(127,z));if(t==0)sc[fg*N+token]=bm/127.f;}

int main(){const char*root="C:/work/models/experiments/bonsai-quality-20261013/five-wide-native/fullk-trace-oracle/";auto load=[&](const char*name,size_t n){std::vector<uint8_t>v(n);FILE*f=fopen((std::string(root)+name).c_str(),"rb");if(!f||fread(v.data(),1,n,f)!=n)exit(4);fclose(f);return v;};auto put=[&](const std::vector<uint8_t>&v){uint8_t*p;CK(cudaMalloc(&p,v.size()));CK(cudaMemcpy(p,v.data(),v.size(),cudaMemcpyHostToDevice));return p;};Args a{};a.m=512;a.w=put(load("fullk-weights.bin",512*DW*2));a.x=put(load("fullk-raw-fp4.bin",N*K/2));a.as=put(load("fullk-as.bin",N*B16));a.gs=(half*)put(load("fullk-gs.bin",512*B128*2));a.us=(half*)put(load("fullk-us.bin",512*B128*2));a.ag=(float*)put(load("fullk-ag.bin",N*4));std::vector<uint64_t>wt(243);std::vector<uint32_t>fp(243),qt(65536);for(int i=0;i<243;i++){int z=i;for(int j=0;j<5;j++){int v=z%3-1;wt[i]|=uint64_t(unsigned(v)&255)<<(8*j);fp[i]|=uint32_t(v<0?10:v>0?2:0)<<(4*j);z/=3;}}int val[8]={0,1,2,3,4,6,8,12};for(int i=0;i<65536;i++)for(int j=0;j<4;j++){int z=(i>>(j*4))&15;qt[i]|=uint32_t(unsigned(val[z&7]*(z&8?-1:1))&255)<<(8*j);}auto pv=[&](const void*p,size_t n){uint8_t*d;CK(cudaMalloc(&d,n));CK(cudaMemcpy(d,p,n,cudaMemcpyHostToDevice));return d;};a.wt=(uint64_t*)pv(wt.data(),wt.size()*8);a.fp=(uint32_t*)pv(fp.data(),fp.size()*4);a.qt=(uint32_t*)pv(qt.data(),qt.size()*4);float *d,*y;CK(cudaMalloc(&d,480*4));CK(cudaMalloc(&y,512*N*4));fp4_ref<<<512/64*(N/8),256>>>(a,y,d);CK(cudaGetLastError());CK(cudaDeviceSynchronize());std::vector<float>h(480);CK(cudaMemcpy(h.data(),d,480*4,cudaMemcpyDeviceToHost));FILE*f=fopen("native-fp4-rounding-trace.bin","wb");fwrite(h.data(),4,h.size(),f);fclose(f);printf("{\"state\":\"observed\",\"cases\":3,\"K\":5120,\"per_case_native64_steps\":80}\n");return 0;}
