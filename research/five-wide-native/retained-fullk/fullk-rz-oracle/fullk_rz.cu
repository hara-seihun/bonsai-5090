#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <type_traits>
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
__device__ __forceinline__ int trunc24(int v){unsigned mag=v<0?unsigned(-v):unsigned(v);int sh=max(0,31-__clz(mag)-23);unsigned z=(mag>>sh)<<sh;return v<0?-int(z):int(z);}
__device__ __forceinline__ float reduce8(const float*v,int mode){if(mode==0){float r=0;for(int j=0;j<8;j++)r+=v[j];return r;}if(mode==1||mode==3){double p=double(v[0])+v[1]+v[2]+v[3];float r=mode==3?__double2float_rz(p):float(p);double z=double(r)+v[4]+v[5]+v[6]+v[7];return mode==3?__double2float_rz(z):float(z);}if(mode==4){int a=0,b=0;
 #pragma unroll
 for(int j=0;j<4;j++){a+=__float2int_rn(v[j]*1024.f);b+=__float2int_rn(v[j+4]*1024.f);}return __int2float_rz(trunc24(a)+b)*0x1p-10f;}float r=(v[0]+v[1])+(v[2]+v[3]);return r+((v[4]+v[5])+(v[6]+v[7]));}
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
__global__ void fp4_ref(Args a,float*out){int warp=threadIdx.x/32,base=(blockIdx.x/(N/8))*64+warp*8,nt=(blockIdx.x%(N/8))*8,g=(threadIdx.x&31)>>2,t=threadIdx.x&3;float ga[2]={},ua[2]={};
 for(int kb=0;kb<B128;kb++){float d[4]={};
 #pragma unroll
 for(int h=0;h<2;h++){int k=kb*128+h*64+8*t;unsigned a0=wp8(a,0,base+g,k),a1=wp8(a,1,base+g,k),a2=wp8(a,0,base+g,k+32),a3=wp8(a,1,base+g,k+32);const uint8_t*x=a.x+(nt+g)*(K/2)+k/2;unsigned b0=*(const unsigned*)x,b1=*(const unsigned*)(x+16);unsigned sb=*(const unsigned*)(a.as+(nt+g)*B16+kb*8+h*4);fm(d,a0,a1,a2,a3,b0,b1,sb);}float wsg=__half2float(a.gs[kb*a.m+base+g]),wsu=__half2float(a.us[kb*a.m+base+g]);
 #pragma unroll
 for(int c=0;c<2;c++){ga[c]=fmaf(d[c],wsg,ga[c]);ua[c]=fmaf(d[c+2],wsu,ua[c]);}}
 #pragma unroll
 for(int c=0;c<2;c++){int tok=nt+2*t+c;out[(size_t)tok*a.m+base+g]=silu(ga[c],ua[c],a.ag[tok]);}}
__global__ void normalize(const float*y,signed char*q,float*sc,int m){int token=blockIdx.x%N,fg=blockIdx.x/N,t=threadIdx.x,i=token*m+fg*128+t;float v=y[i],mx=fabsf(v);for(int k=16;k;k>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,k));__shared__ float am[4];if(!(t&31))am[t/32]=mx;__syncthreads();float bm=fmaxf(fmaxf(am[0],am[1]),fmaxf(am[2],am[3])),iv=bm>0.f?127.f/bm:0.f;int z=__float2int_rn(v*iv);q[i]=max(-127,min(127,z));if(t==0)sc[fg*N+token]=bm/127.f;}
int main(int argc,char**argv){int m=argc>1?atoi(argv[1]):128,context=argc>2?atoi(argv[2]):0;if(m<128||m%128)return 4;size_t tracebytes=size_t(m)*N*B16*4;size_t freeb,totalb;CK(cudaMemGetInfo(&freeb,&totalb));if(freeb<tracebytes+size_t(m)*DW*6+512ull*1024*1024){printf("{\"state\":\"gpu_memory_unavailable\",\"free\":%llu,\"required\":%llu}\n",(unsigned long long)freeb,(unsigned long long)(tracebytes+size_t(m)*DW*6+512ull*1024*1024));return 5;}
 unsigned seed=179199;auto rng=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;};std::vector<uint8_t>w(size_t(m)*DW*2),x(N*K/2),as(N*B16);std::vector<half>gs(m*B128),us(m*B128);std::vector<float>ag(N);for(auto&v:w)v=rng()%243;for(auto&v:x)v=rng()&255;for(auto&v:as)v=context?56+(rng()&7):rng()%127;for(auto&v:gs){__half_raw b;b.x=(uint16_t)(((10+rng()%10)<<10)|(rng()&1023));v=half(b);}for(auto&v:us){__half_raw b;b.x=(uint16_t)(((10+rng()%10)<<10)|(rng()&1023));v=half(b);}for(auto&v:ag)v=1.f+(rng()%16)/16.f;for(int f=0;f<128;f++)for(int p=0;p<2;p++)std::fill(w.begin()+(size_t(p)*m+f)*DW,w.begin()+(size_t(p)*m+f+1)*DW,121);std::fill(x.begin(),x.begin()+K/2,0);if(m>=384)for(int f=256;f<384;f++)for(int p=0;p<2;p++)std::fill(w.begin()+(size_t(p)*m+f)*DW,w.begin()+(size_t(p)*m+f+1)*DW,242);
 std::vector<uint64_t>wt(243);std::vector<uint32_t>fp(243),qt(65536);for(int i=0;i<243;i++){int z=i;for(int j=0;j<5;j++){int v=z%3-1;wt[i]|=uint64_t(unsigned(v)&255)<<(8*j);fp[i]|=uint32_t(v<0?10:v>0?2:0)<<(4*j);z/=3;}}int val[8]={0,1,2,3,4,6,8,12};for(int i=0;i<65536;i++)for(int j=0;j<4;j++){int z=(i>>(j*4))&15,v=val[z&7]*(z&8?-1:1);qt[i]|=uint32_t(unsigned(v)&255)<<(8*j);}
 Args a{};a.m=m;auto put=[&](auto&v){using T=typename std::decay_t<decltype(v)>::value_type;T*p;CK(cudaMalloc(&p,v.size()*sizeof(T)));CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;};a.w=put(w);a.x=put(x);a.as=put(as);a.gs=put(gs);a.us=put(us);a.ag=put(ag);a.wt=put(wt);a.fp=put(fp);a.qt=put(qt);int*trace;CK(cudaMalloc(&trace,tracebytes));unsigned long long*errs;CK(cudaMalloc(&errs,8));CK(cudaMemset(errs,0,8));int n=m*N,ns=m/128*N;float*y[8],*s[8];signed char*qout[8];for(int j=0;j<8;j++){CK(cudaMalloc(&y[j],n*4));CK(cudaMalloc(&s[j],ns*4));CK(cudaMalloc(&qout[j],n));}
 int grid=m/128*(N/8);producer<false><<<grid,256>>>(a,trace);fp4_ref<<<m/64*(N/8),256>>>(a,y[7]);for(int mode=0;mode<5;mode++)finish<<<grid,256>>>(a,trace,y[mode],mode);dp4a_fused<<<(n+255)/256,256>>>(a,y[5],4,errs,trace);folded_fused<<<grid,256>>>(a,y[6],4);for(int j=0;j<8;j++)normalize<<<ns,128>>>(y[j],qout[j],s[j],m);CK(cudaGetLastError());CK(cudaDeviceSynchronize());unsigned long long physical;CK(cudaMemcpy(&physical,errs,8,cudaMemcpyDeviceToHost));std::vector<uint32_t>hy[8],hs[8];std::vector<signed char>hq[8];for(int j=0;j<8;j++){hy[j].resize(n);hs[j].resize(ns);hq[j].resize(n);CK(cudaMemcpy(hy[j].data(),y[j],n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hs[j].data(),s[j],ns*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hq[j].data(),qout[j],n,cudaMemcpyDeviceToHost));}int ey[8]={},eq[8]={},es[8]={};for(int j=0;j<8;j++){for(int i=0;i<n;i++){ey[j]+=hy[j][i]!=hy[7][i];eq[j]+=hq[j][i]!=hq[7][i];}for(int i=0;i<ns;i++)es[j]+=hs[j][i]!=hs[7][i];}printf("{\"state\":\"observed\",\"M\":%d,\"N\":64,\"K\":5120,\"scale_context\":%d,\"physical_errors\":%llu,\"f32_errors\":[%d,%d,%d,%d,%d,%d,%d,%d],\"q8_errors\":[%d,%d,%d,%d,%d,%d,%d,%d],\"q8_scale_errors\":[%d,%d,%d,%d,%d,%d,%d,%d],\"carrier_bytes\":%llu}\n",m,context,physical,ey[0],ey[1],ey[2],ey[3],ey[4],ey[5],ey[6],ey[7],eq[0],eq[1],eq[2],eq[3],eq[4],eq[5],eq[6],eq[7],es[0],es[1],es[2],es[3],es[4],es[5],es[6],es[7],(unsigned long long)tracebytes);
 auto save=[&](const char*name,const void*p,size_t sz){FILE*f=fopen(name,"wb");if(!f||fwrite(p,1,sz,f)!=sz)exit(4);fclose(f);};save("fullk-weights.bin",w.data(),w.size());save("fullk-raw-fp4.bin",x.data(),x.size());save("fullk-as.bin",as.data(),as.size());save("fullk-gs.bin",gs.data(),gs.size()*2);save("fullk-us.bin",us.data(),us.size()*2);save("fullk-ag.bin",ag.data(),ag.size()*4);for(int j=0;j<8;j++){char nm[64];sprintf(nm,"fullk-arm%d-f32.bin",j);save(nm,hy[j].data(),n*4);sprintf(nm,"fullk-arm%d-q8.bin",j);save(nm,hq[j].data(),n);sprintf(nm,"fullk-arm%d-scale.bin",j);save(nm,hs[j].data(),ns*4);}if(m==128){std::vector<int>tr(tracebytes/4);CK(cudaMemcpy(tr.data(),trace,tracebytes,cudaMemcpyDeviceToHost));save("fullk-carrier-trace.bin",tr.data(),tracebytes);}return physical||ey[4]||eq[4]||es[4]?1:0;}
