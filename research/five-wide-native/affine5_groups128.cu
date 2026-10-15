#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#define CK(x) do{auto cuda_status=(x);if(cuda_status!=cudaSuccess){fprintf(stderr,"cuda:%s\n",cudaGetErrorString(cuda_status));exit(2);}}while(0)
constexpr int OFFSET=126819900;
struct Tile {uint16_t w[2][16];uint32_t x[2][8];uint8_t as[2][8],gs[2][16],us[2][16];uint16_t role[16];};
__device__ __forceinline__ int q(int b){int a=int((0x0c08060403020100ull>>((b&7)*8))&255),sg=-(b>>3);return (a^sg)-sg;}
__device__ __forceinline__ uint64_t packx(unsigned x){uint64_t v=0;
 #pragma unroll
 for(int k=0;k<5;k++)v|=uint64_t(unsigned(q((x>>(4*k))&15))&255)<<(8*k);return v;}
__device__ __forceinline__ void mma(int&d0,int&d1,int&d2,int&d3,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));}
__device__ __forceinline__ float scale(unsigned x){__nv_fp8_e4m3 s;s.__x=(__nv_fp8_storage_t)x;return float(s);}
__device__ __forceinline__ float observe_parts(int g0,int u0,int g1,int u1,const Tile&z,int row,int col){float ag0=scale(z.as[0][col]),ag1=scale(z.as[1][col]);float wg0=1.f+z.gs[0][row]*(1.f/16),wg1=1.f+z.gs[1][row]*(1.f/16),wu0=1.f+z.us[0][row]*(1.f/16),wu1=1.f+z.us[1][row]*(1.f/16);float g=((g0*.5f)*ag0)*wg0+((g1*.5f)*ag1)*wg1,u=((u0*.5f)*ag0)*wu0+((u1*.5f)*ag1)*wu1;int r=(z.role[row]>>(2*col))&3;return ((g/(1.f+__expf(-g)))*u)*(r-1);}
__device__ __forceinline__ float observe(int word,const Tile&z,int row,int col){unsigned c=word+OFFSET;int g0=int(c&127)-60,u0=int((c>>7)&127)-60,g1=int((c>>14)&127)-60,u1=int(c>>21)-60;return observe_parts(g0,u0,g1,u1,z,row,col);}
__device__ __forceinline__ int q8(float y){return __float2int_rn(fmaxf(-127.f,fminf(127.f,y)));}
__global__ void native(const Tile*in,const uint64_t*enc,const uint64_t*encu,signed char*out,uint32_t*bits,int*words,int tiles){int wi=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(wi>=tiles)return;int lane=threadIdx.x&31,g=lane>>2,t=lane&3;auto const&z=in[wi];int d0=0,d1=0,d2=0,d3=0;
 #pragma unroll
 for(int j=0;j<2;j++){int group=1-j;unsigned w0=z.w[group][g],w1=z.w[group][g+8];uint64_t p0=enc[w0&255],p1=enc[w1&255],u0=encu[w0>>8],u1=encu[w1>>8],a=packx(z.x[group][g]),a2x=(a<<1)&0xfefefefefeull;unsigned a0=t<2?unsigned(p0>>(32*t)):0,a1=t<2?unsigned(p1>>(32*t)):0,a2=t<2?unsigned(u0>>(32*t)):0,a3=t<2?unsigned(u1>>(32*t)):0,b0=t<2?unsigned(a>>(32*t)):0,b1=t<2?unsigned(a2x>>(32*t)):0;mma(d0,d1,d2,d3,a0,a1,a2,a3,b0,b1);if(j==0){d0*=16384;d1*=16384;d2*=16384;d3*=16384;}}
 int d[4]={d0,d1,d2,d3};
 #pragma unroll
 for(int k=0;k<4;k++){int row=g+(k/2)*8,col=2*t+k%2,i=wi*128+row*8+col;float y=observe(d[k],z,row,col);out[i]=q8(y);if(bits)bits[i]=__float_as_uint(y);if(words)words[i]=d[k];}}
__global__ void direct(const Tile*in,signed char*out,uint32_t*bits,int*words,int tiles){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tiles*128)return;int wi=i/128,row=(i%128)/8,col=i%8;auto const&z=in[wi];int gs[2],us[2];
 #pragma unroll
 for(int j=0;j<2;j++){int g=0,u=0;unsigned w=z.w[j][row],gw=w&255,uw=w>>8,x=z.x[j][col];
 #pragma unroll
 for(int k=0;k<5;k++){int a=q((x>>(4*k))&15);g+=a*(int(gw%3)-1);u+=a*(int(uw%3)-1);gw/=3;uw/=3;}gs[j]=g;us[j]=u;}
 float y=observe_parts(gs[0],us[0],gs[1],us[1],z,row,col);out[i]=q8(y);if(bits)bits[i]=__float_as_uint(y);if(words)words[i]=(gs[1]+128*us[1])*16384+gs[0]+128*us[0];}
__device__ __forceinline__ int dot5(uint64_t w,uint64_t x){return __dp4a(int(w),int(x),0)+int(int8_t(w>>32))*int(int8_t(x>>32));}
__global__ void folded_tile(const Tile*in,const uint64_t*enc,const uint64_t*encu,signed char*out,uint32_t*bits,int tiles){int wi=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(wi>=tiles)return;int lane=threadIdx.x&31,g=lane>>2,t=lane&3;auto const&z=in[wi];int d[4]={};
 #pragma unroll
 for(int j=0;j<2;j++){int group=1-j;unsigned w0=z.w[group][g],w1=z.w[group][g+8];uint64_t p[2]={enc[w0&255],enc[w1&255]},u[2]={encu[w0>>8],encu[w1>>8]},xp=packx(z.x[group][g]),x[2];unsigned lo=xp,hi=xp>>32;
 #pragma unroll
 for(int c=0;c<2;c++){int from=(2*t+c)*4;x[c]=uint64_t(__shfl_sync(0xffffffff,lo,from))|(uint64_t(__shfl_sync(0xffffffff,hi,from))<<32);}
 #pragma unroll
 for(int k=0;k<4;k++)d[k]+=dot5(p[k/2],x[k%2])+dot5(u[k/2],(x[k%2]<<1)&0xfefefefefeull);if(j==0){
 #pragma unroll
 for(int k=0;k<4;k++)d[k]*=16384;}}
 #pragma unroll
 for(int k=0;k<4;k++){int row=g+(k/2)*8,col=2*t+k%2,i=wi*128+row*8+col;float y=observe(d[k],z,row,col);out[i]=q8(y);if(bits)bits[i]=__float_as_uint(y);}}
__global__ void dp4a_direct(const Tile*in,const uint64_t*enc,signed char*out,uint32_t*bits,int tiles){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tiles*128)return;int wi=i/128,row=(i%128)/8,col=i%8;auto const&z=in[wi];int gs[2],us[2];
 #pragma unroll
 for(int j=0;j<2;j++){unsigned w=z.w[j][row];uint64_t x=packx(z.x[j][col]);gs[j]=dot5(enc[w&255],x);us[j]=dot5(enc[w>>8],x);}float y=observe_parts(gs[0],us[0],gs[1],us[1],z,row,col);out[i]=q8(y);if(bits)bits[i]=__float_as_uint(y);}
int main(){std::vector<uint64_t>e(243),s(243),ue(243);for(int id=0;id<243;id++){int v=id;for(int k=0;k<5;k++){e[id]|=uint64_t(unsigned(v%3-1)&255)<<(8*k);ue[id]|=uint64_t(unsigned(64*(v%3-1))&255)<<(8*k);s[id]|=uint64_t(unsigned(v%3-1)&255)<<(8*k);v/=3;}}uint64_t*de,*ds,*du;CK(cudaMalloc(&de,1944));CK(cudaMalloc(&ds,1944));CK(cudaMalloc(&du,1944));CK(cudaMemcpy(du,ue.data(),1944,cudaMemcpyHostToDevice));CK(cudaMemcpy(de,e.data(),1944,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,s.data(),1944,cudaMemcpyHostToDevice));int tiles=8192,n=tiles*128;std::vector<Tile>in(tiles);unsigned seed=765133;auto rng=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;};for(auto&z:in){for(int j=0;j<2;j++){for(auto&w:z.w[j])w=(rng()%243)|((rng()%243)<<8);for(auto&x:z.x[j])x=rng()&0xfffff;for(auto&a:z.as[j])a=rng()%127;for(auto&g:z.gs[j])g=rng()%16;for(auto&u:z.us[j])u=rng()%16;}for(auto&r:z.role){r=0;for(int c=0;c<8;c++)r|=(rng()%3)<<(2*c);}}FILE*f=fopen("groups-inputs.bin","wb");fwrite(in.data(),sizeof(Tile),tiles,f);fclose(f);f=fopen("groups-up64-producer.bin","wb");fwrite(ue.data(),8,243,f);fclose(f);f=fopen("groups-producer.bin","wb");fwrite(e.data(),8,243,f);fclose(f);Tile*di;CK(cudaMalloc(&di,tiles*sizeof(Tile)));CK(cudaMemcpy(di,in.data(),tiles*sizeof(Tile),cudaMemcpyHostToDevice));signed char*out[4];uint32_t*bits[4];int*word[2];for(int j=0;j<4;j++){CK(cudaMalloc(&out[j],n));CK(cudaMalloc(&bits[j],n*4));}for(auto&w:word)CK(cudaMalloc(&w,n*4));native<<<tiles/8,256>>>(di,de,du,out[0],bits[0],word[0],tiles);direct<<<n/256,256>>>(di,out[1],bits[1],word[1],tiles);folded_tile<<<tiles/8,256>>>(di,de,du,out[2],bits[2],tiles);dp4a_direct<<<n/256,256>>>(di,ds,out[3],bits[3],tiles);CK(cudaGetLastError());CK(cudaDeviceSynchronize());std::vector<uint32_t>hb[4];std::vector<signed char>ho[4];for(int j=0;j<4;j++){hb[j].resize(n);ho[j].resize(n);CK(cudaMemcpy(hb[j].data(),bits[j],n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(ho[j].data(),out[j],n,cudaMemcpyDeviceToHost));}std::vector<int>hw[2];for(int j=0;j<2;j++){hw[j].resize(n);CK(cudaMemcpy(hw[j].data(),word[j],n*4,cudaMemcpyDeviceToHost));}int ew=0,eb[4]={},eq[4]={};for(int i=0;i<n;i++){ew+=hw[0][i]!=hw[1][i];for(int j=0;j<4;j++){eb[j]+=hb[j][i]!=hb[1][i];eq[j]+=ho[j][i]!=ho[1][i];}}f=fopen("groups-native-words.bin","wb");fwrite(hw[0].data(),4,n,f);fclose(f);f=fopen("groups-native-f32.bin","wb");fwrite(hb[0].data(),4,n,f);fclose(f);f=fopen("groups-native-q8.bin","wb");fwrite(ho[0].data(),1,n,f);fclose(f);if(ew||eb[0]||eb[2]||eb[3]||eq[0]||eq[2]||eq[3]){printf("{\"state\":\"mismatch\",\"word_errors\":%d,\"f32_errors\":[%d,%d,%d,%d],\"q8_errors\":[%d,%d,%d,%d]}\n",ew,eb[0],eb[1],eb[2],eb[3],eq[0],eq[1],eq[2],eq[3]);return 1;}
 cudaStream_t st;CK(cudaStreamCreate(&st));cudaGraph_t gr[4];cudaGraphExec_t ge[4];for(int arm=0;arm<4;arm++){CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));for(int j=0;j<100;j++){if(arm==0)native<<<tiles/8,256,0,st>>>(di,de,du,out[0],nullptr,nullptr,tiles);else if(arm==1)direct<<<n/256,256,0,st>>>(di,out[1],nullptr,nullptr,tiles);else if(arm==2)folded_tile<<<tiles/8,256,0,st>>>(di,de,du,out[2],nullptr,tiles);else dp4a_direct<<<n/256,256,0,st>>>(di,ds,out[3],nullptr,tiles);}CK(cudaStreamEndCapture(st,&gr[arm]));CK(cudaGraphInstantiate(&ge[arm],gr[arm],nullptr,nullptr,0));CK(cudaGraphLaunch(ge[arm],st));CK(cudaStreamSynchronize(st));}cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));std::vector<float>ms[4];for(int r=0;r<7;r++)for(int s=0;s<4;s++){int arm=(r+s)%4;CK(cudaEventRecord(a,st));CK(cudaGraphLaunch(ge[arm],st));CK(cudaEventRecord(b,st));CK(cudaEventSynchronize(b));float v;CK(cudaEventElapsedTime(&v,a,b));ms[arm].push_back(v*10);}for(auto&v:ms)std::sort(v.begin(),v.end());printf("{\"state\":\"matched\",\"sampled_outputs\":%d,\"word_errors\":%d,\"f32_errors\":[%d,%d,%d,%d],\"q8_errors\":[%d,%d,%d,%d],\"native_us\":%.6f,\"direct_us\":%.6f,\"folded_tile_us\":%.6f,\"dp4a_us\":%.6f,\"source_bytes_per_tile\":%llu,\"native_producer_bytes\":3888,\"native_mmas_per_tile\":2,\"distinct_outputs_per_tile\":128}\n",n,ew,eb[0],eb[1],eb[2],eb[3],eq[0],eq[1],eq[2],eq[3],ms[0][3],ms[1][3],ms[2][3],ms[3][3],(unsigned long long)sizeof(Tile));return 0;}
