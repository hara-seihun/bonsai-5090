#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <algorithm>
#define CK(x) do{auto s=(x);if(s!=cudaSuccess){fprintf(stderr,"cuda %s\n",cudaGetErrorString(s));exit(2);}}while(0)
constexpr unsigned MOD=15015;
__device__ __forceinline__ unsigned compose(unsigned a,unsigned b){
 unsigned lane=threadIdx.x&31,g=lane>>2,t=lane&3,lo=0,hi=0;
 #pragma unroll
 for(int c=0;c<4;c++){
  unsigned v=__shfl_sync(0xffffffff,a,g*4+c);lo|=(v&255)<<(8*c);hi|=(v>>8)<<(8*c);
 }
 unsigned a0=(g<4&&t==g)?lo:0,a1=(g<4&&t==g)?hi:0,a2=(g>=4&&t==g-4)?lo:0,a3=(g>=4&&t==g-4)?hi:0;
 unsigned b0=__shfl_sync(0xffffffff,b,t*4+g/2),b1=__shfl_sync(0xffffffff,b,(t+4)*4+g/2);
 b0=((g&1)?b0>>8:b0&255)<<(8*(g/2));b1=((g&1)?b1>>8:b1&255)<<(8*(g/2));
 int c0=0,c1=0,c2=0,c3=0;
 asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
 unsigned product=unsigned(c0)+((unsigned(c1)+unsigned(c2))<<8)+(unsigned(c3)<<16),r;
 asm("rem.u32 %0,%1,15015;":"=r"(r):"r"(product));return r;
}
constexpr int NA=759375, HS=2097152;
__device__ __constant__ int Q[15]={-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12};
__device__ __forceinline__ unsigned hashkey(unsigned x){return (x*2654435761u)&(HS-1);}
__device__ __forceinline__ int observe(unsigned a,unsigned b,const uint64_t* ht){unsigned key=a+MOD*b,slot=hashkey(key);for(int j=0;j<64;j++){uint64_t v=ht[slot];if(unsigned(v)==key)return int(v>>32)-60;if(unsigned(v)==0xffffffffu)return 999;slot=(slot+1)&(HS-1);}return 998;}
__device__ __forceinline__ float finish(int v,unsigned i){unsigned r=i%126,e=r>>3,m=r&7;float s=e?ldexpf(1.f+m*.125f,int(e)-7):m*(1.f/512.f);float ws=1.f+((i>>7)&15)*(1.f/16.f);return ((v*.5f)*s)*ws;}
__global__ void mixed(const uint32_t* in,const uint32_t*ew,const uint32_t*ea,const uint64_t*ht,float*out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;unsigned x=i<n?in[i]:((379687u<<8)|242),w=ew[x&255],a=ea[x>>8];unsigned r0=compose(w&65535,a&65535),r1=compose(w>>16,a>>16);if(i<n)out[i]=finish(observe(r0,r1,ht),i);}
__global__ void arithmetic(const uint32_t*in,float*out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned x=in[i],w=x&255,a=x>>8;int s=0;
 #pragma unroll
 for(int k=0;k<5;k++){s+=(int(w%3)-1)*Q[a%15];w/=3;a/=15;}out[i]=finish(s,i);}
__global__ void exhaustive(const uint32_t*ew,const uint32_t*ea,const uint64_t*ht,unsigned long long*bad,unsigned*first,unsigned long long*finger,unsigned long long n){unsigned long long i=blockIdx.x*uint64_t(blockDim.x)+threadIdx.x;unsigned wi=i<n?unsigned(i/NA):242,ai=i<n?unsigned(i%NA):379687;unsigned w=ew[wi],a=ea[ai];unsigned r0=compose(w&65535,a&65535),r1=compose(w>>16,a>>16);if(i>=n)return;int got=observe(r0,r1,ht),want=0;unsigned wj=wi,aj=ai;
 #pragma unroll
 for(int k=0;k<5;k++){want+=(int(wj%3)-1)*Q[aj%15];wj/=3;aj/=15;}
 float fv=finish(got,unsigned(i)),ev=finish(want,unsigned(i));if(got!=want||__float_as_uint(fv)!=__float_as_uint(ev)){atomicAdd(bad,1ull);atomicMin(first,unsigned(i));}atomicXor(finger,uint64_t(r0+MOD*r1)*(i+1));}
unsigned crt(const int*v,const int*p){unsigned z=0;for(int k=0;k<5;k++){int f=MOD/p[k],inv=1;while(f*inv%p[k]!=1)inv++;int val=(v[k]%p[k]+p[k])%p[k];z+=val*f*inv;}return z%MOD;}
int main(){int pa[5]={3,5,7,11,13},pb[5]={11,7,13,3,5},q[15]={-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12};std::vector<uint32_t>ew(243),ea(NA);std::vector<uint64_t>ht(HS,0xffffffffffffffffull);int maxprobe=0;
 for(int id=0;id<243;id++){int a=id,v[5];for(int k=0;k<5;k++){v[k]=a%3-1;a/=3;}ew[id]=crt(v,pa)|(crt(v,pb)<<16);}
 for(int id=0;id<NA;id++){int a=id,v[5],sum=0;for(int k=0;k<5;k++){v[k]=q[a%15];sum+=v[k];a/=15;}unsigned r0=crt(v,pa),r1=crt(v,pb);ea[id]=r0|(r1<<16);unsigned key=r0+MOD*r1,slot=key*2654435761u&(HS-1);int probes=1;while(ht[slot]!=0xffffffffffffffffull){if(unsigned(ht[slot])==key){fprintf(stderr,"joint collision %d\n",id);return 3;}slot=(slot+1)&(HS-1);probes++;}ht[slot]=key|(uint64_t(sum+60)<<32);maxprobe=std::max(maxprobe,probes);}
 if(maxprobe>64)return 4;FILE*f=fopen("mixed-weight-encoding.bin","wb");fwrite(ew.data(),4,ew.size(),f);fclose(f);f=fopen("mixed-activation-encoding.bin","wb");fwrite(ea.data(),4,ea.size(),f);fclose(f);f=fopen("mixed-final-observer.bin","wb");fwrite(ht.data(),8,ht.size(),f);fclose(f);
 uint32_t*dw,*da;uint64_t*dh;CK(cudaMalloc(&dw,ew.size()*4));CK(cudaMalloc(&da,ea.size()*4));CK(cudaMalloc(&dh,ht.size()*8));CK(cudaMemcpy(dw,ew.data(),ew.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(da,ea.data(),ea.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dh,ht.data(),ht.size()*8,cudaMemcpyHostToDevice));unsigned long long*bad,*fp;unsigned*first;CK(cudaMalloc(&bad,8));CK(cudaMalloc(&fp,8));CK(cudaMalloc(&first,4));CK(cudaMemset(bad,0,8));CK(cudaMemset(fp,0,8));CK(cudaMemset(first,255,4));unsigned long long total=uint64_t(243)*NA;exhaustive<<<unsigned((total+255)/256),256>>>(dw,da,dh,bad,first,fp,total);CK(cudaGetLastError());CK(cudaDeviceSynchronize());unsigned long long hb,hf;unsigned hi;CK(cudaMemcpy(&hb,bad,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&hf,fp,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&hi,first,4,cudaMemcpyDeviceToHost));
 int n=262144;std::vector<uint32_t>in(n);unsigned seed=715987;for(auto&x:in){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;unsigned a=seed%NA;seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;x=(a<<8)|(seed%243);}uint32_t*di;float*on,*oc;CK(cudaMalloc(&di,n*4));CK(cudaMalloc(&on,n*4));CK(cudaMalloc(&oc,n*4));CK(cudaMemcpy(di,in.data(),n*4,cudaMemcpyHostToDevice));mixed<<<n/256,256>>>(di,dw,da,dh,on,n);arithmetic<<<n/256,256>>>(di,oc,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());std::vector<float>hn(n),hc(n);CK(cudaMemcpy(hn.data(),on,n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hc.data(),oc,n*4,cudaMemcpyDeviceToHost));int wrong=0;for(int i=0;i<n;i++)if(hn[i]!=hc[i])wrong++;
 cudaStream_t st;CK(cudaStreamCreate(&st));cudaGraph_t gr[2];cudaGraphExec_t ge[2];for(int arm=0;arm<2;arm++){CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));for(int j=0;j<100;j++){if(arm==0)mixed<<<n/256,256,0,st>>>(di,dw,da,dh,on,n);else arithmetic<<<n/256,256,0,st>>>(di,oc,n);}CK(cudaStreamEndCapture(st,&gr[arm]));CK(cudaGraphInstantiate(&ge[arm],gr[arm],nullptr,nullptr,0));CK(cudaGraphLaunch(ge[arm],st));CK(cudaStreamSynchronize(st));}
 cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));std::vector<float>ms[2];for(int r=0;r<7;r++)for(int aa=0;aa<2;aa++){int arm=(r+aa)%2;CK(cudaEventRecord(start,st));CK(cudaGraphLaunch(ge[arm],st));CK(cudaEventRecord(end,st));CK(cudaEventSynchronize(end));float v;CK(cudaEventElapsedTime(&v,start,end));ms[arm].push_back(v*10);}for(auto&v:ms)std::sort(v.begin(),v.end());printf("{\"full_pairs\":%llu,\"full_errors\":%llu,\"first_error\":%u,\"native_fingerprint\":%llu,\"chain_errors\":%d,\"native_us\":%.6f,\"direct_us\":%.6f,\"observer_max_probes\":%d,\"weight_producer_bytes\":972,\"activation_producer_bytes\":3037500,\"observer_bytes\":16777216}\n",total,hb,hi,hf,wrong,ms[0][3],ms[1][3],maxprobe);return hb||wrong?1:0;}
