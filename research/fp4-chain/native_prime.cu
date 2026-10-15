#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#define CK(x) do {auto ck_status=(x);if(ck_status!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(ck_status));exit(1);}}while(0)
__constant__ float labels[27];
__device__ float tensor(float a,float b){uint32_t x,y;asm("cvt.rna.tf32.f32 %0,%1;":"=r"(x):"f"(a));asm("cvt.rna.tf32.f32 %0,%1;":"=r"(y):"f"(b));float d[4]={};asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(x),"r"(y));return d[0]*0.125f;}
__global__ void probe(const uint32_t*expected,uint32_t*out,unsigned*mismatch){int n=blockIdx.x;int t=n,ix[4];for(int j=3;j>=0;j--){ix[j]=t%27;t/=27;}float a=tensor(labels[ix[0]],labels[ix[1]]);a=tensor(a,labels[ix[2]]);a=tensor(a,labels[ix[3]]);uint32_t bits=__float_as_uint(a);if(threadIdx.x==0)out[n]=bits;if(bits!=expected[n])atomicAdd(mismatch,1u);}
uint32_t bits(float x){uint32_t b;memcpy(&b,&x,4);return b;}float from(uint32_t b){float x;memcpy(&x,&b,4);return x;}float roundtf(float x){return from((bits(x)+4096u)&0xffffe000u);}
int main(){const int primes[]={2,5,7};float vals[27];int tr[27][3];double largest=pow(70.,5);int shift=int(log2(largest))/2;for(int i=0;i<27;i++){int n=i;double v=1;for(int j=2;j>=0;j--){int d=n%3;n/=3;tr[i][j]=d-1;v*=pow(double(primes[j]),d==2?0:d==0?1:5);}vals[i]=float(ldexp(v,-shift));}
const int N=27*27*27*27;std::vector<uint32_t> expect(N),host(N);std::vector<std::pair<uint32_t,int>> truth;truth.reserve(N);for(int n=0;n<N;n++){int ix[4],t=n;for(int j=3;j>=0;j--){ix[j]=t%27;t/=27;}float a=roundtf(vals[ix[0]])*roundtf(vals[ix[1]]);for(int j=2;j<4;j++)a=roundtf(a)*roundtf(vals[ix[j]]);expect[n]=bits(a);int target=0;for(int j=0;j<3;j++){int p=1;for(int k=0;k<4;k++)p*=tr[ix[k]][j];target=3*target+p+1;}truth.emplace_back(bits(a),target);}
std::sort(truth.begin(),truth.end());unsigned conflicts=0,states=0;for(int i=0;i<N;i++){if(!i||truth[i].first!=truth[i-1].first)states++;else if(truth[i].second!=truth[i-1].second)conflicts++;}
if(conflicts){fprintf(stderr,"CPU target collision %u\n",conflicts);return 3;}
uint32_t *de,*out;unsigned*bad;CK(cudaMalloc(&de,N*4));CK(cudaMalloc(&out,N*4));CK(cudaMalloc(&bad,4));CK(cudaMemset(bad,0,4));CK(cudaMemcpy(de,expect.data(),N*4,cudaMemcpyHostToDevice));CK(cudaMemcpyToSymbol(labels,vals,sizeof(vals)));
cudaEvent_t s,e;CK(cudaEventCreate(&s));CK(cudaEventCreate(&e));CK(cudaEventRecord(s));probe<<<N,32>>>(de,out,bad);CK(cudaGetLastError());CK(cudaEventRecord(e));CK(cudaEventSynchronize(e));float ms;CK(cudaEventElapsedTime(&ms,s,e));unsigned mismatch=0;CK(cudaMemcpy(&mismatch,bad,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(host.data(),out,N*4,cudaMemcpyDeviceToHost));unsigned hostbad=0;for(int i=0;i<N;i++)hostbad+=host[i]!=expect[i];
// Store the finite arbitrary final observer; numeric labels are never factor-decoded.
FILE*f=fopen("native-prime-observer.bin","wb");uint32_t prev=0;bool first=true;for(auto z:truth){if(first||z.first!=prev){fwrite(&z.first,4,1,f);fwrite(&z.second,4,1,f);prev=z.first;first=false;}}fclose(f);
cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));printf("{\"device\":\"%s\",\"cases\":%d,\"lane_checks\":%d,\"mismatches\":%u,\"host_mismatches\":%u,\"decoder_collisions\":%u,\"final_states\":%u,\"input_shift\":%d,\"probe_ms\":%.5f,\"scope\":\"three TF32 MMAs carry three independent four-factor ternary chains; explicit rna conversion and power-of-two normalizations, uniform K8, final-only observer; no independent K proof or speedup\"}\n",p.name,N,N*32,mismatch,hostbad,conflicts,states,shift,ms);
CK(cudaFree(de));CK(cudaFree(out));CK(cudaFree(bad));return mismatch||hostbad?2:0;}
