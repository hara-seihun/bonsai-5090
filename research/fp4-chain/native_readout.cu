#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#define CK(x) do {auto status=(x); if(status!=cudaSuccess){fprintf(stderr,"cuda_error: %s\n",cudaGetErrorString(status));exit(1);}}while(0)
constexpr int N=27*27*27*27;
struct Entry {uint32_t bits;int target;};
static_assert(sizeof(Entry)==8,"observer ABI");
__constant__ float labels[27];
__device__ __forceinline__ void indices(uchar4 c,int* ix){ix[0]=c.x;ix[1]=c.y;ix[2]=c.z;ix[3]=c.w;}
__device__ __forceinline__ int logical(uchar4 c){int ix[4];indices(c,ix);int result=0;for(int j=0;j<3;j++){int product=1;for(int k=0;k<4;k++){product*=ix[k]%3-1;ix[k]/=3;}result+=(product+1)*(j==0?1:j==1?3:9);}return result;}
__device__ __forceinline__ int observe(uint32_t bits,const Entry* dict,int count){int lo=0,hi=count;while(lo<hi){int mid=(lo+hi)/2;if(dict[mid].bits<bits)lo=mid+1;else hi=mid;}return lo<count&&dict[lo].bits==bits?dict[lo].target:-1;}
template<int Slot> __device__ __forceinline__ float tensor(float a,float b){uint32_t x,y;asm("cvt.rna.tf32.f32 %0,%1;":"=r"(x):"f"(a));asm("cvt.rna.tf32.f32 %0,%1;":"=r"(y):"f"(b));float d[4]={};asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(x),"r"(y));return d[Slot]*0.125f;}
template<int Slot,bool Validate> __global__ void native(const uchar4*input,int*output,const Entry*dict,int count,unsigned long long*bad){int lane=threadIdx.x%32,n=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(n>=N)return;uchar4 c=input[n];float a=tensor<Slot>(labels[c.x],labels[c.y]);a=tensor<Slot>(a,labels[c.z]);a=tensor<Slot>(a,labels[c.w]);if(Validate||lane==0){int got=observe(__float_as_uint(a),dict,count);if(Validate){if(got<0)atomicAdd(bad,1ULL);else if(got!=logical(c))atomicAdd(bad+1,1ULL);}if(lane==0)output[n]=got;}}
template<bool Warp> __global__ void direct(const uchar4*input,int*output){int tid=blockIdx.x*blockDim.x+threadIdx.x;int n=Warp?tid/32:tid;if(n>=N||(Warp&&tid%32))return;output[n]=logical(input[n]);}
__global__ void compare(const int*a,const int*b,unsigned long long*bad){int n=blockIdx.x*blockDim.x+threadIdx.x;if(n<N&&a[n]!=b[n])atomicAdd(bad,1ULL);}
float elapsed(cudaEvent_t s,cudaEvent_t e){CK(cudaEventRecord(e));CK(cudaEventSynchronize(e));float ms;CK(cudaEventElapsedTime(&ms,s,e));return ms;}
int main(){FILE*f=fopen("native-prime-observer.bin","rb");if(!f){fprintf(stderr,"observer_unset\n");return 3;}fseek(f,0,SEEK_END);long size=ftell(f);rewind(f);if(size!=149520){fprintf(stderr,"observer_invalid_size\n");return 3;}std::vector<Entry> dict(size/8);if(fread(dict.data(),8,dict.size(),f)!=dict.size()){fprintf(stderr,"observer_read_error\n");return 3;}fclose(f);for(int i=0;i<(int)dict.size();i++)if(dict[i].target<0||dict[i].target>=27||(i&&dict[i].bits<=dict[i-1].bits)){fprintf(stderr,"observer_invalid_entry\n");return 3;}
float vals[27];const int primes[]={2,5,7};for(int i=0;i<27;i++){int n=i;double v=1;for(int j=2;j>=0;j--){int digit=n%3;n/=3;v*=pow(double(primes[j]),digit==2?0:digit==0?1:5);}vals[i]=float(ldexp(v,-15));}CK(cudaMemcpyToSymbol(labels,vals,sizeof(vals)));
std::vector<uchar4> input(N);for(int i=0;i<N;i++){int n=(uint64_t(i)*15485863+17)%N;uchar4 c;c.w=n%27;n/=27;c.z=n%27;n/=27;c.y=n%27;n/=27;c.x=n;input[i]=c;}
uchar4*di;Entry*dd;int *out,*reference;unsigned long long*bad;CK(cudaMalloc(&di,N*4));CK(cudaMalloc(&dd,size));CK(cudaMalloc(&out,N*4));CK(cudaMalloc(&reference,N*4));CK(cudaMalloc(&bad,16));CK(cudaMemcpy(di,input.data(),N*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dd,dict.data(),size,cudaMemcpyHostToDevice));CK(cudaMemset(bad,0,16));direct<false><<<(N+255)/256,256>>>(di,reference);CK(cudaGetLastError());
const int blocks=(N+7)/8;
#define V(S) native<S,true><<<blocks,256>>>(di,out,dd,(int)dict.size(),bad);CK(cudaGetLastError());
V(0) V(1) V(2) V(3)
CK(cudaDeviceSynchronize());unsigned long long counts[2];CK(cudaMemcpy(counts,bad,16,cudaMemcpyDeviceToHost));if(counts[0]||counts[1]){printf("{\"status\":\"native_validation_error\",\"missing\":%llu,\"mismatches\":%llu}\n",counts[0],counts[1]);return 2;}
native<0,false><<<blocks,256>>>(di,out,dd,(int)dict.size(),bad);compare<<<(N+255)/256,256>>>(out,reference,bad+1);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(counts,bad,16,cudaMemcpyDeviceToHost));if(counts[1]){fprintf(stderr,"timed_native_control_mismatch\n");return 2;}
for(int k=0;k<5;k++){native<0,false><<<blocks,256>>>(di,out,dd,(int)dict.size(),bad);direct<true><<<blocks,256>>>(di,out);direct<false><<<(N+255)/256,256>>>(di,out);}CK(cudaDeviceSynchronize());
cudaEvent_t s,e;CK(cudaEventCreate(&s));CK(cudaEventCreate(&e));std::vector<float> times[3];constexpr int repeat=10;
for(int round=0;round<7;round++)for(int order=0;order<3;order++){int route=(round+order)%3;CK(cudaEventRecord(s));for(int k=0;k<repeat;k++){if(route==0)native<0,false><<<blocks,256>>>(di,out,dd,(int)dict.size(),bad);if(route==1)direct<true><<<blocks,256>>>(di,out);if(route==2)direct<false><<<(N+255)/256,256>>>(di,out);}CK(cudaGetLastError());times[route].push_back(elapsed(s,e)/repeat);}
cudaFuncAttributes a,b,c;CK(cudaFuncGetAttributes(&a,native<0,false>));CK(cudaFuncGetAttributes(&b,direct<true>));CK(cudaFuncGetAttributes(&c,direct<false>));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));printf("{\"status\":\"pass\",\"device\":\"%s\",\"cases\":%d,\"accumulator_paths\":4,\"lane_checks\":%llu,\"missing\":%llu,\"mismatches\":%llu,\"dictionary_bytes\":%ld,\"labels_bytes\":108,\"input_bytes\":%d,\"output_bytes\":%d,\"rounds\":7,\"repetitions_per_round\":10,\"timings_ms\":{",p.name,N,4ULL*32*N,counts[0],counts[1],size,N*4,N*4);
const char*names[]={"native_final_observer","warp_direct","lane_direct"};for(int route=0;route<3;route++){std::sort(times[route].begin(),times[route].end());printf("%s\"%s\":{\"median\":%.6f,\"min\":%.6f,\"max\":%.6f}",route?",":"",names[route],times[route][3],times[route][0],times[route][6]);}printf("},\"registers\":[%d,%d,%d],\"local_bytes_per_thread\":[%zu,%zu,%zu],\"shared_bytes_per_block\":[%zu,%zu,%zu],\"scope\":\"shuffled exhaustive ternary four-factor source codes to final three-trit code; producer lookup, RNA conversion, three MMAs, three 1/8 normalizations, final binary-search observer and output included; warm resident timing, uniform K8, no independent-K/model/HTTP claim\"}\n",a.numRegs,b.numRegs,c.numRegs,a.localSizeBytes,b.localSizeBytes,c.localSizeBytes,a.sharedSizeBytes,b.sharedSizeBytes,c.sharedSizeBytes);
CK(cudaEventDestroy(s));CK(cudaEventDestroy(e));CK(cudaFree(di));CK(cudaFree(dd));CK(cudaFree(out));CK(cudaFree(reference));CK(cudaFree(bad));return 0;}
