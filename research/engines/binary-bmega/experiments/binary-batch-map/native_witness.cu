#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(c) do{auto e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
__global__ void witness(int2* out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2*256*256)return;
  unsigned u=i%256,v=(i/256)%256,b=i/(256*256),a=u|(v<<16),sum=0;
  #pragma unroll 1
  for(int j=0;j<128;j++)asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(sum) : "r"(a),"r"(b),"r"(sum));
  int dc0=128*(int(u)-128),dc1=128*(int(v)-128),count=128*b;
  out[i]=make_int2(2*int(sum&65535)-dc0-256*count,2*int(sum>>16)-dc1-256*count);
}
int main(){constexpr int N=2*256*256;int2* d;CK(cudaMalloc(&d,N*sizeof(int2)));witness<<<N/256,256>>>(d);CK(cudaGetLastError());
  std::vector<int2> h(N);CK(cudaMemcpy(h.data(),d,N*sizeof(int2),cudaMemcpyDeviceToHost));
  for(int i=0;i<N;i++){int a=i%256-128,b=(i/256)%256-128,w=2*(i/(256*256))-1;
    if(h[i].x!=128*w*a||h[i].y!=128*w*b){fprintf(stderr,"integer mismatch %d\n",i);return 3;}}
  printf("native mad.lo.u32 K128: all 131072 uniform two-byte × weight-bit cases exact; extrema +/-16384 and lifted sums32640; 262144 integer outputs checked\n");
  CK(cudaFree(d));return 0;
}
