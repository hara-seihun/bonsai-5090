#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstring>
#include "../src/spec_copy.cuh"
int main(){
 size_t N=8*1024*1024;std::vector<unsigned char>src(N),dst(N),expected(N);
 for(size_t i=0;i<N;++i)src[i]=(i*137+i/17)&255;
 unsigned char *s,*d;cudaMalloc(&s,N);cudaMalloc(&d,N);cudaMemcpy(s,src.data(),N,cudaMemcpyHostToDevice);
 int cases=0;
 for(int begin:{0,1,15,31,32,127})for(int len:{1,7,15,17,31,32,127,4097,32769}){
  std::fill(expected.begin(),expected.end(),0xa5);cudaMemset(d,0xa5,N);h5::SpecCopyPlan p;
  for(int k=0;k<10;k++){size_t base=k*700000+begin,pitch=65536;int rows=3;
   h5::spec_copy_add(p,s+base,d+base,len,rows,pitch);
   for(int r=0;r<rows;r++)std::memcpy(expected.data()+base+r*pitch,src.data()+base+r*pitch,len);
  }
  h5::spec_copy_launch(p,0);cudaMemcpy(dst.data(),d,N,cudaMemcpyDeviceToHost);
  if(cudaGetLastError()!=cudaSuccess||dst!=expected){printf("FAIL begin=%d len=%d\n",begin,len);return 1;}cases++;
 }
 printf("PASS %d cases; complete destination and untouched padding checked\n",cases);cudaFree(s);cudaFree(d);
}
