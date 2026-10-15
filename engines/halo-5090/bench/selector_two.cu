#include "kernels.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
namespace h5 {
void launch_selector(const __nv_bfloat16*,const __nv_bfloat16*,const float*,const float*,const int*,const int*,int*,float*,int,cudaStream_t);
void launch_selector_two(const __nv_bfloat16*,const __nv_bfloat16*,const float*,const float*,const int*,const int*,int*,float*,int*,float*,int*,int,int,cudaStream_t);
}
#define CK(x) do{auto ckerr=(x);if(ckerr!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(ckerr));exit(2);}}while(0)
template<class T>T* upload(const std::vector<T>&v){T*p;CK(cudaMalloc(&p,v.size()*sizeof(T)));CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T>T* alloc(int n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
  using namespace h5;
  std::vector<__nv_bfloat16>pred(32*DF_RANK,__float2bfloat16(0)),succ(pred.size(),__float2bfloat16(0));
  pred[DF_RANK]=succ[DF_RANK]=__float2bfloat16(1);pred[2*DF_RANK]=succ[2*DF_RANK]=__float2bfloat16(-1);
  std::vector<float>hp(7*DF_RANK),tv(7*DF_TOPK);std::vector<int>ti(7*DF_TOPK);
  for(int p=0;p<7;p++){hp[p*DF_RANK]=p==2?0:p<2?.02f:1.f;for(int k=0;k<DF_TOPK;k++){ti[p*DF_TOPK+k]=k+1;tv[p*DF_TOPK+k]=k==0?1.f:k==1?(p==2?.99f:p==1?.9f:.8f):-10.f-k;}}
  auto dp=upload(pred),ds=upload(succ);auto dh=upload(hp),dtv=upload(tv);auto dti=upload(ti),an=upload(std::vector<int>{1});
  auto primary=alloc<int>(7),control=alloc<int>(7),alt=alloc<int>(7),branch=alloc<int>(1);auto conf=alloc<float>(7),cc=alloc<float>(7),ac=alloc<float>(7);
  launch_selector(dp,ds,dh,dtv,dti,an,control,cc,1,0);
  launch_selector_two(dp,ds,dh,dtv,dti,an,primary,conf,alt,ac,branch,3,1,0);CK(cudaDeviceSynchronize());
  std::vector<int>p(7),c(7),a(7);int fork;
  CK(cudaMemcpy(p.data(),primary,28,cudaMemcpyDeviceToHost));CK(cudaMemcpy(c.data(),control,28,cudaMemcpyDeviceToHost));CK(cudaMemcpy(a.data(),alt,28,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&fork,branch,4,cudaMemcpyDeviceToHost));
  bool pass=p==c&&fork==2;for(int i=0;i<7;i++){pass&=p[i]==1;pass&=a[i]==(i<2?1:2);}
  printf("{\"selector_guard\":true,\"fork\":%d,\"old_primary_identical\":%s,\"alternate_continuation_uses_own_predecessor\":%s,\"pass\":%s}\n",fork,p==c?"true":"false",a[3]==2?"true":"false",pass?"true":"false");return pass?0:3;
}
