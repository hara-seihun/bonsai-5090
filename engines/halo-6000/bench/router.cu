#include "moe.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
using namespace halo6000;
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}
__global__ void select_kernel(MoeScratch s,float scale){__shared__ float scratch[32];MoeView v{};v.weight_scale=scale;moe_select(v,s,scratch);}
int main(){
 MoeScratch s{};ck(cudaMalloc(&s.logits,Experts*4));ck(cudaMalloc(&s.probs,Experts*4));ck(cudaMalloc(&s.ids,TopExperts*4));ck(cudaMalloc(&s.weights,TopExperts*4));
 size_t errors=0;for(int test=0;test<8;test++){
  std::vector<float>logits(Experts),p(Experts);for(int i=0;i<Experts;i++)logits[i]=test==0?0:test==1?float(i%4):test==2?(i==198?90.f:-90.f):float(((i*37+test*17)%97)-48)/(test==3?100000.f:3.f);
  const float scale=test==7?1.125f:1.f;ck(cudaMemcpy(s.logits,logits.data(),Experts*4,cudaMemcpyHostToDevice));select_kernel<<<1,256>>>(s,scale);ck(cudaGetLastError());ck(cudaMemcpy(p.data(),s.probs,Experts*4,cudaMemcpyDeviceToHost));
  int32_t got[TopExperts];float weights[TopExperts];ck(cudaMemcpy(got,s.ids,sizeof got,cudaMemcpyDeviceToHost));ck(cudaMemcpy(weights,s.weights,sizeof weights,cudaMemcpyDeviceToHost));
  std::vector<int>ids(Experts);for(int i=0;i<Experts;i++)ids[i]=i;std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return p[a]>p[b];});float total=0;for(int j=0;j<TopExperts;j++)total+=p[ids[j]];total=std::max(total,6.103515625e-5f);
  for(int j=0;j<TopExperts;j++){const float ref=p[ids[j]]/total*scale;errors+=got[j]!=ids[j];errors+=memcmp(&ref,&weights[j],4)!=0;}
 }
 printf("{\"cases\":8,\"id_or_weight_bit_errors\":%zu,\"tie_rule\":\"lowest_ID_after_probability_rounding\"}\n",errors);
 for(void*p:{(void*)s.logits,(void*)s.probs,(void*)s.ids,(void*)s.weights})ck(cudaFree(p));return errors?1:0;
}
