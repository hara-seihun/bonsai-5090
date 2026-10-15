#pragma once
#include "moe.cuh"
#include "matvec.cuh"
namespace halo6000 {
template<int Lanes,int RowsPerCTA>
__device__ __forceinline__ void moe_router_candidate(const MoeView&v,MoeScratch s){
 static_assert(RowsPerCTA*Lanes<=256);
 const int group=threadIdx.x/Lanes,lane=threadIdx.x&(Lanes-1);
 if(group<RowsPerCTA)for(int row=blockIdx.x*RowsPerCTA+group;row<v.weights->router.n;row+=gridDim.x*RowsPerCTA){
  float value=row_exact<Lanes>(v.weights->router,v.prepared,v.input,row);
  if(lane==0)s.logits[row]=value;
 }
 if(blockIdx.x==0&&threadIdx.x<32){const auto&w=v.weights->shared_router;float gain=w.format==WeightFormat::F32?float_row(w,v.input,0):quantized_row(w,v.prepared,0);if(threadIdx.x==0)*s.shared_gain=sigmoid(gain);}
}
// Fixed-geometry F32 router roles only: K2048,N256 and shared K2048,N1.
// Retains the logical32-lane FMA/tree, but exposes constants to the compiler.
template<int RowsPerCTA>
__device__ __forceinline__ void moe_router_fixed_f32(const MoeView&v,MoeScratch s){
 static_assert(RowsPerCTA>=1&&RowsPerCTA*32<=256);
 MatrixView router=v.weights->router;router.k=Hidden;router.n=Experts;router.format=WeightFormat::F32;router.row_stride=Hidden*4;
 const int group=threadIdx.x/32,lane=threadIdx.x&31;
 if(group<RowsPerCTA)for(int row=blockIdx.x*RowsPerCTA+group;row<Experts;row+=gridDim.x*RowsPerCTA){
  const float value=row_exact<32>(router,v.prepared,v.input,row);if(!lane)s.logits[row]=value;
 }
 if(blockIdx.x==0&&threadIdx.x<32){MatrixView shared=v.weights->shared_router;shared.k=Hidden;shared.format=WeightFormat::F32;const float gain=float_row(shared,v.input,0);if(threadIdx.x==0)*s.shared_gain=sigmoid(gain);}
}
struct RouterRank {float probability;int id;};
__device__ __forceinline__ bool router_better(RouterRank a,RouterRank b){return a.probability>b.probability||(a.probability==b.probability&&a.id<b.id);}
__device__ __forceinline__ RouterRank router_pick(RouterRank a,RouterRank b,bool best){return router_better(b,a)==best?b:a;}
// Exactly256 threads; existing32-float reduction scratch plus512B staticSMEM.
// Ranking is over rounded probabilities, never raw logits.
__device__ __forceinline__ void moe_select_candidate(const MoeView&v,MoeScratch s,float*scratch){
 __shared__ float top_probability[64];__shared__ int top_id[64];
 const int lane=threadIdx.x&31,warp=threadIdx.x/32;
 float p=s.logits[threadIdx.x];float mx=block_max256(p,scratch);p=expf(p-mx);float sum=block_sum256(p,scratch);p/=sum;
 s.probs[threadIdx.x]=p;
 RouterRank a{p,int(threadIdx.x)};
 #pragma unroll
 for(int k=2;k<=32;k<<=1){
  #pragma unroll
  for(int j=k/2;j;j>>=1){RouterRank b{__shfl_xor_sync(0xffffffff,a.probability,j),__shfl_xor_sync(0xffffffff,a.id,j)};a=router_pick(a,b,((lane&j)==0)^((lane&k)!=0));}
 }
 if(lane<8){top_probability[warp*8+lane]=a.probability;top_id[warp*8+lane]=a.id;}
 __syncthreads();
 if(warp==0){
  RouterRank x[2]={{top_probability[lane],top_id[lane]},{top_probability[lane+32],top_id[lane+32]}};
  #pragma unroll
  for(int k=2;k<=64;k<<=1){
   #pragma unroll
   for(int j=k/2;j;j>>=1){
    RouterRank next[2];
    #pragma unroll
    for(int z=0;z<2;z++){
     RouterRank b=j==32?x[z^1]:RouterRank{__shfl_xor_sync(0xffffffff,x[z].probability,j),__shfl_xor_sync(0xffffffff,x[z].id,j)};
     const int index=lane+z*32;next[z]=router_pick(x[z],b,((index&j)==0)^((index&k)!=0));
    }
    x[0]=next[0];x[1]=next[1];
   }
  }
  float total=0;
  #pragma unroll
  for(int j=0;j<TopExperts;j++){
   const float best=__shfl_sync(0xffffffff,x[0].probability,j);const int id=__shfl_sync(0xffffffff,x[0].id,j);
   if(lane==0){s.ids[j]=id;s.weights[j]=best;total+=best;}
  }
  if(lane==0){total=fmaxf(total,6.103515625e-5f);for(int j=0;j<TopExperts;j++)s.weights[j]=s.weights[j]/total*v.weight_scale;}
 }
}
} // namespace halo6000
