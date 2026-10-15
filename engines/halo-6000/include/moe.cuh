#pragma once
#include "device.cuh"
namespace halo6000 {
struct MoeScratch {
 float*logits=nullptr;float*probs=nullptr;int32_t*ids=nullptr;float*weights=nullptr;float*shared_gain=nullptr;
 float*gate=nullptr;float*up=nullptr;float*activation=nullptr;float*down=nullptr;
 ActivationBuffer prepared;
};
struct MoeView {
 const LayerWeights*weights=nullptr;const float*input=nullptr;ActivationView prepared;
 const float*residual=nullptr;float*output=nullptr;float weight_scale=1.f;
};
__device__ __forceinline__ float block_sum256(float x,float*s){
 x=warp_sum(x);if((threadIdx.x&31)==0)s[threadIdx.x/32]=x;__syncthreads();
 float v=(threadIdx.x&31)<8?s[threadIdx.x&31]:0;v=warp_sum(v);__syncthreads();return v;
}
__device__ __forceinline__ float block_max256(float x,float*s){
 x=warp_max(x);if((threadIdx.x&31)==0)s[threadIdx.x/32]=x;__syncthreads();
 float v=(threadIdx.x&31)<8?s[threadIdx.x&31]:-INFINITY;v=warp_max(v);__syncthreads();return v;
}
__device__ __forceinline__ void moe_router(const MoeView&v,MoeScratch s){
 matvec_one_warp_per_cta(v.weights->router,v.prepared,v.input,s.logits);
 if(blockIdx.x==0&&threadIdx.x<32){const auto&w=v.weights->shared_router;float gain=w.format==WeightFormat::F32?float_row(w,v.input,0):quantized_row(w,v.prepared,0);if(threadIdx.x==0)*s.shared_gain=sigmoid(gain);}
}
__device__ __forceinline__ int block_min256(int x,float*s){
 #pragma unroll
 for(int d=16;d;d>>=1)x=min(x,__shfl_xor_sync(0xffffffff,x,d));
 auto*indices=reinterpret_cast<int*>(s+8);if((threadIdx.x&31)==0)indices[threadIdx.x/32]=x;__syncthreads();
 int v=(threadIdx.x&31)<8?indices[threadIdx.x&31]:Experts;
 #pragma unroll
 for(int d=16;d;d>>=1)v=min(v,__shfl_xor_sync(0xffffffff,v,d));
 __syncthreads();return v;
}
// One CTA,256 threads. Softmax is kept explicit: ordering the raw logits instead
// would silently change the oracle's ties after probability rounding.
__device__ __forceinline__ void moe_select(const MoeView&v,MoeScratch s,float*scratch){
 float p=s.logits[threadIdx.x];float mx=block_max256(p,scratch);p=expf(p-mx);float sum=block_sum256(p,scratch);p/=sum;
 s.probs[threadIdx.x]=p;__syncthreads();
 bool selected=false;float total=0;
 #pragma unroll
 for(int j=0;j<TopExperts;j++){
  const float best_probability=block_max256(selected?-INFINITY:p,scratch);
  const int best_id=block_min256(!selected&&p==best_probability?int(threadIdx.x):Experts,scratch);
  selected|=int(threadIdx.x)==best_id;
  if(threadIdx.x==0){s.ids[j]=best_id;s.weights[j]=best_probability;total+=best_probability;}
 }
 if(threadIdx.x==0){total=fmaxf(total,6.103515625e-5f);for(int j=0;j<TopExperts;j++)s.weights[j]=s.weights[j]/total*v.weight_scale;}
}
__device__ __forceinline__ void moe_gate_up(const MoeView&v,MoeScratch s){
 const int lane=threadIdx.x&31,warp=threadIdx.x/32,warps=blockDim.x/32;
 for(int u=blockIdx.x*warps+warp;u<(TopExperts+1)*ExpertHidden;u+=gridDim.x*warps){
  const int selected=u/ExpertHidden,row=u%ExpertHidden;
  const bool shared=selected==TopExperts;const int expert=shared?0:s.ids[selected];
  const MatrixView&gate=shared?v.weights->shared_gate:v.weights->expert_gate;
  const MatrixView&up=shared?v.weights->shared_up:v.weights->expert_up;
  const float g=gate.format==WeightFormat::F32?float_row(gate,v.input,row,expert):quantized_row(gate,v.prepared,row,expert);
  const float h=up.format==WeightFormat::F32?float_row(up,v.input,row,expert):quantized_row(up,v.prepared,row,expert);
  if(lane==0){if(s.gate)s.gate[u]=g;if(s.up)s.up[u]=h;s.activation[u]=silu(g)*h;}
 }
}
__device__ __forceinline__ void moe_prepare(MoeScratch s){
 ActivationBuffer full=s.prepared;full.k=(TopExperts+1)*ExpertHidden;quantize_phase(s.activation,full);
}
template<int Width>
__device__ __forceinline__ float subgroup_row(const MatrixView&w,ActivationView a,const float*x,int row,int expert,int ar){
 if(w.format==WeightFormat::Q2K)return q2k_row<Width>(w,a,row,expert,ar);
 const int lane=threadIdx.x&(Width-1);float value=0;
 if(w.format==WeightFormat::F32){
  const auto*p=reinterpret_cast<const float*>(static_cast<const char*>(w.data)+expert*w.expert_stride+row*w.row_stride);
  for(int c=lane;c<w.k;c+=Width)value=fmaf(p[c],x[ar*w.k+c],value);
 }else for(int b=lane;b<w.k/128;b+=Width)value+=w.format==WeightFormat::Pane2?pane_group(w,a,row,b,expert,ar):q8_group(w,a,row,b,expert,ar);
 #pragma unroll
 for(int d=Width/2;d;d>>=1)value+=__shfl_xor_sync(0xffffffff,value,d,Width);
 return value;
}
__device__ __forceinline__ void moe_down(const MoeView&v,MoeScratch s){
 const int lane=threadIdx.x&31,warp=threadIdx.x/32,warps=blockDim.x/32,selected=lane/4;
 const ActivationView a=s.prepared.view();
 for(int row=blockIdx.x*warps+warp;row<Hidden;row+=gridDim.x*warps){
  const float raw=subgroup_row<4>(v.weights->expert_down,a,s.activation,row,s.ids[selected],selected);
  if(s.down&&(lane&3)==0)s.down[selected*Hidden+row]=raw;
  const float weighted=raw*s.weights[selected];float routed=0;
  #pragma unroll
  for(int j=0;j<TopExperts;j++){const float term=__shfl_sync(0xffffffff,weighted,j*4);routed=__fadd_rn(routed,term);}
  const MatrixView&w=v.weights->shared_down;
  const float shared=w.format==WeightFormat::F32?float_row(w,s.activation+TopExperts*ExpertHidden,row):quantized_row(w,a,row,0,TopExperts);
  if(lane==0){if(s.down)s.down[TopExperts*Hidden+row]=shared;v.output[row]=(routed+shared*(*s.shared_gain))+v.residual[row];}
 }
}
__device__ __forceinline__ void moe_gate_up_q8(const MoeView&v,MoeScratch s,float*shared){
 for(int tile=blockIdx.x;tile<(TopExperts+1)*(ExpertHidden/32);tile+=gridDim.x){
  const int selected=tile/(ExpertHidden/32),row=(tile%(ExpertHidden/32))*32+threadIdx.x/8;
  const bool is_shared=selected==TopExperts;const int expert=is_shared?0:s.ids[selected];
  const MatrixView&gate=is_shared?v.weights->shared_gate:v.weights->expert_gate;
  const MatrixView&up=is_shared?v.weights->shared_up:v.weights->expert_up;
  const float g=subgroup_row<8>(gate,v.prepared,v.input,row,expert,0);
  const float u=subgroup_row<8>(up,v.prepared,v.input,row,expert,0);
  const float value=silu(g)*u;const int index=selected*ExpertHidden+row;
  if((threadIdx.x&7)==0){if(s.gate)s.gate[index]=g;if(s.up)s.up[index]=u;s.activation[index]=value;shared[threadIdx.x/8]=value;}
  __syncthreads();
  if(threadIdx.x<32){const float x=shared[threadIdx.x],amax=warp_max(fabsf(x)),d=amax/127.f;const int group=tile;
   s.prepared.qs[group*32+threadIdx.x]=amax==0?0:int8_t(roundf(x/d));
   if(threadIdx.x==0)s.prepared.scale[group]=__half2float(__float2half(d));
  }
  __syncthreads();
 }
}
} // namespace halo6000
