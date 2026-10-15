#pragma once
// Device helpers shared by the engine kernels (decode.cu) and the scheduled
// token (schedule.cu): workspace views, rotation dispatch, embedding, argmax.
#include "decode.hpp"
#include "device.cuh"
#include "gdn.cuh"
#include "moe.cuh"
#include "attention.cuh"
#include "rotation.cuh"
#include "matvec.cuh"
#include "head.cuh"
#include "router.cuh"
#include <cooperative_groups.h>
namespace halo6000 {
__device__ __forceinline__ GdnView gdn_view(const LayerWeights&l,StateView state,DecodeWorkspace w,int slot){
 const size_t index=size_t(slot)*LinearLayers+linear_index(l.index);
 return {w.qkv,w.z,w.alpha,w.beta,l.conv,l.ssm_a,l.ssm_dt,l.ssm_norm,
 state.gdn+index*ValueHeads*ValueDim*StateDim,state.conv+index*(ConvTaps-1)*ConvChannels,w.mix};
}
__device__ __forceinline__ GdnScratch gdn_scratch(DecodeWorkspace w){return{w.gdn_prepared,w.decay,w.effective_beta,w.core,w.log_decay,w.softplus};}
__device__ __forceinline__ MoeView moe_view(const LayerWeights&l,DecodeWorkspace w){return{&l,w.normalized,w.hidden_q.view(),w.residual,w.hidden,1.f};}
__device__ __forceinline__ MoeScratch moe_scratch(DecodeWorkspace w){return{w.router_logits,w.router_probs,w.route_ids,w.route_weights,w.shared_gain,w.expert_gate,w.expert_up,w.expert_activation,w.expert_down,w.expert_q};}
__device__ __forceinline__ bool rotate_inputs(const LayerWeights&l,DecodeWorkspace w,bool writer,float*shared){
 if(writer){
  if(!(l.output.flags&RotationEnabled))return false;
  rotation_q8_phase(w.mix,w.mix,Inner,rotation_block(l.output.flags),static_cast<const float*>(l.output.aux),shared,w.inner_q);return true;
 }
 const bool routed=l.expert_down.flags&RotationEnabled,sh=l.shared_down.flags&RotationEnabled;
 if(routed){auto q=w.expert_q;q.rows=TopExperts;rotation_q8_rows_phase(w.expert_activation,w.expert_activation,TopExperts,ExpertHidden,rotation_block(l.expert_down.flags),static_cast<const float*>(l.expert_down.aux),0,q,shared);}
 if(sh){auto q=w.expert_q;q.rows=1;q.qs+=TopExperts*q.q_stride;q.scale+=TopExperts*q.scale_stride;float*x=w.expert_activation+TopExperts*ExpertHidden;rotation_q8_phase(x,x,ExpertHidden,rotation_block(l.shared_down.flags),static_cast<const float*>(l.shared_down.aux),shared,q);}
 return routed||sh;
}
__device__ __forceinline__ void embedding_phase(MatrixView w,float*out,int token){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Hidden;i+=gridDim.x*blockDim.x){
  if(w.format==WeightFormat::F32){out[i]=reinterpret_cast<const float*>(static_cast<const char*>(w.data)+token*w.row_stride)[i];}
  else if(w.format==WeightFormat::Q8Grouped128){const size_t b=size_t(token)*(Hidden/128)+i/128;const auto*p=static_cast<const char*>(w.data)+b*136;out[i]=float(reinterpret_cast<const int8_t*>(p)[i%128])*half_value(reinterpret_cast<const uint16_t*>(p+128)+(i%128)/32);}
  else if(w.format==WeightFormat::Q2K){const auto&b=q2k_weights(w,token)[i/256];const int e=i%256,part=e/128,h=(e%128)/32,offset=e%32,sc=b.scales[part*8+offset/16+2*h],code=(b.qs[part*32+offset]>>(2*h))&3;out[i]=half_value(&b.d)*float((sc&15)*code)-half_value(&b.dmin)*float(sc>>4);}
  else{const size_t b=size_t(token)*(Hidden/128)+i/128;const auto*p=static_cast<const char*>(w.data)+(b>>4)*544;const uint32_t word=reinterpret_cast<const uint32_t*>(p+(b&15)*32)[(i%128)/16];const int e=i%16,code=(word>>(4*(e&7)+(e>>3)))&5;out[i]=float((code&1)?(code&4?-1:1):0)*half_value(reinterpret_cast<const uint16_t*>(p+512)+(b&15));}
 }
}
__device__ __forceinline__ AttentionBuffers attention_buffers(DecodeWorkspace w){return{w.qkv,w.key,w.value,w.query,w.scores,w.mix};}
__device__ __forceinline__ void warp_argmax(float&value,int&id){
 #pragma unroll
 for(int d=16;d;d>>=1){const float v=__shfl_xor_sync(0xffffffff,value,d);const int i=__shfl_xor_sync(0xffffffff,id,d);if(v>value||(v==value&&i<id)){value=v;id=i;}}
}
__device__ __forceinline__ void block_argmax(float value,int id,float*shared,float*out,int*outid){
 warp_argmax(value,id);if((threadIdx.x&31)==0){shared[threadIdx.x/32]=value;reinterpret_cast<int*>(shared+8)[threadIdx.x/32]=id;}__syncthreads();
 if(threadIdx.x==0){float best=-INFINITY;int bestid=Vocab;for(int j=0;j<8;j++){float v=shared[j];int i=reinterpret_cast<int*>(shared+8)[j];if(v>best||(v==best&&i<bestid)){best=v;bestid=i;}}*out=best;*outid=bestid;}
}
} // namespace halo6000
