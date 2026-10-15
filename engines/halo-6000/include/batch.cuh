#pragma once
#include "batch.hpp"
#include "phases.cuh"
#include "matmul.cuh"
#include "batch-moe.cuh"
namespace halo6000 {
__device__ __forceinline__ int batch_block(){return blockIdx.y*gridDim.x+blockIdx.x;}
__device__ __forceinline__ int batch_blocks(){return gridDim.x*gridDim.y;}
__device__ __forceinline__ PreparedActivationView batch_activation(PreparedActivationView a,int rows){a.columns=rows;a.input_experts=1;return a;}
__device__ __forceinline__ void batch_pack(PreparedActivationView a,int rows){
 a=batch_activation(a,rows);uint64_t count=matmul_activation_bytes(rows,a.source.k)/4;
 for(uint64_t i=uint64_t(batch_block())*blockDim.x+threadIdx.x;i<count;i+=uint64_t(batch_blocks())*blockDim.x)
  const_cast<uint32_t*>(a.operands)[i]=prepare_matmul_activation_word(a,i);
}
template<WeightFormat F,int Groups>
__device__ __forceinline__ void batch_matrix_mma(const BatchArgs&a,const PreparedMatmulView&w,PreparedActivationView input,float*out,float*shared){
 input=batch_activation(input,a.workspace.rows);int rt=w.source.n/16,ct=(input.columns+7)/8;
 for(int tile=batch_block();tile<rt*ct;tile+=batch_blocks())matmul_q8_splitk_phase<F,Groups>(w,input,out,tile%rt,tile/rt,0,0,1,shared);
}
__device__ __forceinline__ void batch_matrix(const BatchArgs&a,const PreparedMatmulView&w,PreparedActivationView input,const float*x,float*out,float*shared){
 const int k=w.source.k;bool mma=w.operands&&w.weight_scales&&(k==512||k==2048||k==4096);
 if(mma){
  if(w.source.format==WeightFormat::Pane2){if(k==512)batch_matrix_mma<WeightFormat::Pane2,4>(a,w,input,out,shared);else if(k==2048)batch_matrix_mma<WeightFormat::Pane2,16>(a,w,input,out,shared);else batch_matrix_mma<WeightFormat::Pane2,32>(a,w,input,out,shared);}
  else {if(k==512)batch_matrix_mma<WeightFormat::Q8Grouped128,4>(a,w,input,out,shared);else if(k==2048)batch_matrix_mma<WeightFormat::Q8Grouped128,16>(a,w,input,out,shared);else batch_matrix_mma<WeightFormat::Q8Grouped128,32>(a,w,input,out,shared);}
 }else{
  int r=blockIdx.y;if(a.workspace.positions[r]<0)return;ActivationView local=input.source;local.qs+=r*local.q_stride;local.scale+=r*local.scale_stride;local.rows=1;
  matvec_exact_phase<16>(w.source,local,x+uint64_t(r)*k,out+uint64_t(r)*w.source.n);
 }
}
template<uint32_t K>H6_HOST_DEVICE constexpr int batch_shared_floats(){
 constexpr uint32_t large=(1u<<int(EpochKind::OutProj))|(1u<<int(EpochKind::RotWriter))|(1u<<int(EpochKind::RotDown));
 constexpr uint32_t mma=(1u<<int(EpochKind::GdnProj))|(1u<<int(EpochKind::AttnProj))|(1u<<int(EpochKind::Head))|(1u<<int(EpochKind::GateUp));
 return (K&large)?4096:(K&mma)?2048:(K&(1u<<int(EpochKind::Down)))?512:32;
}
// All row-local math calls the unchanged canonical helpers. Dense MMA is a
// collective over flattened x*y CTAs; the row dimension still equals active rows.
// Epoch boundaries, not this dispatcher, own grid-wide publication.
template<uint32_t KindMask=0xffffffffu>
__device__ __forceinline__ void run_batch_epoch(const BatchArgs&a,Epoch ep,float*shared){
 const BatchView&b=a.workspace;const int row=blockIdx.y;const DecodeWorkspace&w=b.row_views[row];
 const ModelWeights&m=*a.model;const LayerWeights&l=m.layer[ep.layer<0?0:ep.layer];
 const auto&prepared=*m.prepared;const PreparedLayer&p=prepared.layer[ep.layer<0?0:ep.layer];
 if(!(KindMask&(1u<<ep.kind)))return;
 if(ep.kind==int(BatchEpochKind::PackHidden)){if constexpr(KindMask&(1u<<int(BatchEpochKind::PackHidden)))batch_pack(b.base.hidden_mma,b.rows);return;}
 if(ep.kind==int(BatchEpochKind::PackInner)){if constexpr(KindMask&(1u<<int(BatchEpochKind::PackInner)))batch_pack(b.base.inner_mma,b.rows);return;}
 if(ep.kind==int(BatchEpochKind::BuildBuckets)){if constexpr(KindMask&(1u<<int(BatchEpochKind::BuildBuckets)))batch_build_buckets(b);return;}
 if(ep.kind==int(BatchEpochKind::PublishBuckets)){if constexpr(KindMask&(1u<<int(BatchEpochKind::PublishBuckets)))batch_publish_buckets(b,ep.layer);return;}
 if(ep.kind==int(BatchEpochKind::GateUpQuant)){if constexpr(KindMask&(1u<<int(BatchEpochKind::GateUpQuant)))batch_bucket_gate_up_quant(b);return;}
 if(ep.kind==int(BatchEpochKind::DownCombine)){if constexpr(KindMask&(1u<<int(BatchEpochKind::DownCombine)))batch_bucket_down_combine(b);return;}
 if(b.expert_policy==BatchExpertPolicy::BucketedExact){if(ep.kind==int(EpochKind::GateUp)){if constexpr(KindMask&(1u<<int(EpochKind::GateUp)))batch_bucket_gate_up(a,p,shared);return;}if(ep.kind==int(EpochKind::Down)){if constexpr(KindMask&(1u<<int(EpochKind::Down)))batch_bucket_down(a,p,shared);return;}}
 #define BATCH_CASE(name) case EpochKind::name: if constexpr((KindMask&(1u<<int(EpochKind::name)))==0)return;
 switch(EpochKind(ep.kind)){
 BATCH_CASE(GdnProj)
  batch_matrix(a,p.qkv,b.base.hidden_mma,b.base.normalized,b.base.qkv,shared);
  batch_matrix(a,p.z,b.base.hidden_mma,b.base.normalized,b.base.z,shared);
  batch_matrix(a,p.alpha,b.base.hidden_mma,b.base.normalized,b.base.alpha,shared);
  batch_matrix(a,p.beta,b.base.hidden_mma,b.base.normalized,b.base.beta,shared);return;
 BATCH_CASE(AttnProj)
  batch_matrix(a,p.q,b.base.hidden_mma,b.base.normalized,b.base.qkv,shared);
  batch_matrix(a,p.k,b.base.hidden_mma,b.base.normalized,b.base.key,shared);
  batch_matrix(a,p.v,b.base.hidden_mma,b.base.normalized,b.base.value,shared);return;
 BATCH_CASE(OutProj)batch_matrix(a,p.output,b.base.inner_mma,b.base.mix,b.base.projected,shared);return;
 BATCH_CASE(Head)batch_matrix(a,prepared.head,b.base.hidden_mma,b.base.normalized,b.base.logits,shared);return;
 default:break;
 }
 const int slot=b.slots[row];
 if(ep.kind==int(EpochKind::Embed)){
  if constexpr((KindMask&(1u<<int(EpochKind::Embed)))==0)return;
  int position=a.state.positions[slot];
  if(position<0||position>=a.state.context){if(blockIdx.x==0&&threadIdx.x==0){b.positions[row]=-1;*w.token=-1;atomicCAS(b.error,0,1);}return;}
  if(blockIdx.x==0&&threadIdx.x==0)b.positions[row]=position;
  int token=b.tokens[row];if(token==-1)token=*w.token;
  if(token<0||token>=Vocab){if(blockIdx.x==0&&threadIdx.x==0){b.positions[row]=-1;atomicCAS(b.error,0,2);}return;}
  embedding_phase(m.embedding,w.hidden,token);return;
 }
 int position=b.positions[row];if(position<0)return;
 switch(EpochKind(ep.kind)){
 BATCH_CASE(InNorm)if(blockIdx.x==0){norm_phase<Hidden>(w.hidden,l.input_norm,w.normalized,shared);quantize_cta(w.normalized,w.hidden_q);}break;
 BATCH_CASE(ConvPrep)gdn_conv_prepare(gdn_view(l,a.state,w,slot),gdn_scratch(w));break;
 BATCH_CASE(Recur)gdn_recur(gdn_view(l,a.state,w,slot),gdn_scratch(w));break;
 BATCH_CASE(FinishQ)gdn_finish_q8(gdn_view(l,a.state,w,slot),gdn_scratch(w),w.inner_q);break;
 BATCH_CASE(RotWriter)rotate_inputs(l,w,true,shared);break;
 BATCH_CASE(ResNorm)if(blockIdx.x==0){for(int i=threadIdx.x;i<Hidden;i+=blockDim.x)w.residual[i]=w.hidden[i]+w.projected[i];__syncthreads();norm_phase<Hidden>(w.residual,l.post_norm,w.normalized,shared);quantize_cta(w.normalized,w.hidden_q);}break;
 BATCH_CASE(Router)if(l.router.format==WeightFormat::F32&&l.shared_router.format==WeightFormat::F32)moe_router_fixed_f32<2>(moe_view(l,w),moe_scratch(w));else moe_router(moe_view(l,w),moe_scratch(w));break;
 BATCH_CASE(Select)if(blockIdx.x==0)moe_select_candidate(moe_view(l,w),moe_scratch(w),shared);break;
 BATCH_CASE(GateUp)moe_gate_up_q8(moe_view(l,w),moe_scratch(w),shared);break;
 BATCH_CASE(RotDown)rotate_inputs(l,w,false,shared);break;
 BATCH_CASE(Down)moe_down(moe_view(l,w),moe_scratch(w));break;
 BATCH_CASE(NormRope)attention_norm_rope_append_phase(l,a.state,attention_index(l.index),slot,position,attention_buffers(w));break;
 BATCH_CASE(Scores)attention_scores_phase(a.state,attention_index(l.index),slot,position,attention_buffers(w));break;
 BATCH_CASE(Softmax)attention_softmax_phase(a.state,position,attention_buffers(w));break;
 BATCH_CASE(ValueGate)attention_value_gate_q8_phase(a.state,attention_index(l.index),slot,position,attention_buffers(w),w.inner_q);break;
 BATCH_CASE(FinalNorm)if(blockIdx.x==0){norm_phase<Hidden>(w.hidden,m.final_norm,w.normalized,shared);quantize_cta(w.normalized,w.hidden_q);}break;
 BATCH_CASE(ArgPart){float best=-INFINITY;int id=Vocab;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Vocab;i+=gridDim.x*blockDim.x){float v=w.logits[i];if(v>best||(v==best&&i<id)){best=v;id=i;}}block_argmax(best,id,shared,w.max_values+blockIdx.x,w.max_ids+blockIdx.x);}break;
 BATCH_CASE(ArgFinish)if(blockIdx.x==0){float best=-INFINITY;int id=Vocab;for(int i=threadIdx.x;i<gridDim.x;i+=blockDim.x){float v=w.max_values[i];int ix=w.max_ids[i];if(v>best||(v==best&&ix<id)){best=v;id=ix;}}block_argmax(best,id,shared,w.max_values,w.token);if(threadIdx.x==0)a.state.positions[slot]=position+1;}break;
 default:break;
 }
 #undef BATCH_CASE
}
}
