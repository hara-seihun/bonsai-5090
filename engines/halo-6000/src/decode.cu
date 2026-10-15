#include "phases.cuh"
namespace halo6000 {
template<int Lanes>
__device__ __forceinline__ void projection_exact(const LayerWeights&l,DecodeWorkspace w){
 const int groups=blockDim.x/Lanes,lane=threadIdx.x&(Lanes-1);const bool gdn=l.kind==LayerKind::GDN;
 const int total=gdn?ConvChannels+Inner+2*ValueHeads:2*QueryHeads*HeadDim+2*KVHeads*HeadDim;
 for(int unit=blockIdx.x*groups+threadIdx.x/Lanes;unit<total;unit+=gridDim.x*groups){
  const MatrixView*matrix;float*out;int row=unit;
  if(gdn){if(row<ConvChannels){matrix=&l.qkv;out=w.qkv;}else if((row-=ConvChannels)<Inner){matrix=&l.z;out=w.z;}else if((row-=Inner)<ValueHeads){matrix=&l.alpha;out=w.alpha;}else{row-=ValueHeads;matrix=&l.beta;out=w.beta;}}
  else{if(row<2*QueryHeads*HeadDim){matrix=&l.q;out=w.qkv;}else if((row-=2*QueryHeads*HeadDim)<KVHeads*HeadDim){matrix=&l.k;out=w.key;}else{row-=KVHeads*HeadDim;matrix=&l.v;out=w.value;}}
  MatrixView fixed=*matrix;fixed.k=Hidden;float value=row_exact<Lanes>(fixed,w.hidden_q.view(),w.normalized,row);if(!lane)out[row]=value;
 }
}
template<int Lanes=32>
__device__ __forceinline__ void layer_phase(const LayerWeights&l,StateView state,DecodeWorkspace w,int slot,LayerPhase phase,float*shared){
 switch(phase){
 case LayerPhase::InputNorm:if(blockIdx.x==0)norm_phase<Hidden>(w.hidden,l.input_norm,w.normalized,shared);break;
 case LayerPhase::InputQuant:quantize_phase(w.normalized,w.hidden_q);break;
 case LayerPhase::Projection:
  if constexpr(Lanes==32){
   matvec_phase(l.qkv,w.hidden_q.view(),w.normalized,w.qkv);matvec_phase(l.z,w.hidden_q.view(),w.normalized,w.z);
   if(threadIdx.x<32)for(int unit=blockIdx.x;unit<2*ValueHeads;unit+=gridDim.x){const bool beta=unit>=ValueHeads;const auto&matrix=beta?l.beta:l.alpha;const int row=unit%ValueHeads;const float value=matrix.format==WeightFormat::F32?float_row(matrix,w.normalized,row):quantized_row(matrix,w.hidden_q.view(),row);if(threadIdx.x==0)(beta?w.beta:w.alpha)[row]=value;}
  }else projection_exact<Lanes>(l,w);break;
 case LayerPhase::Conv:gdn_conv(gdn_view(l,state,w,slot),gdn_scratch(w));break;
 case LayerPhase::PrepareGdn:gdn_prepare(gdn_view(l,state,w,slot),gdn_scratch(w));break;
 case LayerPhase::Recurrence:gdn_recur(gdn_view(l,state,w,slot),gdn_scratch(w));break;
 case LayerPhase::FinishGdn:gdn_finish(gdn_view(l,state,w,slot),gdn_scratch(w));break;
 case LayerPhase::InnerQuant:quantize_phase(w.mix,w.inner_q);break;
 case LayerPhase::OutputProjection:if constexpr(Lanes==32)matvec_phase(l.output,w.inner_q.view(),w.mix,w.projected);else{MatrixView fixed=l.output;fixed.k=Inner;fixed.n=Hidden;matvec_exact_phase<Lanes>(fixed,w.inner_q.view(),w.mix,w.projected);}break;
 case LayerPhase::ResidualNorm:
  if(blockIdx.x==0){for(int i=threadIdx.x;i<Hidden;i+=blockDim.x)w.residual[i]=w.hidden[i]+w.projected[i];__syncthreads();norm_phase<Hidden>(w.residual,l.post_norm,w.normalized,shared);}break;
 case LayerPhase::PostQuant:quantize_phase(w.normalized,w.hidden_q);break;
 case LayerPhase::Router:
  if constexpr(Lanes==32)moe_router(moe_view(l,w),moe_scratch(w));
  else if(l.router.format==WeightFormat::F32&&l.shared_router.format==WeightFormat::F32)moe_router_fixed_f32<2>(moe_view(l,w),moe_scratch(w));
  else moe_router(moe_view(l,w),moe_scratch(w));break;
 case LayerPhase::Select:if(blockIdx.x==0){if constexpr(Lanes==32)moe_select(moe_view(l,w),moe_scratch(w),shared);else moe_select_candidate(moe_view(l,w),moe_scratch(w),shared);}break;
 case LayerPhase::GateUp:moe_gate_up(moe_view(l,w),moe_scratch(w));break;
 case LayerPhase::ExpertQuant:moe_prepare(moe_scratch(w));break;
 case LayerPhase::Down:moe_down(moe_view(l,w),moe_scratch(w));break;
 }
}
__device__ __forceinline__ bool skipped_coarse_phase(int p){return p==int(LayerPhase::InputQuant)||p==int(LayerPhase::PrepareGdn)||p==int(LayerPhase::InnerQuant)||p==int(LayerPhase::PostQuant)||p==int(LayerPhase::ExpertQuant);}
template<int Lanes=32>
__device__ __forceinline__ void coarse_phase(const LayerWeights&l,StateView state,DecodeWorkspace w,int slot,LayerPhase phase,float*shared){
 switch(phase){
 case LayerPhase::InputNorm:if(blockIdx.x==0){norm_phase<Hidden>(w.hidden,l.input_norm,w.normalized,shared);quantize_cta(w.normalized,w.hidden_q);}break;
 case LayerPhase::Conv:gdn_conv_prepare(gdn_view(l,state,w,slot),gdn_scratch(w));break;
 case LayerPhase::FinishGdn:gdn_finish_q8(gdn_view(l,state,w,slot),gdn_scratch(w),w.inner_q);break;
 case LayerPhase::ResidualNorm:if(blockIdx.x==0){for(int i=threadIdx.x;i<Hidden;i+=blockDim.x)w.residual[i]=w.hidden[i]+w.projected[i];__syncthreads();norm_phase<Hidden>(w.residual,l.post_norm,w.normalized,shared);quantize_cta(w.normalized,w.hidden_q);}break;
 case LayerPhase::GateUp:moe_gate_up_q8(moe_view(l,w),moe_scratch(w),shared);break;
 default:layer_phase<Lanes>(l,state,w,slot,phase,shared);break;
 }
}
__global__ void rotation_control_kernel(const ModelWeights*model,DecodeWorkspace w,int layer,bool writer){__shared__ float shared[4096];rotate_inputs(model->layer[layer],w,writer,shared);}
template<bool Rotate,int Lanes=32>
__global__ void gdn_layer_kernel(const ModelWeights*model,StateView state,DecodeWorkspace w,int layer,int slot){
 __shared__ float shared[Rotate?4096:32];const auto grid=cooperative_groups::this_grid();
 for(int p=int(LayerPhase::InputNorm);p<=int(LayerPhase::Down);p++){
  if(skipped_coarse_phase(p))continue;
  coarse_phase<Lanes>(model->layer[layer],state,w,slot,LayerPhase(p),shared);
  if(p<int(LayerPhase::Down))grid.sync();
  if constexpr(Rotate){if((p==int(LayerPhase::FinishGdn)||p==int(LayerPhase::GateUp))&&rotate_inputs(model->layer[layer],w,p==int(LayerPhase::FinishGdn),shared))grid.sync();}
 }
}
__global__ void gdn_phase_kernel(const ModelWeights*model,StateView state,DecodeWorkspace w,int layer,int slot,LayerPhase phase){__shared__ float shared[32];layer_phase(model->layer[layer],state,w,slot,phase,shared);}
__global__ void embedding_kernel(MatrixView w,float*out,int token){embedding_phase(w,out,token);}
template<int Lanes=32>
__device__ __forceinline__ void attention_epoch(const LayerWeights&l,StateView state,DecodeWorkspace w,int slot,int position,int phase,float*shared){
 const auto b=attention_buffers(w);const int index=attention_index(l.index);
 switch(phase){
 case 0:coarse_phase<Lanes>(l,state,w,slot,LayerPhase::InputNorm,shared);break;
 case 1:if constexpr(Lanes==32)attention_project_phase(l,w.hidden_q.view(),w.normalized,b);else projection_exact<Lanes>(l,w);break;
 case 2:attention_norm_rope_append_phase(l,state,index,slot,position,b);break;
 case 3:attention_append_phase(state,index,slot,position,b);break;
 case 4:attention_scores_phase(state,index,slot,position,b);break;
 case 5:attention_softmax_phase(state,position,b);break;
 case 6:attention_value_gate_q8_phase(state,index,slot,position,b,w.inner_q);break;
 case 7:quantize_phase(w.mix,w.inner_q);break;
 case 8:coarse_phase<Lanes>(l,state,w,slot,LayerPhase::OutputProjection,shared);break;
 case 9:coarse_phase<Lanes>(l,state,w,slot,LayerPhase::ResidualNorm,shared);break;
 case 10:coarse_phase<Lanes>(l,state,w,slot,LayerPhase::Router,shared);break;
 case 11:coarse_phase<Lanes>(l,state,w,slot,LayerPhase::Select,shared);break;
 case 12:coarse_phase<Lanes>(l,state,w,slot,LayerPhase::GateUp,shared);break;
 case 13:coarse_phase<Lanes>(l,state,w,slot,LayerPhase::Down,shared);break;
 }
}
__device__ __forceinline__ void timestamp(DecodeWorkspace w,int i){if(w.timestamps&&blockIdx.x==0&&threadIdx.x==0){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));w.timestamps[i]=t;}}
template<bool Rotate,int Lanes=32>
__global__ void token_kernel(const ModelWeights*model,StateView state,DecodeWorkspace w,int token,int position,int slot){
 __shared__ float shared[Rotate?4096:32];const auto grid=cooperative_groups::this_grid();if(token==-1)token=*w.token;
 timestamp(w,0);embedding_phase(model->embedding,w.hidden,token);grid.sync();timestamp(w,1);
 for(int layer=0;layer<Layers;layer++){
  const auto&l=model->layer[layer];
  if(l.kind==LayerKind::GDN){for(int p=0;p<=int(LayerPhase::Down);p++){
   if(skipped_coarse_phase(p))continue;coarse_phase<Lanes>(l,state,w,slot,LayerPhase(p),shared);grid.sync();
   if constexpr(Rotate){if((p==int(LayerPhase::FinishGdn)||p==int(LayerPhase::GateUp))&&rotate_inputs(l,w,p==int(LayerPhase::FinishGdn),shared))grid.sync();}
   timestamp(w,2+layer*16+p);
  }}else{for(int p=0;p<14;p++){
   if(p==3||p==7)continue;attention_epoch<Lanes>(l,state,w,slot,position,p,shared);grid.sync();
   if constexpr(Rotate){if((p==6||p==12)&&rotate_inputs(l,w,p==6,shared))grid.sync();}
   timestamp(w,2+layer*16+p);
  }}
 }
 if(blockIdx.x==0){norm_phase<Hidden>(w.hidden,model->final_norm,w.normalized,shared);quantize_cta(w.normalized,w.hidden_q);}grid.sync();timestamp(w,642);
 if constexpr(Lanes==32)matvec_phase(model->head,w.hidden_q.view(),w.normalized,w.logits);
 else if(model->head.format==WeightFormat::Q8Grouped128)head_q8_phase<16>(model->head,w.hidden_q.view(),w.logits);
 else matvec_exact_phase<Lanes>(model->head,w.hidden_q.view(),w.normalized,w.logits);
 grid.sync();timestamp(w,643);
 float best=-INFINITY;int bestid=Vocab;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Vocab;i+=gridDim.x*blockDim.x){const float v=w.logits[i];if(v>best||(v==best&&i<bestid)){best=v;bestid=i;}}
 block_argmax(best,bestid,shared,w.max_values+blockIdx.x,w.max_ids+blockIdx.x);grid.sync();timestamp(w,644);
 if(blockIdx.x==0){best=-INFINITY;bestid=Vocab;for(int i=threadIdx.x;i<gridDim.x;i+=blockDim.x){const float v=w.max_values[i];const int id=w.max_ids[i];if(v>best||(v==best&&id<bestid)){best=v;bestid=id;}}block_argmax(best,bestid,shared,w.max_values,w.token);if(threadIdx.x==0)state.positions[slot]=position+1;}timestamp(w,645);
}
Status run_token(const DeviceModel&m,DecodeWorkspace w,int token,int position,int slot,int blocks,cudaStream_t stream){
 if(token< -1||token>=Vocab)return Error{ErrorCode::Geometry,"token outside vocabulary"};if(slot<0||slot>=m.state.slots||position<0||position>=m.state.context||w.context<m.state.context||blocks<1||blocks>1024)return Error{ErrorCode::Capacity,"token state/grid outside capacity (grid1..1024, subject to cooperative residency)"};
 if(m.matvec_lanes!=8&&m.matvec_lanes!=16&&m.matvec_lanes!=32)return Error{ErrorCode::Unsupported,"row lanes8/16/32 required"};
 if(m.matvec_lanes!=32&&!m.compressed_rows_supported)return Error{ErrorCode::Unsupported,"compressed row ownership requires F32/PANE/Q8 projections and head"};
 const auto kernel=m.matvec_lanes==8?(m.has_rotation?token_kernel<true,8>:token_kernel<false,8>):m.matvec_lanes==16?(m.has_rotation?token_kernel<true,16>:token_kernel<false,16>):(m.has_rotation?token_kernel<true,32>:token_kernel<false,32>);
 const int variant=(m.has_rotation?3:0)+(m.matvec_lanes==8?0:m.matvec_lanes==16?1:2);
 int active=m.token_residency[variant].load(std::memory_order_relaxed);if(!active){auto s=cuda_status(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,256,0),"token occupancy");if(!s.ok())return s;m.token_residency[variant].store(active,std::memory_order_relaxed);}if(blocks>active*m.multiprocessors)return Error{ErrorCode::Capacity,"whole token grid exceeds compiled cooperative residency"};
 const ModelWeights*model=m.device_weights;StateView state=m.state;void*args[]={&model,&state,&w,&token,&position,&slot};return cuda_status(cudaLaunchCooperativeKernel((void*)kernel,blocks,256,args,0,stream),"whole token");
}
Status embed(const DeviceModel&m,DecodeWorkspace w,int token,cudaStream_t stream){if(token<0||token>=Vocab)return Error{ErrorCode::Geometry,"token outside vocabulary"};embedding_kernel<<<8,256,0,stream>>>(m.weights.embedding,w.hidden,token);return cuda_status(cudaGetLastError(),"embedding");}
static Status layer_check(const DeviceModel&m,DecodeWorkspace w,int layer,int slot,int blocks){
 if(layer<0||layer>=Layers||attention_layer(layer))return Error{ErrorCode::Geometry,"GDN layer required"};if(slot<0||slot>=m.state.slots||blocks<1)return Error{ErrorCode::Capacity,"slot/grid out of range"};if(w.hidden_q.group!=32||w.inner_q.group!=32||w.expert_q.group!=32)return Error{ErrorCode::Unsupported,"only strict32 activation scale is implemented"};return success();
}
Status run_gdn_layer(const DeviceModel&m,DecodeWorkspace w,int layer,int slot,int blocks,cudaStream_t stream){
 if(m.matvec_lanes!=8&&m.matvec_lanes!=16&&m.matvec_lanes!=32)return Error{ErrorCode::Unsupported,"row lanes8/16/32 required"};
 if(m.matvec_lanes!=32&&!m.compressed_rows_supported)return Error{ErrorCode::Unsupported,"compressed row ownership requires F32/PANE/Q8 projections and head"};
 const auto kernel=m.matvec_lanes==8?(m.has_rotation?gdn_layer_kernel<true,8>:gdn_layer_kernel<false,8>):m.matvec_lanes==16?(m.has_rotation?gdn_layer_kernel<true,16>:gdn_layer_kernel<false,16>):(m.has_rotation?gdn_layer_kernel<true,32>:gdn_layer_kernel<false,32>);
 auto s=layer_check(m,w,layer,slot,blocks);if(!s.ok())return s;const int variant=(m.has_rotation?3:0)+(m.matvec_lanes==8?0:m.matvec_lanes==16?1:2);int active=m.layer_residency[variant].load(std::memory_order_relaxed);if(!active){s=cuda_status(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,256,0),"layer occupancy");if(!s.ok())return s;m.layer_residency[variant].store(active,std::memory_order_relaxed);}if(blocks>active*m.multiprocessors)return Error{ErrorCode::Capacity,"persistent layer grid exceeds compiled cooperative residency"};
 const ModelWeights*model=m.device_weights;StateView state=m.state;void*args[]={&model,&state,&w,&layer,&slot};return cuda_status(cudaLaunchCooperativeKernel((void*)kernel,blocks,256,args,0,stream),"persistent GDN layer");
}
Status run_gdn_phase(const DeviceModel&m,DecodeWorkspace w,int layer,int slot,LayerPhase phase,int blocks,cudaStream_t stream){
 auto s=layer_check(m,w,layer,slot,blocks);if(!s.ok())return s;gdn_phase_kernel<<<blocks,256,0,stream>>>(m.device_weights,m.state,w,layer,slot,phase);s=cuda_status(cudaGetLastError(),"GDN control phase");if(!s.ok())return s;
 if(m.has_rotation&&(phase==LayerPhase::InnerQuant||phase==LayerPhase::ExpertQuant)){rotation_control_kernel<<<blocks,256,0,stream>>>(m.device_weights,w,layer,phase==LayerPhase::InnerQuant);return cuda_status(cudaGetLastError(),"rotation control phase");}return success();
}
} // namespace halo6000
