#include "batch.cuh"
#include <algorithm>
#include <string>
namespace halo6000 {
BatchWorkspace::~BatchWorkspace(){for(auto i=allocations_.rbegin();i!=allocations_.rend();++i)cudaFree(*i);if(host_inputs_)cudaFreeHost(host_inputs_);}
Result<std::unique_ptr<BatchWorkspace>>BatchWorkspace::create(const DeviceModel&m,int capacity,BatchExpertPolicy experts){
 if(capacity<1||capacity>MaxRows||capacity>m.state.slots)return Error{ErrorCode::Capacity,"batch capacity1..min(128,state.slots) required"};
 if(!m.weights.prepared)return Error{ErrorCode::Unsupported,"prepare_matmul(false) before creating batch workspace"};
 if(!m.compressed_rows_supported)return Error{ErrorCode::Unsupported,"batch dense sources require canonical F32/PANE/Q8 support"};
 if(experts==BatchExpertPolicy::BucketedExact)for(const auto&p:m.prepared.layer){
  if(!p.expert_gate.operands||!p.expert_up.operands||!p.shared_gate.operands||!p.shared_up.operands||!p.shared_down.operands||(!p.expert_down.operands&&p.expert_down.source.format!=WeightFormat::Q2K))return Error{ErrorCode::Unsupported,"bucketed experts require prepare_matmul(true), PANE/Q8 GU/shared and PANE/Q8/native Q2_K down"};
 }
 auto w=std::make_unique<BatchWorkspace>();w->owner_=m.device_weights;w->model_slots_=m.state.slots;w->view.capacity=capacity;w->view.expert_policy=experts;w->view.base.context=m.state.context;w->host_rows_.resize(capacity);
 for(auto&r:w->host_rows_)r.context=m.state.context;
 auto allocate=[&](size_t bytes)->Result<void*>{void*p=nullptr;auto s=cuda_status(cudaMalloc(&p,bytes),"allocate batch workspace");if(!s.ok())return s.error();w->allocations_.push_back(p);w->bytes+=bytes;return p;};
 auto rows=[&](auto member,size_t width)->Status{
  using Pointer=std::remove_reference_t<decltype(w->view.base.*member)>;using T=std::remove_pointer_t<Pointer>;
  auto p=allocate(size_t(capacity)*width*sizeof(T));if(!p.ok())return p.error();auto*base=static_cast<T*>(p.value());w->view.base.*member=base;
  for(int r=0;r<capacity;r++)w->host_rows_[r].*member=base+size_t(r)*width;return success();
 };
 #define B_ALLOC(call) do{auto s=(call);if(!s.ok())return s.error();}while(0)
 for(auto p:{&DecodeWorkspace::hidden,&DecodeWorkspace::residual,&DecodeWorkspace::normalized,&DecodeWorkspace::projected})B_ALLOC(rows(p,Hidden));
 B_ALLOC(rows(&DecodeWorkspace::qkv,ConvChannels));B_ALLOC(rows(&DecodeWorkspace::z,Inner));B_ALLOC(rows(&DecodeWorkspace::alpha,ValueHeads));B_ALLOC(rows(&DecodeWorkspace::beta,ValueHeads));B_ALLOC(rows(&DecodeWorkspace::mix,Inner));
 B_ALLOC(rows(&DecodeWorkspace::logits,Vocab));B_ALLOC(rows(&DecodeWorkspace::token,1));
 for(auto p:{&DecodeWorkspace::router_logits,&DecodeWorkspace::router_probs})B_ALLOC(rows(p,Experts));
 B_ALLOC(rows(&DecodeWorkspace::route_ids,TopExperts));B_ALLOC(rows(&DecodeWorkspace::route_weights,TopExperts));B_ALLOC(rows(&DecodeWorkspace::shared_gain,1));
 for(auto p:{&DecodeWorkspace::expert_gate,&DecodeWorkspace::expert_up,&DecodeWorkspace::expert_activation})B_ALLOC(rows(p,(TopExperts+1)*ExpertHidden));
 B_ALLOC(rows(&DecodeWorkspace::expert_down,(TopExperts+1)*Hidden));B_ALLOC(rows(&DecodeWorkspace::phase_scratch,32));B_ALLOC(rows(&DecodeWorkspace::gdn_prepared,ConvChannels));
 for(auto p:{&DecodeWorkspace::log_decay,&DecodeWorkspace::softplus,&DecodeWorkspace::decay,&DecodeWorkspace::effective_beta})B_ALLOC(rows(p,ValueHeads));
 B_ALLOC(rows(&DecodeWorkspace::core,Inner));B_ALLOC(rows(&DecodeWorkspace::max_values,1024));B_ALLOC(rows(&DecodeWorkspace::max_ids,1024));
 B_ALLOC(rows(&DecodeWorkspace::key,KVHeads*HeadDim));B_ALLOC(rows(&DecodeWorkspace::value,KVHeads*HeadDim));B_ALLOC(rows(&DecodeWorkspace::query,QueryHeads*HeadDim));B_ALLOC(rows(&DecodeWorkspace::scores,size_t(QueryHeads)*m.state.context));
 auto activation=[&](ActivationBuffer DecodeWorkspace::*member,int k,int experts)->Status{
  ActivationBuffer a;a.k=k;a.rows=capacity*experts;a.group=32;a.q_stride=k;a.scale_stride=k/32;
  auto q=allocate(size_t(a.rows)*k);if(!q.ok())return q.error();auto s=allocate(size_t(a.rows)*(k/32)*4);if(!s.ok())return s.error();a.qs=static_cast<int8_t*>(q.value());a.scale=static_cast<float*>(s.value());w->view.base.*member=a;
  for(int r=0;r<capacity;r++){auto local=a;local.rows=experts;local.qs+=size_t(r)*experts*k;local.scale+=size_t(r)*experts*(k/32);w->host_rows_[r].*member=local;}return success();
 };
 B_ALLOC(activation(&DecodeWorkspace::hidden_q,Hidden,1));B_ALLOC(activation(&DecodeWorkspace::inner_q,Inner,1));B_ALLOC(activation(&DecodeWorkspace::expert_q,ExpertHidden,TopExperts+1));
 auto mma=[&](PreparedActivationView&v,const ActivationBuffer&a)->Status{v.source=a.view();v.columns=capacity;v.input_experts=1;auto p=allocate(matmul_activation_bytes(capacity,a.k));if(!p.ok())return p.error();v.operands=static_cast<uint32_t*>(p.value());return success();};
 B_ALLOC(mma(w->view.base.hidden_mma,w->view.base.hidden_q));B_ALLOC(mma(w->view.base.inner_mma,w->view.base.inner_q));
 auto rv=allocate(capacity*sizeof(DecodeWorkspace));if(!rv.ok())return rv.error();w->view.row_views=static_cast<DecodeWorkspace*>(rv.value());B_ALLOC(cuda_status(cudaMemcpy(const_cast<DecodeWorkspace*>(w->view.row_views),w->host_rows_.data(),capacity*sizeof(DecodeWorkspace),cudaMemcpyHostToDevice),"publish batch row slices"));
 auto ints=[&](int n)->Result<int32_t*>{auto p=allocate(size_t(n)*4);if(!p.ok())return p.error();return static_cast<int32_t*>(p.value());};
 auto slots=ints(capacity),tokens=ints(capacity),positions=ints(capacity),error=ints(1);if(!slots.ok())return slots.error();if(!tokens.ok())return tokens.error();if(!positions.ok())return positions.error();if(!error.ok())return error.error();
 w->view.slots=slots.value();w->view.tokens=tokens.value();w->view.positions=positions.value();w->view.error=error.value();
 if(experts==BatchExpertPolicy::BucketedExact){
  auto bucket_allocate=[&](auto&pointer,size_t count)->Status{using T=std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>;auto p=allocate(count*sizeof(T));if(!p.ok())return p.error();pointer=static_cast<T*>(p.value());return success();};
  auto&b=w->view;b.roster_stride=capacity*TopExperts;b.tile_capacity=capacity*TopExperts+(capacity+7)/8;
  B_ALLOC(bucket_allocate(b.bucket_counts,Experts+1));B_ALLOC(bucket_allocate(b.bucket_occurrences,size_t(Experts+1)*b.roster_stride));B_ALLOC(bucket_allocate(b.bucket_hidden_rows,size_t(Experts+1)*b.roster_stride));B_ALLOC(bucket_allocate(b.bucket_map,capacity*(TopExperts+1)));B_ALLOC(bucket_allocate(b.bucket_tile_count,1));B_ALLOC(bucket_allocate(b.bucket_tiles,b.tile_capacity));B_ALLOC(bucket_allocate(b.bucket_stats,Layers));
  B_ALLOC(bucket_allocate(b.bucket_gate,size_t(b.tile_capacity)*8*ExpertHidden));B_ALLOC(bucket_allocate(b.bucket_up,size_t(b.tile_capacity)*8*ExpertHidden));B_ALLOC(bucket_allocate(b.bucket_down,size_t(b.tile_capacity)*8*Hidden));
 }
 B_ALLOC(cuda_status(cudaHostAlloc(&w->host_inputs_,size_t(capacity)*8,cudaHostAllocDefault),"pin batch configuration"));
 auto add=[&](EpochKind k,int layer){w->host_epochs_.push_back({int16_t(k),int16_t(layer)});};
 auto pack=[&](BatchEpochKind k,int layer){w->host_epochs_.push_back({int16_t(k),int16_t(layer)});};
 add(EpochKind::Embed,-1);
 for(int layer=0;layer<Layers;layer++){
  const auto&l=m.weights.layer[layer];add(EpochKind::InNorm,layer);pack(BatchEpochKind::PackHidden,layer);
  if(l.kind==LayerKind::GDN){add(EpochKind::GdnProj,layer);add(EpochKind::ConvPrep,layer);add(EpochKind::Recur,layer);add(EpochKind::FinishQ,layer);}
  else {add(EpochKind::AttnProj,layer);add(EpochKind::NormRope,layer);add(EpochKind::Scores,layer);add(EpochKind::Softmax,layer);add(EpochKind::ValueGate,layer);}
  if(l.output.flags&RotationEnabled)add(EpochKind::RotWriter,layer);
  pack(BatchEpochKind::PackInner,layer);add(EpochKind::OutProj,layer);add(EpochKind::ResNorm,layer);add(EpochKind::Router,layer);add(EpochKind::Select,layer);
  if(experts==BatchExpertPolicy::BucketedExact){pack(BatchEpochKind::BuildBuckets,layer);pack(BatchEpochKind::PublishBuckets,layer);}
  add(EpochKind::GateUp,layer);if(experts==BatchExpertPolicy::BucketedExact)pack(BatchEpochKind::GateUpQuant,layer);
  if((l.expert_down.flags|l.shared_down.flags)&RotationEnabled)add(EpochKind::RotDown,layer);
  add(EpochKind::Down,layer);if(experts==BatchExpertPolicy::BucketedExact)pack(BatchEpochKind::DownCombine,layer);
 }
 add(EpochKind::FinalNorm,-1);pack(BatchEpochKind::PackHidden,-1);add(EpochKind::Head,-1);add(EpochKind::ArgPart,-1);add(EpochKind::ArgFinish,-1);
 auto ep=allocate(w->host_epochs_.size()*sizeof(Epoch));if(!ep.ok())return ep.error();w->device_epochs=static_cast<Epoch*>(ep.value());B_ALLOC(cuda_status(cudaMemcpy(const_cast<Epoch*>(w->device_epochs),w->host_epochs_.data(),w->host_epochs_.size()*sizeof(Epoch),cudaMemcpyHostToDevice),"publish batch epochs"));
 #undef B_ALLOC
 return std::move(w);
}
Status BatchWorkspace::configure(std::span<const BatchInput>inputs,cudaStream_t stream){
 if(inputs.empty()||inputs.size()>size_t(view.capacity))return Error{ErrorCode::Capacity,"active batch rows exceed workspace capacity"};
 std::vector<bool>seen(model_slots_);std::vector<int32_t>slots;
 for(size_t r=0;r<inputs.size();r++){const auto&i=inputs[r];if(i.slot<0||i.slot>=model_slots_||seen[i.slot])return Error{ErrorCode::Capacity,"batch slots must be distinct and in range"};if(i.token< -1||i.token>=Vocab)return Error{ErrorCode::Geometry,"batch token outside vocabulary"};if(i.token==-1&&(!has_output_||r>=configured_slots_.size()||configured_slots_[r]!=i.slot))return Error{ErrorCode::Geometry,"feedback requires preceding output for unchanged row/slot mapping"};seen[i.slot]=true;slots.push_back(i.slot);}
 // Pin source storage survives capture and asynchronous copies; configuration
 // changes require prior work completion, stated by the public contract.
 for(size_t r=0;r<inputs.size();r++){host_inputs_[r]=inputs[r].slot;host_inputs_[view.capacity+r]=inputs[r].token;}
 auto s=cuda_status(cudaMemcpyAsync(const_cast<int32_t*>(view.slots),host_inputs_,inputs.size()*4,cudaMemcpyHostToDevice,stream),"upload batch slots");if(!s.ok())return s;
 s=cuda_status(cudaMemcpyAsync(const_cast<int32_t*>(view.tokens),host_inputs_+view.capacity,inputs.size()*4,cudaMemcpyHostToDevice,stream),"upload batch inputs");if(!s.ok())return s;
 if(slots!=configured_slots_)has_output_=false;
 view.rows=inputs.size();configured_slots_=std::move(slots);return success();
}
Status BatchWorkspace::finish(cudaStream_t stream)const{
 int error=0;auto s=cuda_status(cudaMemcpyAsync(&error,view.error,4,cudaMemcpyDeviceToHost,stream),"read batch error");if(!s.ok())return s;s=cuda_status(cudaStreamSynchronize(stream),"finish batch token");if(!s.ok())return s;
 if(error)return Error{error==1?ErrorCode::Capacity:ErrorCode::Geometry,error==1?"batch slot exhausted its context; invalid rows were not advanced":"invalid GPU feedback token"};return success();
}
__global__ void batch_token_kernel(const __grid_constant__ BatchArgs a){
 __shared__ float shared[4096];const auto grid=cooperative_groups::this_grid();
 for(int i=0;i<a.count;i++){run_batch_epoch(a,a.epochs[i],shared);grid.sync();}
}
Status run_batch_token(const DeviceModel&m,BatchWorkspace&w,int blocks_x,cudaStream_t stream){
 if(m.device_weights!=w.owner_||!m.weights.prepared)return Error{ErrorCode::Geometry,"batch workspace belongs to another/unprepared model"};
 if(w.view.rows<1||w.view.rows>w.view.capacity)return Error{ErrorCode::Capacity,"configure batch rows before forward"};
 if(blocks_x==0)blocks_x=std::max(1,m.multiprocessors/w.view.rows);
 if(blocks_x<1||blocks_x>1024)return Error{ErrorCode::Capacity,"batch x grid1..1024 required"};
 int active=0;auto s=cuda_status(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,batch_token_kernel,256,0),"batch cooperative residency");if(!s.ok())return s;
 if(blocks_x*w.view.rows>active*m.multiprocessors)return Error{ErrorCode::Capacity,"batch x*y grid exceeds compiled cooperative residency"};
 s=cuda_status(cudaMemsetAsync(w.view.error,0,4,stream),"clear batch token error");if(!s.ok())return s;
 BatchArgs a{m.device_weights,m.state,w.view,w.device_epochs,w.epoch_count()};void*args[]={&a};
 s=cuda_status(cudaLaunchCooperativeKernel((void*)batch_token_kernel,dim3(blocks_x,w.view.rows),256,args,0,stream),"native batch token");if(s.ok())w.has_output_=true;return s;
}
}
