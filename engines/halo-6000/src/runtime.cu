#include "runtime.hpp"
#include "matmul.cuh"
#include <string>
namespace halo6000 {
Status cuda_status(cudaError_t e,const char*op){if(e!=cudaSuccess)return Error{ErrorCode::Cuda,std::string(op)+": "+cudaGetErrorString(e)};return success();}
static Result<void*>allocate(size_t bytes,std::vector<void*>&owner){void*p=nullptr;auto s=cuda_status(cudaMalloc(&p,bytes),"cudaMalloc");if(!s.ok())return s.error();owner.push_back(p);return p;}
DeviceModel::~DeviceModel(){for(auto it=allocations_.rbegin();it!=allocations_.rend();++it)cudaFree(*it);}
Workspace::~Workspace(){for(auto it=allocations_.rbegin();it!=allocations_.rend();++it)cudaFree(*it);}
Status DeviceModel::upload(const HostImage&image){
 auto source=Gguf::open(image.provenance.source_path);if(!source.ok())return source.error();if(!(source.value().storage->identity==image.provenance.source))return Error{ErrorCode::CacheMismatch,"source changed before upload"};
 std::unordered_map<std::string,MatrixView>tensors;
 for(const auto&t:image.tensors)if(t.format!=WeightFormat::F32&&t.format!=WeightFormat::Pane2&&t.format!=WeightFormat::Q8Grouped128&&t.format!=WeightFormat::Q2K)return Error{ErrorCode::Unsupported,"native inference format not wired: "+t.name};
 for(const auto&t:image.tensors){auto p=allocate(t.bytes,allocations_);if(!p.ok())return p.error();auto s=cuda_status(cudaMemcpy(p.value(),t.data,t.bytes,cudaMemcpyHostToDevice),t.name.c_str());if(!s.ok())return s;auto v=t.view();v.data=p.value();tensors.emplace(t.name,v);weight_bytes+=t.bytes;image.storage->discard(t.data,t.bytes);}
 auto matrix=[&](const std::string&name,MatrixView&dest)->Status{
  auto it=tensors.find(name);if(it==tensors.end())return Error{ErrorCode::Geometry,"missing tensor "+name};dest=it->second;
  if(dest.format==WeightFormat::Q2K&&name!="token_embd.weight"&&name.find("ffn_down_")==std::string::npos)compressed_rows_supported=false;
  const std::string prefix="halo6000.rotation."+name;auto block=source.value().metadata.find(prefix+".block_size");
  if(block!=source.value().metadata.end()){
   auto b=block->second.integer();if(!b||!supported_rotation_block(int(*b))||dest.k%*b)return Error{ErrorCode::Geometry,"invalid rotation block "+name};
   int log2=0;while((1<<log2)<*b)log2++;dest.flags=rotation_flags(log2);has_rotation=true;
   auto sign=source.value().metadata.find(prefix+".signs_tensor");if(sign!=source.value().metadata.end()){
    if(sign->second.strings.size()!=1)return Error{ErrorCode::Geometry,"invalid rotation signs name "+name};auto s=tensors.find(sign->second.strings[0]);
    if(s==tensors.end()||s->second.format!=WeightFormat::F32||s->second.bytes!=size_t(dest.k)*4)return Error{ErrorCode::Geometry,"missing F32 rotation signs "+name};dest.aux=s->second.data;
   }
  }
  return success();
 };
 auto vector=[&](const std::string&name,const float*&dest)->Status{auto it=tensors.find(name);if(it==tensors.end()||it->second.format!=WeightFormat::F32)return Error{ErrorCode::Geometry,"missing/F32 tensor required "+name};dest=static_cast<const float*>(it->second.data);return success();};
 #define H6_BIND(call) do{auto result=(call);if(!result.ok())return result;}while(0)
 H6_BIND(matrix("token_embd.weight",weights.embedding));H6_BIND(matrix("output.weight",weights.head));H6_BIND(vector("output_norm.weight",weights.final_norm));
 for(int i=0;i<Layers;i++){
  auto&l=weights.layer[i];l.index=i;l.kind=attention_layer(i)?LayerKind::Attention:LayerKind::GDN;std::string p="blk."+std::to_string(i)+".";
  H6_BIND(vector(p+"attn_norm.weight",l.input_norm));H6_BIND(vector(p+"post_attention_norm.weight",l.post_norm));
  H6_BIND(matrix(p+"ffn_gate_inp.weight",l.router));H6_BIND(matrix(p+"ffn_gate_inp_shexp.weight",l.shared_router));
  H6_BIND(matrix(p+"ffn_gate_exps.weight",l.expert_gate));H6_BIND(matrix(p+"ffn_up_exps.weight",l.expert_up));H6_BIND(matrix(p+"ffn_down_exps.weight",l.expert_down));
  H6_BIND(matrix(p+"ffn_gate_shexp.weight",l.shared_gate));H6_BIND(matrix(p+"ffn_up_shexp.weight",l.shared_up));H6_BIND(matrix(p+"ffn_down_shexp.weight",l.shared_down));
  if(l.kind==LayerKind::GDN){
   H6_BIND(matrix(p+"attn_qkv.weight",l.qkv));H6_BIND(matrix(p+"attn_gate.weight",l.z));H6_BIND(matrix(p+"ssm_alpha.weight",l.alpha));H6_BIND(matrix(p+"ssm_beta.weight",l.beta));H6_BIND(matrix(p+"ssm_out.weight",l.output));
   H6_BIND(vector(p+"ssm_a",l.ssm_a));H6_BIND(vector(p+"ssm_dt.bias",l.ssm_dt));H6_BIND(vector(p+"ssm_conv1d.weight",l.conv));H6_BIND(vector(p+"ssm_norm.weight",l.ssm_norm));
  }else{
   H6_BIND(matrix(p+"attn_q.weight",l.q));H6_BIND(matrix(p+"attn_k.weight",l.k));H6_BIND(matrix(p+"attn_v.weight",l.v));H6_BIND(matrix(p+"attn_output.weight",l.output));H6_BIND(vector(p+"attn_q_norm.weight",l.q_norm));H6_BIND(vector(p+"attn_k_norm.weight",l.k_norm));
  }
 }
 #undef H6_BIND
 auto p=allocate(sizeof(ModelWeights),allocations_);if(!p.ok())return p.error();device_weights=static_cast<ModelWeights*>(p.value());return cuda_status(cudaMemcpy(device_weights,&weights,sizeof weights,cudaMemcpyHostToDevice),"publish model descriptor");
}
Result<std::unique_ptr<DeviceModel>>DeviceModel::create(const HostImage&image,int slots,int context){
 if(slots<1||slots>MaxRows||context<1||context>131072)return Error{ErrorCode::Capacity,"slots1..128/context1..131072 required"};
 int device=0;auto status=cuda_status(cudaGetDevice(&device),"current device");if(!status.ok())return status.error();cudaDeviceProp prop{};status=cuda_status(cudaGetDeviceProperties(&prop,device),"device properties");if(!status.ok())return status.error();if(prop.major!=12||prop.minor!=0)return Error{ErrorCode::Unsupported,"engine requires sm_120"};
 auto m=std::make_unique<DeviceModel>();m->device_ordinal=device;m->multiprocessors=prop.multiProcessorCount;status=m->upload(image);if(!status.ok())return status.error();m->state.slots=slots;m->state.context=context;
 auto state_alloc=[&](size_t bytes,auto*&dest)->Status{auto p=allocate(bytes,m->allocations_);if(!p.ok())return p.error();dest=reinterpret_cast<std::remove_reference_t<decltype(dest)>>(p.value());m->state_bytes+=bytes;return success();};
 status=state_alloc(size_t(slots)*GdnFloatsPerSlot*4,m->state.gdn);if(!status.ok())return status.error();status=state_alloc(size_t(slots)*ConvFloatsPerSlot*4,m->state.conv);if(!status.ok())return status.error();
 const size_t kvbytes=size_t(AttentionLayers)*slots*context*KVHeads*HeadDim*2;
 status=state_alloc(kvbytes,m->state.key);if(!status.ok())return status.error();status=state_alloc(kvbytes,m->state.value);if(!status.ok())return status.error();status=state_alloc(size_t(slots)*4,m->state.positions);if(!status.ok())return status.error();
 status=m->reset();if(!status.ok())return status.error();return std::move(m);
}
Status DeviceModel::reset(int slot,cudaStream_t stream){
 if(slot>=state.slots||slot< -1)return Error{ErrorCode::Capacity,"invalid state slot"};const int first=slot<0?0:slot,count=slot<0?state.slots:1;
 auto s=cuda_status(cudaMemsetAsync(state.gdn+size_t(first)*GdnFloatsPerSlot,0,size_t(count)*GdnFloatsPerSlot*4,stream),"reset GDN");if(!s.ok())return s;
 s=cuda_status(cudaMemsetAsync(state.conv+size_t(first)*ConvFloatsPerSlot,0,size_t(count)*ConvFloatsPerSlot*4,stream),"reset convolution");if(!s.ok())return s;
 for(int l=0;l<AttentionLayers;l++){size_t off=(size_t(l)*state.slots+first)*state.context*KVHeads*HeadDim,n=size_t(count)*state.context*KVHeads*HeadDim*2;s=cuda_status(cudaMemsetAsync(state.key+off,0,n,stream),"reset key");if(!s.ok())return s;s=cuda_status(cudaMemsetAsync(state.value+off,0,n,stream),"reset value");if(!s.ok())return s;}
 s=cuda_status(cudaMemsetAsync(state.positions+first,0,size_t(count)*4,stream),"reset position");if(!s.ok())return s;
 return cuda_status(cudaStreamSynchronize(stream),"publish reset state");
}
Status DeviceModel::prepare_matmul(bool routed_experts,cudaStream_t stream){
 if(weights.prepared){if(routed_experts&&!prepared.layer[0].expert_gate.operands)return Error{ErrorCode::Unsupported,"prepared matmul scope is immutable after publication"};return success();}
 struct Pending {std::vector<void*>allocations;bool committed=false;~Pending(){if(!committed)for(void*p:allocations)cudaFree(p);}} pending;
 PreparedModel image{};size_t bytes=sizeof image;
 auto prepare=[&](const MatrixView&source,PreparedMatmulView&dest)->Status{
  dest.source=source;if(!supported_matmul_source(source))return success();auto sizes=matmul_weight_bytes(source);
  auto op=allocate(sizes.operands,pending.allocations);if(!op.ok())return op.error();auto scales=allocate(sizes.scales,pending.allocations);if(!scales.ok())return scales.error();
  dest.operands=static_cast<uint32_t*>(op.value());dest.weight_scales=static_cast<uint16_t*>(scales.value());
  uint64_t units=uint64_t(source.n/16)*(source.k/32)*source.experts*32;unsigned blocks=unsigned((units+255)/256);
  if(source.format==WeightFormat::Pane2)prepare_matmul_weights<WeightFormat::Pane2><<<blocks,256,0,stream>>>(source,static_cast<uint32_t*>(op.value()),static_cast<uint16_t*>(scales.value()));
  else prepare_matmul_weights<WeightFormat::Q8Grouped128><<<blocks,256,0,stream>>>(source,static_cast<uint32_t*>(op.value()),static_cast<uint16_t*>(scales.value()));
  auto s=cuda_status(cudaGetLastError(),"prepare matmul weights");if(!s.ok())return s;bytes+=sizes.operands+sizes.scales;return success();
 };
 #define H6_PREP(source,dest) do{auto s=prepare(source,dest);if(!s.ok())return s;}while(0)
 H6_PREP(weights.head,image.head);
 for(int i=0;i<Layers;i++){const auto&l=weights.layer[i];auto&p=image.layer[i];
  if(l.kind==LayerKind::GDN){H6_PREP(l.qkv,p.qkv);H6_PREP(l.z,p.z);H6_PREP(l.alpha,p.alpha);H6_PREP(l.beta,p.beta);}else{H6_PREP(l.q,p.q);H6_PREP(l.k,p.k);H6_PREP(l.v,p.v);}
  H6_PREP(l.output,p.output);H6_PREP(l.shared_gate,p.shared_gate);H6_PREP(l.shared_up,p.shared_up);H6_PREP(l.shared_down,p.shared_down);
  if(routed_experts){H6_PREP(l.expert_gate,p.expert_gate);H6_PREP(l.expert_up,p.expert_up);H6_PREP(l.expert_down,p.expert_down);}
 }
 #undef H6_PREP
 auto descriptor=allocate(sizeof image,pending.allocations);if(!descriptor.ok())return descriptor.error();auto s=cuda_status(cudaMemcpyAsync(descriptor.value(),&image,sizeof image,cudaMemcpyHostToDevice,stream),"upload prepared model");if(!s.ok())return s;
 s=cuda_status(cudaStreamSynchronize(stream),"finish matmul preparation");if(!s.ok())return s;
 ModelWeights published=weights;published.prepared=static_cast<PreparedModel*>(descriptor.value());s=cuda_status(cudaMemcpy(device_weights,&published,sizeof published,cudaMemcpyHostToDevice),"publish prepared matmul");if(!s.ok())return s;
 allocations_.insert(allocations_.end(),pending.allocations.begin(),pending.allocations.end());pending.committed=true;weights=published;prepared=image;prepared_bytes=bytes;return success();
}
Result<std::unique_ptr<Workspace>>Workspace::create(int context,bool profile){
 if(context<1||context>131072)return Error{ErrorCode::Capacity,"invalid workspace context"};auto w=std::make_unique<Workspace>();w->view.context=context;
 auto floats=[&](float*&dest,size_t n)->Status{auto p=allocate(n*4,w->allocations_);if(!p.ok())return p.error();dest=static_cast<float*>(p.value());w->bytes+=n*4;return success();};
 auto ints=[&](int32_t*&dest,size_t n)->Status{auto p=allocate(n*4,w->allocations_);if(!p.ok())return p.error();dest=static_cast<int32_t*>(p.value());w->bytes+=n*4;return success();};
 auto activation=[&](ActivationBuffer&b,int k,int rows)->Status{b.k=k;b.rows=rows;b.group=32;b.q_stride=k;b.scale_stride=k/32;auto p=allocate(size_t(k)*rows,w->allocations_);if(!p.ok())return p.error();b.qs=static_cast<int8_t*>(p.value());w->bytes+=size_t(k)*rows;return floats(b.scale,size_t(k/32)*rows);};
 #define H6_ALLOC(call) do{auto s=(call);if(!s.ok())return s.error();}while(0)
 auto&v=w->view;
 if(profile){auto p=allocate(646*sizeof(uint64_t),w->allocations_);if(!p.ok())return p.error();v.timestamps=static_cast<uint64_t*>(p.value());w->bytes+=646*sizeof(uint64_t);H6_ALLOC(cuda_status(cudaMemset(v.timestamps,0,646*sizeof(uint64_t)),"clear timestamps"));}
 for(auto p:{&v.hidden,&v.residual,&v.normalized,&v.projected})H6_ALLOC(floats(*p,Hidden));
 H6_ALLOC(floats(v.qkv,ConvChannels));H6_ALLOC(floats(v.z,Inner));H6_ALLOC(floats(v.alpha,ValueHeads));H6_ALLOC(floats(v.beta,ValueHeads));H6_ALLOC(floats(v.mix,Inner));
 H6_ALLOC(floats(v.logits,Vocab));H6_ALLOC(ints(v.token,1));H6_ALLOC(activation(v.hidden_q,Hidden,1));H6_ALLOC(activation(v.inner_q,Inner,1));H6_ALLOC(activation(v.expert_q,ExpertHidden,TopExperts+1));
 auto operand=[&](PreparedActivationView&a,const ActivationBuffer&b)->Status{a.source=b.view();a.columns=1;a.input_experts=b.rows;size_t n=matmul_activation_bytes(1,b.k,b.rows);auto p=allocate(n,w->allocations_);if(!p.ok())return p.error();a.operands=static_cast<uint32_t*>(p.value());w->bytes+=n;return success();};
 H6_ALLOC(operand(v.hidden_mma,v.hidden_q));H6_ALLOC(operand(v.inner_mma,v.inner_q));H6_ALLOC(operand(v.expert_mma,v.expert_q));
 H6_ALLOC(floats(v.router_logits,Experts));H6_ALLOC(floats(v.router_probs,Experts));H6_ALLOC(ints(v.route_ids,TopExperts));H6_ALLOC(floats(v.route_weights,TopExperts));H6_ALLOC(floats(v.shared_gain,1));
 H6_ALLOC(floats(v.expert_gate,(TopExperts+1)*ExpertHidden));H6_ALLOC(floats(v.expert_up,(TopExperts+1)*ExpertHidden));H6_ALLOC(floats(v.expert_activation,(TopExperts+1)*ExpertHidden));H6_ALLOC(floats(v.expert_down,(TopExperts+1)*Hidden));
 H6_ALLOC(floats(v.phase_scratch,32));H6_ALLOC(floats(v.gdn_prepared,ConvChannels));H6_ALLOC(floats(v.log_decay,ValueHeads));H6_ALLOC(floats(v.softplus,ValueHeads));H6_ALLOC(floats(v.decay,ValueHeads));H6_ALLOC(floats(v.effective_beta,ValueHeads));H6_ALLOC(floats(v.core,Inner));
 H6_ALLOC(floats(v.max_values,1024));H6_ALLOC(ints(v.max_ids,1024));
 H6_ALLOC(floats(v.key,KVHeads*HeadDim));H6_ALLOC(floats(v.value,KVHeads*HeadDim));H6_ALLOC(floats(v.query,QueryHeads*HeadDim));H6_ALLOC(floats(v.scores,size_t(QueryHeads)*context));
 #undef H6_ALLOC
 return std::move(w);
}
} // namespace halo6000
