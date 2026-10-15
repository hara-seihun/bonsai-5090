#include "debug.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <initializer_list>
namespace halo6000 {
static std::string json_string(const std::string&s){std::string r="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){r+='\\';r+=c;}else if(c<32){const char*hex="0123456789abcdef";r+="\\u00";r+=hex[c>>4];r+=hex[c&15];}else r+=c;}return r+'"';}
class LayerDump {
 std::string directory_;std::ostringstream entries_;bool comma_=false;int serial_=0;
public:
 int position=0;
 explicit LayerDump(std::string dir):directory_(std::move(dir)){}
 Status tensor(const std::string&name,const void*ptr,std::initializer_list<int>shape,const char*type="f32"){
  size_t count=1;for(int n:shape)count*=n;std::vector<uint8_t>bytes(count*4);auto s=cuda_status(cudaMemcpy(bytes.data(),ptr,bytes.size(),cudaMemcpyDeviceToHost),"debug tensor copy");if(!s.ok())return s;
  std::string filename=std::to_string(serial_++)+"-"+name+".bin";std::ofstream file(directory_+"/"+filename,std::ios::binary);file.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());if(!file)return Error{ErrorCode::Io,"debug tensor write "+filename};
  if(comma_)entries_<<',';comma_=true;
  entries_<<"{\"step\":"<<position<<",\"position\":"<<position<<",\"batch\":1,\"name\":"<<json_string(name)<<",\"type\":"<<json_string(type)<<",\"ne\":[";bool first=true;for(int n:shape){if(!first)entries_<<',';first=false;entries_<<n;}entries_<<"],\"bytes\":"<<bytes.size()<<",\"file\":"<<json_string(filename)<<",\"order\":\"ne0 fastest, packed contiguous; reshape C with reversed ne\"}";return success();
 }
 Status finish(const std::string&model,size_t modelbytes,const std::vector<int32_t>&tokens,const std::vector<int32_t>&argmax={}){
  std::ofstream f(directory_+"/manifest.json");f<<"{\"version\":1,\"callback_diagnostics_not_benchmark\":true,\"mode\":\"debug-layer0-control\",\"model\":"<<json_string(model)<<",\"model_bytes\":"<<modelbytes<<",\"tokens_first16\":[";for(size_t i=0;i<tokens.size();i++){if(i)f<<',';f<<tokens[i];}f<<"],\"schedule\":[";for(size_t i=0;i<tokens.size();i++){if(i)f<<',';f<<"{\"step\":"<<i<<",\"position\":"<<i<<",\"batch\":1";if(!argmax.empty())f<<",\"argmax\":"<<argmax[i]<<",\"logit_file\":\"step"<<i<<"-logits.f32\"";f<<'}';}f<<"],\"logits_available\":"<<(argmax.empty()?"false":"true")<<",\"conv_state_update_boundary\":\"destination precommit, true posthistory is conv_state_last\",\"tensors\":["<<entries_.str()<<"]}\n";if(!f)return Error{ErrorCode::Io,"debug manifest write"};return success();
 }
};
Status dump_layer0(const DeviceModel&m,DecodeWorkspace w,const std::string&model,const std::string&tokensfile,const std::string&output,int steps,int blocks){
 if(steps<1||steps>512)return Error{ErrorCode::Capacity,"debug steps1..512 required"};std::error_code ec;std::filesystem::create_directories(output,ec);if(ec)return Error{ErrorCode::Io,"debug directory: "+ec.message()};
 std::ifstream input(tokensfile,std::ios::binary);std::vector<int32_t>tokens(steps);input.read(reinterpret_cast<char*>(tokens.data()),steps*4);if(!input)return Error{ErrorCode::Io,"read debug tokens "+tokensfile};auto s=m.state.gdn?success():Status(Error{ErrorCode::Invalid,"uninitialized model state"});if(!s.ok())return s;
 LayerDump d(output);
 #define H6_DUMP(call) do{auto status=(call);if(!status.ok())return status;}while(0)
 for(int position=0;position<steps;position++){
  d.position=position;H6_DUMP(embed(m,w,tokens[position]));H6_DUMP(d.tensor("model.input_embed",w.hidden,{Hidden,1,1,1}));
  H6_DUMP(d.tensor("state_predelta-0",m.state.gdn,{128,128,32,1}));H6_DUMP(d.tensor("conv_states-0",m.state.conv,{24576,1,1,1}));
  H6_DUMP(d.tensor("conv_states_reshaped-0",m.state.conv,{3,8192,1,1}));H6_DUMP(d.tensor("conv_state_update-0",m.state.conv,{24576,1,1,1}));
  for(int phase=0;phase<=int(LayerPhase::Down);phase++){
   H6_DUMP(run_gdn_phase(m,w,0,0,LayerPhase(phase),blocks));
   switch(LayerPhase(phase)){
   case LayerPhase::InputNorm:H6_DUMP(d.tensor("attn_norm-0",w.normalized,{Hidden,1,1,1}));break;
   case LayerPhase::Projection:
    H6_DUMP(d.tensor("linear_attn_qkv_mixed-0",w.qkv,{8192,1,1,1}));H6_DUMP(d.tensor("qkv_mixed_transposed-0",w.qkv,{1,8192,1,1}));H6_DUMP(d.tensor("z-0",w.z,{4096,1,1,1}));H6_DUMP(d.tensor("alpha-0",w.alpha,{32,1,1,1}));H6_DUMP(d.tensor("beta-0",w.beta,{1,32,1,1}));break;
   case LayerPhase::Conv:
    H6_DUMP(d.tensor("conv_output_silu-0",w.gdn_prepared,{8192,1,1,1}));H6_DUMP(d.tensor("q_conv-0",w.gdn_prepared,{128,16,1,1}));H6_DUMP(d.tensor("k_conv-0",w.gdn_prepared+2048,{128,16,1,1}));H6_DUMP(d.tensor("conv_state_last-0",m.state.conv,{3,8192,1,1}));break;
   case LayerPhase::PrepareGdn:
    H6_DUMP(d.tensor("q_conv_predelta-0",w.gdn_prepared,{128,16,1,1}));H6_DUMP(d.tensor("k_conv_predelta-0",w.gdn_prepared+2048,{128,16,1,1}));H6_DUMP(d.tensor("v_conv_predelta-0",w.gdn_prepared+4096,{128,32,1,1}));H6_DUMP(d.tensor("gate-0",w.log_decay,{32,1,1,1}));H6_DUMP(d.tensor("a_softplus-0",w.softplus,{32,1,1,1}));H6_DUMP(d.tensor("beta_sigmoid-0",w.effective_beta,{1,32,1,1}));break;
   case LayerPhase::Recurrence:H6_DUMP(d.tensor("new_state-0",m.state.gdn,{128,128,32,1}));H6_DUMP(d.tensor("attn_output-0",w.core,{128,32,1,1}));break;
   case LayerPhase::FinishGdn:H6_DUMP(d.tensor("final_output-0",w.mix,{4096,1,1,1}));break;
   case LayerPhase::OutputProjection:H6_DUMP(d.tensor("linear_attn_out-0",w.projected,{Hidden,1,1,1}));break;
   case LayerPhase::ResidualNorm:H6_DUMP(d.tensor("attn_residual-0",w.residual,{Hidden,1,1,1}));H6_DUMP(d.tensor("attn_post_norm-0",w.normalized,{Hidden,1,1,1}));break;
   case LayerPhase::Router:H6_DUMP(d.tensor("ffn_moe_logits-0",w.router_logits,{256,1,1,1}));H6_DUMP(d.tensor("shared_expert_gate_sigmoid-0",w.shared_gain,{1,1,1,1}));break;
   case LayerPhase::Select:H6_DUMP(d.tensor("ffn_moe_probs-0",w.router_probs,{256,1,1,1}));H6_DUMP(d.tensor("ffn_moe_topk-0",w.route_ids,{8,1,1,1},"i32"));H6_DUMP(d.tensor("ffn_moe_weights_norm-0",w.route_weights,{8,1,1,1}));break;
   case LayerPhase::GateUp:
    H6_DUMP(d.tensor("ffn_moe_gate-0",w.expert_gate,{512,8,1,1}));H6_DUMP(d.tensor("ffn_moe_up-0",w.expert_up,{512,8,1,1}));H6_DUMP(d.tensor("ffn_moe_swiglu-0",w.expert_activation,{512,8,1,1}));
    H6_DUMP(d.tensor("ffn_gate-0",w.expert_gate+4096,{512,1,1,1}));H6_DUMP(d.tensor("ffn_up-0",w.expert_up+4096,{512,1,1,1}));H6_DUMP(d.tensor("ffn_swiglu-0",w.expert_activation+4096,{512,1,1,1}));break;
   case LayerPhase::Down:H6_DUMP(d.tensor("ffn_moe_down-0",w.expert_down,{Hidden,8,1,1}));H6_DUMP(d.tensor("ffn_shexp-0",w.expert_down+8*Hidden,{Hidden,1,1,1}));H6_DUMP(d.tensor("l_out-0",w.hidden,{Hidden,1,1,1}));break;
   default:break;
   }
  }
 }
 #undef H6_DUMP
 auto bytes=std::filesystem::file_size(model,ec);if(ec)return Error{ErrorCode::Io,"source size: "+ec.message()};return d.finish(model,bytes,tokens);
}
Status dump_forward(const DeviceModel&m,DecodeWorkspace w,const std::string&model,const std::string&tokensfile,const std::string&output,int steps,int blocks){
 if(steps<1||steps>m.state.context)return Error{ErrorCode::Capacity,"forward steps exceed context"};std::error_code ec;std::filesystem::create_directories(output,ec);if(ec)return Error{ErrorCode::Io,"forward directory: "+ec.message()};
 std::ifstream input(tokensfile,std::ios::binary);std::vector<int32_t>tokens(steps),argmax(steps);input.read(reinterpret_cast<char*>(tokens.data()),steps*4);if(!input)return Error{ErrorCode::Io,"read forward tokens"};LayerDump d(output);
 for(int p=0;p<steps;p++){
  d.position=p;auto s=run_token(m,w,tokens[p],p,0,blocks);if(!s.ok())return s;
  s=d.tensor("l_out-39",w.hidden,{Hidden,1,1,1});if(!s.ok())return s;s=d.tensor("h_nextn",w.normalized,{Hidden,1,1,1});if(!s.ok())return s;s=d.tensor("result_norm",w.normalized,{Hidden,1,1,1});if(!s.ok())return s;
  std::vector<float>logits(Vocab);s=cuda_status(cudaMemcpy(logits.data(),w.logits,Vocab*4,cudaMemcpyDeviceToHost),"dump forward logits");if(!s.ok())return s;s=cuda_status(cudaMemcpy(&argmax[p],w.token,4,cudaMemcpyDeviceToHost),"dump argmax");if(!s.ok())return s;
  std::ofstream file(output+"/step"+std::to_string(p)+"-logits.f32",std::ios::binary);file.write(reinterpret_cast<const char*>(logits.data()),Vocab*4);if(!file)return Error{ErrorCode::Io,"write forward logits"};
 }
 auto bytes=std::filesystem::file_size(model,ec);if(ec)return Error{ErrorCode::Io,"forward source size"};return d.finish(model,bytes,tokens,argmax);
}
} // namespace halo6000
