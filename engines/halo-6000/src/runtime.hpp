#pragma once
#include "cache.hpp"
#include "matmul.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <array>
#include <atomic>
namespace halo6000 {
struct PreparedLayer {
 PreparedMatmulView qkv,z,alpha,beta,output,q,k,v;
 PreparedMatmulView expert_gate,expert_up,expert_down,shared_gate,shared_up,shared_down;
};
struct PreparedModel {PreparedMatmulView head;PreparedLayer layer[Layers];};
static_assert(std::is_trivially_copyable_v<PreparedModel>);
class DeviceModel {
 std::vector<void*>allocations_;
 Status upload(const HostImage&image);
public:
 ModelWeights weights{};ModelWeights*device_weights=nullptr;StateView state{};
 size_t weight_bytes=0,state_bytes=0;bool has_rotation=false,compressed_rows_supported=true;int matvec_lanes=32;
 int device_ordinal=0,multiprocessors=0;
 PreparedModel prepared{};size_t prepared_bytes=0;
 mutable std::array<std::atomic<int>,6> token_residency{},layer_residency{};
 ~DeviceModel();
 DeviceModel()=default;DeviceModel(const DeviceModel&)=delete;DeviceModel&operator=(const DeviceModel&)=delete;
 static Result<std::unique_ptr<DeviceModel>>create(const HostImage&image,int slots,int context);
 Status reset(int slot=-1,cudaStream_t stream=nullptr);
 Status prepare_matmul(bool routed_experts=false,cudaStream_t stream=nullptr);
};
struct DecodeWorkspace {
 float*hidden=nullptr;float*residual=nullptr;float*normalized=nullptr;
 float*qkv=nullptr;float*z=nullptr;float*alpha=nullptr;float*beta=nullptr;float*mix=nullptr;float*projected=nullptr;
 float*logits=nullptr;int32_t*token=nullptr;
 ActivationBuffer hidden_q,inner_q,expert_q;
 PreparedActivationView hidden_mma,inner_mma,expert_mma;
 float*router_logits=nullptr;float*router_probs=nullptr;int32_t*route_ids=nullptr;float*route_weights=nullptr;float*shared_gain=nullptr;
 float*expert_gate=nullptr;float*expert_up=nullptr;float*expert_activation=nullptr;float*expert_down=nullptr;
 float*phase_scratch=nullptr;float*gdn_prepared=nullptr;float*decay=nullptr;float*effective_beta=nullptr;float*core=nullptr;float*log_decay=nullptr;float*softplus=nullptr;
 float*key=nullptr;float*value=nullptr;float*query=nullptr;float*scores=nullptr;int32_t context=0;
 float*max_values=nullptr;int32_t*max_ids=nullptr;uint64_t*timestamps=nullptr;
};
class Workspace {
 std::vector<void*>allocations_;
public:
 DecodeWorkspace view{};size_t bytes=0;
 ~Workspace();Workspace()=default;Workspace(const Workspace&)=delete;Workspace&operator=(const Workspace&)=delete;
 static Result<std::unique_ptr<Workspace>>create(int context=4096,bool profile=false);
};
Status cuda_status(cudaError_t error,const char*operation);
} // namespace halo6000
