#pragma once
#include "runtime.hpp"
#include "schedule.hpp"
#include <span>
#include <vector>
namespace halo6000 {
// Configuration is immutable until queued work finishes. GPU positions belong
// to StateView; token=-1 consumes the preceding argmax for the same row/slot.
struct BatchInput {int32_t slot=0,token=0;};
enum class BatchExpertPolicy:int32_t {RowParallel=0,BucketedExact=1};
struct BucketTile {int32_t expert=0,offset=0,columns=0;};
struct BucketStats {int32_t active_experts=0,routed_occurrences=0,routed_tiles=0,padded_routed_columns=0,shared_rows=0,total_tiles=0,max_roster=0,singleton_experts=0,pair_experts=0;};
struct BatchView {
 DecodeWorkspace base{};const DecodeWorkspace*row_views=nullptr;
 const int32_t*slots=nullptr;const int32_t*tokens=nullptr;
 int32_t*positions=nullptr;int32_t*error=nullptr;
 int32_t rows=0,capacity=0;
 BatchExpertPolicy expert_policy=BatchExpertPolicy::RowParallel;
 int32_t roster_stride=0,tile_capacity=0;
 int32_t*bucket_counts=nullptr;int32_t*bucket_occurrences=nullptr;int32_t*bucket_hidden_rows=nullptr;int32_t*bucket_map=nullptr;int32_t*bucket_tile_count=nullptr;
 BucketTile*bucket_tiles=nullptr;BucketStats*bucket_stats=nullptr;
 float*bucket_gate=nullptr;float*bucket_up=nullptr;float*bucket_down=nullptr;
};
struct BatchArgs {const ModelWeights*model=nullptr;StateView state{};BatchView workspace{};const Epoch*epochs=nullptr;int32_t count=0;};
enum class BatchEpochKind:int {PackHidden=int(EpochKind::Count),PackInner,BuildBuckets,PublishBuckets,GateUpQuant,DownCombine,Count};
inline constexpr const char*batch_epoch_kind_name(int kind){switch(BatchEpochKind(kind)){case BatchEpochKind::PackHidden:return "pack_hidden";case BatchEpochKind::PackInner:return "pack_inner";case BatchEpochKind::BuildBuckets:return "build_buckets";case BatchEpochKind::PublishBuckets:return "publish_buckets";case BatchEpochKind::GateUpQuant:return "gateup_quant";case BatchEpochKind::DownCombine:return "down_combine";default:return epoch_kind_name(kind);}}
static_assert(int(BatchEpochKind::Count)<=32);
class BatchWorkspace {
 std::vector<void*>allocations_;std::vector<DecodeWorkspace>host_rows_;
 int32_t*host_inputs_=nullptr;std::vector<int32_t>configured_slots_;
 const ModelWeights*owner_=nullptr;int model_slots_=0;
 bool has_output_=false;std::vector<Epoch>host_epochs_;
 friend Status run_batch_token(const DeviceModel&,BatchWorkspace&,int,cudaStream_t);
public:
 BatchView view{};const Epoch*device_epochs=nullptr;size_t bytes=0;
 ~BatchWorkspace();BatchWorkspace()=default;BatchWorkspace(const BatchWorkspace&)=delete;BatchWorkspace&operator=(const BatchWorkspace&)=delete;
 static Result<std::unique_ptr<BatchWorkspace>>create(const DeviceModel&model,int capacity,BatchExpertPolicy experts=BatchExpertPolicy::RowParallel);
 // Outside capture; caller must finish prior use before changing configuration.
 Status configure(std::span<const BatchInput>inputs,cudaStream_t stream=nullptr);
 // Synchronizes and checks async capacity errors. Invalid rows never advance.
 Status finish(cudaStream_t stream=nullptr)const;
 const std::vector<DecodeWorkspace>&host_rows()const{return host_rows_;}
 int epoch_count()const{return int(host_epochs_.size());}
};
// blocks_x=0 chooses floor(SMs/rows); grid.y always equals active rows.
// Capture-safe after configuration. s1 runtime/kernel APIs are untouched.
Status run_batch_token(const DeviceModel&model,BatchWorkspace&workspace,int blocks_x=0,cudaStream_t stream=nullptr);
}
