#pragma once
#include "runtime.hpp"
namespace halo6000 {
enum class LayerPhase:int {InputNorm=0,InputQuant,Projection,Conv,PrepareGdn,Recurrence,FinishGdn,InnerQuant,OutputProjection,ResidualNorm,PostQuant,Router,Select,GateUp,ExpertQuant,Down};
Status run_token(const DeviceModel&model,DecodeWorkspace workspace,int token,int position,int slot=0,int blocks=188,cudaStream_t stream=nullptr);
Status embed(const DeviceModel&model,DecodeWorkspace workspace,int token,cudaStream_t stream=nullptr);
Status run_gdn_layer(const DeviceModel&model,DecodeWorkspace workspace,int layer,int slot,int blocks=188,cudaStream_t stream=nullptr);
Status run_gdn_phase(const DeviceModel&model,DecodeWorkspace workspace,int layer,int slot,LayerPhase phase,int blocks=188,cudaStream_t stream=nullptr);
} // namespace halo6000
