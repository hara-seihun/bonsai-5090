#pragma once
// Scheduled whole token: the token as a sequence of epochs, cut into segments.
// Each segment is one kernel launch (cooperative with grid.sync between its
// epochs when it holds more than one); a token is one CUDA graph replay.
// Body choices and segment kernel classes come from include/tune.hpp.
// The C ABI lets the tuner's resident driver dlopen many compiled variants
// against one loaded model.
#include "runtime.hpp"
#include <cstdint>
namespace halo6000 {
enum class EpochKind:int {
 Embed=0,InNorm,GdnProj,ConvPrep,Recur,FinishQ,RotWriter,OutProj,ResNorm,Router,Select,GateUp,RotDown,Down,
 AttnProj,NormRope,Scores,Softmax,ValueGate,FinalNorm,Head,ArgPart,ArgFinish,Count
};
inline constexpr const char*epoch_kind_name(int k){
 constexpr const char*n[]={"embed","in_norm","gdn_proj","conv_prep","recur","finish_q","rot_writer","out_proj","res_norm","router","select","gate_up","rot_down","down",
  "attn_proj","norm_rope","scores","softmax","value_gate","final_norm","head","arg_part","arg_finish"};
 return k>=0&&k<int(EpochKind::Count)?n[k]:"?";
}
struct Epoch {int16_t kind,layer;};
} // namespace halo6000
extern "C" {
// Version 2 (v1 + step_token). All entry points return 0 on success; on failure they write a
// message into err (when non-null) and return nonzero.
struct HaloTuneApi {
 int version;
 const char*config_id;
 // grids: per kernel class grid (0 = default). teacher: device int32[context]
 // or null for greedy feedback through workspace.token.
 void*(*create)(const halo6000::DeviceModel*model,const halo6000::DecodeWorkspace*w,const int*grids,int ngrids,const int32_t*teacher,int slot,char*err,int errlen);
 int(*step)(void*ctx,cudaStream_t stream,char*err,int errlen);
 // Writes epoch-start globaltimer stamps (epochs+1 values) after one profiled step.
 int(*profile)(void*ctx,cudaStream_t stream,uint64_t*stamps,int capacity,char*err,int errlen);
 // JSON description: classes (kinds, min_blocks, registers, residency, grid), segments, epochs.
 int(*describe)(void*ctx,char*buf,int len);
 void(*destroy)(void*ctx);
 // One token on `stream`: token>=0 feeds that token (prompt / teacher forcing),
 // token<0 feeds the previous argmax from workspace.token (greedy generation).
 // Requires a context created with teacher==nullptr. Logits land in
 // workspace.logits, the argmax in workspace.token, state.positions[slot] advances.
 int(*step_token)(void*ctx,int32_t token,cudaStream_t stream,char*err,int errlen);
};
const HaloTuneApi*halo_tune_api();
}
