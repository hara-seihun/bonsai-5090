#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__CUDACC__)
#define H6_HOST_DEVICE __host__ __device__
#else
#define H6_HOST_DEVICE
#endif
namespace halo6000 {
inline constexpr int Layers=40,LinearLayers=30,AttentionLayers=10,Hidden=2048,Vocab=248320;
inline constexpr int Experts=256,TopExperts=8,ExpertHidden=512,SharedHidden=512;
inline constexpr int KeyHeads=16,ValueHeads=32,StateDim=128,ValueDim=128,Inner=4096,ConvChannels=8192,ConvTaps=4;
inline constexpr int QueryHeads=16,KVHeads=2,HeadDim=256,RotaryDim=64,FullInterval=4;
inline constexpr int WeightGroup=128,DefaultActivationGroup=32,MaxRows=128;
inline constexpr float NormEpsilon=1e-6f,RopeBase=1e7f,AttentionScale=1.f/16.f;
inline constexpr size_t GdnFloatsPerSlot=size_t(LinearLayers)*ValueHeads*ValueDim*StateDim;
inline constexpr size_t ConvFloatsPerSlot=size_t(LinearLayers)*(ConvTaps-1)*ConvChannels;
H6_HOST_DEVICE constexpr bool attention_layer(int layer){return (layer+1)%FullInterval==0;}
H6_HOST_DEVICE constexpr int attention_index(int layer){return layer/FullInterval;}
H6_HOST_DEVICE constexpr int linear_index(int layer){return layer-layer/FullInterval;}
enum class WeightFormat:uint32_t {F32=0,Pane2=1,Q8Grouped128=2,Sop4=3,LookupG3=4,LookupG4=5,LookupG5=6,Q2K=7,Q4_0=8};
// Q4_0 tag8 is the aligned grouped128 cache, not raw GGUF18-byte blocks.
inline constexpr uint32_t RotationEnabled=1u,RotationBlockShift=8,RotationBlockMask=0x1fu<<RotationBlockShift;
H6_HOST_DEVICE constexpr uint32_t rotation_flags(int log2_block){return RotationEnabled|(uint32_t(log2_block)<<RotationBlockShift);}
H6_HOST_DEVICE constexpr int rotation_block(uint32_t flags){return 1<<((flags&RotationBlockMask)>>RotationBlockShift);}
H6_HOST_DEVICE constexpr bool supported_rotation_block(int b){return b==128||b==256||b==512||b==4096;}
// Data/scale/aux are immutable device pointers. Format owns its address calculation.
// F32 row_stride is bytes; PANE group tiling can use row_stride=0 (not a flat row image).
struct MatrixView {
 const void*data=nullptr;const uint16_t*scale=nullptr;const void*aux=nullptr;
 int32_t n=0,k=0,experts=1;WeightFormat format=WeightFormat::F32;
 uint32_t scale_group=0,flags=0;uint64_t row_stride=0,expert_stride=0,bytes=0;
};
// Scales are EFFECTIVE dequant scales. Strict32 mode stores half-rounded d asF32;
// integer quantization itself used unrounded amax/127. Strides count elements.
struct ActivationView {
 const int8_t*qs=nullptr;const float*scale=nullptr;
 int32_t k=0,rows=0,group=DefaultActivationGroup;uint32_t flags=0;
 int64_t q_stride=0,scale_stride=0;
};
struct ActivationBuffer {
 int8_t*qs=nullptr;float*scale=nullptr;
 int32_t k=0,rows=0,group=DefaultActivationGroup;uint32_t flags=0;
 int64_t q_stride=0,scale_stride=0;
 H6_HOST_DEVICE ActivationView view()const{return{qs,scale,k,rows,group,flags,q_stride,scale_stride};}
};
enum class LayerKind:uint32_t {GDN=0,Attention=1};
struct LayerWeights {
 LayerKind kind=LayerKind::GDN;uint32_t index=0;
 const float*input_norm=nullptr;const float*post_norm=nullptr;
 MatrixView qkv,z,alpha,beta,output,q,k,v;
 const float*ssm_a=nullptr;const float*ssm_dt=nullptr;const float*conv=nullptr;const float*ssm_norm=nullptr;
 const float*q_norm=nullptr;const float*k_norm=nullptr;
 MatrixView router,expert_gate,expert_up,expert_down,shared_gate,shared_up,shared_down,shared_router;
};
struct PreparedModel;
struct ModelWeights {
 MatrixView embedding,head;const float*final_norm=nullptr;LayerWeights layer[Layers];
 const PreparedModel*prepared=nullptr;
};
struct StateView {
 float*gdn=nullptr;float*conv=nullptr;uint16_t*key=nullptr;uint16_t*value=nullptr;
 int32_t*positions=nullptr;int32_t slots=0,context=0;
};
static_assert(sizeof(MatrixView)==72,"matrix ABI");
static_assert(sizeof(ModelWeights)<65536,"model descriptors must fit the intended constant-memory budget");
} // namespace halo6000
