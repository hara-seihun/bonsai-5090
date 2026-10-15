#pragma once
#include "model.hpp"
#include <type_traits>

namespace halo6000 {
enum class InputPolicy:uint32_t {Q8Group32=0,MXF4Group32=1,NVF4Group16=2};
enum class WeightOperandLayout:uint32_t {K32S8=0,K64FP4=1};
struct PreparedMatmulView {
 MatrixView source;
 const uint32_t*operands=nullptr;
 const uint16_t*weight_scales=nullptr;
 InputPolicy input_policy=InputPolicy::Q8Group32;
 WeightOperandLayout operand_layout=WeightOperandLayout::K32S8;
};
struct PreparedActivationView {
 ActivationView source;
 const uint32_t*operands=nullptr;
 const uint8_t*block_scales=nullptr;
 const float*outer_scale=nullptr;
 int64_t outer_scale_stride=1;
 const int32_t*row_indices=nullptr;
 int32_t columns=0,input_experts=1;
 InputPolicy input_policy=InputPolicy::Q8Group32;
};
struct MatmulBytes {uint64_t operands=0,scales=0;};
H6_HOST_DEVICE inline bool supported_matmul_source(const MatrixView&w){return w.n>0&&w.n%16==0&&w.k>0&&w.k%128==0&&w.experts>0&&(w.format==WeightFormat::Pane2||w.format==WeightFormat::Q8Grouped128);}
H6_HOST_DEVICE inline MatmulBytes matmul_weight_bytes(const MatrixView&w){uint64_t values=uint64_t(w.n)*w.k*w.experts;return {w.format==WeightFormat::Pane2?values/4:values,w.format==WeightFormat::Pane2?values/128*2:values/32*2};}
H6_HOST_DEVICE inline uint64_t matmul_activation_bytes(int columns,int k,int experts=1){return uint64_t((columns+7)/8)*(k/32)*64*4*experts;}
H6_HOST_DEVICE inline bool exact_q8_matmul(const PreparedMatmulView&w,const PreparedActivationView&a){return supported_matmul_source(w.source)&&w.operands&&w.weight_scales&&a.operands&&a.source.qs&&a.source.scale&&a.source.k==w.source.k&&a.source.group==32&&a.columns>0&&a.input_experts>0&&w.input_policy==InputPolicy::Q8Group32&&a.input_policy==InputPolicy::Q8Group32&&w.operand_layout==WeightOperandLayout::K32S8;}
static_assert(std::is_standard_layout_v<PreparedMatmulView> && std::is_trivially_copyable_v<PreparedMatmulView>);
static_assert(std::is_standard_layout_v<PreparedActivationView> && std::is_trivially_copyable_v<PreparedActivationView>);
static_assert(sizeof(PreparedMatmulView)==96 && sizeof(PreparedActivationView)==104,"prepared matmul POD ABI");
}
