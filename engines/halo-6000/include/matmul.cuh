#pragma once
#include "matmul.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace halo6000 {
__device__ __forceinline__ uint32_t mm_prmt(uint32_t x){uint32_t v;asm("prmt.b32 %0,0x00000100,0x0000ff00,%1;":"=r"(v):"r"(x));return v;}
__device__ __forceinline__ const uint8_t*mm_expert(const MatrixView&w,int expert){uint64_t bytes=w.expert_stride?w.expert_stride:(w.format==WeightFormat::Q8Grouped128&&w.row_stride?uint64_t(w.n)*w.row_stride:uint64_t(w.n)*w.k/128*(w.format==WeightFormat::Pane2?34:136));return static_cast<const uint8_t*>(w.data)+expert*bytes;}
template<WeightFormat F>__device__ __forceinline__ uint32_t mm_source_word(const MatrixView&w,int row,int col,int expert){
 const uint8_t*base=mm_expert(w,expert);
 if constexpr(F==WeightFormat::Pane2){
  uint64_t b=uint64_t(row)*(w.k/128)+col/128;
  const uint32_t*p=reinterpret_cast<const uint32_t*>(base+(b/16)*544+(b%16)*32);
  uint32_t v=p[(col%128)/16];int part=(col%16)/4;
  uint32_t selector=(v>>(part>=2?1:0))&0x55555555;
  return mm_prmt(selector>>(part%2*16));
 }else{
  uint64_t stride=w.row_stride?w.row_stride:uint64_t(w.k/128)*136;
  const uint8_t*p=base+row*stride+(col/128)*136+col%128;
  return *reinterpret_cast<const uint32_t*>(p);
 }
}
template<WeightFormat F>__device__ __forceinline__ uint16_t mm_source_scale(const MatrixView&w,int row,int group,int expert){
 const uint8_t*base=mm_expert(w,expert);
 if constexpr(F==WeightFormat::Pane2){uint64_t b=uint64_t(row)*(w.k/128)+group;return *reinterpret_cast<const uint16_t*>(base+(b/16)*544+512+(b%16)*2);}
 else {uint64_t stride=w.row_stride?w.row_stride:uint64_t(w.k/128)*136;return *reinterpret_cast<const uint16_t*>(base+row*stride+(group/4)*136+128+(group%4)*2);}
}

// Immutable load-time image. PANE remains2.125bpw; Q8 remains8.5bpw. Rotation
// signs stay in source.aux; this map never rotates or requantizes weights.
template<WeightFormat F>__global__ void prepare_matmul_weights(MatrixView w,uint32_t*out,uint16_t*scales){
 static_assert(F==WeightFormat::Pane2||F==WeightFormat::Q8Grouped128);
 uint64_t id=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,tile=id/32;
 int lane=id%32,ktiles=w.k/32;
 if(tile>=uint64_t(w.n/16)*ktiles*w.experts)return;
 int expert=tile/(uint64_t(w.n/16)*ktiles),m=tile/ktiles%(w.n/16),g=tile%ktiles;
 uint32_t code=0;
 #pragma unroll
 for(int j=0;j<4;j++){
  int row=m*16+lane/4+(j%2?8:0),col=g*32+(lane%4)*4+(j>=2?16:0);
  uint32_t v=mm_source_word<F>(w,row,col,expert);
  if constexpr(F==WeightFormat::Q8Grouped128)out[(tile*32+lane)*4+j]=v;
  else {
   #pragma unroll
   for(int i=0;i<4;i++){int value=int(int8_t(v>>(i*8))),e=j*4+i;code|=uint32_t(value<0?5:value>0?1:0)<<(4*(e%8)+(e/8));}
  }
 }
 if constexpr(F==WeightFormat::Pane2)out[tile*32+lane]=code;
 if(lane%4==0){
  #pragma unroll
  for(int h=0;h<2;h++){
   int row=m*16+lane/4+h*8;
   if constexpr(F==WeightFormat::Pane2){if(g%4==0)scales[(uint64_t(expert)*w.n+row)*(w.k/128)+g/4]=mm_source_scale<F>(w,row,g/4,expert);}
   else scales[(uint64_t(expert)*w.n+row)*(w.k/32)+g]=mm_source_scale<F>(w,row,g,expert);
  }
 }
}
__device__ __forceinline__ int mm_activation_row(const PreparedActivationView&a,int column,int expert){return a.row_indices?a.row_indices[expert*a.columns+column]:column*a.input_experts+expert;}

// Producer-owned B image, built once and shared by every matrix reading that
// activation. INT8 and effective group32 scales were already created after D_H.
__device__ __forceinline__ uint32_t prepare_matmul_activation_word(PreparedActivationView a,uint64_t id){
 int ct=(a.columns+7)/8,kt=a.source.k/32;
 int h=id%2,lane=id/2%32,g=id/64%kt,t=id/(64*kt)%ct,e=id/(uint64_t(ct)*kt*64),column=t*8+lane/4;
 uint32_t v=0;
 if(column<a.columns){int ar=mm_activation_row(a,column,e);if(ar>=0){const int8_t*q=a.source.qs+ar*a.source.q_stride+g*32+(lane%4)*4+h*16;v=*reinterpret_cast<const uint32_t*>(q);}}
 return v;
}
// Grid-striding phase for a resident owner; producer quantization must finish
// before entry and the consumer needs the owner's whole-grid publication barrier.
__device__ __forceinline__ void prepare_matmul_activation_phase(PreparedActivationView a,uint32_t*out){
 uint64_t count=matmul_activation_bytes(a.columns,a.source.k,a.input_experts)/4;
 for(uint64_t id=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;id<count;id+=uint64_t(gridDim.x)*blockDim.x)out[id]=prepare_matmul_activation_word(a,id);
}
static __global__ void prepare_matmul_activations(PreparedActivationView a,uint32_t*out){prepare_matmul_activation_phase(a,out);}
template<WeightFormat F>__device__ __forceinline__ void mm_operand(const uint32_t*p,int lane,uint32_t a[4]){
 if constexpr(F==WeightFormat::Pane2){uint32_t v=p[lane],lo=v&0x55555555,hi=(v>>1)&0x55555555;a[0]=mm_prmt(lo);a[1]=mm_prmt(lo>>16);a[2]=mm_prmt(hi);a[3]=mm_prmt(hi>>16);}
 else {
  #pragma unroll
  for(int j=0;j<4;j++)a[j]=p[lane*4+j];
 }
}
template<bool DirectActivation=false>
__device__ __forceinline__ void mm_input_fragment(const PreparedActivationView&a,int column_tile,int k,int input_expert,int lane,uint32_t&b0,uint32_t&b1){
 if constexpr(DirectActivation){
  b0=0;b1=0;int column=column_tile*8+lane/4;
  if(column<a.columns){int ar=mm_activation_row(a,column,input_expert);if(ar>=0){const int8_t*q=a.source.qs+uint64_t(ar)*a.source.q_stride+k*32+(lane%4)*4;b0=*reinterpret_cast<const uint32_t*>(q);b1=*reinterpret_cast<const uint32_t*>(q+16);}}
 }else{int ct=(a.columns+7)/8;uint64_t p=((uint64_t(input_expert)*ct+column_tile)*(a.source.k/32)+k)*64+lane*2;b0=a.operands[p];b1=a.operands[p+1];}
}
struct MatmulIntegerGuard {const int32_t*reference=nullptr;unsigned long long*different=nullptr;int32_t first_row=0,rows=0;};

// Output [column][expert_count][N], last index contiguous. group128 retains the
// source's inner group32 scale application; outer groups accumulate sequentially
// rather than the scalar warp's tree. Quantify this difference, don't call it bits.
template<WeightFormat F,int TokenTiles=1,bool Guard=false>
__global__ void matmul_q8(PreparedMatmulView w,PreparedActivationView a,float*out,int first_expert=0,int expert_count=1,MatmulIntegerGuard guard={}){
 static_assert(F==WeightFormat::Pane2||F==WeightFormat::Q8Grouped128);
 int lane=threadIdx.x%32,warp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(warp>=w.source.n/16)return;
 if constexpr(Guard){if(warp*16>=guard.first_row+guard.rows||(warp+1)*16<=guard.first_row)return;}
 int expert=first_expert+blockIdx.z,input_expert=a.input_experts==1?0:blockIdx.z,ct=(a.columns+7)/8,kt=w.source.k/32;
 float result[TokenTiles][4]={};
 for(int group=0;group<w.source.k/128;group++){
  float partial[TokenTiles][4]={};
  #pragma unroll
  for(int h=0;h<4;h++){
   int k=group*4+h;uint32_t weights[4];constexpr int words=F==WeightFormat::Pane2?32:128;
   mm_operand<F>(w.operands+((uint64_t(expert)*(w.source.n/16)+warp)*kt+k)*words,lane,weights);
   #pragma unroll
   for(int t=0;t<TokenTiles;t++){
    int tile=blockIdx.y*TokenTiles+t;uint32_t b0=0,b1=0;
    if(tile<ct){uint64_t p=((uint64_t(input_expert)*ct+tile)*kt+k)*64+lane*2;b0=a.operands[p];b1=a.operands[p+1];}
    int c[4]={};
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(weights[0]),"r"(weights[1]),"r"(weights[2]),"r"(weights[3]),"r"(b0),"r"(b1));
    #pragma unroll
    for(int j=0;j<4;j++){
     int row=warp*16+lane/4+(j>=2?8:0),col=tile*8+(lane%4)*2+j%2;
     if(col<a.columns){
      if constexpr(Guard){if(row>=guard.first_row&&row<guard.first_row+guard.rows){uint64_t index=((uint64_t(col)*expert_count+blockIdx.z)*guard.rows+row-guard.first_row)*kt+k;if(c[j]!=guard.reference[index])atomicAdd(guard.different,1ull);}}
      int ar=mm_activation_row(a,col,input_expert);float ds=ar<0?0.f:a.source.scale[ar*a.source.scale_stride+k];
      if constexpr(F==WeightFormat::Q8Grouped128){uint16_t bits=w.weight_scales[(uint64_t(expert)*w.source.n+row)*kt+k];ds=__fmul_rn(ds,__half2float(*reinterpret_cast<half*>(&bits)));}
      partial[t][j]=__fmaf_rn(float(c[j]),ds,partial[t][j]);
     }
    }
   }
  }
  #pragma unroll
  for(int t=0;t<TokenTiles;t++){
   #pragma unroll
   for(int j=0;j<4;j++){
    float value=partial[t][j];
    if constexpr(F==WeightFormat::Pane2){int row=warp*16+lane/4+(j>=2?8:0);uint16_t bits=w.weight_scales[(uint64_t(expert)*w.source.n+row)*(w.source.k/128)+group];value=__fmul_rn(value,__half2float(*reinterpret_cast<half*>(&bits)));}
    result[t][j]=__fadd_rn(result[t][j],value);
   }
  }
 }
 #pragma unroll
 for(int t=0;t<TokenTiles;t++){
  #pragma unroll
  for(int j=0;j<4;j++){int row=warp*16+lane/4+(j>=2?8:0),col=(blockIdx.y*TokenTiles+t)*8+(lane%4)*2+j%2;if(col<a.columns)out[(uint64_t(col)*expert_count+blockIdx.z)*w.source.n+row]=result[t][j];}
 }
}

// Cooperating CTA K128 groups; original four K32 dots and scale boundaries.
// Final reverse-butterfly equals quantized_row's XOR tree at K512/2048/4096.
// Resident owners may call the phase: all threads enter, blockDim128 or256.
template<WeightFormat F,int Groups,bool Guard=false,bool SingleColumn=false,bool DirectActivation=false>
__device__ __forceinline__ void matmul_q8_splitk_phase(PreparedMatmulView w,PreparedActivationView a,float*out,int row_tile,int column_tile,int expert,int output_expert,int expert_count,float*scratch,MatmulIntegerGuard guard={}){
 static_assert(Groups==4||Groups==16||Groups==32);
 int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32,kt=Groups*4,ct=(a.columns+7)/8,input_expert=a.input_experts==1?0:output_expert;
 for(int group=warp;group<Groups;group+=warps){
  float partial[4]={};
  #pragma unroll
  for(int h=0;h<4;h++){
   int k=group*4+h;constexpr int words=F==WeightFormat::Pane2?32:128;uint32_t weights[4];
   mm_operand<F>(w.operands+((uint64_t(expert)*(w.source.n/16)+row_tile)*kt+k)*words,lane,weights);
   uint32_t b0,b1;mm_input_fragment<DirectActivation>(a,column_tile,k,input_expert,lane,b0,b1);int c[4]={};
   asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(weights[0]),"r"(weights[1]),"r"(weights[2]),"r"(weights[3]),"r"(b0),"r"(b1));
   #pragma unroll
   for(int j=0;j<4;j++){
    int row=row_tile*16+lane/4+(j>=2?8:0),col=column_tile*8+(lane%4)*2+j%2;
    if(SingleColumn?col==0:col<a.columns){
     if constexpr(Guard){if(row>=guard.first_row&&row<guard.first_row+guard.rows){uint64_t index=((uint64_t(col)*expert_count+output_expert)*guard.rows+row-guard.first_row)*kt+k;if(c[j]!=guard.reference[index])atomicAdd(guard.different,1ull);}}
     int ar=mm_activation_row(a,col,input_expert);float ds=ar<0?0.f:a.source.scale[ar*a.source.scale_stride+k];
     if constexpr(F==WeightFormat::Q8Grouped128){uint16_t bits=w.weight_scales[(uint64_t(expert)*w.source.n+row)*kt+k];ds=__fmul_rn(ds,__half2float(*reinterpret_cast<half*>(&bits)));}
     partial[j]=__fmaf_rn(float(c[j]),ds,partial[j]);
    }
   }
  }
  #pragma unroll
  for(int j=0;j<4;j++){
   int row=row_tile*16+lane/4+(j>=2?8:0),col=(lane%4)*2+j%2;
   if constexpr(SingleColumn){if(col!=0)continue;}
   float v=partial[j];
   if constexpr(F==WeightFormat::Pane2){uint16_t bits=w.weight_scales[(uint64_t(expert)*w.source.n+row)*Groups+group];v=__fmul_rn(v,__half2float(*reinterpret_cast<half*>(&bits)));}
   if constexpr(SingleColumn){if(col==0)scratch[group*16+row%16]=__fadd_rn(0.f,v);}
   else scratch[group*128+col*16+row%16]=__fadd_rn(0.f,v);
  }
 }
 __syncthreads();
 constexpr int Outputs=SingleColumn?16:128;
 if(threadIdx.x<Outputs){
  float values[Groups];
  #pragma unroll
  for(int g=0;g<Groups;g++)values[g]=scratch[g*Outputs+threadIdx.x];
  #pragma unroll
  for(int d=Groups/2;d;d>>=1){
   #pragma unroll
   for(int g=0;g<d;g++)values[g]=__fadd_rn(values[g],values[g+d]);
  }
  int row=row_tile*16+threadIdx.x%16,col=column_tile*8+threadIdx.x/16;
  if(col<a.columns)out[(uint64_t(col)*expert_count+output_expert)*w.source.n+row]=values[0];
 }
 __syncthreads();
}
// Wide grid=N/16*ceil(columns/8)*experts, or resident188 grid-striding.
// This is a CTA phase, not a cooperative whole-grid synchronization boundary.
template<WeightFormat F,int Groups,bool Guard=false>
__global__ void matmul_q8_splitk(PreparedMatmulView w,PreparedActivationView a,float*out,int first_expert=0,int expert_count=1,MatmulIntegerGuard guard={}){
 __shared__ float scratch[Groups*128];int rt=w.source.n/16,ct=(a.columns+7)/8;
 for(uint64_t tile=blockIdx.x;tile<uint64_t(rt)*ct*expert_count;tile+=gridDim.x){
  int row=tile%rt,col=tile/rt%ct,e=tile/(uint64_t(rt)*ct);
  if constexpr(Guard){if(row*16>=guard.first_row+guard.rows||(row+1)*16<=guard.first_row)continue;}
  matmul_q8_splitk_phase<F,Groups,Guard>(w,a,out,row,col,first_expert+e,e,expert_count,scratch,guard);
 }
}
// Decode-only phase: same integer MMA, scales and logical32 tree; just16 real
// output lanes instead of128. Scratch Groups*16 floats (256B/1KiB/2KiB).
template<WeightFormat F,int Groups,bool Guard=false>
__device__ __forceinline__ void matmul_q8_s1_phase(PreparedMatmulView w,PreparedActivationView a,float*out,int row_tile,int expert,int output_expert,int expert_count,float*scratch,MatmulIntegerGuard guard={}){
 matmul_q8_splitk_phase<F,Groups,Guard,true>(w,a,out,row_tile,0,expert,output_expert,expert_count,scratch,guard);
}
template<WeightFormat F,int Groups,bool Guard=false>
__global__ void matmul_q8_s1(PreparedMatmulView w,PreparedActivationView a,float*out,int first_expert=0,int expert_count=1,MatmulIntegerGuard guard={}){
 __shared__ float scratch[Groups*16];int rt=w.source.n/16;
 if(a.columns!=1||w.source.k!=Groups*128){asm volatile("trap;");return;}
 for(uint64_t tile=blockIdx.x;tile<uint64_t(rt)*expert_count;tile+=gridDim.x){
  int row=tile%rt,e=tile/rt;
  if constexpr(Guard){if(row*16>=guard.first_row+guard.rows||(row+1)*16<=guard.first_row)continue;}
  matmul_q8_s1_phase<F,Groups,Guard>(w,a,out,row,first_expert+e,e,expert_count,scratch,guard);
 }
}
}
