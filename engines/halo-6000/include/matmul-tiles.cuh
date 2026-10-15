#pragma once
#include "matmul.cuh"

namespace halo6000 {
// Each K128 group uses one A-register load for TokenTiles B tiles. The exact
// per-column scale/FMA/tree map is unchanged; scratch is TokenTiles*Groups*128.
template<WeightFormat F,int Groups,int TokenTiles,bool Guard=false>
__device__ __forceinline__ void matmul_q8_splitk_tiles_phase(
 PreparedMatmulView w,PreparedActivationView a,float*out,int row_tile,
 int first_column_tile,int expert,int output_expert,int expert_count,
 float*scratch,MatmulIntegerGuard guard={}) {
 static_assert(F==WeightFormat::Pane2||F==WeightFormat::Q8Grouped128);
 static_assert(Groups==4||Groups==16||Groups==32);
 static_assert(TokenTiles==2||TokenTiles==4);
 int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32;
 int kt=Groups*4,ct=(a.columns+7)/8,input_expert=a.input_experts==1?0:output_expert;
 for(int group=warp;group<Groups;group+=warps){
  float partial[TokenTiles][4]={};
  #pragma unroll
  for(int h=0;h<4;h++){
   int k=group*4+h;constexpr int words=F==WeightFormat::Pane2?32:128;
   uint32_t weights[4];
   mm_operand<F>(w.operands+((uint64_t(expert)*(w.source.n/16)+row_tile)*kt+k)*words,lane,weights);
   #pragma unroll
   for(int t=0;t<TokenTiles;t++){
    int tile=first_column_tile+t;uint32_t b0=0,b1=0;
    if(tile<ct){uint64_t p=((uint64_t(input_expert)*ct+tile)*kt+k)*64+lane*2;b0=a.operands[p];b1=a.operands[p+1];}
    int c[4]={};
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(weights[0]),"r"(weights[1]),"r"(weights[2]),"r"(weights[3]),"r"(b0),"r"(b1));
    #pragma unroll
    for(int j=0;j<4;j++){
     int row=row_tile*16+lane/4+(j>=2?8:0),col=tile*8+(lane%4)*2+j%2;
     if(col<a.columns){
      if constexpr(Guard){if(row>=guard.first_row&&row<guard.first_row+guard.rows){uint64_t index=((uint64_t(col)*expert_count+output_expert)*guard.rows+row-guard.first_row)*kt+k;if(c[j]!=guard.reference[index])atomicAdd(guard.different,1ull);}}
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
    int row=row_tile*16+lane/4+(j>=2?8:0),col=(lane%4)*2+j%2;float value=partial[t][j];
    if constexpr(F==WeightFormat::Pane2){uint16_t bits=w.weight_scales[(uint64_t(expert)*w.source.n+row)*Groups+group];value=__fmul_rn(value,__half2float(*reinterpret_cast<half*>(&bits)));}
    scratch[(t*Groups+group)*128+col*16+row%16]=__fadd_rn(0.f,value);
   }
  }
 }
 __syncthreads();
 for(int output=threadIdx.x;output<TokenTiles*128;output+=blockDim.x){
  int t=output/128,o=output%128;float values[Groups];
  #pragma unroll
  for(int g=0;g<Groups;g++)values[g]=scratch[(t*Groups+g)*128+o];
  #pragma unroll
  for(int d=Groups/2;d;d>>=1){
   #pragma unroll
   for(int g=0;g<d;g++)values[g]=__fadd_rn(values[g],values[g+d]);
  }
  int row=row_tile*16+o%16,col=(first_column_tile+t)*8+o/16;
  if(col<a.columns)out[(uint64_t(col)*expert_count+output_expert)*w.source.n+row]=values[0];
 }
 __syncthreads();
}

template<WeightFormat F,int Groups,int TokenTiles,bool Guard=false>
__global__ void matmul_q8_splitk_tiles(PreparedMatmulView w,PreparedActivationView a,
 float*out,int first_expert=0,int expert_count=1,MatmulIntegerGuard guard={}) {
 extern __shared__ float scratch[];
 int rt=w.source.n/16,ct=((a.columns+7)/8+TokenTiles-1)/TokenTiles;
 for(uint64_t tile=blockIdx.x;tile<uint64_t(rt)*ct*expert_count;tile+=gridDim.x){
  int row=tile%rt,col=tile/rt%ct,e=tile/(uint64_t(rt)*ct);
  if constexpr(Guard){if(row*16>=guard.first_row+guard.rows||(row+1)*16<=guard.first_row)continue;}
  matmul_q8_splitk_tiles_phase<F,Groups,TokenTiles,Guard>(w,a,out,row,col*TokenTiles,first_expert+e,e,expert_count,scratch,guard);
 }
}

// Call once at setup; the K4096/T4 wrapper needs64KiB dynamic shared memory.
template<WeightFormat F,int Groups,int TokenTiles,bool Guard=false>
inline cudaError_t configure_matmul_q8_splitk_tiles(){
 constexpr int bytes=TokenTiles*Groups*128*sizeof(float);
 return cudaFuncSetAttribute(matmul_q8_splitk_tiles<F,Groups,TokenTiles,Guard>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
}
}
