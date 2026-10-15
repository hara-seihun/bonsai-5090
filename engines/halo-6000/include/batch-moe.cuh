#pragma once
#include "batch.hpp"
#include "matmul.cuh"
#include "moe.cuh"
namespace halo6000 {
__device__ __forceinline__ int bucket_block(){return blockIdx.y*gridDim.x+blockIdx.x;}
__device__ __forceinline__ int bucket_blocks(){return gridDim.y*gridDim.x;}
__device__ __forceinline__ void batch_build_buckets(const BatchView&b){
 for(int e=bucket_block()*blockDim.x+threadIdx.x;e<=Experts;e+=bucket_blocks()*blockDim.x){
  int count=0;
  for(int r=0;r<b.rows;r++)if(b.positions[r]>=0){
   if(e==Experts){b.bucket_occurrences[e*b.roster_stride+count]=r*(TopExperts+1)+TopExperts;b.bucket_hidden_rows[e*b.roster_stride+count++]=r;}
   else for(int j=0;j<TopExperts;j++)if(b.base.route_ids[r*TopExperts+j]==e){b.bucket_occurrences[e*b.roster_stride+count]=r*(TopExperts+1)+j;b.bucket_hidden_rows[e*b.roster_stride+count++]=r;}
  }
  b.bucket_counts[e]=count;
 }
}
__device__ __forceinline__ void batch_publish_buckets(const BatchView&b,int layer){
 if(bucket_block()!=0||threadIdx.x!=0)return;BucketStats stats;int tile=0;
 for(int e=0;e<=Experts;e++){
  int count=b.bucket_counts[e];
  if(e<Experts){stats.routed_occurrences+=count;stats.active_experts+=count>0;stats.singleton_experts+=count==1;stats.pair_experts+=count==2;stats.max_roster=max(stats.max_roster,count);stats.routed_tiles+=(count+7)/8;stats.padded_routed_columns+=((count+7)/8)*8;}
  else stats.shared_rows=count;
  for(int first=0;first<count;first+=8){int columns=min(8,count-first);b.bucket_tiles[tile]={e,first,columns};for(int j=0;j<columns;j++){int occurrence=b.bucket_occurrences[e*b.roster_stride+first+j];b.bucket_map[occurrence]=tile*8+j;}tile++;}
 }
 stats.total_tiles=tile;b.bucket_stats[layer]=stats;*b.bucket_tile_count=tile;
}
template<WeightFormat F,int Groups>
__device__ __forceinline__ void batch_bucket_mma(const PreparedMatmulView&w,PreparedActivationView a,float*out,int row_tile,int expert,float*shared){matmul_q8_splitk_phase<F,Groups,false,false,true>(w,a,out,row_tile,0,expert,0,1,shared);}
__device__ __forceinline__ PreparedActivationView bucket_input(const BatchView&b,BucketTile tile,bool down){
 PreparedActivationView a;a.source=(down?b.base.expert_q:b.base.hidden_q).view();a.columns=tile.columns;a.input_experts=1;a.row_indices=(down?b.bucket_occurrences:b.bucket_hidden_rows)+tile.expert*b.roster_stride+tile.offset;return a;
}
__device__ __forceinline__ void batch_bucket_gate_up(const BatchArgs&a,const PreparedLayer&p,float*shared){
 const BatchView&b=a.workspace;const int rowtiles=ExpertHidden/16;
 for(int unit=bucket_block();unit<(*b.bucket_tile_count)*rowtiles;unit+=bucket_blocks()){
  int tile_id=unit/rowtiles,row_tile=unit%rowtiles;BucketTile tile=b.bucket_tiles[tile_id];bool shared_expert=tile.expert==Experts;int expert=shared_expert?0:tile.expert;
  const auto&gate=shared_expert?p.shared_gate:p.expert_gate;const auto&up=shared_expert?p.shared_up:p.expert_up;auto input=bucket_input(b,tile,false);
  float*g=b.bucket_gate+uint64_t(tile_id)*8*ExpertHidden,*u=b.bucket_up+uint64_t(tile_id)*8*ExpertHidden;
  if(gate.source.format==WeightFormat::Pane2)batch_bucket_mma<WeightFormat::Pane2,16>(gate,input,g,row_tile,expert,shared);else batch_bucket_mma<WeightFormat::Q8Grouped128,16>(gate,input,g,row_tile,expert,shared);
  if(up.source.format==WeightFormat::Pane2)batch_bucket_mma<WeightFormat::Pane2,16>(up,input,u,row_tile,expert,shared);else batch_bucket_mma<WeightFormat::Q8Grouped128,16>(up,input,u,row_tile,expert,shared);
 }
}
__device__ __forceinline__ void batch_bucket_gate_up_quant(const BatchView&b){
 int r=blockIdx.y;if(b.positions[r]<0)return;int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32;
 for(int group=blockIdx.x*warps+warp;group<(TopExperts+1)*(ExpertHidden/32);group+=gridDim.x*warps){
  int selected=group/(ExpertHidden/32),row=(group%(ExpertHidden/32))*32+lane,occurrence=r*(TopExperts+1)+selected,index=b.bucket_map[occurrence];
  float g=b.bucket_gate[uint64_t(index)*ExpertHidden+row],u=b.bucket_up[uint64_t(index)*ExpertHidden+row],value=silu(g)*u;
  b.base.expert_gate[uint64_t(occurrence)*ExpertHidden+row]=g;b.base.expert_up[uint64_t(occurrence)*ExpertHidden+row]=u;b.base.expert_activation[uint64_t(occurrence)*ExpertHidden+row]=value;
  float amax=warp_max(fabsf(value)),d=amax/127.f;
  b.base.expert_q.qs[uint64_t(occurrence)*ExpertHidden+row]=amax==0?0:int8_t(roundf(value/d));if(lane==0)b.base.expert_q.scale[uint64_t(occurrence)*(ExpertHidden/32)+group%(ExpertHidden/32)]=__half2float(__float2half(d));
 }
}
// The affine native Q2_K slice and LG4 tree are unchanged. Several input rows
// meet each immutable weight slice before advancing to the next slice.
__device__ __forceinline__ void batch_bucket_q2_down(const MatrixView&w,PreparedActivationView a,float*out,int row_tile,int expert){
 int lane=threadIdx.x%4,groups=blockDim.x/4,row=row_tile*groups+threadIdx.x/4;
 const Q2KBlock*weights=q2k_weights(w,row,expert);float result[8]={};
 for(int i=lane;i<w.k/16;i+=4){int block=i/16;
  #pragma unroll
  for(int c=0;c<8;c++)if(c<a.columns){int ar=mm_activation_row(a,c,0);result[c]+=q2k_slice(weights[block],a.source.qs+uint64_t(ar)*a.source.q_stride+block*256,a.source.scale+uint64_t(ar)*a.source.scale_stride+block*8,i%16);}
 }
 #pragma unroll
 for(int c=0;c<8;c++)if(c<a.columns){float value=result[c];value+=__shfl_down_sync(0xffffffff,value,2,4);value+=__shfl_down_sync(0xffffffff,value,1,4);if(!lane)out[uint64_t(c)*w.n+row]=value;}
}
__device__ __forceinline__ void batch_bucket_down(const BatchArgs&a,const PreparedLayer&p,float*shared){
 const BatchView&b=a.workspace;int mma_rowtiles=Hidden/16;
 for(int unit=bucket_block();unit<(*b.bucket_tile_count)*mma_rowtiles;unit+=bucket_blocks()){
  int tile_id=unit/mma_rowtiles,row_tile=unit%mma_rowtiles;BucketTile tile=b.bucket_tiles[tile_id];bool shared_expert=tile.expert==Experts;
  const auto&w=shared_expert?p.shared_down:p.expert_down;if(w.source.format==WeightFormat::Q2K)continue;
  auto input=bucket_input(b,tile,true);float*out=b.bucket_down+uint64_t(tile_id)*8*Hidden;int expert=shared_expert?0:tile.expert;
  if(w.source.format==WeightFormat::Pane2)batch_bucket_mma<WeightFormat::Pane2,4>(w,input,out,row_tile,expert,shared);else batch_bucket_mma<WeightFormat::Q8Grouped128,4>(w,input,out,row_tile,expert,shared);
 }
 int q2_rowtiles=Hidden/(blockDim.x/4);
 for(int unit=bucket_block();unit<(*b.bucket_tile_count)*q2_rowtiles;unit+=bucket_blocks()){
  int tile_id=unit/q2_rowtiles,row_tile=unit%q2_rowtiles;BucketTile tile=b.bucket_tiles[tile_id];if(tile.expert==Experts||p.expert_down.source.format!=WeightFormat::Q2K)continue;
  batch_bucket_q2_down(p.expert_down.source,bucket_input(b,tile,true),b.bucket_down+uint64_t(tile_id)*8*Hidden,row_tile,tile.expert);
 }
}
__device__ __forceinline__ void batch_bucket_down_combine(const BatchView&b){
 int r=blockIdx.y;if(b.positions[r]<0)return;const DecodeWorkspace&w=b.row_views[r];
 for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<Hidden;row+=gridDim.x*blockDim.x){
  float routed=0;
  #pragma unroll
  for(int j=0;j<TopExperts;j++){int occurrence=r*(TopExperts+1)+j;float raw=b.bucket_down[uint64_t(b.bucket_map[occurrence])*Hidden+row];w.expert_down[j*Hidden+row]=raw;float weighted=raw*w.route_weights[j];routed=__fadd_rn(routed,weighted);}
  int occurrence=r*(TopExperts+1)+TopExperts;float raw=b.bucket_down[uint64_t(b.bucket_map[occurrence])*Hidden+row];w.expert_down[TopExperts*Hidden+row]=raw;w.hidden[row]=(routed+raw*(*w.shared_gain))+w.residual[row];
 }
}
}
