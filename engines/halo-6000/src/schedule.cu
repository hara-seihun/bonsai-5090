// Scheduled whole token (see schedule.hpp). Bodies are the engine's existing,
// bit-exact phase helpers; this file only chooses among them (tune.hpp), cuts
// the epoch sequence into segment kernels and replays the token as a graph.
#include "schedule.hpp"
#include "tune.hpp"
#include "phases.cuh"
#include "matmul.cuh"
#include "router-bf16.cuh"
#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
namespace halo6000 {
namespace {
__host__ __device__ constexpr uint32_t bit(EpochKind k){return 1u<<int(k);}
constexpr uint32_t RotKinds=bit(EpochKind::RotWriter)|bit(EpochKind::RotDown);
static_assert(int(EpochKind::Count)<=32,"epoch kinds fit a mask");
static_assert(tune::segment_count>=1&&tune::segment_count<=32,"segment classes");
inline constexpr int ArgPartitions=1024;  // fixed virtual partitions; argmax is order-invariant

struct SegArgs {
 const ModelWeights*model;StateView state;DecodeWorkspace w;const Epoch*epochs;const int32_t*teacher;uint64_t*stamps;const uint16_t*router_bf16;
 int first,last,total,slot;
};
__device__ __forceinline__ uint64_t globaltimer(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}

// Rows of one matrix with L physical lanes per row. offset spreads the tails of
// consecutive matrices in one epoch over different warp groups, as one
// concatenated unit list would; it never changes a row's arithmetic.
template<int L>
__device__ __forceinline__ void rows_exact(const MatrixView&m,int n,int k,ActivationView a,const float*x,float*out,int offset){
 const int per=blockDim.x/L,groups=gridDim.x*per,gid=blockIdx.x*per+threadIdx.x/L,lane=threadIdx.x&(L-1);
 MatrixView f=m;f.k=k;
 for(int row=((gid-offset)%groups+groups)%groups;row<n;row+=groups){const float v=row_exact<L>(f,a,x,row);if(!lane)out[row]=v;}
}
__device__ __forceinline__ void norm_quant_cta(const float*x,const float*gain,DecodeWorkspace w,float*shared){
 norm_phase<Hidden>(x,gain,w.normalized,shared);quantize_cta(w.normalized,w.hidden_q);
}
// Shared layout for fused prologues (floats): [0,32) reduction scratch,
// then a CTA-private normalized row, its Q8 bytes and scales, and a private
// router selection. Rotation reuses the area from 0 as its own scratch.
inline constexpr int ShNorm=32,ShQs=ShNorm+Hidden,ShDs=ShQs+Hidden/4,ShProbs=ShDs+Hidden/32,ShIds=ShProbs+Experts,ShWeights=ShIds+TopExperts,ShTile=ShWeights+TopExperts,ShFused=ShTile+32;
// Prepared-MMA activation operands (CTA-private, up to K4096) and split-K scratch.
inline constexpr int ShMma=ShFused,ShMmaScratch=ShMma+8192,ShMmaEnd=ShMmaScratch+512;
template<uint32_t K>__host__ __device__ constexpr bool uses_mma(){
 return (tune::mma_proj&&(K&(bit(EpochKind::GdnProj)|bit(EpochKind::AttnProj))))||(tune::mma_out&&(K&bit(EpochKind::OutProj)))||(tune::mma_head&&(K&bit(EpochKind::Head)));
}
template<uint32_t K>__host__ __device__ constexpr int shared_floats(){
 int n=32;
 if((K&RotKinds)!=0)n=4096;
 constexpr uint32_t fused=bit(EpochKind::GdnProj)|bit(EpochKind::AttnProj)|bit(EpochKind::Router)|bit(EpochKind::GateUp);
 if((tune::fuse_norm||tune::fuse_select)&&(K&fused)!=0&&n<ShFused)n=ShFused;
 if(uses_mma<K>()&&n<ShMmaEnd)n=ShMmaEnd;
 return n;
}
// Each CTA builds its own MMA B operands from the published (or CTA-private) Q8
// activation; no cross-CTA operand buffer, so no extra publication barrier.
__device__ __forceinline__ PreparedActivationView mma_local(PreparedActivationView a,ActivationView q,float*shared){
 a.source=q;a.columns=1;a.input_experts=1;a.row_indices=nullptr;
 uint32_t*out=reinterpret_cast<uint32_t*>(shared+ShMma);
 const uint64_t count=matmul_activation_bytes(1,q.k,1)/4;
 for(uint64_t id=threadIdx.x;id<count;id+=blockDim.x)out[id]=prepare_matmul_activation_word(a,id);
 __syncthreads();a.operands=out;return a;
}
template<int Groups>
__device__ __forceinline__ void mma_rows(const PreparedMatmulView&w,const PreparedActivationView&a,float*out,int offset,float*shared){
 const int tiles=w.source.n/16;float*scratch=shared+ShMmaScratch;
 for(int t=((int(blockIdx.x)-offset)%int(gridDim.x)+int(gridDim.x))%int(gridDim.x);t<tiles;t+=gridDim.x){
  if(w.source.format==WeightFormat::Pane2)matmul_q8_s1_phase<WeightFormat::Pane2,Groups>(w,a,out,t,0,0,1,scratch);
  else matmul_q8_s1_phase<WeightFormat::Q8Grouped128,Groups>(w,a,out,t,0,0,1,scratch);
 }
}
__device__ __forceinline__ ActivationBuffer local_q8(float*shared){
 ActivationBuffer a;a.qs=reinterpret_cast<int8_t*>(shared+ShQs);a.scale=shared+ShDs;a.k=Hidden;a.rows=1;a.group=32;a.q_stride=Hidden;a.scale_stride=Hidden/32;return a;
}
// Every CTA computes the same norm+Q8 with the same instructions on the same
// inputs, so its private copy is bit-identical to the one-owner epoch. Block 0
// publishes the global copies later epochs and debug dumps read.
__device__ __forceinline__ void norm_local(const float*x,const float*add,const float*gain,float*shared,const DecodeWorkspace&w,float*residual_out){
 float*row=shared+ShNorm;
 if(add){for(int i=threadIdx.x;i<Hidden;i+=blockDim.x)row[i]=x[i]+add[i];}else{for(int i=threadIdx.x;i<Hidden;i+=blockDim.x)row[i]=x[i];}
 __syncthreads();
 norm_phase<Hidden>(row,gain,row,shared);
 ActivationBuffer q=local_q8(shared);quantize_cta(row,q);
 __syncthreads();
 if(blockIdx.x==0){
  for(int i=threadIdx.x;i<Hidden;i+=blockDim.x){w.normalized[i]=row[i];w.hidden_q.qs[i]=q.qs[i];if(residual_out)residual_out[i]=x[i]+add[i];}
  for(int i=threadIdx.x;i<Hidden/32;i+=blockDim.x)w.hidden_q.scale[i]=q.scale[i];
 }
}
// L2 prefetch of a byte range, split over the grid, one 128-byte line per thread step.
__device__ __forceinline__ void prefetch_l2(const void*p,size_t bytes){
 if(!p||!bytes)return;
 const char*b=static_cast<const char*>(p);const size_t lines=(bytes+127)/128;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<lines;i+=size_t(gridDim.x)*blockDim.x)asm volatile("prefetch.global.L2 [%0];"::"l"(b+i*128));
}
__device__ __forceinline__ void prefetch_matrix(const MatrixView&m){prefetch_l2(m.data,m.bytes);}
__device__ __forceinline__ void prefetch_expert(const MatrixView&m,int e){const size_t per=m.bytes/(m.experts>0?m.experts:1);prefetch_l2(static_cast<const char*>(m.data)+size_t(e)*per,per);}
__device__ __forceinline__ void prefetch_projection(const LayerWeights&l){
 if(l.kind==LayerKind::GDN){prefetch_matrix(l.qkv);prefetch_matrix(l.z);prefetch_matrix(l.alpha);prefetch_matrix(l.beta);}
 else{prefetch_matrix(l.q);prefetch_matrix(l.k);prefetch_matrix(l.v);}
}

// ---- Body genes: more independent rows in flight per lane group. Every row keeps
// the canonical arithmetic: each logical lane's serial group sum (starting from
// +0.0f exactly as row_exact) and the descending-XOR tree; only issue order and
// load instructions change.
template<bool NC,class T>__device__ __forceinline__ T ldw(const T*p){if constexpr(NC)return __ldg(p);else return *p;}
template<bool NC>
__device__ __forceinline__ float pane_group_g(const MatrixView&w,ActivationView a,int row,int group,int expert=0,int ar=0){
 const size_t b=(size_t(expert)*w.n+row)*(w.k/128)+group;
 const auto*base=static_cast<const char*>(w.data)+(b>>4)*544;
 const auto*words=reinterpret_cast<const uint32_t*>(base+(b&15)*32);
 uint32_t wv[8];
 if constexpr(NC){const uint4 lo=__ldg(reinterpret_cast<const uint4*>(words)),hi=__ldg(reinterpret_cast<const uint4*>(words)+1);wv[0]=lo.x;wv[1]=lo.y;wv[2]=lo.z;wv[3]=lo.w;wv[4]=hi.x;wv[5]=hi.y;wv[6]=hi.z;wv[7]=hi.w;}
 else{
  #pragma unroll
  for(int i=0;i<8;i++)wv[i]=words[i];
 }
 const int8_t*q=a.qs+ar*a.q_stride+group*128;const float*s=a.scale+ar*a.scale_stride+group*4;
 float v=0;
 #pragma unroll
 for(int c=0;c<4;c++){
  int isum=0;
  #pragma unroll
  for(int h=0;h<2;h++){
   const uint32_t p=wv[2*c+h],lo=p&0x55555555u,hi=(p>>1)&0x55555555u;
   isum=__dp4a(pane_operand(lo),load_i8x4(q+c*32+h*16),isum);
   isum=__dp4a(pane_operand(lo>>16),load_i8x4(q+c*32+h*16+4),isum);
   isum=__dp4a(pane_operand(hi),load_i8x4(q+c*32+h*16+8),isum);
   isum=__dp4a(pane_operand(hi>>16),load_i8x4(q+c*32+h*16+12),isum);
  }
  v+=s[c]*float(isum);
 }
 return half_value(reinterpret_cast<const uint16_t*>(base+512+(b&15)*2))*v;
}
template<bool NC>
__device__ __forceinline__ float group_g(const MatrixView&w,ActivationView a,int row,int group,int expert=0,int ar=0){
 return w.format==WeightFormat::Pane2?pane_group_g<NC>(w,a,row,group,expert,ar):q8_group(w,a,row,group,expert,ar);
}
// Logical-lane partials of one row (row_exact's first half), F32/PANE/Q8.
template<int L,bool NC>
__device__ __forceinline__ void row_parts(const MatrixView&w,ActivationView a,const float*x,int row,float*part){
 constexpr int Slots=32/L;const int lane=threadIdx.x&(L-1);
 #pragma unroll
 for(int s=0;s<Slots;s++){
  float v=0;const int logical=lane+s*L;
  if(w.format==WeightFormat::F32){const auto*p=reinterpret_cast<const float*>(static_cast<const char*>(w.data)+row*w.row_stride);
   for(int i=logical;i<w.k;i+=32)v=fmaf(ldw<NC>(p+i),x[i],v);
  }else{for(int b=logical;b<w.k/128;b+=32)v+=group_g<NC>(w,a,row,b);}
  part[s]=v;
 }
}
// row_exact's second half: fold slots, then the XOR tree over L physical lanes.
template<int L>
__device__ __forceinline__ float row_fold(float*part){
 #pragma unroll
 for(int d=16;d>=L;d>>=1){
  #pragma unroll
  for(int s=0;s<d/L;s++)part[s]+=part[s+d/L];
 }
 float v=part[0];
 #pragma unroll
 for(int d=L/2;d;d>>=1)v+=__shfl_xor_sync(0xffffffffu,v,d,L);
 return v;
}
// R rows per lane group per iteration; out-of-range rows recompute the last row
// and are discarded so every lane enters every shuffle.
template<int L,int R,bool NC>
__device__ __forceinline__ void rows_multi(const MatrixView&m,int n,int k,ActivationView a,const float*x,float*out,int offset){
 if constexpr(R==1){rows_exact<L>(m,n,k,a,x,out,offset);return;}
 if(m.format!=WeightFormat::F32&&m.format!=WeightFormat::Pane2&&m.format!=WeightFormat::Q8Grouped128){rows_exact<L>(m,n,k,a,x,out,offset);return;}
 constexpr int Slots=32/L;
 const int per=blockDim.x/L,groups=gridDim.x*per,gid=blockIdx.x*per+threadIdx.x/L,lane=threadIdx.x&(L-1);
 MatrixView f=m;f.k=k;
 const int start=((gid-offset)%groups+groups)%groups;
 // Uniform trip count across the warp: every lane group iterates while any row remains.
 const int trips=(n+groups*R-1)/(groups*R);
 for(int t=0;t<trips;t++){
  float part[R][Slots];
  #pragma unroll
  for(int j=0;j<R;j++){const int row=start+(t*R+j)*groups;row_parts<L,NC>(f,a,x,row<n?row:n-1,part[j]);}
  #pragma unroll
  for(int j=0;j<R;j++){const int row=start+(t*R+j)*groups;const float v=row_fold<L>(part[j]);if(!lane&&row<n)out[row]=v;}
 }
}
// moe_down with R output rows per warp in flight (PANE routed+shared down only;
// other formats use the canonical moe_down).
template<int R,bool NC>
__device__ __forceinline__ void down_multi(const MoeView&v,MoeScratch s){
 const LayerWeights&l=*v.weights;
 if constexpr(R==1){moe_down(v,s);return;}
 if(l.expert_down.format!=WeightFormat::Pane2||l.shared_down.format!=WeightFormat::Pane2||l.expert_down.k!=ExpertHidden){moe_down(v,s);return;}
 const int lane=threadIdx.x&31,warp=threadIdx.x/32,warps=blockDim.x/32,selected=lane/4,sub=lane&3;
 const ActivationView a=s.prepared.view();const int expert=s.ids[selected];const float wsel=s.weights[selected];const float gain=*s.shared_gain;
 const int total=gridDim.x*warps,first=blockIdx.x*warps+warp,trips=(Hidden+total*R-1)/(total*R);
 for(int t=0;t<trips;t++){
  float routed_part[R],shared_part[R];
  #pragma unroll
  for(int j=0;j<R;j++){
   int row=first+(t*R+j)*total;if(row>=Hidden)row=Hidden-1;
   float rv=0;rv+=pane_group_g<NC>(l.expert_down,a,row,sub,expert,selected);routed_part[j]=rv;
   float sv=0;if(lane<ExpertHidden/128)sv+=pane_group_g<NC>(l.shared_down,a,row,lane,0,TopExperts);shared_part[j]=sv;
  }
  #pragma unroll
  for(int j=0;j<R;j++){
   const int row=first+(t*R+j)*total;
   float raw=routed_part[j];
   #pragma unroll
   for(int d=2;d;d>>=1)raw+=__shfl_xor_sync(0xffffffffu,raw,d,4);
   if(s.down&&sub==0&&row<Hidden)s.down[selected*Hidden+row]=raw;
   const float weighted=raw*wsel;float routed=0;
   #pragma unroll
   for(int e=0;e<TopExperts;e++){const float term=__shfl_sync(0xffffffffu,weighted,e*4);routed=__fadd_rn(routed,term);}
   const float shared=warp_sum(shared_part[j]);
   if(lane==0&&row<Hidden){if(s.down)s.down[TopExperts*Hidden+row]=shared;v.output[row]=(routed+shared*gain)+v.residual[row];}
  }
 }
}
template<int R,bool NC>
__device__ __forceinline__ void router_multi(const MoeView&v,MoeScratch s){
 MatrixView router=v.weights->router;router.k=Hidden;router.n=Experts;router.format=WeightFormat::F32;router.row_stride=Hidden*4;
 constexpr int Warps=tune::router_rows>0?tune::router_rows:2;
 // Warps rows-groups per CTA (as moe_router_fixed_f32<Warps>), each with R rows in flight.
 const int group=threadIdx.x/32,lane=threadIdx.x&31,groups=gridDim.x*Warps,gid=blockIdx.x*Warps+group;
 if(group<Warps){
  const int trips=(Experts+groups*R-1)/(groups*R);
  for(int t=0;t<trips;t++){
   float part[R][1];
   #pragma unroll
   for(int j=0;j<R;j++){const int row=gid+(t*R+j)*groups;row_parts<32,NC>(router,v.prepared,v.input,row<Experts?row:Experts-1,part[j]);}
   #pragma unroll
   for(int j=0;j<R;j++){const int row=gid+(t*R+j)*groups;const float val=row_fold<32>(part[j]);if(!lane&&row<Experts)s.logits[row]=val;}
  }
 }
 if(blockIdx.x==0&&threadIdx.x<32){MatrixView shared=v.weights->shared_router;shared.k=Hidden;shared.format=WeightFormat::F32;const float gain=float_row(shared,v.input,0);if(threadIdx.x==0)*s.shared_gain=sigmoid(gain);}
}
template<uint32_t K>
__device__ __forceinline__ void run_epoch(const SegArgs a,Epoch ep,int position,float*shared){
 const ModelWeights*model=a.model;const DecodeWorkspace w=a.w;const StateView state=a.state;const int slot=a.slot;
 const LayerWeights&l=model->layer[ep.layer<0?0:ep.layer];
 #define H6_CASE(name) case EpochKind::name:if constexpr((K&bit(EpochKind::name))!=0)
 switch(EpochKind(ep.kind)){
 H6_CASE(Embed){const int token=a.teacher?a.teacher[position]:*w.token;embedding_phase(model->embedding,w.hidden,token);}break;
 H6_CASE(InNorm){if(blockIdx.x==0)norm_quant_cta(w.hidden,l.input_norm,w,shared);}break;
 H6_CASE(GdnProj){
  if constexpr((tune::prefetch&1)!=0){prefetch_matrix(l.output);prefetch_matrix(l.router);prefetch_matrix(l.shared_router);}
  ActivationView h=w.hidden_q.view();const float*x=w.normalized;
  if constexpr(tune::fuse_norm!=0){norm_local(w.hidden,nullptr,l.input_norm,shared,w,nullptr);h=local_q8(shared).view();x=shared+ShNorm;}
  if constexpr(tune::mma_proj!=0){
   const auto&pl=model->prepared->layer[ep.layer];const PreparedActivationView pa=mma_local(w.hidden_mma,h,shared);
   mma_rows<16>(pl.qkv,pa,w.qkv,0,shared);mma_rows<16>(pl.z,pa,w.z,ConvChannels/16,shared);
  }else{
   rows_multi<tune::qkv_lanes,tune::proj_rows,tune::load_nc!=0>(l.qkv,ConvChannels,Hidden,h,x,w.qkv,0);
   rows_multi<tune::z_lanes,tune::proj_rows,tune::load_nc!=0>(l.z,Inner,Hidden,h,x,w.z,ConvChannels);
  }
  rows_exact<tune::ab_lanes>(l.alpha,ValueHeads,Hidden,h,x,w.alpha,ConvChannels+Inner);
  rows_exact<tune::ab_lanes>(l.beta,ValueHeads,Hidden,h,x,w.beta,ConvChannels+Inner+ValueHeads);
 }break;
 H6_CASE(ConvPrep){gdn_conv_prepare(gdn_view(l,state,w,slot),gdn_scratch(w));}break;
 H6_CASE(Recur){gdn_recur(gdn_view(l,state,w,slot),gdn_scratch(w));}break;
 H6_CASE(FinishQ){gdn_finish_q8(gdn_view(l,state,w,slot),gdn_scratch(w),w.inner_q);}break;
 H6_CASE(RotWriter){rotate_inputs(l,w,true,shared);}break;
 H6_CASE(OutProj){
  if constexpr(tune::mma_out!=0){const PreparedActivationView pa=mma_local(w.inner_mma,w.inner_q.view(),shared);mma_rows<32>(model->prepared->layer[ep.layer].output,pa,w.projected,0,shared);}
  else rows_multi<tune::out_lanes,tune::out_rows,tune::load_nc!=0>(l.output,Hidden,Inner,w.inner_q.view(),w.mix,w.projected,0);
 }break;
 H6_CASE(ResNorm){if(blockIdx.x==0){for(int i=threadIdx.x;i<Hidden;i+=blockDim.x)w.residual[i]=w.hidden[i]+w.projected[i];__syncthreads();norm_quant_cta(w.residual,l.post_norm,w,shared);}}break;
 H6_CASE(Router){
  if constexpr((tune::prefetch&8)!=0){prefetch_matrix(l.shared_gate);prefetch_matrix(l.shared_up);prefetch_matrix(l.shared_down);}
  MoeView v=moe_view(l,w);
  if constexpr(tune::fuse_norm!=0){norm_local(w.hidden,w.projected,l.post_norm,shared,w,w.residual);v.input=shared+ShNorm;v.prepared=local_q8(shared).view();}
  if constexpr(tune::router_bf16!=0){moe_router_bf16<(tune::router_rows>0?tune::router_rows:2)>(a.router_bf16,ep.layer,v,moe_scratch(w));}
  else if constexpr(tune::router_rows>0){
   if(l.router.format==WeightFormat::F32&&l.shared_router.format==WeightFormat::F32){
    if constexpr(tune::router_mrows>1||tune::load_nc)router_multi<tune::router_mrows,tune::load_nc!=0>(v,moe_scratch(w));
    else moe_router_fixed_f32<tune::router_rows>(v,moe_scratch(w));
   }
   else moe_router(v,moe_scratch(w));
  }else moe_router(v,moe_scratch(w));
 }break;
 H6_CASE(Select){if(blockIdx.x==0){if constexpr(tune::select_body==1)moe_select_candidate(moe_view(l,w),moe_scratch(w),shared);else moe_select(moe_view(l,w),moe_scratch(w),shared);}}break;
 H6_CASE(GateUp){
  MoeScratch sc=moe_scratch(w);float*tile=shared;
  if constexpr(tune::fuse_select!=0){
   MoeScratch mine=sc;if(blockIdx.x!=0){mine.probs=shared+ShProbs;mine.ids=reinterpret_cast<int32_t*>(shared+ShIds);mine.weights=shared+ShWeights;}
   if constexpr(tune::select_body==1)moe_select_candidate(moe_view(l,w),mine,shared);else moe_select(moe_view(l,w),mine,shared);
   __syncthreads();sc=mine;tile=shared+ShTile;
  }
  if constexpr((tune::prefetch&2)!=0){for(int j=0;j<TopExperts;j++)prefetch_expert(l.expert_down,sc.ids[j]);}
  moe_gate_up_q8(moe_view(l,w),sc,tile);
 }break;
 H6_CASE(RotDown){rotate_inputs(l,w,false,shared);}break;
 H6_CASE(Down){
  if constexpr((tune::prefetch&4)!=0){if(ep.layer+1<Layers)prefetch_projection(model->layer[ep.layer+1]);}
  down_multi<tune::down_rows,tune::load_nc!=0>(moe_view(l,w),moe_scratch(w));
 }break;
 H6_CASE(AttnProj){
  if constexpr((tune::prefetch&1)!=0){prefetch_matrix(l.output);prefetch_matrix(l.router);prefetch_matrix(l.shared_router);}
  ActivationView h=w.hidden_q.view();const float*x=w.normalized;
  if constexpr(tune::fuse_norm!=0){norm_local(w.hidden,nullptr,l.input_norm,shared,w,nullptr);h=local_q8(shared).view();x=shared+ShNorm;}
  if constexpr(tune::mma_proj!=0){
   const auto&pl=model->prepared->layer[ep.layer];const PreparedActivationView pa=mma_local(w.hidden_mma,h,shared);
   mma_rows<16>(pl.q,pa,w.qkv,0,shared);mma_rows<16>(pl.k,pa,w.key,2*QueryHeads*HeadDim/16,shared);mma_rows<16>(pl.v,pa,w.value,(2*QueryHeads*HeadDim+KVHeads*HeadDim)/16,shared);
  }else{
   rows_multi<tune::qkv_lanes,tune::proj_rows,tune::load_nc!=0>(l.q,2*QueryHeads*HeadDim,Hidden,h,x,w.qkv,0);
   rows_multi<tune::z_lanes,tune::proj_rows,tune::load_nc!=0>(l.k,KVHeads*HeadDim,Hidden,h,x,w.key,2*QueryHeads*HeadDim);
   rows_multi<tune::z_lanes,tune::proj_rows,tune::load_nc!=0>(l.v,KVHeads*HeadDim,Hidden,h,x,w.value,2*QueryHeads*HeadDim+KVHeads*HeadDim);
  }
 }break;
 H6_CASE(NormRope){attention_norm_rope_append_phase(l,state,attention_index(ep.layer),slot,position,attention_buffers(w));}break;
 H6_CASE(Scores){attention_scores_phase(state,attention_index(ep.layer),slot,position,attention_buffers(w));}break;
 H6_CASE(Softmax){attention_softmax_phase(state,position,attention_buffers(w));}break;
 H6_CASE(ValueGate){attention_value_gate_q8_phase(state,attention_index(ep.layer),slot,position,attention_buffers(w),w.inner_q);}break;
 H6_CASE(FinalNorm){if(blockIdx.x==0)norm_quant_cta(w.hidden,model->final_norm,w,shared);}break;
 H6_CASE(Head){
  if constexpr(tune::mma_head!=0){const PreparedActivationView pa=mma_local(w.hidden_mma,w.hidden_q.view(),shared);mma_rows<16>(model->prepared->head,pa,w.logits,0,shared);}
  else if(model->head.format==WeightFormat::Q8Grouped128)head_q8_phase<tune::head_lanes,tune::head_contiguous!=0>(model->head,w.hidden_q.view(),w.logits);
  else matvec_exact_phase<16>(model->head,w.hidden_q.view(),w.normalized,w.logits);
 }break;
 H6_CASE(ArgPart){
  for(int p=blockIdx.x;p<ArgPartitions;p+=gridDim.x){
   float best=-INFINITY;int bestid=Vocab;
   for(int i=p*blockDim.x+threadIdx.x;i<Vocab;i+=ArgPartitions*blockDim.x){const float v=w.logits[i];if(v>best||(v==best&&i<bestid)){best=v;bestid=i;}}
   block_argmax(best,bestid,shared,w.max_values+p,w.max_ids+p);__syncthreads();
  }
 }break;
 H6_CASE(ArgFinish){
  if(blockIdx.x==0){
   float best=-INFINITY;int bestid=Vocab;
   for(int i=threadIdx.x;i<ArgPartitions;i+=blockDim.x){const float v=w.max_values[i];const int id=w.max_ids[i];if(v>best||(v==best&&id<bestid)){best=v;bestid=id;}}
   block_argmax(best,bestid,shared,w.max_values,w.token);if(threadIdx.x==0)state.positions[slot]=position+1;
  }
 }break;
 default:break;
 }
 #undef H6_CASE
}
template<int Class>
__global__ void __launch_bounds__(256,tune::segment_min_blocks[Class]) segment_kernel(SegArgs a){
 constexpr uint32_t K=tune::segment_kinds[Class];
 __shared__ float shared[shared_floats<K>()];
 const int position=a.state.positions[a.slot];
 for(int e=a.first;e<a.last;e++){
  if(a.stamps&&blockIdx.x==0&&threadIdx.x==0)a.stamps[e]=globaltimer();
  run_epoch<K>(a,a.epochs[e],position,shared);
  if(e+1<a.last)cooperative_groups::this_grid().sync();
 }
 if(a.stamps&&a.last==a.total&&blockIdx.x==0&&threadIdx.x==0)a.stamps[a.total]=globaltimer();
}
// RNE BF16 copy of every F32 router (rows 0..255) and shared router (row 256).
__global__ void router_bf16_kernel(const ModelWeights*m,uint16_t*image,int*lossy){
 const size_t total=size_t(Layers)*RouterBf16LayerElements;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x){
  const int layer=int(i/RouterBf16LayerElements);const size_t r=i%RouterBf16LayerElements;const int row=int(r/Hidden),col=int(r%Hidden);
  const auto&l=m->layer[layer];const MatrixView&w=row<Experts?l.router:l.shared_router;const int rr=row<Experts?row:0;
  const float x=reinterpret_cast<const float*>(static_cast<const char*>(w.data)+size_t(rr)*(w.row_stride?w.row_stride:Hidden*4))[col];
  if(__float_as_uint(x)&0xffffu)*lossy=1;   // only lossless images are admitted
  image[i]=uint16_t(__float_as_uint(x)>>16);
 }
}
__global__ void set_token_kernel(int32_t*dst,int32_t token){*dst=token;}
using KernelFn=void(*)(SegArgs);
template<int...I>constexpr std::array<KernelFn,sizeof...(I)>kernel_table(std::integer_sequence<int,I...>){return{segment_kernel<I>...};}
const auto Kernels=kernel_table(std::make_integer_sequence<int,tune::segment_count>{});

struct Segment {int first,last,cls;bool cooperative;};
struct Context {
 const DeviceModel*model=nullptr;DecodeWorkspace w{};const int32_t*teacher=nullptr;int slot=0;
 std::vector<Epoch>epochs;Epoch*device_epochs=nullptr;uint64_t*stamps=nullptr;
 std::vector<Segment>segments;std::vector<int>grid,active;std::vector<cudaFuncAttributes>attr;std::vector<int>uses;
 cudaStream_t capture=nullptr;cudaGraphExec_t exec=nullptr,profiled=nullptr;
 bool persisting=false;uint16_t*router_bf16=nullptr;
 ~Context(){if(router_bf16)cudaFree(router_bf16);if(persisting){cudaCtxResetPersistingL2Cache();cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,0);}if(exec)cudaGraphExecDestroy(exec);if(profiled)cudaGraphExecDestroy(profiled);if(capture)cudaStreamDestroy(capture);if(device_epochs)cudaFree(device_epochs);if(stamps)cudaFree(stamps);}
};
void fail(char*err,int len,const std::string&m){if(err&&len>0){snprintf(err,size_t(len),"%s",m.c_str());}}
bool cuda_ok(cudaError_t e,const char*op,char*err,int len){if(e==cudaSuccess)return true;fail(err,len,std::string(op)+": "+cudaGetErrorString(e));return false;}

std::vector<Epoch>epoch_sequence(const ModelWeights&m){
 std::vector<Epoch>e;auto add=[&](EpochKind k,int layer){e.push_back({int16_t(k),int16_t(layer)});};
 add(EpochKind::Embed,-1);
 for(int i=0;i<Layers;i++){
  const auto&l=m.layer[i];const bool writer=l.output.flags&RotationEnabled,down=(l.expert_down.flags|l.shared_down.flags)&RotationEnabled;
  if(!tune::fuse_norm)add(EpochKind::InNorm,i);
  if(l.kind==LayerKind::GDN){add(EpochKind::GdnProj,i);add(EpochKind::ConvPrep,i);add(EpochKind::Recur,i);add(EpochKind::FinishQ,i);}
  else{add(EpochKind::AttnProj,i);add(EpochKind::NormRope,i);add(EpochKind::Scores,i);add(EpochKind::Softmax,i);add(EpochKind::ValueGate,i);}
  if(writer)add(EpochKind::RotWriter,i);
  add(EpochKind::OutProj,i);if(!tune::fuse_norm)add(EpochKind::ResNorm,i);add(EpochKind::Router,i);if(!tune::fuse_select)add(EpochKind::Select,i);add(EpochKind::GateUp,i);
  if(down)add(EpochKind::RotDown,i);
  add(EpochKind::Down,i);
 }
 add(EpochKind::FinalNorm,-1);add(EpochKind::Head,-1);add(EpochKind::ArgPart,-1);add(EpochKind::ArgFinish,-1);
 return e;
}
int popcount(uint32_t x){return __builtin_popcount(x);}

// Persisting L2 window for a segment: the first router (or recurrence) epoch it
// holds decides it; one window per kernel launch. Hints only, never numerics.
struct Window {const void*base=nullptr;size_t bytes=0;};
Window segment_window(const Context&c,const Segment&s){
 const ModelWeights&m=c.model->weights;
 for(int e=s.first;e<s.last;e++){
  const auto k=EpochKind(c.epochs[e].kind);const int layer=c.epochs[e].layer;
  if(tune::l2_router&&k==EpochKind::Router){
   if(tune::router_bf16)return {c.router_bf16+size_t(layer)*RouterBf16LayerElements,RouterBf16LayerBytes};
   return {m.layer[layer].router.data,m.layer[layer].router.bytes};
  }
  if(tune::l2_state&&k==EpochKind::Recur){const size_t per=size_t(ValueHeads)*ValueDim*StateDim;return {c.model->state.gdn+(size_t(c.slot)*LinearLayers+linear_index(layer))*per,per*4};}
 }
 return {};
}
bool build_graph(Context&c,bool profile,cudaGraphExec_t&out,char*err,int len){
 if(!cuda_ok(cudaStreamBeginCapture(c.capture,cudaStreamCaptureModeThreadLocal),"begin capture",err,len))return false;
 bool ok=true;
 for(const auto&s:c.segments){
  SegArgs a{c.model->device_weights,c.model->state,c.w,c.device_epochs,c.teacher,profile?c.stamps:nullptr,c.router_bf16,s.first,s.last,int(c.epochs.size()),c.slot};
  const KernelFn fn=Kernels[s.cls];const int g=c.grid[s.cls];
  cudaLaunchAttribute attrs[2];int na=0;
  if(s.cooperative){attrs[na].id=cudaLaunchAttributeCooperative;attrs[na].val.cooperative=1;na++;}
  const Window win=segment_window(c,s);
  if(win.bytes){attrs[na].id=cudaLaunchAttributeAccessPolicyWindow;auto&v=attrs[na].val.accessPolicyWindow;v.base_ptr=const_cast<void*>(win.base);v.num_bytes=win.bytes;v.hitRatio=tune::l2_ratio/100.f;v.hitProp=cudaAccessPropertyPersisting;v.missProp=cudaAccessPropertyStreaming;na++;}
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(g);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=0;cfg.stream=c.capture;cfg.attrs=attrs;cfg.numAttrs=na;
  cudaError_t e=cudaLaunchKernelEx(&cfg,fn,a);
  if(e!=cudaSuccess){fail(err,len,std::string("segment launch: ")+cudaGetErrorString(e));ok=false;break;}
 }
 cudaGraph_t graph=nullptr;const cudaError_t end=cudaStreamEndCapture(c.capture,&graph);
 if(!ok){if(graph)cudaGraphDestroy(graph);return false;}
 if(!cuda_ok(end,"end capture",err,len))return false;
 const cudaError_t inst=cudaGraphInstantiate(&out,graph,0);cudaGraphDestroy(graph);
 return cuda_ok(inst,"graph instantiate",err,len);
}

void*create(const DeviceModel*model,const DecodeWorkspace*w,const int*grids,int ngrids,const int32_t*teacher,int slot,char*err,int len){
 auto c=std::make_unique<Context>();c->model=model;c->w=*w;c->teacher=teacher;c->slot=slot;
 if(!model||slot<0||slot>=model->state.slots){fail(err,len,"invalid model/slot");return nullptr;}
 if(w->hidden_q.group!=32||w->inner_q.group!=32||w->expert_q.group!=32){fail(err,len,"strict32 activations required");return nullptr;}
 if((tune::mma_proj||tune::mma_out||tune::mma_head)&&!model->weights.prepared){fail(err,len,"MMA genes need DeviceModel::prepare_matmul()");return nullptr;}
 c->epochs=epoch_sequence(model->weights);
 if(const char*limit=getenv("HALO_TUNE_EPOCHS")){const int n=atoi(limit);if(n>0&&n<int(c->epochs.size()))c->epochs.resize(n);}
 // Cut: a kernel boundary precedes every epoch whose kind is in boundary_kinds.
 for(int e=0;e<int(c->epochs.size());e++){
  const int k=c->epochs[e].kind;
  if(e==0||(tune::boundary_kinds>>k&1u))c->segments.push_back({e,e+1,-1,false});else c->segments.back().last=e+1;
 }
 c->uses.assign(tune::segment_count,0);
 for(auto&s:c->segments){
  uint32_t mask=0;for(int e=s.first;e<s.last;e++)mask|=1u<<c->epochs[e].kind;
  int best=-1;for(int i=0;i<tune::segment_count;i++)if((tune::segment_kinds[i]&mask)==mask&&(best<0||popcount(tune::segment_kinds[i])<popcount(tune::segment_kinds[best])))best=i;
  if(best<0){char m[96];snprintf(m,sizeof m,"no segment class covers kinds 0x%x",mask);fail(err,len,m);return nullptr;}
  s.cls=best;s.cooperative=s.last-s.first>1;c->uses[best]++;
 }
 cudaDeviceProp prop{};if(!cuda_ok(cudaGetDeviceProperties(&prop,0),"device",err,len))return nullptr;
 c->grid.assign(tune::segment_count,0);c->active.assign(tune::segment_count,0);c->attr.resize(tune::segment_count);
 for(int i=0;i<tune::segment_count;i++){
  if(!cuda_ok(cudaFuncGetAttributes(&c->attr[i],(const void*)Kernels[i]),"attributes",err,len))return nullptr;
  if(!cuda_ok(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&c->active[i],Kernels[i],256,0),"occupancy",err,len))return nullptr;
  // grids[i]>0: exact grid; <0: |grid| clamped to cooperative residency; 0: min_blocks x SMs.
  const int req=i<ngrids&&grids&&grids[i]?grids[i]:tune::segment_grids[i];
  int want=req>0?req:req<0?-req:std::max(1,std::min(tune::segment_min_blocks[i],c->active[i]))*prop.multiProcessorCount;
  if(req<0){bool coop=false;for(const auto&s:c->segments)coop|=s.cls==i&&s.cooperative;if(coop)want=std::min(want,c->active[i]*prop.multiProcessorCount);}
  c->grid[i]=want;
 }
 for(const auto&s:c->segments)if(s.cooperative&&c->grid[s.cls]>c->active[s.cls]*prop.multiProcessorCount){
  char m[160];snprintf(m,sizeof m,"class %d grid %d exceeds cooperative residency %d",s.cls,c->grid[s.cls],c->active[s.cls]*prop.multiProcessorCount);fail(err,len,m);return nullptr;
 }
 const size_t n=c->epochs.size();
 if(!cuda_ok(cudaMalloc(&c->device_epochs,n*sizeof(Epoch)),"epochs",err,len)||!cuda_ok(cudaMemcpy(c->device_epochs,c->epochs.data(),n*sizeof(Epoch),cudaMemcpyHostToDevice),"epochs upload",err,len))return nullptr;
 if(!cuda_ok(cudaMalloc(&c->stamps,(n+1)*sizeof(uint64_t)),"stamps",err,len))return nullptr;
 if(tune::router_bf16){
  for(int i=0;i<Layers;i++)if(model->weights.layer[i].router.format!=WeightFormat::F32||model->weights.layer[i].shared_router.format!=WeightFormat::F32){fail(err,len,"router_bf16 needs F32 routers");return nullptr;}
  if(!cuda_ok(cudaMalloc(&c->router_bf16,RouterBf16ImageBytes),"router bf16 image",err,len))return nullptr;
  int*lossy=nullptr;if(!cuda_ok(cudaMalloc(&lossy,4),"router bf16 flag",err,len))return nullptr;cudaMemset(lossy,0,4);
  router_bf16_kernel<<<1024,256>>>(model->device_weights,c->router_bf16,lossy);int bad=0;
  const bool built=cuda_ok(cudaDeviceSynchronize(),"router bf16 build",err,len)&&cuda_ok(cudaMemcpy(&bad,lossy,4,cudaMemcpyDeviceToHost),"router bf16 flag",err,len);cudaFree(lossy);
  if(!built)return nullptr;
  if(bad){fail(err,len,"router_bf16 refused: this model has router coefficients that BF16 cannot represent exactly");return nullptr;}
 }
 if(!cuda_ok(cudaStreamCreateWithFlags(&c->capture,cudaStreamNonBlocking),"capture stream",err,len))return nullptr;
 if(tune::l2_router||tune::l2_state){
  if(!cuda_ok(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,size_t(prop.persistingL2CacheMaxSize)),"persisting L2 limit",err,len))return nullptr;
  c->persisting=true;
 }
 if(!build_graph(*c,false,c->exec,err,len))return nullptr;
 return c.release();
}
int step(void*ctx,cudaStream_t s,char*err,int len){auto*c=static_cast<Context*>(ctx);return cuda_ok(cudaGraphLaunch(c->exec,s),"graph launch",err,len)?0:1;}
int step_token(void*ctx,int32_t token,cudaStream_t s,char*err,int len){
 auto*c=static_cast<Context*>(ctx);
 if(c->teacher){fail(err,len,"step_token needs a context created without a teacher array");return 1;}
 if(token>=Vocab){fail(err,len,"token outside vocabulary");return 1;}
 if(token>=0){set_token_kernel<<<1,1,0,s>>>(c->w.token,token);if(!cuda_ok(cudaGetLastError(),"token input",err,len))return 1;}
 return cuda_ok(cudaGraphLaunch(c->exec,s),"graph launch",err,len)?0:1;
}
int profile(void*ctx,cudaStream_t s,uint64_t*out,int capacity,char*err,int len){
 auto*c=static_cast<Context*>(ctx);const int n=int(c->epochs.size())+1;if(capacity<n){fail(err,len,"stamp capacity");return 1;}
 if(!c->profiled&&!build_graph(*c,true,c->profiled,err,len))return 1;
 if(!cuda_ok(cudaGraphLaunch(c->profiled,s),"profile launch",err,len)||!cuda_ok(cudaStreamSynchronize(s),"profile sync",err,len))return 1;
 return cuda_ok(cudaMemcpy(out,c->stamps,n*sizeof(uint64_t),cudaMemcpyDeviceToHost),"stamps read",err,len)?0:1;
}
int describe(void*ctx,char*buf,int len){
 auto*c=static_cast<Context*>(ctx);std::string j="{\"config\":\"";j+=tune::config_id;j+="\",\"epochs\":"+std::to_string(c->epochs.size())+",\"segments\":"+std::to_string(c->segments.size())+",\"classes\":[";
 for(int i=0;i<tune::segment_count;i++){
  char b[256];snprintf(b,sizeof b,"%s{\"kinds\":%u,\"min_blocks\":%d,\"registers\":%d,\"local_bytes\":%zu,\"static_smem\":%zu,\"active_per_sm\":%d,\"grid\":%d,\"segments\":%d}",i?",":"",tune::segment_kinds[i],tune::segment_min_blocks[i],c->attr[i].numRegs,c->attr[i].localSizeBytes,c->attr[i].sharedSizeBytes,c->active[i],c->grid[i],c->uses[i]);j+=b;
 }
 j+="],\"epoch_kinds\":[";for(size_t e=0;e<c->epochs.size();e++){j+=(e?",":"")+std::to_string(c->epochs[e].kind);}
 j+="],\"segment_first\":[";for(size_t s=0;s<c->segments.size();s++){j+=(s?",":"")+std::to_string(c->segments[s].first);}
 j+="]}";
 if(int(j.size())+1>len)return 1;memcpy(buf,j.c_str(),j.size()+1);return 0;
}
void destroy(void*ctx){delete static_cast<Context*>(ctx);}
const HaloTuneApi Api{2,tune::config_id,create,step,profile,describe,destroy,step_token};
} // namespace
} // namespace halo6000
extern "C" __attribute__((visibility("default"))) const HaloTuneApi*halo_tune_api(){return &halo6000::Api;}
