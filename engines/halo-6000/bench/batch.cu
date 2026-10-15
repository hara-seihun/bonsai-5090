#include "batch.hpp"
#include "decode.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
using namespace halo6000;
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s line%d\n",#x,cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
void ok(Status s){if(!s.ok()){fprintf(stderr,"batch: %s\n",s.error().message.c_str());exit(1);}}
template<class T>T*allocate(size_t n){T*p;CU(cudaMalloc(&p,n*sizeof(T)));return p;}
struct Allocation {void*p;size_t bytes;const char*name;};
std::vector<Allocation>state_ranges(StateView s){size_t kv=size_t(AttentionLayers)*s.slots*s.context*KVHeads*HeadDim*2;return{{s.gdn,size_t(s.slots)*GdnFloatsPerSlot*4,"GDN"},{s.conv,size_t(s.slots)*ConvFloatsPerSlot*4,"conv"},{s.key,kv,"K"},{s.value,kv,"V"},{s.positions,size_t(s.slots)*4,"positions"}};}
struct StateCopy {
 std::vector<Allocation>ranges;
 explicit StateCopy(StateView s){ranges=state_ranges(s);for(auto&r:ranges){void*p;CU(cudaMalloc(&p,r.bytes));r.p=p;}}
 ~StateCopy(){for(auto&r:ranges)cudaFree(r.p);}
 void save(StateView s,cudaStream_t stream){auto src=state_ranges(s);for(size_t i=0;i<ranges.size();i++)CU(cudaMemcpyAsync(ranges[i].p,src[i].p,src[i].bytes,cudaMemcpyDeviceToDevice,stream));}
 void restore(StateView s,cudaStream_t stream){auto dst=state_ranges(s);for(size_t i=0;i<ranges.size();i++)CU(cudaMemcpyAsync(dst[i].p,ranges[i].p,dst[i].bytes,cudaMemcpyDeviceToDevice,stream));}
};
struct Difference {unsigned long long bits=0,first=~0ull,nonfinite=0;};
__global__ void compare_words(const uint32_t*ref,const uint32_t*got,uint64_t n,Difference*d,bool floats){for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(gridDim.x)*blockDim.x){if(ref[i]!=got[i]){atomicAdd(&d->bits,1ull);atomicMin(&d->first,(unsigned long long)i);}if(floats&&(got[i]&0x7f800000u)==0x7f800000u)atomicAdd(&d->nonfinite,1ull);}}
Difference compare(const void*ref,const void*got,size_t bytes,const char*kind,int step,int grid,cudaStream_t stream,bool floats=false){Difference initial;Difference*device=allocate<Difference>(1);CU(cudaMemcpyAsync(device,&initial,sizeof initial,cudaMemcpyHostToDevice,stream));compare_words<<<188,256,0,stream>>>(static_cast<const uint32_t*>(ref),static_cast<const uint32_t*>(got),bytes/4,device,floats);Difference result;CU(cudaMemcpyAsync(&result,device,sizeof result,cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));CU(cudaFree(device));printf("{\"guard\":\"%s\",\"step\":%d,\"grid_x\":%d,\"words\":%zu,\"different_bits_words\":%llu,\"nonfinite\":%llu,\"first_word\":%llu}\n",kind,step,grid,bytes/4,result.bits,result.nonfinite,result.first);return result;}
struct Reference {std::unique_ptr<StateCopy>state;float*logits;int32_t*tokens;Reference(StateView s,int rows):state(std::make_unique<StateCopy>(s)),logits(allocate<float>(size_t(rows)*Vocab)),tokens(allocate<int32_t>(rows)){}~Reference(){cudaFree(logits);cudaFree(tokens);}};
int teacher(int row,int position){return 100+(row*997+position*31+position*position*7)%20000;}
std::vector<BatchInput>inputs(const std::vector<int>&slots,const std::vector<int>&prefix,int step,bool feedback=false){std::vector<BatchInput>v;for(size_t r=0;r<slots.size();r++)v.push_back({slots[r],feedback?-1:teacher(r,prefix[r]+step)});return v;}
void check_reference(DeviceModel&m,BatchWorkspace&w,const Reference&ref,int step,int grid,cudaStream_t stream){
 bool bad=false;auto d=compare(ref.logits,w.view.base.logits,size_t(w.view.rows)*Vocab*4,"logits",step,grid,stream,true);bad|=d.bits||d.nonfinite;
 d=compare(ref.tokens,w.view.base.token,size_t(w.view.rows)*4,"argmax",step,grid,stream);bad|=d.bits;
 auto got=state_ranges(m.state);for(size_t i=0;i<got.size();i++){d=compare(ref.state->ranges[i].p,got[i].p,got[i].bytes,got[i].name,step,grid,stream);bad|=d.bits;}
 if(bad)exit(2);
}
struct Events {cudaEvent_t start,end;Events(){CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));}~Events(){cudaEventDestroy(start);cudaEventDestroy(end);}float elapsed(){CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,start,end));return ms;}};
int main(int argc,char**argv){
 if(argc<3){fprintf(stderr,"usage: batch MODEL.gguf ROWS [steps32] [context128] [bucketed0]\n");return 1;}setvbuf(stdout,nullptr,_IOLBF,0);std::string model=argv[1];int rows=atoi(argv[2]),steps=argc>3?atoi(argv[3]):32,context=argc>4?atoi(argv[4]):128;bool bucketed=argc>5&&atoi(argv[5]);
 if(rows<1||rows>128||steps<1||steps+12>=context)return 1;int slots_count=std::min(128,rows+3);cudaStream_t stream;CU(cudaStreamCreate(&stream));
 auto start=std::chrono::steady_clock::now();auto image=load_cached(model,"/workspace/models/halo6000-cache",false,true);if(!image.ok()){fprintf(stderr,"cache: %s\n",image.error().message.c_str());return 1;}
 auto made=DeviceModel::create(image.value(),slots_count,context);if(!made.ok()){fprintf(stderr,"model: %s\n",made.error().message.c_str());return 1;}auto m=std::move(made.value());m->matvec_lanes=16;ok(m->prepare_matmul(bucketed,stream));
 auto ws=BatchWorkspace::create(*m,rows,bucketed?BatchExpertPolicy::BucketedExact:BatchExpertPolicy::RowParallel);if(!ws.ok()){fprintf(stderr,"workspace: %s\n",ws.error().message.c_str());return 1;}auto w=std::move(ws.value());
 std::vector<std::unique_ptr<Workspace>>serial;for(int r=0;r<rows;r++){auto s=Workspace::create(context);if(!s.ok())return 1;serial.push_back(std::move(s.value()));}
 printf("{\"native_batch\":true,\"rows\":%d,\"state_slots\":%d,\"context\":%d,\"weight_bytes\":%zu,\"prepared_bytes\":%zu,\"workspace_bytes\":%zu,\"epochs\":%d,\"load_s\":%.3f,\"bucketed\":%s,\"policy\":\"exact original tree, Q8 group32, native Q2_K LG4; metadata/gather/quant included\"}\n",rows,slots_count,context,m->weight_bytes,m->prepared_bytes,w->bytes,w->epoch_count(),std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),bucketed?"true":"false");
 // Host negative-input guards precede any forward. Distinct-slot ownership is mandatory.
 if(w->configure({}).ok())return 2;std::vector<BatchInput>invalid{{0,100},{0,101}};if(rows>=2&&w->configure(invalid).ok())return 2;
 invalid={{slots_count,100}};if(w->configure(invalid).ok())return 2;invalid={{0,-1}};if(w->configure(invalid).ok())return 2;
 printf("{\"CPU_negative_guards\":true}\n");
 std::vector<int>slots(rows),prefix(rows);int multiplier=5;while(std::gcd(multiplier,slots_count)!=1)multiplier+=2;
 for(int r=0;r<rows;r++){slots[r]=(r*multiplier+3)%slots_count;prefix[r]=r%3;for(int p=0;p<prefix[r];p++)ok(run_token(*m,serial[r]->view,teacher(r,p),p,slots[r],188,stream));}
 StateCopy initial(m->state);initial.save(m->state,stream);CU(cudaStreamSynchronize(stream));
 constexpr int guard_steps=3;std::vector<std::unique_ptr<Reference>>refs;
 for(int p=0;p<guard_steps;p++){
  auto ref=std::make_unique<Reference>(m->state,rows);
  for(int r=0;r<rows;r++){ok(run_token(*m,serial[r]->view,teacher(r,prefix[r]+p),prefix[r]+p,slots[r],188,stream));CU(cudaMemcpyAsync(ref->logits+size_t(r)*Vocab,serial[r]->view.logits,Vocab*4,cudaMemcpyDeviceToDevice,stream));CU(cudaMemcpyAsync(ref->tokens+r,serial[r]->view.token,4,cudaMemcpyDeviceToDevice,stream));}
  ref->state->save(m->state,stream);refs.push_back(std::move(ref));
 }
 CU(cudaStreamSynchronize(stream));int default_grid=std::max(1,m->multiprocessors/rows);std::vector<int>grids={1,default_grid};if(default_grid>1)grids.push_back(std::max(1,default_grid/2));std::sort(grids.begin(),grids.end());grids.erase(std::unique(grids.begin(),grids.end()),grids.end());
 for(int grid:grids){initial.restore(m->state,stream);CU(cudaStreamSynchronize(stream));for(int p=0;p<guard_steps;p++){ok(w->configure(inputs(slots,prefix,p),stream));ok(run_batch_token(*m,*w,grid,stream));ok(w->finish(stream));check_reference(*m,*w,*refs[p],p,grid,stream);}}
 // Drop half the rows, reset/reuse one previously active slot and change row order.
 initial.restore(m->state,stream);CU(cudaStreamSynchronize(stream));int kept=std::max(1,rows/2);std::vector<int>subset(slots.begin(),slots.begin()+kept),subprefix(prefix.begin(),prefix.begin()+kept);std::reverse(subset.begin(),subset.end());std::reverse(subprefix.begin(),subprefix.end());ok(m->reset(subset[0],stream));subprefix[0]=0;
 StateCopy reset_start(m->state);reset_start.save(m->state,stream);Reference reset_ref(m->state,kept);
 for(int r=0;r<kept;r++){ok(run_token(*m,serial[r]->view,teacher(r,subprefix[r]),subprefix[r],subset[r],188,stream));CU(cudaMemcpyAsync(reset_ref.logits+size_t(r)*Vocab,serial[r]->view.logits,Vocab*4,cudaMemcpyDeviceToDevice,stream));CU(cudaMemcpyAsync(reset_ref.tokens+r,serial[r]->view.token,4,cudaMemcpyDeviceToDevice,stream));}reset_ref.state->save(m->state,stream);reset_start.restore(m->state,stream);CU(cudaStreamSynchronize(stream));
 ok(w->configure(inputs(subset,subprefix,0),stream));ok(run_batch_token(*m,*w,0,stream));ok(w->finish(stream));check_reference(*m,*w,reset_ref,99,std::max(1,m->multiprocessors/kept),stream);
 // One fixed GPU graph advances ragged slots, feedback tokens and positions.
 initial.restore(m->state,stream);CU(cudaStreamSynchronize(stream));ok(w->configure(inputs(slots,prefix,0),stream));ok(run_batch_token(*m,*w,0,stream));ok(w->finish(stream));ok(w->configure(inputs(slots,prefix,0,true),stream));CU(cudaStreamSynchronize(stream));
 cudaGraph_t graph;cudaGraphExec_t executable;CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));ok(run_batch_token(*m,*w,0,stream));CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));for(int i=0;i<8;i++)CU(cudaGraphLaunch(executable,stream));CU(cudaStreamSynchronize(stream));
 Events timer;CU(cudaEventRecord(timer.start,stream));for(int i=0;i<steps;i++)CU(cudaGraphLaunch(executable,stream));CU(cudaEventRecord(timer.end,stream));float batch_ms=timer.elapsed();ok(w->finish(stream));
 std::vector<int32_t>final_positions(slots_count);CU(cudaMemcpy(final_positions.data(),m->state.positions,slots_count*4,cudaMemcpyDeviceToHost));for(int r=0;r<rows;r++)if(final_positions[slots[r]]!=prefix[r]+1+8+steps)return 2;
 printf("{\"timing\":\"native_batched_graph_decode\",\"rows\":%d,\"grid_x\":%d,\"steps\":%d,\"gpu_ms\":%.6f,\"aggregate_tokens_per_second\":%.6f,\"per_step_ms\":%.6f,\"feedback_GPU\":true,\"independent_positions\":true}\n",rows,default_grid,steps,batch_ms,rows*steps*1000./batch_ms,batch_ms/steps);
 if(bucketed){std::vector<BucketStats>stats(Layers);CU(cudaMemcpy(stats.data(),w->view.bucket_stats,Layers*sizeof(BucketStats),cudaMemcpyDeviceToHost));for(int l=0;l<Layers;l++){const auto&s=stats[l];printf("{\"bucket_layer\":%d,\"active_experts\":%d,\"routed_occurrences\":%d,\"routed_tiles\":%d,\"padded_routed_columns\":%d,\"shared_rows\":%d,\"total_tiles\":%d,\"max_roster\":%d,\"singleton_experts\":%d,\"pair_experts\":%d}\n",l,s.active_experts,s.routed_occurrences,s.routed_tiles,s.padded_routed_columns,s.shared_rows,s.total_tiles,s.max_roster,s.singleton_experts,s.pair_experts);}}
 Reference graph_ref(m->state,rows);graph_ref.state->save(m->state,stream);CU(cudaMemcpyAsync(graph_ref.logits,w->view.base.logits,size_t(rows)*Vocab*4,cudaMemcpyDeviceToDevice,stream));CU(cudaMemcpyAsync(graph_ref.tokens,w->view.base.token,rows*4,cudaMemcpyDeviceToDevice,stream));
 // Same native canonical s1 kernels, row-serial over independent slots.
 initial.restore(m->state,stream);CU(cudaStreamSynchronize(stream));for(int r=0;r<rows;r++)ok(run_token(*m,serial[r]->view,teacher(r,prefix[r]),prefix[r],slots[r],188,stream));for(int p=1;p<=8;p++)for(int r=0;r<rows;r++)ok(run_token(*m,serial[r]->view,-1,prefix[r]+p,slots[r],188,stream));CU(cudaStreamSynchronize(stream));
 CU(cudaEventRecord(timer.start,stream));for(int p=9;p<9+steps;p++)for(int r=0;r<rows;r++)ok(run_token(*m,serial[r]->view,-1,prefix[r]+p,slots[r],188,stream));CU(cudaEventRecord(timer.end,stream));float serial_ms=timer.elapsed();
 printf("{\"timing\":\"serial_native_independent_slots\",\"rows\":%d,\"steps\":%d,\"gpu_ms\":%.6f,\"aggregate_tokens_per_second\":%.6f,\"batch_speedup\":%.6f}\n",rows,steps,serial_ms,rows*steps*1000./serial_ms,serial_ms/batch_ms);
 bool graph_bad=false;for(int r=0;r<rows;r++){auto d=compare(graph_ref.logits+size_t(r)*Vocab,serial[r]->view.logits,Vocab*4,"graph_feedback_logits",r,default_grid,stream,true);graph_bad|=d.bits||d.nonfinite;d=compare(graph_ref.tokens+r,serial[r]->view.token,4,"graph_feedback_argmax",r,default_grid,stream);graph_bad|=d.bits;}
 auto final_state=state_ranges(m->state);for(size_t i=0;i<final_state.size();i++){auto d=compare(graph_ref.state->ranges[i].p,final_state[i].p,final_state[i].bytes,"graph_feedback_state",i,default_grid,stream);graph_bad|=d.bits;}if(graph_bad)return 2;
 // An exhausted slot and an invalid device feedback id must not mutate state.
 int exhausted=context;CU(cudaMemcpyAsync(m->state.positions+slots[0],&exhausted,4,cudaMemcpyHostToDevice,stream));StateCopy rejected_state(m->state);rejected_state.save(m->state,stream);CU(cudaStreamSynchronize(stream));
 std::vector<BatchInput>single{{slots[0],100}};ok(w->configure(single,stream));if(rows>1){auto unseeded=single;unseeded[0].token=-1;if(w->configure(unseeded,stream).ok())return 2;}ok(run_batch_token(*m,*w,0,stream));auto rejected=w->finish(stream);if(rejected.ok()||rejected.error().code!=ErrorCode::Capacity)return 2;
 auto reject_ranges=state_ranges(m->state);for(size_t i=0;i<reject_ranges.size();i++){auto d=compare(rejected_state.ranges[i].p,reject_ranges[i].p,reject_ranges[i].bytes,"capacity_no_state_mutation",i,188,stream);if(d.bits)return 2;}
 int zero=0,bogus=Vocab;CU(cudaMemcpyAsync(m->state.positions+slots[0],&zero,4,cudaMemcpyHostToDevice,stream));CU(cudaMemcpyAsync(w->view.base.token,&bogus,4,cudaMemcpyHostToDevice,stream));rejected_state.save(m->state,stream);CU(cudaStreamSynchronize(stream));single[0].token=-1;ok(w->configure(single,stream));ok(run_batch_token(*m,*w,0,stream));rejected=w->finish(stream);if(rejected.ok()||rejected.error().code!=ErrorCode::Geometry)return 2;
 for(size_t i=0;i<reject_ranges.size();i++){auto d=compare(rejected_state.ranges[i].p,reject_ranges[i].p,reject_ranges[i].bytes,"invalid_feedback_no_state_mutation",i,188,stream);if(d.bits)return 2;}
 CU(cudaGraphExecDestroy(executable));CU(cudaGraphDestroy(graph));CU(cudaStreamDestroy(stream));printf("{\"passed\":true,\"rows\":%d,\"bitwise_full_logits_and_state\":true,\"arbitrary_slots_ragged_prefix_reset_reuse\":true}\n",rows);return 0;
}
