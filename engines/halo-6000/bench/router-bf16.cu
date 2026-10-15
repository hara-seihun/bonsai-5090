#include "router-bf16.cuh"
#include "router.cuh"
#include "gguf.hpp"
#include <cuda_bf16.h>
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>
using namespace halo6000;
using json=nlohmann::json;namespace fs=std::filesystem;
void need(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T take(Result<T>r){if(!r.ok())throw std::runtime_error(r.error().message);return std::move(r.value());}
template<class T>std::vector<T>read(const fs::path&p,size_t n){std::vector<T>v(n);std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));need(bool(f)&&f.peek()==EOF,"Fixture extent: "+p.string());return v;}
void save(const fs::path&p,const json&j){std::ofstream f(p);f<<j.dump(2)<<'\n';need(bool(f),"JSON output failed");}
uint16_t rne(float f){uint32_t u;memcpy(&u,&f,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
uint64_t hash(const void*p,size_t n){auto*q=static_cast<const uint8_t*>(p);uint64_t h=14695981039346656037ULL;for(size_t i=0;i<n;i++){h^=q[i];h*=1099511628211ULL;}return h;}
__global__ void round_reference(const float*src,float*rounded,uint16_t*bits,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){__nv_bfloat16 b=__float2bfloat16_rn(src[i]);rounded[i]=__bfloat162float(b);bits[i]=__bfloat16_as_ushort(b);}}
__global__ void f32_router(const LayerWeights*w,const float*x,int layer,MoeScratch s){MoeView v{};v.weights=w+layer;v.input=x;moe_router_fixed_f32<2>(v,s);}
template<int R>__global__ void bf16_router(const uint16_t*w,const float*x,int layer,MoeScratch s){MoeView v{};v.input=x;moe_router_bf16<R>(w,layer,v,s);}
__global__ void select(MoeScratch s){__shared__ float scratch[32];MoeView v{};moe_select(v,s,scratch);}
__global__ void f32_sweep(const LayerWeights*w,const float*x,MoeScratch s){for(int l=0;l<Layers;l++){MoeView v{};v.weights=w+l;v.input=x+l*Hidden;moe_router_fixed_f32<2>(v,s);__syncthreads();}}
template<int R>__global__ void bf16_sweep(const uint16_t*w,const float*x,MoeScratch s){for(int l=0;l<Layers;l++){MoeView v{};v.input=x+l*Hidden;moe_router_bf16<R>(w,l,v,s);__syncthreads();}}
__global__ void flush(uint32_t*p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=uint32_t(i);}
struct Snapshot {float logits[Experts],probs[Experts];int32_t ids[TopExperts];float weights[TopExperts],gain;};
struct Scratch {
 Snapshot*data;MoeScratch s{};
 Scratch(){ck(cudaMalloc(&data,sizeof(Snapshot)));s.logits=data->logits;s.probs=data->probs;s.ids=data->ids;s.weights=data->weights;s.shared_gain=&data->gain;}
 ~Scratch(){cudaFree(data);}
 Snapshot get(){Snapshot a;ck(cudaMemcpy(&a,data,sizeof a,cudaMemcpyDeviceToHost));return a;}
};
struct Device {
 std::vector<void*>p;
 template<class T>T*alloc(size_t n){T*x;ck(cudaMalloc(&x,n*sizeof(T)));p.push_back(x);return x;}
 ~Device(){for(void*x:p)cudaFree(x);}
};
struct Stats {
 size_t frames=0,order_flips=0,set_flips=0,changed_slots=0,retained=0,n=0,inf_kl=0;
 double logit_max=0,logit_sq=0,prob_max=0,prob_sq=0,weight_max=0,gain_max=0,kl=0;
 void add(const Snapshot&a,const Snapshot&b){++frames;bool order=false;int common=0;
  for(int i=0;i<Experts;i++){double d=double(a.logits[i])-b.logits[i];logit_max=std::max(logit_max,std::abs(d));logit_sq+=d*d;d=double(a.probs[i])-b.probs[i];prob_max=std::max(prob_max,std::abs(d));prob_sq+=d*d;++n;
   if(a.probs[i]>0){if(b.probs[i]<=0)++inf_kl;else kl+=double(a.probs[i])*std::log(double(a.probs[i])/b.probs[i]);}}
  for(int j=0;j<TopExperts;j++){order|=a.ids[j]!=b.ids[j];changed_slots+=a.ids[j]!=b.ids[j];common+=std::find(b.ids,b.ids+TopExperts,a.ids[j])!=b.ids+TopExperts;weight_max=std::max(weight_max,std::abs(double(a.weights[j])-b.weights[j]));}
  order_flips+=order;set_flips+=common<TopExperts;retained+=common;gain_max=std::max(gain_max,std::abs(double(a.gain)-b.gain));
 }
 json result()const{return {{"frames",frames},{"logits_maxabs",logit_max},{"logits_rms",std::sqrt(logit_sq/n)},{"probs_maxabs",prob_max},{"probs_rms",std::sqrt(prob_sq/n)},{"top8_order_changed_frames",order_flips},{"top8_order_flip_rate",double(order_flips)/frames},{"top8_set_changed_frames",set_flips},{"top8_set_flip_rate",double(set_flips)/frames},{"top8_changed_slots",changed_slots},{"mean_top8_retained_fraction",double(retained)/(frames*TopExperts)},{"normalized_weight_maxabs_slotwise",weight_max},{"shared_gain_maxabs",gain_max},{"mean_kl_f32_to_candidate",inf_kl?json(nullptr):json(kl/frames)},{"infinite_kl_terms",inf_kl}};}
};
float price(const std::function<void(cudaStream_t)>&launch,uint32_t*trash,size_t trash_bytes,bool cold){
 for(int i=0;i<8;i++)launch(0);ck(cudaDeviceSynchronize());cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));cudaGraph_t g;cudaGraphExec_t e;
 ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
 if(cold)flush<<<768,256,0,st>>>(trash,trash_bytes/4);
 ck(cudaEventRecordWithFlags(a,st,cudaEventRecordExternal));int count=cold?1:40;for(int i=0;i<count;i++)launch(st);ck(cudaEventRecordWithFlags(b,st,cudaEventRecordExternal));ck(cudaStreamEndCapture(st,&g));ck(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
 std::vector<float>times;for(int i=0;i<12;i++){ck(cudaGraphLaunch(e,st));ck(cudaStreamSynchronize(st));float ms;ck(cudaEventElapsedTime(&ms,a,b));if(i>=3)times.push_back(ms*1000/count);}std::sort(times.begin(),times.end());
 ck(cudaGraphExecDestroy(e));ck(cudaGraphDestroy(g));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));ck(cudaStreamDestroy(st));return times[4];
}
int main(int argc,char**argv){try{
 need(argc==4,"Usage: router-bf16 MODEL.gguf FIXTURE_DIR OUTPUT_DIR");fs::path fixture=argv[2],out=argv[3];fs::create_directories(out);
 auto g=take(Gguf::open(argv[1]));need(g.validate_model().ok(),"Model geometry invalid");
 std::ifstream meta(fixture/"fixture.json");json fm;meta>>fm;need(fm.at("actual_per_layer_inputs")==true&&fm.at("shape")==json::array({16,40,2048}),"Genuine all-layer fixture required");need(fm.at("model_bytes")==g.storage->identity.size&&fs::canonical(fm.at("model").get<std::string>())==fs::canonical(argv[1]),"Fixture model identity mismatch");
 auto inputs=read<float>(fixture/"inputs.f32",16*Layers*Hidden);auto oracle_logits=read<float>(fixture/"oracle-logits.f32",16*Layers*Experts),oracle_probs=read<float>(fixture/"oracle-probs.f32",16*Layers*Experts),oracle_weights=read<float>(fixture/"oracle-weights.f32",16*Layers*TopExperts),oracle_gain=read<float>(fixture/"oracle-shared.f32",16*Layers);auto oracle_ids=read<int32_t>(fixture/"oracle-ids.i32",16*Layers*TopExperts);
 const size_t elements=Layers*RouterBf16LayerElements;std::vector<float>source(elements);std::vector<uint16_t>packed(elements);
 for(int l=0;l<Layers;l++)for(int shared=0;shared<2;shared++){
  auto*t=g.find("blk."+std::to_string(l)+(shared?".ffn_gate_inp_shexp.weight":".ffn_gate_inp.weight"));need(t&&t->type==0&&t->shape==(shared?std::vector<uint64_t>{Hidden}:std::vector<uint64_t>{Hidden,Experts}),"Source router must be original F32 geometry");
  memcpy(source.data()+l*RouterBf16LayerElements+(shared?Experts*Hidden:0),t->data,t->bytes);
 }
 for(size_t i=0;i<elements;i++){need(std::isfinite(source[i]),"Nonfinite source coefficient");packed[i]=rne(source[i]);}
 int l2,persist,window;ck(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,0));ck(cudaDeviceGetAttribute(&persist,cudaDevAttrMaxPersistingL2CacheSize,0));ck(cudaDeviceGetAttribute(&window,cudaDevAttrMaxAccessPolicyWindowSize,0));size_t current;ck(cudaDeviceGetLimit(&current,cudaLimitPersistingL2CacheSize));
 json capacity={{"l2_bytes",l2},{"max_persisting_bytes",persist},{"max_window_bytes",window},{"initial_context_persisting_bytes",current},{"bf16_image_bytes",RouterBf16ImageBytes},{"f32_image_bytes",elements*4},{"gdn_state_bytes",GdnFloatsPerSlot*4},{"bf16_plus_gdn_state_bytes",RouterBf16ImageBytes+GdnFloatsPerSlot*4},{"changes_cache_policy",false}};save(out/"capacity.json",capacity);std::cout<<capacity.dump()<<std::endl;
 Device mem;auto*ds=mem.alloc<float>(elements),*dr=mem.alloc<float>(elements),*dx=mem.alloc<float>(inputs.size());auto*db=mem.alloc<uint16_t>(elements),*dg=mem.alloc<uint16_t>(elements);
 ck(cudaMemcpy(ds,source.data(),elements*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,packed.data(),elements*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dx,inputs.data(),inputs.size()*4,cudaMemcpyHostToDevice));round_reference<<<768,256>>>(ds,dr,dg,elements);ck(cudaGetLastError());
 std::vector<uint16_t>gpu_bits(elements);ck(cudaMemcpy(gpu_bits.data(),dg,elements*2,cudaMemcpyDeviceToHost));size_t coefficient_errors=0;for(size_t i=0;i<elements;i++)coefficient_errors+=gpu_bits[i]!=packed[i];need(coefficient_errors==0,"CPU RNE != independent CUDA BF16 conversion");
 auto views=[&](float*data){std::vector<LayerWeights>w(Layers);for(int l=0;l<Layers;l++){auto&v=w[l];v.index=l;v.router.data=data+l*RouterBf16LayerElements;v.router.format=WeightFormat::F32;v.router.n=Experts;v.router.k=Hidden;v.router.row_stride=Hidden*4;v.shared_router=v.router;v.shared_router.data=data+l*RouterBf16LayerElements+Experts*Hidden;v.shared_router.n=1;}auto*dw=mem.alloc<LayerWeights>(Layers);ck(cudaMemcpy(dw,w.data(),w.size()*sizeof(LayerWeights),cudaMemcpyHostToDevice));return dw;};
 auto*original=views(ds),*rounded=views(dr);Scratch sf,sr,sb;std::vector<Stats>layer(Layers);Stats total,oracle_baseline,oracle_bf16;json frames=json::array();size_t bit_errors=0,guard_cases=0;
 for(int step=0;step<16;step++)for(int l=0;l<Layers;l++){
  size_t frame=step*Layers+l;auto*x=dx+frame*Hidden;f32_router<<<188,256>>>(original,x,l,sf.s);select<<<1,256>>>(sf.s);Snapshot f=sf.get();f32_router<<<188,256>>>(rounded,x,l,sr.s);select<<<1,256>>>(sr.s);Snapshot r=sr.get();
  Snapshot oracle{};memcpy(oracle.logits,oracle_logits.data()+frame*Experts,Experts*4);memcpy(oracle.probs,oracle_probs.data()+frame*Experts,Experts*4);memcpy(oracle.ids,oracle_ids.data()+frame*TopExperts,TopExperts*4);memcpy(oracle.weights,oracle_weights.data()+frame*TopExperts,TopExperts*4);oracle.gain=oracle_gain[frame];oracle_baseline.add(oracle,f);
  Snapshot b{};
  for(int grid:{64,128,188,256})for(int rows:{1,2,4,8}){
   switch(rows){case 1:bf16_router<1><<<grid,256>>>(db,x,l,sb.s);break;case 2:bf16_router<2><<<grid,256>>>(db,x,l,sb.s);break;case 4:bf16_router<4><<<grid,256>>>(db,x,l,sb.s);break;case 8:bf16_router<8><<<grid,256>>>(db,x,l,sb.s);break;}
   select<<<1,256>>>(sb.s);ck(cudaGetLastError());auto candidate=sb.get();auto*a=reinterpret_cast<const uint8_t*>(&r);auto*c=reinterpret_cast<const uint8_t*>(&candidate);for(size_t i=0;i<sizeof(Snapshot);i+=4)bit_errors+=memcmp(a+i,c+i,4)!=0;++guard_cases;if(grid==188&&rows==2)b=candidate;
  }
  total.add(f,b);layer[l].add(f,b);oracle_bf16.add(oracle,b);std::vector<int32_t>fids(f.ids,f.ids+8),bids(b.ids,b.ids+8);auto fa=fids,ba=bids;std::sort(fa.begin(),fa.end());std::sort(ba.begin(),ba.end());
  double mx=0;for(int i=0;i<Experts;i++)mx=std::max(mx,std::abs(double(f.logits[i])-b.logits[i]));frames.push_back({{"step",step},{"layer",l},{"f32_top8",fids},{"bf16_top8",bids},{"set_changed",fa!=ba},{"logits_maxabs",mx}});
 }
 need(bit_errors==0,"BF16 helper differs from independent rounded-F32 coefficient reference");json layers=json::array();for(int l=0;l<Layers;l++){auto a=layer[l].result();a["layer"]=l;layers.push_back(a);}
 json result={{"variant","router-bf16-rne-fp32-fma-tree32"},{"source",argv[1]},{"source_router_f32_fnv64",hash(source.data(),source.size()*4)},{"coefficient_conversion_cases",elements},{"coefficient_bit_errors",coefficient_errors},{"bf16_reference_guard_cases",guard_cases},{"bf16_reference_output_bit_errors",bit_errors},{"actual_layer_teacher_frames",640},{"fixture",fm},{"fixed_input_diagnostic_not_end_to_end_quality",true},{"bf16_vs_original_f32",total.result()},{"original_f32_vs_oracle",oracle_baseline.result()},{"bf16_vs_oracle_router_callbacks",oracle_bf16.result()},{"per_layer",layers}};save(out/"routing.json",result);save(out/"frames.json",frames);std::cout<<json({{"guard_cases",guard_cases},{"output_bit_errors",bit_errors},{"routing",total.result()}}).dump()<<std::endl;
 fs::path image=out/"router-bf16-rne-v1.bin";if(fs::exists(image)){auto prior=read<uint16_t>(image,elements);need(prior==packed,"Existing immutable image differs");}else{fs::path temporary=image.string()+".tmp";{std::ofstream f(temporary,std::ios::binary);f.write(reinterpret_cast<const char*>(packed.data()),packed.size()*2);need(bool(f),"BF16 image write failed");}fs::rename(temporary,image);}
 auto&id=g.storage->identity;save(out/"image.json",{{"format_version",1},{"numerical_variant","router-bf16-rne-fp32-fma-tree32"},{"source",fs::canonical(argv[1]).string()},{"source_stat",{{"device",id.device},{"inode",id.inode},{"size",id.size},{"mtime_ns",id.mtime_ns},{"ctime_ns",id.ctime_ns}}},{"dtype","uint16 BF16 RNE little endian"},{"shape",{Layers,Experts+1,Hidden}},{"layer_bytes",RouterBf16LayerBytes},{"bytes",RouterBf16ImageBytes},{"payload",image.string()},{"fp32_source_retained",true},{"is_halo_weight_cache",false},{"reference_guard_pass",true}});
 auto*trash=mem.alloc<uint32_t>(256ul*1024*1024/4);json timing=json::array();
 for(int rows:{0,1,2,4,8}){
  auto run=[&](cudaStream_t st){switch(rows){case 0:f32_sweep<<<188,256,0,st>>>(original,dx,sf.s);break;case 1:bf16_sweep<1><<<188,256,0,st>>>(db,dx,sb.s);break;case 2:bf16_sweep<2><<<188,256,0,st>>>(db,dx,sb.s);break;case 4:bf16_sweep<4><<<188,256,0,st>>>(db,dx,sb.s);break;case 8:bf16_sweep<8><<<188,256,0,st>>>(db,dx,sb.s);break;}};
  json t={{"variant",rows?"BF16_RNE":"F32_compact_original"},{"rows_per_cta",rows?rows:2},{"layers_per_sweep",40},{"grid",188},{"threads",256},{"hot_sweep_us",price(run,trash,256ul*1024*1024,false)},{"cold_sweep_us",price(run,trash,256ul*1024*1024,true)},{"includes_select",false},{"includes_state_or_other_model_weights",false}};timing.push_back(t);std::cout<<t.dump()<<std::endl;
 }
 save(out/"timing.json",timing);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
