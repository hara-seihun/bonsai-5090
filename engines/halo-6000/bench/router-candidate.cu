#include "router.cuh"
#include "gguf.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <random>
#include <functional>
#include <fstream>
using namespace halo6000;
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}
__global__ void router_ref(MoeView v,MoeScratch s,int repeat){for(int i=0;i<repeat;i++){moe_router(v,s);__syncthreads();}}
template<int L,int R>__global__ void router_new(MoeView v,MoeScratch s,int repeat){for(int i=0;i<repeat;i++){moe_router_candidate<L,R>(v,s);__syncthreads();}}
template<int R>__global__ void router_fixed(MoeView v,MoeScratch s,int repeat){for(int i=0;i<repeat;i++){moe_router_fixed_f32<R>(v,s);__syncthreads();}}
template<bool New>__global__ void select_run(MoeView v,MoeScratch s,int repeat){__shared__ float tmp[32];for(int i=0;i<repeat;i++){if constexpr(New)moe_select_candidate(v,s,tmp);else moe_select(v,s,tmp);__syncthreads();}}
__global__ void prep(const float*x,ActivationBuffer a){quantize_cta(x,a);}
__global__ void flush(uint32_t*p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=uint32_t(i);}
struct Scratch {
 MoeScratch s;
 Scratch(){ck(cudaMalloc(&s.logits,Experts*4));ck(cudaMalloc(&s.probs,Experts*4));ck(cudaMalloc(&s.ids,TopExperts*4));ck(cudaMalloc(&s.weights,TopExperts*4));ck(cudaMalloc(&s.shared_gain,4));}
 ~Scratch(){for(void*p:{(void*)s.logits,(void*)s.probs,(void*)s.ids,(void*)s.weights,(void*)s.shared_gain})ck(cudaFree(p));}
};
static std::vector<uint8_t>copy(const void*p,size_t n){std::vector<uint8_t>x(n);ck(cudaMemcpy(x.data(),p,n,cudaMemcpyDeviceToHost));return x;}
static size_t bits(const void*a,const void*b,size_t n){auto x=copy(a,n),y=copy(b,n);size_t e=0;for(size_t i=0;i<n;i+=4)e+=memcmp(x.data()+i,y.data()+i,4)!=0;return e;}
using Launch=std::function<void(MoeView,MoeScratch,int,int,cudaStream_t)>;
struct Variant {const char*name;int lanes,rows;Launch run;bool fixed_f32=false;};
template<int L,int R>Variant variant(){return {"virtual",L,R,[](auto v,auto s,int blocks,int repeat,cudaStream_t stream){router_new<L,R><<<blocks,256,0,stream>>>(v,s,repeat);}};}
template<int R>Variant fixed_variant(){return {"fixed_f32",32,R,[](auto v,auto s,int blocks,int repeat,cudaStream_t stream){router_fixed<R><<<blocks,256,0,stream>>>(v,s,repeat);},true};}
struct Weight {
 void*router=nullptr,*shared=nullptr;LayerWeights*device=nullptr;LayerWeights host{};
 ~Weight(){ck(cudaFree(router));ck(cudaFree(shared));ck(cudaFree(device));}
};
static void load(const Gguf&g,int layer,Weight&w,bool q8_shape_probe=false){
 auto one=[&](const std::string&name,void*&device,MatrixView&view,int rows,bool require_q8){
  auto*t=g.find(name);if(!t||t->shape[0]!=Hidden||t->elements()/Hidden<size_t(rows)||(require_q8&&t->type!=8)){fprintf(stderr,"Invalid actual weight fixture %s\n",name.c_str());exit(2);}std::vector<uint8_t>data;
  WeightFormat fmt;const size_t elements=size_t(rows)*Hidden;
  if(t->type==0){fmt=WeightFormat::F32;data.assign(t->data,t->data+elements*4);}
  else if(t->type==8){fmt=WeightFormat::Q8Grouped128;data.resize(elements/128*136);auto status=repack_q8(t->data,elements/128,data.data(),data.size());if(!status.ok()){fprintf(stderr,"Repack: %s\n",status.error().message.c_str());exit(2);}}
  else{fprintf(stderr,"Unsupported router source type%u\n",t->type);exit(2);}
  ck(cudaFree(device));ck(cudaMalloc(&device,data.size()));ck(cudaMemcpy(device,data.data(),data.size(),cudaMemcpyHostToDevice));
  view={};view.data=device;view.k=Hidden;view.n=rows;view.experts=1;view.format=fmt;view.bytes=data.size();view.row_stride=fmt==WeightFormat::F32?Hidden*4:Hidden/128*136;view.scale_group=fmt==WeightFormat::F32?0:32;
 };
 std::string p="blk."+std::to_string(layer)+".";
 // Hybrid preserves F32 routers. This explicit Q8 shape probe uses existing
 // attention/gate weight bytes, not a newly quantized or model-router matrix.
 one(p+(q8_shape_probe?(attention_layer(layer)?"attn_q.weight":"attn_gate.weight"):"ffn_gate_inp.weight"),w.router,w.host.router,Experts,q8_shape_probe);
 one(p+"ffn_gate_inp_shexp.weight",w.shared,w.host.shared_router,1,false);
 if(!q8_shape_probe&&w.host.router.format!=WeightFormat::F32){fprintf(stderr,"Expected source F32 router; update experiment labels\n");exit(2);}
 if(!w.device)ck(cudaMalloc(&w.device,sizeof(LayerWeights)));ck(cudaMemcpy(w.device,&w.host,sizeof(LayerWeights),cudaMemcpyHostToDevice));
}
static float graph_us(const std::function<void(cudaStream_t)>&f){
 for(int i=0;i<10;i++)f(0);ck(cudaDeviceSynchronize());
 cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 cudaGraph_t graph;cudaGraphExec_t exec;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<100;i++)f(stream);ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));std::vector<float>times;
 for(int i=0;i<9;i++){ck(cudaEventRecord(a,stream));ck(cudaGraphLaunch(exec,stream));ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));times.push_back(ms*10);}
 std::sort(times.begin(),times.end());ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));return times[4];
}
static float cold_us(const std::function<void(cudaStream_t)>&f,uint32_t*trash,size_t bytes){
 cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));std::vector<float>times;
 cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));cudaGraph_t graph;cudaGraphExec_t exec;
 ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
 flush<<<768,256,0,stream>>>(trash,bytes/4);ck(cudaEventRecordWithFlags(a,stream,cudaEventRecordExternal));f(stream);ck(cudaEventRecordWithFlags(b,stream,cudaEventRecordExternal));
 ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 for(int i=0;i<13;i++){ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));float ms;ck(cudaEventElapsedTime(&ms,a,b));if(i>=3)times.push_back(ms*1000);}
 std::sort(times.begin(),times.end());ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return times[times.size()/2];
}
int main(int argc,char**argv){
 if(argc!=4){fprintf(stderr,"Usage: router-candidate V1.gguf HYBRID.gguf ACTUAL_INPUTS.f32\n");return 2;}
 std::vector<float>actual(16*Hidden);std::ifstream inputs(argv[3],std::ios::binary);inputs.read(reinterpret_cast<char*>(actual.data()),actual.size()*4);if(!inputs||inputs.peek()!=EOF){fprintf(stderr,"Actual inputs must contain16x2048 float32\n");return 2;}
 auto r0=Gguf::open(argv[1]),r1=Gguf::open(argv[2]);if(!r0.ok()||!r1.ok()){fprintf(stderr,"GGUF load failed\n");return 2;}auto&v1=r0.value();auto&hybrid=r1.value();
 std::vector<Variant>variants={{"reference",32,1,[](auto v,auto s,int blocks,int repeat,cudaStream_t stream){router_ref<<<blocks,256,0,stream>>>(v,s,repeat);}},variant<32,1>(),variant<16,1>(),variant<8,1>(),variant<4,1>(),variant<32,2>(),variant<16,2>(),variant<8,2>(),variant<4,2>(),variant<32,4>(),variant<16,4>(),variant<8,4>(),variant<4,4>(),variant<32,8>(),fixed_variant<1>(),fixed_variant<2>(),fixed_variant<4>(),fixed_variant<8>()};
 Scratch ref,got;float*x=nullptr;ActivationBuffer q;q.k=Hidden;ck(cudaMalloc(&x,Hidden*4));ck(cudaMalloc(&q.qs,Hidden));ck(cudaMalloc(&q.scale,Hidden/32*4));
 MoeView v{};v.input=x;v.prepared=q.view();
 std::mt19937 rng(87);size_t selector_errors=0,router_errors=0,cases=0;
 for(int test=0;test<512;test++){
  std::vector<float>logits(Experts);for(int j=0;j<Experts;j++)logits[j]=test==0?0:test==1?float(j%4):test==2?(j==198?90.f:-90.f):test==3?float(j%32)*1e-8f:test==4?-1000.f+float(j%16):test==5?float(j%2)*200.f:float(int(rng()%2001)-1000)/(test%3==0?1000000.f:test%3==1?7.f:1.f);
  v.weight_scale=test%7==0?1.125f:1.f;
  ck(cudaMemcpy(ref.s.logits,logits.data(),Experts*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(got.s.logits,logits.data(),Experts*4,cudaMemcpyHostToDevice));
  select_run<false><<<1,256>>>(v,ref.s,1);select_run<true><<<1,256>>>(v,got.s,1);ck(cudaGetLastError());
  selector_errors+=bits(ref.s.probs,got.s.probs,Experts*4)+bits(ref.s.ids,got.s.ids,TopExperts*4)+bits(ref.s.weights,got.s.weights,TopExperts*4);
 }
 for(int format=0;format<3;format++){
  auto&source=format?hybrid:v1;Weight w;
  for(int layer=0;layer<Layers;layer++){
   load(source,layer,w,format==2);v.weights=w.device;
   for(int activation=0;activation<19;activation++){
    std::vector<float>input(Hidden);if(activation>=3)std::copy_n(actual.data()+(activation-3)*Hidden,Hidden,input.data());else for(auto&value:input)value=float(int(rng()%20001)-10000)/(activation==0?3000.f:activation==1?3e8f:37.f);
    ck(cudaMemcpy(x,input.data(),Hidden*4,cudaMemcpyHostToDevice));prep<<<1,256>>>(x,q);
    router_ref<<<188,256>>>(v,ref.s,1);select_run<false><<<1,256>>>(v,ref.s,1);
    for(auto&candidate:variants){if(candidate.fixed_f32&&format==2)continue;candidate.run(v,got.s,188,1,0);select_run<true><<<1,256>>>(v,got.s,1);ck(cudaGetLastError());
     router_errors+=bits(ref.s.logits,got.s.logits,Experts*4)+bits(ref.s.shared_gain,got.s.shared_gain,4)+bits(ref.s.probs,got.s.probs,Experts*4)+bits(ref.s.ids,got.s.ids,TopExperts*4)+bits(ref.s.weights,got.s.weights,TopExperts*4);++cases;
    }
   }
  }
 }
 printf("{\"guard\":true,\"selector_cases\":512,\"selector_bit_errors\":%zu,\"router_cases\":%zu,\"router_bit_errors\":%zu,\"layers_each_source\":40,\"weight_sources\":[\"v1_router_F32\",\"hybrid_router_F32\",\"hybrid_Q8_shape_probe_NOT_router\"],\"synthetic_activation_cases\":3,\"actual_callback_activation_cases\":16}\n",selector_errors,cases,router_errors);fflush(stdout);
 if(selector_errors||router_errors)return 1;
 uint32_t*trash;const size_t trash_bytes=256ul*1024*1024;ck(cudaMalloc(&trash,trash_bytes));
 Weight select_weight;load(v1,0,select_weight);v.weights=select_weight.device;router_ref<<<188,256>>>(v,got.s,1);
 for(bool newer:{false,true}){
  auto run=[&](int repeats,cudaStream_t stream){if(newer)select_run<true><<<1,256,0,stream>>>(v,got.s,repeats);else select_run<false><<<1,256,0,stream>>>(v,got.s,repeats);};
  printf("{\"phase\":\"select\",\"variant\":\"%s\",\"graph_hot_us\":%.6f,\"in_kernel_us\":%.6f}\n",newer?"network":"reference",graph_us([&](auto stream){run(1,stream);}),graph_us([&](auto stream){run(1000,stream);})/1000);fflush(stdout);
 }
 for(int format=0;format<3;format++){
  Weight w;load(format?hybrid:v1,0,w,format==2);v.weights=w.device;
  for(auto&candidate:variants){
   if(candidate.fixed_f32&&format==2)continue;
   auto run=[&](int reps,cudaStream_t stream){candidate.run(v,got.s,188,reps,stream);};
   float hot=graph_us([&](auto stream){run(1,stream);}),cold=cold_us([&](auto stream){run(1,stream);},trash,trash_bytes),phase=graph_us([&](auto stream){run(100,stream);})/100;
   printf("{\"phase\":\"router\",\"format\":\"%s\",\"variant\":\"%s\",\"lanes\":%d,\"rows_per_cta\":%d,\"grid\":188,\"threads\":256,\"graph_hot_us\":%.6f,\"cold_us\":%.6f,\"in_kernel_us\":%.6f}\n",format==2?"Q8Grouped128_shape_probe":format==1?"hybrid_router_F32":"v1_router_F32",candidate.name,candidate.lanes,candidate.rows,hot,cold,phase);fflush(stdout);
  }
 }
 ck(cudaFree(trash));ck(cudaFree(x));ck(cudaFree(q.qs));ck(cudaFree(q.scale));return 0;
}
