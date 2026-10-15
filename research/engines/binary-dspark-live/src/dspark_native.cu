#include "dspark_native.cuh"
#include "dspark_kernels.cuh"
#include "gguf.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <numeric>
namespace h5 {
DsparkVocabSelection read_dspark_vocab(const char* path, int source_vocab) {
 using Code=DsparkVocabErrorCode;
 if(source_vocab<=0)return DsparkVocabError{Code::invalid_source_vocab,"invalid source domain"};
 std::vector<int> ids;
 if(!path){ids.resize(source_vocab);std::iota(ids.begin(),ids.end(),0);return ids;}
 if(!*path)return DsparkVocabError{Code::empty_path,"H5_DSPARK_VOCAB is set to an empty path"};
 std::ifstream input(path,std::ios::binary);
 if(!input)return DsparkVocabError{Code::open_failed,"cannot open candidate ID file: "+std::string(path)};
 std::vector<unsigned char> seen(source_vocab,0);int id=0;bool active=false;char c;
 while(input.get(c)){
  if(c>='0'&&c<='9'){
   const int digit=c-'0';
   if(id>(source_vocab-1)/10 || (id==(source_vocab-1)/10 && digit>(source_vocab-1)%10))
    return DsparkVocabError{Code::out_of_range,"candidate ID exceeds source domain"};
   id=id*10+digit;active=true;
  }else if(c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'){
   if(active){if(seen[id])return DsparkVocabError{Code::duplicate_id,"duplicate candidate ID "+std::to_string(id)};seen[id]=1;ids.push_back(id);id=0;active=false;}
  }else return DsparkVocabError{Code::malformed_id,"expected whitespace-separated unsigned decimal IDs"};
 }
 if(!input.eof())return DsparkVocabError{Code::read_failed,"cannot read candidate ID file: "+std::string(path)};
 if(active){if(seen[id])return DsparkVocabError{Code::duplicate_id,"duplicate candidate ID "+std::to_string(id)};ids.push_back(id);}
 if(ids.empty())return DsparkVocabError{Code::empty_selection,"candidate ID file is empty"};
 std::sort(ids.begin(),ids.end());return ids;
}
namespace {
void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(std::string("DSpark CUDA: ")+cudaGetErrorString(e)); }
void blas(cublasStatus_t e) { if(e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("DSpark cuBLAS status "+std::to_string((int)e)); }
constexpr int D=5120,V=248320,TAPS=5,H=40,KV=4,HD=128,LAYERS=6,FF=5120,RANK=256;
__global__ void bias_silu(float* x,const float* bias,int rows,int width,bool silu) {
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(rows)*width)return;
 float v=x[i]+bias[i%width];x[i]=silu?v/(1+expf(-v)):v;
}
__global__ void confidence_input(const float* hidden,const __nv_bfloat16* a,const int* prev,float* x,int rows,int* error) {
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(rows)*(D+RANK))return;
 int r=i/(D+RANK),c=i%(D+RANK);if(prev[r]<0||prev[r]>=V){atomicMax(error,int(dspark::DeviceError::token_out_of_range));x[i]=nanf("");return;}x[i]=c<D?hidden[size_t(r)*D+c]:__bfloat162float(a[size_t(prev[r])*RANK+c-D]);
}
__global__ void sigmoid_conf(float* x,float bias,int rows){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows)x[i]=1.f/(1.f+expf(-(x[i]+bias)));}
}
struct NativeDspark::State {
 struct Mat { __nv_bfloat16* w=nullptr;int n=0,k=0; };
 struct Layer {Mat qkv,kv,o,gate,up,down;float *an,*fn,*qn,*kn;};
 struct Resources {
 cublasHandle_t handle=nullptr;std::vector<void*> allocations;
 ~Resources(){if(handle)cublasDestroy(handle);for(void*p:allocations)cudaFree(p);}
 } resources;
 cudaStream_t stream; cublasHandle_t& handle=resources.handle; std::vector<void*>& allocations=resources.allocations;
 std::vector<int> taps,output_ids;int slots,context,maxrows,proposals,block,mask,ring,output_count,completed_rows=0;
 int* candidate_ids=nullptr;
 float eps,rope,conf_bias; bool snr_enabled;
 size_t weights=0;
 Mat fc,head,snr1,snr2,conf;Layer layer[LAYERS];
 __nv_bfloat16 *embedding,*markov_a,*markov_b,*activation;
 float *hnorm,*onorm,*snr_bias1,*snr_bias2,*snr_rows,*mode=nullptr;
 float *features,*ctx_hidden,*hidden,*normal,*qkv,*attention,*qr,*gate,*up,*product,*projected,*base_logits,*step_logits,*confidence_in,*conf_step,*d_conf;
 __half *keys,*values;int *tags,*error,*positions,*slot_ids,*cache_indices,*ids,*offsets,*block_ids,*previous,*winners;
 dspark::AttentionBlock* blocks;
 std::vector<float> host_conf;
 std::vector<int> host_winners;
 dspark::KvGeometry geometry()const{return {slots,LAYERS,ring,H,KV,HD};}
 template<class X>X* alloc(size_t count){X*p;check(cudaMalloc(&p,count*sizeof(X)));allocations.push_back(p);check(cudaMemsetAsync(p,0,count*sizeof(X),stream));return p;}
 template<class X>X* upload(const void* data,size_t count){X*p=alloc<X>(count);check(cudaMemcpyAsync(p,data,count*sizeof(X),cudaMemcpyHostToDevice,stream));weights+=count*sizeof(X);return p;}
 int integer(const Gguf& g,const char* key) {auto it=g.kv.find(key);if(it==g.kv.end()||it->second.ints.size()!=1)throw std::runtime_error(std::string("DSpark missing scalar ")+key);return it->second.ints[0];}
 float real(const Gguf&g,const char*key){auto it=g.kv.find(key);if(it==g.kv.end()||it->second.nums.size()!=1||!std::isfinite(it->second.nums[0]))throw std::runtime_error(std::string("DSpark missing finite scalar ")+key);return it->second.nums[0];}
 void require(const Gguf&g,const char*key,int expected){if(integer(g,key)!=expected)throw std::runtime_error(std::string("unsupported DSpark geometry: ")+key);}
 const GTensor& tensor(const Gguf& g,const std::string& name,std::vector<uint64_t> shape,uint32_t type){const auto&t=g.t(name);if(t.ne!=shape||t.type!=type)throw std::runtime_error("DSpark tensor shape/type: "+name);return t;}
 float* norm(const Gguf&g,const std::string&name,int width){const auto&t=tensor(g,name,{uint64_t(width)},0);return upload<float>(t.data,width);}
 Mat matrix(const Gguf&g,const std::vector<std::string>& names,int k,int n) {
 Mat m{alloc<__nv_bfloat16>(size_t(k)*n),n,k};int off=0;
 for(const auto&name:names){const auto&t=g.t(name);if(t.ne.size()!=2||t.ne[0]!=uint64_t(k)||t.type!=30)throw std::runtime_error("DSpark BF16 matrix: "+name);int nr=t.ne[1];if(off+nr>n)throw std::runtime_error("DSpark joined matrix overflow");check(cudaMemcpyAsync(m.w+size_t(off)*k,t.data,t.bytes,cudaMemcpyHostToDevice,stream));weights+=t.bytes;off+=nr;}
 if(off!=n)throw std::runtime_error("DSpark joined matrix width mismatch");return m;
 }
 Mat output_matrix(const Gguf&g,const std::string&name,int k) {
 if(output_count==V)return matrix(g,{name},k,V);
 const auto&t=tensor(g,name,{uint64_t(k),uint64_t(V)},30);
 std::vector<__nv_bfloat16> selected(size_t(output_count)*k);
 for(int i=0;i<output_count;i++)std::memcpy(selected.data()+size_t(i)*k,
     static_cast<const uint8_t*>(t.data)+size_t(output_ids[i])*k*sizeof(__nv_bfloat16),size_t(k)*sizeof(__nv_bfloat16));
 Mat m{upload<__nv_bfloat16>(selected.data(),selected.size()),output_count,k};
 check(cudaStreamSynchronize(stream));return m;
 }
 void project(const Mat&m,const float*x,float*y,int rows){if(rows<1||rows>maxrows||!m.w)throw std::runtime_error("DSpark projection capacity");check(dspark::launch_f32_to_bf16(x,activation,rows,m.k,stream));const float one=1,zero=0;blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,m.n,rows,m.k,&one,m.w,CUDA_R_16BF,m.k,activation,CUDA_R_16BF,m.k,&zero,y,CUDA_R_32F,m.n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));}
 void rms(const float*x,const float*w,float*y,int rows,int width){check(dspark::launch_rms_norm(x,w,y,rows,width,eps,stream));}
 void add(const float*x,const float*r,float*y,int rows,int width){check(dspark::launch_add(x,r,y,rows,width,stream));}
 void put_metadata(const std::vector<int>&p,const std::vector<int>&s,const std::vector<int>&ci){check(cudaMemcpyAsync(positions,p.data(),p.size()*4,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(slot_ids,s.data(),s.size()*4,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(cache_indices,ci.data(),ci.size()*4,cudaMemcpyHostToDevice,stream));}
 void qkv_cache(int l,int rows,bool emit){dspark::QkvArgs a{};a.qkv=qkv;a.q_norm=layer[l].qn;a.k_norm=layer[l].kn;a.positions=positions;a.batch_slots=slot_ids;a.cache_indices=cache_indices;a.q_rotated=emit?qr:nullptr;a.keys=keys;a.values=values;a.cache_positions=tags;a.device_error=error;a.geometry=geometry();a.rows=rows;a.layer=l;a.qkv_stride=emit?D+2*KV*HD:2*KV*HD;a.q_offset=0;a.k_offset=emit?D:0;a.v_offset=a.k_offset+KV*HD;a.n_rot=HD;a.emit_q=emit;a.pairing=dspark::RopePairing::split_half;a.epsilon=eps;a.rope_base=rope;a.rope_scale=1.f;check(dspark::launch_qkv_rope_cache(a,stream));}
 State(const std::string&path,int ns,int ctx,int rows,int prop,cudaStream_t st):stream(st),slots(ns),context(ctx),maxrows(rows),proposals(prop) {
 Gguf g;g.open(path);if(g.kv.at("general.architecture").strs.at(0)!="dspark")throw std::runtime_error("expected DSpark GGUF");
 require(g,"dspark.block_count",LAYERS);require(g,"dspark.embedding_length",D);require(g,"dspark.feed_forward_length",FF);require(g,"dspark.attention.head_count",H);require(g,"dspark.attention.head_count_kv",KV);require(g,"dspark.attention.key_length",HD);require(g,"dspark.attention.value_length",HD);require(g,"dspark.vocab_size",V);
 block=integer(g,"dspark.dspark.block_size");mask=integer(g,"dspark.dspark.mask_token_id");require(g,"dspark.dspark.markov_rank",RANK);
 if(block!=4||mask<0||mask>=V||ns<1||ns>INT_MAX/block||prop<1||prop>block||rows<ns*block||ctx<block+1||ctx>INT_MAX-block)throw std::runtime_error("DSpark requires block4, proposals1..4, sufficient rows/context");
 for(const char*key:{"dspark.dspark.hidden_correction","dspark.dspark.markov_deploy_exposure"})if(g.kv.count(key)&&integer(g,key))throw std::runtime_error("DSpark hidden-correction checkpoint is not supported by this native reference route");
 require(g,"dspark.dspark.confidence_head",1);require(g,"dspark.dspark.confidence_head_with_markov",1);require(g,"dspark.dspark.log_snr_conditioning",1);snr_enabled=true;
 eps=real(g,"dspark.attention.layer_norm_rms_epsilon");rope=real(g,"dspark.rope.freq_base");if(eps<=0||rope<=0)throw std::runtime_error("DSpark invalid norm/rope parameters");
 taps=g.kv.at("dspark.dspark.target_layers").ints.size()==TAPS?std::vector<int>(g.kv.at("dspark.dspark.target_layers").ints.begin(),g.kv.at("dspark.dspark.target_layers").ints.end()):std::vector<int>{};
 if(taps!=std::vector<int>{1,16,31,46,61})throw std::runtime_error("unsupported DSpark target tap contract");
 auto selection=read_dspark_vocab(std::getenv("H5_DSPARK_VOCAB"),V);
 if(auto* failure=std::get_if<DsparkVocabError>(&selection))throw DsparkVocabException(*failure);
 output_ids=std::move(std::get<std::vector<int>>(selection));output_count=int(output_ids.size());
 if(output_count==1&&output_ids[0]==mask)throw DsparkVocabException({DsparkVocabErrorCode::only_mask,"selection contains only the suppressed mask token"});
 if(output_count!=V)candidate_ids=upload<int>(output_ids.data(),output_ids.size());
 ring=context+block;blas(cublasCreate(&handle));blas(cublasSetStream(handle,stream));blas(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));blas(cublasSetWorkspace(handle,alloc<uint8_t>(8*1024*1024),8*1024*1024));
 embedding=matrix(g,{"token_embd.weight"},D,V).w;head=output_matrix(g,"output.weight",D);fc=matrix(g,{"dspark.fc.weight"},TAPS*D,D);hnorm=norm(g,"dspark.hidden_norm.weight",D);onorm=norm(g,"output_norm.weight",D);
 markov_a=matrix(g,{"dspark.markov_head_a.weight"},RANK,V).w;markov_b=output_matrix(g,"dspark.markov_head_b.weight",RANK).w;
 conf=matrix(g,{"dspark.confidence_head.weight"},D+RANK,1);conf_bias=*reinterpret_cast<const float*>(tensor(g,"dspark.confidence_head.bias",{1},0).data);if(!std::isfinite(conf_bias))throw std::runtime_error("DSpark nonfinite confidence bias");
 snr1=matrix(g,{"dspark.log_snr_fc1.weight"},128,D);snr2=matrix(g,{"dspark.log_snr_fc2.weight"},D,D);snr_bias1=norm(g,"dspark.log_snr_fc1.bias",D);snr_bias2=norm(g,"dspark.log_snr_fc2.bias",D);
 if(g.has("dspark.mode_embedding.bias"))mode=norm(g,"dspark.mode_embedding.bias",D);
 for(int l=0;l<LAYERS;++l){std::string p="blk."+std::to_string(l)+".";auto&y=layer[l];y.qkv=matrix(g,{p+"attn_q.weight",p+"attn_k.weight",p+"attn_v.weight"},D,D+2*KV*HD);y.kv=matrix(g,{p+"attn_k.weight",p+"attn_v.weight"},D,2*KV*HD);y.o=matrix(g,{p+"attn_output.weight"},D,D);y.gate=matrix(g,{p+"ffn_gate.weight"},D,FF);y.up=matrix(g,{p+"ffn_up.weight"},D,FF);y.down=matrix(g,{p+"ffn_down.weight"},FF,D);y.an=norm(g,p+"attn_norm.weight",D);y.fn=norm(g,p+"ffn_norm.weight",D);y.qn=norm(g,p+"attn_q_norm.weight",HD);y.kn=norm(g,p+"attn_k_norm.weight",HD);}
 activation=alloc<__nv_bfloat16>(size_t(maxrows)*TAPS*D);features=alloc<float>(size_t(maxrows)*TAPS*D);ctx_hidden=alloc<float>(size_t(maxrows)*D);hidden=alloc<float>(size_t(maxrows)*D);normal=alloc<float>(size_t(maxrows)*D);qkv=alloc<float>(size_t(maxrows)*(D+2*KV*HD));attention=alloc<float>(size_t(maxrows)*D);qr=alloc<float>(size_t(maxrows)*D);gate=alloc<float>(size_t(maxrows)*FF);up=alloc<float>(size_t(maxrows)*FF);product=alloc<float>(size_t(maxrows)*FF);projected=alloc<float>(size_t(maxrows)*D);base_logits=alloc<float>(size_t(slots)*block*output_count);step_logits=alloc<float>(size_t(slots)*output_count);confidence_in=alloc<float>(size_t(slots)*(D+RANK));conf_step=alloc<float>(slots);d_conf=alloc<float>(size_t(slots)*prop);
 keys=alloc<__half>(size_t(slots)*LAYERS*ring*KV*HD);values=alloc<__half>(size_t(slots)*LAYERS*ring*KV*HD);tags=alloc<int>(size_t(slots)*LAYERS*ring);check(cudaMemsetAsync(tags,255,size_t(slots)*LAYERS*ring*4,stream));error=alloc<int>(1);positions=alloc<int>(maxrows);slot_ids=alloc<int>(maxrows);cache_indices=alloc<int>(maxrows);ids=alloc<int>(maxrows);offsets=alloc<int>(maxrows);block_ids=alloc<int>(maxrows);blocks=alloc<dspark::AttentionBlock>(slots);previous=alloc<int>(slots);winners=alloc<int>(slots);host_conf.resize(size_t(slots)*prop);host_winners.resize(size_t(slots)*prop);
 const float lo=real(g,"dspark.dspark.min_log_snr"),hi=real(g,"dspark.dspark.max_log_snr");if(!(hi>lo))throw std::runtime_error("DSpark invalid log-SNR bounds");float freq[256];for(int row=0;row<2;row++)for(int i=0;i<64;i++){float t=row?0.f:1000.f,angle=t*expf(-logf(10000.f)*float(i)/64.f);freq[row*128+i]=sinf(angle);freq[row*128+64+i]=cosf(angle);}float*fe=upload<float>(freq,256);snr_rows=alloc<float>(2*D);project(snr1,fe,normal,2);bias_silu<<<(2*D+255)/256,256,0,stream>>>(normal,snr_bias1,2,D,true);project(snr2,normal,snr_rows,2);bias_silu<<<(2*D+255)/256,256,0,stream>>>(snr_rows,snr_bias2,2,D,false);check(cudaGetLastError());check(cudaStreamSynchronize(stream));
 fprintf(stderr,"native DSpark BF16: own embedding/head, six layers, taps1/16/31/46/61, block4, output %d/%d, full Markov source domain, %.3fGB weights; raw tap FC contract\n",output_count,V,weights/1e9);
 }
};
NativeDspark::NativeDspark(const std::string&p,int s,int c,int r,int n,cudaStream_t st):state(std::make_unique<State>(p,s,c,r,n,st)){}
NativeDspark::~NativeDspark()=default;
const std::vector<int>&NativeDspark::capture_layers()const{return state->taps;}
const float*NativeDspark::confidence()const{return state->host_conf.data();}
int NativeDspark::query_rows()const{return state->completed_rows;}
int NativeDspark::output_vocab()const{return state->output_count;}
const std::vector<int>&NativeDspark::output_ids()const{return state->output_ids;}
cudaError_t NativeDspark::read_base_logits(float*out,size_t elements)const{auto&s=*state;if(!out||s.completed_rows<=0||elements!=size_t(s.completed_rows)*s.output_count)return cudaErrorInvalidValue;return cudaMemcpy(out,s.base_logits,elements*sizeof(float),cudaMemcpyDeviceToHost);}
void NativeDspark::reset(int slot){auto&s=*state;if(slot<0||slot>=s.slots)throw std::runtime_error("DSpark invalid reset slot");check(cudaMemsetAsync(s.tags+size_t(slot)*LAYERS*s.ring,255,size_t(LAYERS)*s.ring*4,s.stream));}
void NativeDspark::copy_slot(int src,int dst){auto&s=*state;if(src<0||dst<0||src>=s.slots||dst>=s.slots)throw std::runtime_error("DSpark invalid copy slot");size_t n=size_t(LAYERS)*s.ring*KV*HD;check(cudaMemcpyAsync(s.keys+dst*n,s.keys+src*n,n*2,cudaMemcpyDeviceToDevice,s.stream));check(cudaMemcpyAsync(s.values+dst*n,s.values+src*n,n*2,cudaMemcpyDeviceToDevice,s.stream));n=size_t(LAYERS)*s.ring;check(cudaMemcpyAsync(s.tags+dst*n,s.tags+src*n,n*4,cudaMemcpyDeviceToDevice,s.stream));}
void NativeDspark::run(const float*taps,int captured,int ns,const int*slots,const int*counts,const int*row0,const int*pos0,const int*anchors,const int*starts,int*drafts){
 auto&s=*state;if(!taps||!slots||!counts||!row0||!pos0||!anchors||!starts||captured<0||captured>s.maxrows||ns<1||ns>s.slots)throw std::runtime_error("DSpark invalid call");
 s.completed_rows=0;check(cudaMemsetAsync(s.error,0,4,s.stream));std::vector<int> ip,is,ci;int ni=0;
 for(int b=0;b<ns;b++){if(slots[b]<0||slots[b]>=s.slots||counts[b]<0||row0[b]<0||row0[b]>captured||counts[b]>captured-row0[b]||pos0[b]<0||pos0[b]>s.context||counts[b]>s.context-pos0[b]||anchors[b]<-1||anchors[b]>=V)throw std::runtime_error("DSpark invalid ingestion metadata");for(int a=0;a<b;a++)if(slots[a]==slots[b])throw std::runtime_error("DSpark duplicate slot");if(counts[b]>s.maxrows-ni)throw std::runtime_error("DSpark ingestion capacity");if(counts[b])check(cudaMemcpyAsync(s.features+size_t(ni)*TAPS*D,taps+size_t(row0[b])*TAPS*D,size_t(counts[b])*TAPS*D*4,cudaMemcpyDeviceToDevice,s.stream));for(int j=0;j<counts[b];j++){ip.push_back(pos0[b]+j);is.push_back(slots[b]);ci.push_back((pos0[b]+j)%s.ring);}ni+=counts[b];}
 if(ni){s.put_metadata(ip,is,ci);s.project(s.fc,s.features,s.ctx_hidden,ni);s.rms(s.ctx_hidden,s.hnorm,s.ctx_hidden,ni,D);for(int l=0;l<LAYERS;l++){s.project(s.layer[l].kv,s.ctx_hidden,s.qkv,ni);s.qkv_cache(l,ni,false);}}
 std::vector<int> draft_ip,draft_is,draft_ci;std::vector<int> token,off,bid,prev;std::vector<dspark::AttentionBlock> hb;int nb=0;
 for(int b=0;b<ns;b++)if(anchors[b]>=0){if(starts[b]<0||starts[b]>s.context-s.block)throw std::runtime_error("DSpark block exceeds target context");hb.push_back({slots[b],starts[b],0,nb*s.block,s.block});prev.push_back(anchors[b]);for(int j=0;j<s.block;j++){draft_ip.push_back(starts[b]+j);draft_is.push_back(slots[b]);draft_ci.push_back((starts[b]+j)%s.ring);token.push_back(j?s.mask:anchors[b]);off.push_back(j);bid.push_back(nb);}nb++;}
 const int rows=nb*s.block;if(rows>s.maxrows||(!drafts&&nb))throw std::runtime_error("DSpark block capacity/output");
 if(nb){s.put_metadata(draft_ip,draft_is,draft_ci);check(cudaMemcpyAsync(s.ids,token.data(),rows*4,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.offsets,off.data(),rows*4,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.block_ids,bid.data(),rows*4,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.blocks,hb.data(),nb*sizeof(hb[0]),cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.previous,prev.data(),nb*4,cudaMemcpyHostToDevice,s.stream));
 dspark::EmbeddingArgs e{};e.table_bf16=s.embedding;e.ids=s.ids;e.block_offsets=s.offsets;e.snr_two_rows=s.snr_rows;e.mode=s.mode;e.output=s.hidden;e.device_error=s.error;e.rows=rows;e.width=D;e.vocab=V;e.block_rows=s.block;check(dspark::launch_embedding(e,s.stream));
 for(int l=0;l<LAYERS;l++){auto&y=s.layer[l];s.rms(s.hidden,y.an,s.normal,rows,D);s.project(y.qkv,s.normal,s.qkv,rows);s.qkv_cache(l,rows,true);dspark::AttentionArgs a{};a.q_rotated=s.qr;a.keys=s.keys;a.values=s.values;a.cache_positions=s.tags;a.positions=s.positions;a.batch_slots=s.slot_ids;a.cache_indices=s.cache_indices;a.block_ids=s.block_ids;a.blocks=s.blocks;a.output=s.attention;a.device_error=s.error;a.geometry=s.geometry();a.rows=rows;a.layer=l;a.block_count=nb;a.context_window=s.context;a.qk_scale=1.f/sqrtf(float(HD));check(dspark::launch_attention(a,s.stream));s.project(y.o,s.attention,s.projected,rows);s.add(s.projected,s.hidden,s.hidden,rows,D);s.rms(s.hidden,y.fn,s.normal,rows,D);s.project(y.gate,s.normal,s.gate,rows);s.project(y.up,s.normal,s.up,rows);check(dspark::launch_silu_mul(s.gate,s.up,s.product,rows,FF,s.stream));s.project(y.down,s.product,s.projected,rows);s.add(s.projected,s.hidden,s.hidden,rows,D);}
 s.rms(s.hidden,s.onorm,s.normal,rows,D);s.project(s.head,s.normal,s.base_logits,rows);
 for(int j=0;j<s.proposals;j++){for(int b=0;b<nb;b++){check(cudaMemcpyAsync(s.step_logits+size_t(b)*s.output_count,s.base_logits+size_t(b*s.block+j)*s.output_count,size_t(s.output_count)*4,cudaMemcpyDeviceToDevice,s.stream));check(cudaMemcpyAsync(s.hidden+size_t(b)*D,s.normal+size_t(b*s.block+j)*D,size_t(D)*4,cudaMemcpyDeviceToDevice,s.stream));}
 confidence_input<<<(nb*(D+RANK)+255)/256,256,0,s.stream>>>(s.hidden,s.markov_a,s.previous,s.confidence_in,nb,s.error);s.project(s.conf,s.confidence_in,s.conf_step,nb);sigmoid_conf<<<(nb+255)/256,256,0,s.stream>>>(s.conf_step,s.conf_bias,nb);for(int b=0;b<nb;b++)check(cudaMemcpyAsync(s.d_conf+size_t(b)*s.proposals+j,s.conf_step+b,4,cudaMemcpyDeviceToDevice,s.stream));
 dspark::MarkovArgs m{};m.base_logits=s.step_logits;m.codebook_a=s.markov_a;m.codebook_b=s.markov_b;m.source_ids=s.previous;m.logits=s.step_logits;m.device_error=s.error;m.rows=nb;m.source_vocab=V;m.output_count=s.output_count;m.output_ids=s.candidate_ids;m.rank=RANK;m.mask_id=s.mask;check(dspark::launch_markov(m,s.stream));check(dspark::launch_candidate_argmax(s.step_logits,s.candidate_ids,s.winners,s.error,nb,s.output_count,V,s.stream));for(int b=0;b<nb;b++)check(cudaMemcpyAsync(&s.host_winners[size_t(b)*s.proposals+j],s.winners+b,4,cudaMemcpyDeviceToHost,s.stream));check(cudaMemcpyAsync(s.previous,s.winners,nb*4,cudaMemcpyDeviceToDevice,s.stream));}
 check(cudaMemcpyAsync(s.host_conf.data(),s.d_conf,size_t(nb)*s.proposals*4,cudaMemcpyDeviceToHost,s.stream));}
 int error=0;check(cudaMemcpyAsync(&error,s.error,4,cudaMemcpyDeviceToHost,s.stream));check(cudaStreamSynchronize(s.stream));if(error)throw std::runtime_error("DSpark device contract failure "+std::to_string(error));for(int i=0;i<nb*s.proposals;i++)if(!std::isfinite(s.host_conf[i]))throw std::runtime_error("DSpark nonfinite learned confidence");if(nb)std::copy_n(s.host_winners.data(),nb*s.proposals,drafts);s.completed_rows=rows;
}
}
