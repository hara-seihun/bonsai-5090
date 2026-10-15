#include "kernels.cuh"
#include "model.h"
#include "experiments/binary-bitserial/bitserial.cuh"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
using namespace h5;
extern "C" { void* h5_create(const char*,int,int,int); void h5_reset(void*,int); void h5_keep(void*,int,int); int h5_step(void*,int,const int*,const int*,const int*,int*); }
extern "C" void real_prep(const PrepArgs&,int,cudaStream_t) asm("__real__ZN2h511launch_prepERKNS_8PrepArgsEiP11CUstream_st");
extern "C" cudaError_t real_produce(bitserial::Route,const float*,int,int,bitserial::Carrier,cudaStream_t) asm("__real__ZN9bitserial7produceENS_5RouteEPKfiiNS_7CarrierEP11CUstream_st");
static int layer=0, regime=0, call=0, position=0, packet_rows=0; static bool enabled=false;
static std::vector<int8_t> q; static std::vector<float> sx; static std::vector<int32_t> dc; static std::vector<bitserial::InputStatus> status; static std::ofstream output;
static bool selected(){return layer==0||layer==30||layer==61;}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> static void copy(std::vector<T>& h,const T* p,size_t n,cudaStream_t s){if(!p)throw std::runtime_error("missing carrier field");h.resize(n);ck(cudaMemcpyAsync(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s));}
static void bytes(const void* p,size_t n){output.write((const char*)p,n);if(!output)throw std::runtime_error("capture write failed");}
extern "C" cudaError_t wrapped_produce(bitserial::Route,const float*,int,int,bitserial::Carrier,cudaStream_t) asm("__wrap__ZN9bitserial7produceENS_5RouteEPKfiiNS_7CarrierEP11CUstream_st");
extern "C" cudaError_t wrapped_produce(bitserial::Route route,const float* h,int k,int rows,bitserial::Carrier out,cudaStream_t s){auto e=real_produce(route,h,k,rows,out,s);if(e!=cudaSuccess)return e;if(enabled&&selected()&&k==D){if(route!=bitserial::Route::Dp4a01)throw std::runtime_error("not original A8 producer");copy(q,out.q,size_t(rows)*D,s);copy(sx,out.scales,size_t(rows)*D/128,s);copy(dc,out.dc,size_t(rows)*D/128,s);copy(status,out.status,size_t(rows)*D/128,s);ck(cudaStreamSynchronize(s));packet_rows=rows;}return e;}
extern "C" void wrapped_prep(const PrepArgs&,int,cudaStream_t) asm("__wrap__ZN2h511launch_prepERKNS_8PrepArgsEiP11CUstream_st");
extern "C" void wrapped_prep(const PrepArgs& a,int rows,cudaStream_t s){bool nonlinear=(a.flags&(P_SILU_MUL|P_SILU_IL))!=0;if(enabled&&nonlinear){if(a.n!=FF||!(a.flags&P_SILU_MUL)||(a.flags&P_SILU_IL)||a.rowmap||a.early||a.sig||a.done_sig)throw std::runtime_error("unsupported FFN observer route");if(selected()){if(packet_rows!=rows)throw std::runtime_error("missing postnorm A8 packet");int head[]={0x52464231,regime,rows,call,layer,D,position};bytes(head,sizeof head);for(int r=0;r<rows;r++){bytes(q.data()+r*D,D);bytes(sx.data()+r*(D/128),D/128*4);bytes(dc.data()+r*(D/128),D/128*4);bytes(status.data()+r*(D/128),D/128*4);}}++layer;packet_rows=0;}real_prep(a,rows,s);}
static std::string text(const std::string& p){std::ifstream f(p);if(!f)throw std::runtime_error("input unavailable: "+p);return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char** argv){try{if(argc!=5)throw std::runtime_error("usage: row-fibers MODEL WIKI PROMPTS_DIR OUTPUT");
 const char* names[]={"H5_WEIGHTS","H5_ACT","H5_NOGRAPH","H5_PDL","H5_KV","H5_STATE","H5_A8_GU","H5_A8_FFN","H5_A8_PROJECT","H5_TAIL_PREP","H5_ORDERED_PREP"};const char* vals[]={"packed1","a8","1","0","f16","f16","off","off","fp4-digits-ragged","0","off"};for(int i=0;i<11;i++)setenv(names[i],vals[i],1);
 llama_log_set([](ggml_log_level,const char*,void*){},nullptr);llama_backend_init();auto vp=llama_model_default_params();vp.vocab_only=true;vp.n_gpu_layers=0;auto* model=llama_model_load_from_file(argv[1],vp);if(!model)throw std::runtime_error("vocab failed");auto* vocab=llama_model_get_vocab(model);
 auto tokenize=[&](const std::string& t){std::vector<int> ids(t.size()+16);int n=llama_tokenize(vocab,t.data(),t.size(),ids.data(),ids.size(),true,true);if(n<32)throw std::runtime_error("prompt below 32 tokens");ids.resize(n);return ids;};auto wiki=tokenize(text(argv[2]));if(wiki.size()<128)throw std::runtime_error("short wiki");std::vector<std::vector<int>> prompts;for(int i=0;i<32;i++)prompts.push_back(tokenize(text(std::string(argv[3])+"/"+std::to_string(i)+".txt")));
 std::ofstream tokenfile(std::string(argv[4])+".tokens",std::ios::binary);for(int i=-1;i<32;i++){const auto& ids=i<0?wiki:prompts[i];int n=std::min(int(ids.size()),i<0?128:64);int head[]={i,n};tokenfile.write((char*)head,sizeof head);tokenfile.write((char*)ids.data(),n*4);}if(!tokenfile)throw std::runtime_error("token write failed");tokenfile.close();
 output.open(argv[4],std::ios::binary);if(!output)throw std::runtime_error("output unavailable");auto* engine=h5_create(argv[1],32,512,32);if(!engine)throw std::runtime_error("engine failed");
 auto run=[&](int rr,int mode){regime=mode;for(int s=0;s<32;s++){h5_reset(engine,s);h5_keep(engine,s,0);}std::vector<int> slots(rr),tokens(rr),pos(rr),out(rr);int nc=mode==0?128/rr:64;if(mode==1)for(int r=0;r<rr;r++)nc=std::min(nc,int(prompts[r].size()));for(call=0;call<nc;call++){layer=0;packet_rows=0;position=mode==0?call*rr:call;for(int r=0;r<rr;r++){slots[r]=mode==0?0:r;pos[r]=mode==0?call*rr+r:call;tokens[r]=mode==0?wiki[pos[r]]:prompts[mode==2?0:r][call];}enabled=true;int e=h5_step(engine,rr,slots.data(),tokens.data(),pos.data(),out.data());enabled=false;if(e||layer!=NLAYER)throw std::runtime_error("engine step or FFN count failed");}fprintf(stderr,"row_fibers regime=%d rows=%d calls=%d\n",mode,rr,nc);};for(int rr:{16,32})for(int mode:{0,1,2})run(rr,mode);output.close();llama_model_free(model);llama_backend_free();return 0;
 }catch(const std::exception& e){fprintf(stderr,"row_fibers: %s\n",e.what());return 3;}}
