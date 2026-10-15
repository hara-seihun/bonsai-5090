#include "kernels.cuh"
#include "model.h"
#include "../experiments/binary-bitserial/bitserial.cuh"
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
using namespace h5;
extern "C" {
void* h5_create(const char*,int,int,int);
void h5_reset(void*,int);
void h5_keep(void*,int,int);
int h5_step_logits(void*,int,const int*,const int*,const int*,int*,float*);
}
extern "C" void real_prep(const PrepArgs&,int,cudaStream_t) asm("__real__ZN2h511launch_prepERKNS_8PrepArgsEiP11CUstream_st");
static bool enabled=false, save_capture=false;
static int layer=0, position=0, quant_levels=0, prefix_boundary=0;
static float drop_fraction=0;
static const float* normalized=nullptr;
static std::ofstream captures;
static void ck(cudaError_t code) { if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
static void bytes(const void* src,size_t size) { captures.write((const char*)src,size); if(!captures) throw std::runtime_error("capture write failed"); }
template<class T> static void device_packet(const T* ptr,size_t n,cudaStream_t st) {
  std::vector<T> host(n); ck(cudaMemcpyAsync(host.data(),ptr,n*sizeof(T),cudaMemcpyDeviceToHost,st)); ck(cudaStreamSynchronize(st)); bytes(host.data(),n*sizeof(T));
}
__global__ void recode_hidden(const float* h,bitserial::Carrier out,int levels,float drop) {
  __shared__ float mag[128]; int t=threadIdx.x, i=blockIdx.x*128+t;
  float v=h[i]; mag[t]=fabsf(v); __syncthreads();
  for(int stride=64;stride;stride>>=1) { if(t<stride) mag[t]=fmaxf(mag[t],mag[t+stride]); __syncthreads(); }
  float m=mag[0];
  int half=(levels-1)/2;
  bool finite=isfinite(v); __shared__ int sums[128]; __shared__ int errors[128];
  errors[t]=finite?0:1; __syncthreads();
  for(int stride=64;stride;stride>>=1) { if(t<stride) errors[t]|=errors[t+stride]; __syncthreads(); }
  unsigned error=errors[0]?1u:(m>0 && (m<1e-30f || m>1e30f)?2u:0u);
  if(drop>0 && fabsf(v)<drop*m) v=0;
  int q=error||m==0?0:__float2int_rn(v*(float(half)/m));
  out.q[i]=int8_t(q); sums[t]=q; __syncthreads();
  for(int stride=64;stride;stride>>=1) { if(t<stride) sums[t]+=sums[t+stride]; __syncthreads(); }
  if(t==0) { out.scales[blockIdx.x]=m==0||error?0:m/half; out.dc[blockIdx.x]=sums[0]; out.status[blockIdx.x]=bitserial::InputStatus(error); if(error&&out.error) atomicMax((unsigned*)out.error,error); }
}
extern "C" cudaError_t real_produce(bitserial::Route,const float*,int,int,bitserial::Carrier,cudaStream_t) asm("__real__ZN9bitserial7produceENS_5RouteEPKfiiNS_7CarrierEP11CUstream_st");
extern "C" cudaError_t wrapped_produce(bitserial::Route,const float*,int,int,bitserial::Carrier,cudaStream_t) asm("__wrap__ZN9bitserial7produceENS_5RouteEPKfiiNS_7CarrierEP11CUstream_st");
extern "C" cudaError_t wrapped_produce(bitserial::Route route,const float* h,int k,int rows,bitserial::Carrier out,cudaStream_t st) {
  if(!enabled || k!=FF || (!quant_levels && !drop_fraction)) return real_produce(route,h,k,rows,out,st);
  if(rows!=1 || route!=bitserial::Route::Dp4a01 || !out.q || !out.scales || !out.dc || !out.status) return cudaErrorInvalidValue;
  recode_hidden<<<FF/128,128,0,st>>>(h,out,quant_levels?quant_levels:255,drop_fraction);
  return cudaGetLastError();
}
extern "C" void wrapped_prep(const PrepArgs& a,int rows,cudaStream_t st) asm("__wrap__ZN2h511launch_prepERKNS_8PrepArgsEiP11CUstream_st");
extern "C" void wrapped_prep(const PrepArgs& a,int rows,cudaStream_t st) {
  if(!enabled) { real_prep(a,rows,st); return; }
  if(rows!=1 || a.rowmap || a.early || a.sig || a.done_sig) throw std::runtime_error("observer requires one unflagged row");
  bool nonlinear=(a.flags&(P_SILU_MUL|P_SILU_IL))!=0;
  if(nonlinear) {
    if(layer>=NLAYER || a.n!=FF || !(a.flags&P_SILU_MUL) || (a.flags&P_SILU_IL) || !a.out_f32 || a.ldo!=FF || !normalized || a.ldx!=GU || a.x2!=a.x+FF)
      throw std::runtime_error("unsupported GU observer route");
    if(save_capture && position>=prefix_boundary-1 && (layer==0 || layer==30 || layer==61)) {
      int header[]={position,layer,rows,D,FF}; bytes(header,sizeof header);
      device_packet(normalized,D,st);
      device_packet(a.xq,D,st); device_packet(a.xs,D/128,st);
      device_packet(a.x,GU,st);
    }
  }
  real_prep(a,rows,st);
  if((a.flags&P_NORM) && a.n==D && a.out_f32 && a.ldo==D) normalized=a.out_f32;
  if(nonlinear) {
    if(save_capture && position>=prefix_boundary-1 && (layer==0 || layer==30 || layer==61)) device_packet(a.out_f32,FF,st);
    ++layer;
  }
}
static std::string text(const char* path) { std::ifstream f(path); if(!f) throw std::runtime_error("corpus unavailable"); return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()}; }
int main(int argc,char** argv) {
  try {
    if(argc!=7 && argc!=8) { fprintf(stderr,"usage: whole-map-capture MODEL CORPUS PREFIX SCORE PREFIX_OUT MODE(capture|levels:N|drop:F|baseline) [PRISM_REF]\n"); return 2; }
    int prefix=std::stoi(argv[3]), scored=std::stoi(argv[4]); if(prefix<1 || scored<1 || scored>256) throw std::runtime_error("invalid finite token count");
    prefix_boundary=prefix;
    std::string mode=argv[6]; save_capture=mode=="capture";
    if(mode.rfind("levels:",0)==0) { quant_levels=std::stoi(mode.substr(7)); if(quant_levels<3 || quant_levels>255 || quant_levels%2!=1) throw std::runtime_error("invalid odd level count"); }
    else if(mode.rfind("drop:",0)==0) { drop_fraction=std::stof(mode.substr(5)); if(!(drop_fraction>0 && drop_fraction<1)) throw std::runtime_error("invalid drop threshold"); }
    else if(mode!="capture" && mode!="baseline") throw std::runtime_error("unknown mode");
    setenv("H5_WEIGHTS","packed1",1); setenv("H5_ACT","a8",1); setenv("H5_NOGRAPH","1",1); setenv("H5_PDL","0",1); setenv("H5_KV","f16",1); setenv("H5_STATE","f16",1);
    unsetenv("H5_A8_GU"); unsetenv("H5_GU_FUSE"); unsetenv("H5_GU_SPLIT"); unsetenv("H5_GU_PLAIN_ROWS"); unsetenv("H5_TAIL_PREP"); unsetenv("H5_ORDERED_PREP");
    llama_log_set([](ggml_log_level,const char*,void*){},nullptr); llama_backend_init();
    auto vp=llama_model_default_params(); vp.vocab_only=true; vp.n_gpu_layers=0; auto* model=llama_model_load_from_file(argv[1],vp); if(!model) throw std::runtime_error("vocabulary load failed");
    const auto* vocab=llama_model_get_vocab(model); auto corpus=text(argv[2]); std::vector<int> ids(corpus.size()+16);
    int n=llama_tokenize(vocab,corpus.data(),corpus.size(),ids.data(),ids.size(),true,true); if(n<prefix+scored+1) throw std::runtime_error("corpus tokenization insufficient"); ids.resize(n);
    std::ifstream reference;
    if(argc==8) {
      reference.open(argv[7],std::ios::binary); int rh[3]{}; reference.read((char*)rh,sizeof rh);
      if(!reference || rh[1]!=prefix || rh[2]!=VOCAB || rh[0]<prefix+scored) throw std::runtime_error("reference header mismatch");
      std::vector<int> rt(rh[0]); reference.read((char*)rt.data(),rt.size()*sizeof(int));
      if(!reference || !std::equal(rt.begin(),rt.begin()+prefix+scored,ids.begin())) throw std::runtime_error("reference token mismatch");
    }
    auto* engine=h5_create(argv[1],1,512,1); if(!engine) throw std::runtime_error("engine load failed"); h5_reset(engine,0); h5_keep(engine,0,0);
    std::ofstream logits_file;
    if(argc==7) {
      logits_file.open(std::string(argv[5])+".logits",std::ios::binary); std::ofstream labels_file(std::string(argv[5])+".tokens",std::ios::binary);
      if(!logits_file || !labels_file) throw std::runtime_error("output unavailable");
      int header[]={scored,VOCAB,prefix}; logits_file.write((char*)header,sizeof header); labels_file.write((char*)ids.data()+prefix*sizeof(int),scored*sizeof(int));
    }
    if(save_capture) { captures.open(std::string(argv[5])+".capture",std::ios::binary); int ch[]={0x574d4631,scored,3,D,FF}; bytes(ch,sizeof ch); }
    std::vector<float> logits(VOCAB), ref(VOCAB); int slot=0,out=0, agree=0; double nll=0, rnll=0, kl=0; auto started=std::chrono::steady_clock::now();
    for(position=0;position<prefix+scored-1;++position) {
      layer=0; normalized=nullptr; enabled=true;
      h5_keep(engine,0,0);
      if(h5_step_logits(engine,1,&slot,&ids[position],&position,&out,logits.data())) throw std::runtime_error("engine step failed");
      if(layer!=NLAYER) throw std::runtime_error("missing FFN observations");
      if(position>=prefix-1) {
        if(argc==7) logits_file.write((char*)logits.data(),logits.size()*sizeof(float));
        float mx=*std::max_element(logits.begin(),logits.end()); double sum=0; for(float v:logits) sum+=std::exp(double(v-mx)); double lz=mx+std::log(sum); nll+=lz-logits[ids[position+1]];
        if(argc==8) {
          reference.read((char*)ref.data(),ref.size()*sizeof(float)); if(!reference) throw std::runtime_error("reference logits missing");
          double rm=*std::max_element(ref.begin(),ref.end()), rs=0; for(float v:ref) rs+=std::exp(double(v)-rm); double rz=rm+std::log(rs); rnll+=rz-ref[ids[position+1]];
          for(int j=0;j<VOCAB;++j) kl+=std::exp(double(ref[j])-rz)*(double(ref[j])-rz-double(logits[j])+lz);
          agree+=std::max_element(ref.begin(),ref.end())-ref.begin()==std::max_element(logits.begin(),logits.end())-logits.begin();
        }
      }
    }
    if(argc==7 && !logits_file) throw std::runtime_error("logits write failed");
    printf("{\"mode\":\"%s\",\"prefix\":%d,\"scored\":%d,\"ppl\":%.10g,\"ref_ppl\":%.10g,\"kl_ref\":%.10g,\"top1_agreement\":%d,\"seconds\":%.6f,\"capture_layers\":[0,30,61],\"observations_per_step\":%d}\n",mode.c_str(),prefix,scored,std::exp(nll/scored),std::exp(rnll/scored),kl/scored,agree,std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count(),layer);
    llama_model_free(model); llama_backend_free(); return 0;
  } catch(const std::exception& e) { fprintf(stderr,"whole-map: %s\n",e.what()); return 3; }
}
