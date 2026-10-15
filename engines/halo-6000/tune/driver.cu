// halo-autotune resident driver. Loads the model once, computes the reference
// (engine run_token, current default lanes) logits for G teacher-forced steps,
// then for every job line dlopens a compiled schedule variant, guards it
// bit-for-bit against the reference and times warm s1 decode.
//
// tune-driver --model GGUF --cache-dir DIR --teacher tokens.i32 --jobs JOBS.jsonl --out RESULTS.jsonl
//   job line: {"id":"...","so":"/abs/variant.so","grids":[...],"profile":false,"time_on_fail":false}
#include "decode.hpp"
#include "schedule.hpp"
#include "batch.hpp"
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
using namespace halo6000;
using json=nlohmann::json;
static void die(const std::string&m){fprintf(stderr,"tune-driver: %s\n",m.c_str());exit(2);}
static void check(const Status&s){if(!s.ok())die(s.error().message);}
static void cu(cudaError_t e,const char*op){if(e!=cudaSuccess)die(std::string(op)+": "+cudaGetErrorString(e));}

// DeviceModel::reset memsets on the legacy stream; our work runs on a
// non-blocking stream, so wait for the reset before any step.
static void reset_sync(DeviceModel&m){check(m.reset());cu(cudaDeviceSynchronize(),"reset sync");}
struct Bench {const DeviceModel*m;DecodeWorkspace w;std::vector<int32_t>tokens;int guard_steps,warm,steps,reps,ref_lanes,ref_grid;cudaStream_t stream;};

// Timed s1 decode: reset, warm steps, then `steps` timed steps; returns tok/s per rep.
template<class Step>static std::vector<double>time_runs(DeviceModel&m,Bench&b,Step step){
 std::vector<double>rates;cudaEvent_t s,e;cu(cudaEventCreate(&s),"event");cu(cudaEventCreate(&e),"event");
 for(int r=0;r<b.reps;r++){
  reset_sync(m);
  for(int p=0;p<b.warm;p++)step(p);
  cu(cudaStreamSynchronize(b.stream),"warm sync");
  cu(cudaEventRecord(s,b.stream),"record");
  for(int p=b.warm;p<b.warm+b.steps;p++)step(p);
  cu(cudaEventRecord(e,b.stream),"record");cu(cudaEventSynchronize(e),"timing sync");
  float ms=0;cu(cudaEventElapsedTime(&ms,s,e),"elapsed");rates.push_back(b.steps*1000.0/ms);
 }
 cudaEventDestroy(s);cudaEventDestroy(e);return rates;
}
struct StateSnap{std::vector<uint8_t>gdn,conv,key,value,pos;};
static StateSnap snap_state(const DeviceModel&m){
 StateSnap s;const auto&st=m.state;const size_t kv=size_t(AttentionLayers)*st.slots*st.context*KVHeads*HeadDim*2;
 auto get=[](std::vector<uint8_t>&v,const void*p,size_t n){v.resize(n);cu(cudaMemcpy(v.data(),p,n,cudaMemcpyDeviceToHost),"state snapshot");};
 get(s.gdn,st.gdn,size_t(st.slots)*GdnFloatsPerSlot*4);get(s.conv,st.conv,size_t(st.slots)*ConvFloatsPerSlot*4);get(s.key,st.key,kv);get(s.value,st.value,kv);get(s.pos,st.positions,size_t(st.slots)*4);return s;
}
static json diff_state(const StateSnap&a,const StateSnap&b){
 auto d=[](const std::vector<uint8_t>&x,const std::vector<uint8_t>&y){return x.size()==y.size()&&!memcmp(x.data(),y.data(),x.size());};
 json r{{"gdn",d(a.gdn,b.gdn)},{"conv",d(a.conv,b.conv)},{"key",d(a.key,b.key)},{"value",d(a.value,b.value)},{"positions",d(a.pos,b.pos)}};
 auto first=[](const std::vector<uint8_t>&x,const std::vector<uint8_t>&y,int parts){if(x.size()!=y.size())return -2;const size_t c=x.size()/parts;for(int i=0;i<parts;i++)if(memcmp(x.data()+i*c,y.data()+i*c,c))return i;return -1;};
 r["first_gdn_layer"]=first(a.gdn,b.gdn,LinearLayers);r["first_conv_layer"]=first(a.conv,b.conv,LinearLayers);r["first_kv_layer"]=first(a.key,b.key,AttentionLayers);
 r["exact"]=r["gdn"].get<bool>()&&r["conv"].get<bool>()&&r["key"].get<bool>()&&r["value"].get<bool>()&&r["positions"].get<bool>();return r;
}
static std::vector<std::pair<std::string,std::vector<uint8_t>>>snap_work(const DecodeWorkspace&w){
 std::vector<std::pair<std::string,std::vector<uint8_t>>>v;auto get=[&](const char*n,const void*p,size_t bytes){std::vector<uint8_t>b(bytes);cu(cudaMemcpy(b.data(),p,bytes,cudaMemcpyDeviceToHost),"work snapshot");v.push_back({n,std::move(b)});};
 get("hidden",w.hidden,Hidden*4);get("residual",w.residual,Hidden*4);get("normalized",w.normalized,Hidden*4);get("qkv",w.qkv,ConvChannels*4);get("z",w.z,Inner*4);
 get("alpha",w.alpha,ValueHeads*4);get("beta",w.beta,ValueHeads*4);get("mix",w.mix,Inner*4);get("projected",w.projected,Hidden*4);
 get("hidden_qs",w.hidden_q.qs,Hidden);get("hidden_ds",w.hidden_q.scale,Hidden/32*4);get("inner_qs",w.inner_q.qs,Inner);get("inner_ds",w.inner_q.scale,Inner/32*4);
 get("expert_qs",w.expert_q.qs,9*ExpertHidden);get("expert_ds",w.expert_q.scale,9*ExpertHidden/32*4);
 get("router_logits",w.router_logits,Experts*4);get("router_probs",w.router_probs,Experts*4);get("route_ids",w.route_ids,TopExperts*4);get("route_weights",w.route_weights,TopExperts*4);get("shared_gain",w.shared_gain,4);
 get("expert_activation",w.expert_activation,9*ExpertHidden*4);get("expert_gate",w.expert_gate,9*ExpertHidden*4);get("expert_up",w.expert_up,9*ExpertHidden*4);get("expert_down",w.expert_down,9*Hidden*4);
 get("gdn_prepared",w.gdn_prepared,ConvChannels*4);get("decay",w.decay,ValueHeads*4);get("effective_beta",w.effective_beta,ValueHeads*4);get("core",w.core,Inner*4);
 get("key",w.key,KVHeads*HeadDim*4);get("value",w.value,KVHeads*HeadDim*4);get("query",w.query,QueryHeads*HeadDim*4);
 return v;
}
static double median(std::vector<double>v){std::sort(v.begin(),v.end());return v.empty()?0:v[v.size()/2];}

// ---- batch mode (--rows R): guard and time scheduled batch tokens against run_batch_token.
extern "C" {
struct HaloTuneBatchApi {
 int version;const char*config_id;
 void*(*create)(const DeviceModel*model,BatchWorkspace*workspace,const int*grids,int ngrids,char*err,int errlen);
 int(*step)(void*ctx,cudaStream_t stream,char*err,int errlen);
 int(*profile)(void*ctx,cudaStream_t stream,uint64_t*stamps,int capacity,char*err,int errlen);
 int(*describe)(void*ctx,char*buf,int len);
 void(*destroy)(void*ctx);
};
}
static int main_batch(DeviceModel&m,const std::vector<int32_t>&chunks,int rows,const std::string&jobs,FILE*o,int guard_steps,int warm,int steps,int reps,bool ref_time,double load_s){
 auto made=BatchWorkspace::create(m,rows);if(!made.ok())die(made.error().message);auto bw=std::move(made.value());
 cudaStream_t st;cu(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");
 auto tok=[&](int r,int p){return chunks[(r%16)*512+(p+7*(r/16))%512];};
 auto teach=[&](int p){std::vector<BatchInput>in(rows);for(int r=0;r<rows;r++)in[r]={r,tok(r,p)};check(bw->configure(in,st));};
 auto feedback=[&](){std::vector<BatchInput>in(rows);for(int r=0;r<rows;r++)in[r]={r,-1};check(bw->configure(in,st));};
 const size_t nlog=size_t(rows)*Vocab;
 auto logits=[&](std::vector<float>&v){v.resize(nlog);cu(cudaMemcpyAsync(v.data(),bw->view.base.logits,nlog*4,cudaMemcpyDeviceToHost,st),"batch logits");cu(cudaStreamSynchronize(st),"batch logits sync");};
 auto tokens=[&](){std::vector<int32_t>t(rows);cu(cudaMemcpy(t.data(),bw->view.base.token,rows*4,cudaMemcpyDeviceToHost),"batch tokens");return t;};
 // Timing: warm steps teacher-forced, then `steps` greedy feedback replays.
 auto time_batch=[&](auto run)->std::vector<double>{
  std::vector<double>rates;cudaEvent_t a,b;cu(cudaEventCreate(&a),"ev");cu(cudaEventCreate(&b),"ev");
  for(int r=0;r<reps;r++){
   reset_sync(m);for(int p=0;p<warm;p++){teach(p);if(!run())return {};cu(cudaStreamSynchronize(st),"warm");}
   feedback();cu(cudaStreamSynchronize(st),"feedback cfg");
   cu(cudaEventRecord(a,st),"rec");for(int p=0;p<steps;p++)if(!run())return {};cu(cudaEventRecord(b,st),"rec");cu(cudaEventSynchronize(b),"sync");
   float ms=0;cu(cudaEventElapsedTime(&ms,a,b),"elapsed");rates.push_back(double(rows)*steps*1000.0/ms);
  }
  cudaEventDestroy(a);cudaEventDestroy(b);return rates;
 };
 reset_sync(m);std::vector<std::vector<float>>ref(guard_steps);std::vector<std::vector<int32_t>>ref_tok(guard_steps);
 for(int p=0;p<guard_steps;p++){teach(p);check(run_batch_token(m,*bw,0,st));check(bw->finish(st));logits(ref[p]);ref_tok[p]=tokens();}
 const StateSnap ref_state=snap_state(m);
 {json r{{"kind","reference"},{"rows",rows},{"load_seconds",load_s},{"guard_steps",guard_steps}};
  if(ref_time){auto rates=time_batch([&](){return run_batch_token(m,*bw,0,st).ok();});r["tok_s"]=median(rates);r["tok_s_reps"]=rates;}
  fprintf(o,"%s\n",r.dump().c_str());fflush(o);fprintf(stderr,"%s\n",r.dump().c_str());}
 std::ifstream jf(jobs);std::string line;void*lib=nullptr;std::string libpath;const HaloTuneBatchApi*api=nullptr;char err[512];std::vector<float>got;
 while(std::getline(jf,line)){
  if(line.empty())continue;json job=json::parse(line);json r{{"id",job.value("id","")},{"so",job["so"]},{"rows",rows}};const std::string so=job["so"];
  if(so!=libpath){if(lib)dlclose(lib);lib=dlopen(so.c_str(),RTLD_NOW|RTLD_LOCAL);libpath=so;api=nullptr;
   if(!lib){r["error"]=std::string("dlopen: ")+dlerror();fprintf(o,"%s\n",r.dump().c_str());fflush(o);libpath.clear();continue;}
   auto fn=reinterpret_cast<const HaloTuneBatchApi*(*)()>(dlsym(lib,"halo_tune_batch_api"));if(fn)api=fn();}
  if(!api){r["error"]="missing halo_tune_batch_api";fprintf(o,"%s\n",r.dump().c_str());fflush(o);continue;}
  r["config"]=api->config_id;std::vector<int>grids=job.value("grids",std::vector<int>{});
  teach(0);cu(cudaStreamSynchronize(st),"cfg");
  void*ctx=api->create(&m,bw.get(),grids.data(),int(grids.size()),err,sizeof err);
  if(!ctx){r["error"]=std::string("create: ")+err;fprintf(o,"%s\n",r.dump().c_str());fflush(o);fprintf(stderr,"%s\n",r.dump().c_str());continue;}
  {static char desc[1<<20];if(!api->describe(ctx,desc,sizeof desc))r["describe"]=json::parse(desc);}
  reset_sync(m);size_t bad=0;double maxabs=0;int bad_tokens=0,first_bad=-1;bool ok=true;
  for(int p=0;p<guard_steps&&ok;p++){
   teach(p);if(api->step(ctx,st,err,sizeof err)){r["error"]=std::string("step: ")+err;ok=false;break;}
   auto fin=bw->finish(st);if(!fin.ok()){r["error"]="finish: "+fin.error().message;ok=false;break;}
   logits(got);size_t badp=0;for(size_t i=0;i<nlog;i++)if(memcmp(&got[i],&ref[p][i],4)){badp++;maxabs=std::max(maxabs,double(std::fabs(got[i]-ref[p][i])));}
   auto t=tokens();for(int q=0;q<rows;q++)bad_tokens+=t[q]!=ref_tok[p][q];bad+=badp;if(badp&&first_bad<0)first_bad=p;
  }
  json state=ok?diff_state(ref_state,snap_state(m)):json{{"exact",false}};
  const bool exact=ok&&!bad&&!bad_tokens&&state["exact"].get<bool>();
  r["guard"]={{"exact",exact},{"state",state},{"bad_logits",bad},{"maxabs",maxabs},{"bad_tokens",bad_tokens},{"first_bad_step",first_bad}};
  if(ok&&exact&&job.value("time",true)){bool good=true;auto rates=time_batch([&](){if(good&&api->step(ctx,st,err,sizeof err))good=false;return good;});if(good&&!rates.empty()){r["tok_s"]=median(rates);r["tok_s_reps"]=rates;}else r["error"]=std::string("timing: ")+err;}
  const int preps=job.contains("profile")?(job["profile"].is_boolean()?(job["profile"].get<bool>()?1:0):job["profile"].get<int>()):0;
  if(ok&&exact&&preps>0){
   reset_sync(m);for(int p=0;p<warm;p++){teach(p);api->step(ctx,st,err,sizeof err);cu(cudaStreamSynchronize(st),"warm");}
   const int n=int(r["describe"]["epochs"]);std::vector<double>us(n,0.0);double total=0;std::vector<uint64_t>sv(8192);bool pok=true;
   feedback();cu(cudaStreamSynchronize(st),"fb");
   for(int q=0;q<preps;q++){if(api->profile(ctx,st,sv.data(),int(sv.size()),err,sizeof err)){r["profile_error"]=err;pok=false;break;}for(int i=0;i<n;i++)us[i]+=(sv[i+1]-sv[i])/1000.0/preps;total+=(sv[n]-sv[0])/1000.0/preps;}
   if(pok){r["epoch_us"]=us;r["profile_token_us"]=total;}
  }
  api->destroy(ctx);const cudaError_t sticky=cudaGetLastError();if(sticky!=cudaSuccess)r["cuda_error"]=cudaGetErrorString(sticky);
  fprintf(o,"%s\n",r.dump().c_str());fflush(o);
  json brief{{"id",r["id"]},{"exact",exact},{"tok_s",r.value("tok_s",0.0)}};if(r.contains("error"))brief["error"]=r["error"];fprintf(stderr,"%s\n",brief.dump().c_str());
  if(sticky!=cudaSuccess&&std::string(cudaGetErrorString(sticky)).find("illegal")!=std::string::npos)die("sticky CUDA error; restart driver");
 }
 if(lib)dlclose(lib);fclose(o);return 0;
}

int main(int argc,char**argv){
 // Lazy module loading places a re-dlopened variant's kernels differently from its
 // first load (measured: one schedule 291.8 first load vs 305.0 reloaded). Eager
 // loading makes every candidate's placement independent of load order and matches
 // a freshly started engine.
 setenv("CUDA_MODULE_LOADING","EAGER",1);
 std::string model,cache,teacher,jobs,out;int guard_steps=16,warm=40,steps=128,reps=3,ref_lanes=32,ref_grid=188,context=1024,rows=0;bool ref_time=true;
 for(int i=1;i+1<argc;i+=2){
  std::string k=argv[i],v=argv[i+1];
  if(k=="--model")model=v;else if(k=="--cache-dir")cache=v;else if(k=="--teacher")teacher=v;else if(k=="--jobs")jobs=v;else if(k=="--out")out=v;
  else if(k=="--guard-steps")guard_steps=std::stoi(v);else if(k=="--warm")warm=std::stoi(v);else if(k=="--steps")steps=std::stoi(v);else if(k=="--reps")reps=std::stoi(v);
  else if(k=="--ref-lanes")ref_lanes=std::stoi(v);else if(k=="--ref-grid")ref_grid=std::stoi(v);else if(k=="--ref-time")ref_time=v!="0";else if(k=="--rows")rows=std::stoi(v);else die("unknown flag "+k);
 }
 if(model.empty()||cache.empty()||teacher.empty()||jobs.empty()||out.empty())die("--model --cache-dir --teacher --jobs --out required");
 auto t0=std::chrono::steady_clock::now();
 auto image=load_cached(model,cache,false,true);if(!image.ok())die(image.error().message);
 auto made=DeviceModel::create(image.value(),std::max(1,rows),context);if(!made.ok())die(made.error().message);auto m=std::move(made.value());
 check(m->prepare_matmul(false));cu(cudaDeviceSynchronize(),"prepare matmul");
 auto ws=Workspace::create(context,false);if(!ws.ok())die(ws.error().message);auto w=std::move(ws.value());
 const double load_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
 std::vector<int32_t>tokens(context);{std::ifstream f(teacher,std::ios::binary);f.read(reinterpret_cast<char*>(tokens.data()),std::min(context,512)*4);if(!f)die("read teacher tokens");}
 for(int p=512;p<context;p++)tokens[p]=tokens[p%512];
 int32_t*dteacher=nullptr;cu(cudaMalloc(&dteacher,context*4),"teacher alloc");cu(cudaMemcpy(dteacher,tokens.data(),context*4,cudaMemcpyHostToDevice),"teacher upload");
 if(rows>0){std::vector<int32_t>chunks(16*512);std::ifstream f(teacher,std::ios::binary);f.read(reinterpret_cast<char*>(chunks.data()),chunks.size()*4);if(!f)die("read teacher chunks");
  FILE*o=fopen(out.c_str(),"a");if(!o)die("open out");return main_batch(*m,chunks,rows,jobs,o,guard_steps,warm,steps,reps,ref_time,load_s);}
 Bench b{m.get(),w->view,tokens,guard_steps,warm,steps,reps,ref_lanes,ref_grid,nullptr};cu(cudaStreamCreateWithFlags(&b.stream,cudaStreamNonBlocking),"stream");
 if(warm+steps>context||guard_steps>context)die("steps exceed context");
 FILE*o=fopen(out.c_str(),"a");if(!o)die("open out");
 // Reference logits from the engine's own run_token.
 m->matvec_lanes=ref_lanes;reset_sync(*m);
 std::vector<std::vector<float>>ref(guard_steps,std::vector<float>(Vocab));std::vector<int32_t>ref_tok(guard_steps);
 for(int p=0;p<guard_steps;p++){check(run_token(*m,w->view,tokens[p],p,0,ref_grid,b.stream));cu(cudaMemcpyAsync(ref[p].data(),w->view.logits,Vocab*4,cudaMemcpyDeviceToHost,b.stream),"ref logits");cu(cudaMemcpyAsync(&ref_tok[p],w->view.token,4,cudaMemcpyDeviceToHost,b.stream),"ref token");cu(cudaStreamSynchronize(b.stream),"ref sync");}
 const StateSnap ref_state=snap_state(*m);
 {json r{{"kind","reference"},{"lanes",ref_lanes},{"grid",ref_grid},{"load_seconds",load_s},{"guard_steps",guard_steps},{"argmax",ref_tok}};
  if(ref_time){auto rates=time_runs(*m,b,[&](int p){check(run_token(*m,w->view,tokens[p],p,0,ref_grid,b.stream));});r["tok_s"]=median(rates);r["tok_s_reps"]=rates;}
  fprintf(o,"%s\n",r.dump().c_str());fflush(o);fprintf(stderr,"%s\n",r.dump().c_str());}
 if(getenv("HALO_BISECT_A")){
  const char*paths[2]={getenv("HALO_BISECT_A"),getenv("HALO_BISECT_B")};int steps_b=getenv("HALO_BISECT_STEPS")?atoi(getenv("HALO_BISECT_STEPS")):1;
  for(int n:std::vector<int>{2,3,4,5,6,7,8,9,10,11,12,13,14,20,26,40,80,200,400,600}){
   std::vector<std::vector<std::pair<std::string,std::vector<uint8_t>>>>works;std::vector<StateSnap>states;
   for(int v=0;v<2;v++){
    setenv("HALO_TUNE_EPOCHS",std::to_string(n).c_str(),1);void*lib=dlopen(paths[v],RTLD_NOW|RTLD_LOCAL);if(!lib)die(dlerror());
    auto api=reinterpret_cast<const HaloTuneApi*(*)()>(dlsym(lib,"halo_tune_api"))();char e[512];void*ctx=api->create(m.get(),&w->view,nullptr,0,dteacher,0,e,sizeof e);if(!ctx)die(e);
    reset_sync(*m);cu(cudaMemset(w->view.qkv,0,ConvChannels*4),"clear");
    for(int s2=0;s2<steps_b;s2++){if(api->step(ctx,b.stream,e,sizeof e))die(e);}cu(cudaStreamSynchronize(b.stream),"bisect sync");
    works.push_back(snap_work(w->view));states.push_back(snap_state(*m));api->destroy(ctx);dlclose(lib);
   }
   json r{{"epochs",n},{"state",diff_state(states[0],states[1])}};std::vector<std::string>bad;
   for(size_t i=0;i<works[0].size();i++)if(memcmp(works[0][i].second.data(),works[1][i].second.data(),works[0][i].second.size()))bad.push_back(works[0][i].first);
   r["work_diff"]=bad;fprintf(o,"%s\n",r.dump().c_str());fflush(o);
  }
  unsetenv("HALO_TUNE_EPOCHS");fclose(o);return 0;
 }
 std::ifstream jf(jobs);std::string line;void*lib=nullptr;std::string libpath;const HaloTuneApi*api=nullptr;char err[512];
 std::vector<float>got(Vocab);
 while(std::getline(jf,line)){
  if(line.empty())continue;json job=json::parse(line);json r{{"id",job.value("id","")},{"so",job["so"]}};
  const std::string so=job["so"];
  if(so!=libpath){if(lib)dlclose(lib);lib=dlopen(so.c_str(),RTLD_NOW|RTLD_LOCAL);libpath=so;api=nullptr;
   if(!lib){r["error"]=std::string("dlopen: ")+dlerror();fprintf(o,"%s\n",r.dump().c_str());fflush(o);libpath.clear();continue;}
   auto fn=reinterpret_cast<const HaloTuneApi*(*)()>(dlsym(lib,"halo_tune_api"));if(fn)api=fn();}
  if(!api||api->version<1){r["error"]="missing halo_tune_api";fprintf(o,"%s\n",r.dump().c_str());fflush(o);continue;}
  r["config"]=api->config_id;
  std::vector<int>grids=job.value("grids",std::vector<int>{});
  err[0]=0;void*ctx=api->create(m.get(),&w->view,grids.data(),int(grids.size()),dteacher,0,err,sizeof err);
  if(!ctx){r["error"]=std::string("create: ")+err;fprintf(o,"%s\n",r.dump().c_str());fflush(o);fprintf(stderr,"%s\n",r.dump().c_str());continue;}
  {static char desc[1<<20];if(!api->describe(ctx,desc,sizeof desc))r["describe"]=json::parse(desc);}
  // Guard: every logit of every guard step bit-identical to the reference.
  reset_sync(*m);size_t bad=0;double maxabs=0;int bad_tokens=0,first_bad=-1;bool launch_ok=true;
  for(int p=0;p<guard_steps&&launch_ok;p++){
   if(api->step(ctx,b.stream,err,sizeof err)){r["error"]=std::string("step: ")+err;launch_ok=false;break;}
   int32_t tok=0;cu(cudaMemcpyAsync(got.data(),w->view.logits,Vocab*4,cudaMemcpyDeviceToHost,b.stream),"logits");cu(cudaMemcpyAsync(&tok,w->view.token,4,cudaMemcpyDeviceToHost,b.stream),"token");
   const cudaError_t e=cudaStreamSynchronize(b.stream);if(e!=cudaSuccess){r["error"]=std::string("guard sync: ")+cudaGetErrorString(e);launch_ok=false;break;}
   size_t badp=0;for(int i=0;i<Vocab;i++)if(memcmp(&got[i],&ref[p][i],4)){badp++;maxabs=std::max(maxabs,double(std::fabs(got[i]-ref[p][i])));}
   bad+=badp;if(badp&&first_bad<0)first_bad=p;bad_tokens+=tok!=ref_tok[p];
  }
  json state=launch_ok?diff_state(ref_state,snap_state(*m)):json{{"exact",false}};
  const bool exact=launch_ok&&bad==0&&bad_tokens==0&&state["exact"].get<bool>();
  r["guard"]={{"exact",exact},{"state",state},{"bad_logits",bad},{"maxabs",maxabs},{"bad_tokens",bad_tokens},{"first_bad_step",first_bad}};
  if(launch_ok&&(exact||job.value("time_on_fail",false))){
   bool ok=true;
   if(job.value("time",true)){
    auto rates=time_runs(*m,b,[&](int){if(ok&&api->step(ctx,b.stream,err,sizeof err))ok=false;});
    if(ok){r["tok_s"]=median(rates);r["tok_s_reps"]=rates;}else r["error"]=std::string("timing step: ")+err;
   }
   const int preps=job.contains("profile")?(job["profile"].is_boolean()?(job["profile"].get<bool>()?1:0):job["profile"].get<int>()):0;
   if(ok&&preps>0){
    reset_sync(*m);for(int p=0;p<warm;p++)api->step(ctx,b.stream,err,sizeof err);
    const int n=int(r["describe"]["epochs"]);std::vector<double>us(n,0.0);double total=0;bool pok=true;
    std::vector<uint64_t>st(8192);
    for(int q=0;q<preps&&pok;q++){
     if(api->profile(ctx,b.stream,st.data(),int(st.size()),err,sizeof err)){r["profile_error"]=err;pok=false;break;}
     for(int i=0;i<n;i++)us[i]+=(st[i+1]-st[i])/1000.0/preps;total+=(st[n]-st[0])/1000.0/preps;
    }
    if(pok){r["epoch_us"]=us;r["profile_token_us"]=total;}
   }
  }
  api->destroy(ctx);
  const cudaError_t sticky=cudaGetLastError();if(sticky!=cudaSuccess)r["cuda_error"]=cudaGetErrorString(sticky);
  fprintf(o,"%s\n",r.dump().c_str());fflush(o);
  json brief{{"id",r["id"]},{"exact",exact},{"tok_s",r.value("tok_s",0.0)}};if(r.contains("error"))brief["error"]=r["error"];fprintf(stderr,"%s\n",brief.dump().c_str());
  if(sticky!=cudaSuccess&&std::string(cudaGetErrorString(sticky)).find("illegal")!=std::string::npos)die("sticky CUDA error; restart driver");
 }
 if(lib)dlclose(lib);fclose(o);return 0;
}
