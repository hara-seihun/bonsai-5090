#include "decode.hpp"
#include "debug.hpp"
#include "generate.hpp"
#include "intrinsic.hpp"
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
using namespace halo6000;
static bool report(const Status&s){if(s.ok())return true;fprintf(stderr,"halo-6000: %s\n",s.error().message.c_str());return false;}
static std::vector<float>read_device(const float*p,size_t n){std::vector<float>v(n);auto s=cuda_status(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost),"read guard output");if(!report(s))exit(2);return v;}
int main(int argc,char**argv){
 if(argc<2||!strcmp(argv[1],"--help")){puts("usage: halo-6000 load|layer-test|debug-layer|forward-test|bench|profile|dump-logits|serve --model GGUF [--cache-dir DIR] [--context N] [--steps N] [--grid N]\nNative CUDA inference; serve is serialized greedy, not native GPU batching.");return argc<2?2:0;}
 std::string command=argv[1],model,cache,tokens_file,output_dir,layer_output_dir,spec,host="127.0.0.1";
 int context=command=="serve"?4096:1024,slots=command=="serve"?8:1,port=8080,steps=command=="bench"?128:4,grid=188,warm=40,row_lanes=32;
 for(int i=2;i<argc;i++){
  const std::string key=argv[i];if(key=="--raw-content")continue;if(i+1==argc){fprintf(stderr,"missing value for %s\n",key.c_str());return 2;}const std::string value=argv[++i];
  if(key=="--spec")spec=value;else if(key=="--host")host=value;else if(key=="--thinking"){if(value!="off"){fprintf(stderr,"only thinking off is supported\n");return 2;}}else if(key=="--model")model=value;else if(key=="--cache-dir")cache=value;else if(key=="--tokens-file")tokens_file=value;else if(key=="--output-dir")output_dir=value;else if(key=="--layer-output-dir")layer_output_dir=value;else if(key=="--context"||key=="--context-per-slot"||key=="--slots"||key=="--port"||key=="--steps"||key=="--grid"||key=="--warm"||key=="--row-lanes"){char*end=nullptr;long n=strtol(value.c_str(),&end,10);if(!end||*end||n<1||n>131072){fprintf(stderr,"invalid integer %s\n",value.c_str());return 2;}if(key=="--context"||key=="--context-per-slot")context=n;else if(key=="--slots")slots=n;else if(key=="--port")port=n;else if(key=="--steps")steps=n;else if(key=="--warm")warm=n;else if(key=="--row-lanes")row_lanes=n;else grid=n;}
  else{fprintf(stderr,"unknown flag %s\n",key.c_str());return 2;}
 }
 if(command!="load"&&command!="layer-test"&&command!="debug-layer"&&command!="forward-test"&&command!="bench"&&command!="profile"&&command!="dump-logits"&&command!="serve"){fprintf(stderr,"halo-6000: command %s is not implemented\n",command.c_str());return 2;}
 if(model.empty()){fprintf(stderr,"--model GGUF is required\n");return 2;}
 if(cache.empty()){const char*configured=getenv("HALO6000_CACHE_DIR");if(configured&&*configured)cache=configured;else{auto dir=std::filesystem::path(model).parent_path();if(dir.empty())dir=".";if(dir.filename()=="gguf")dir=dir.parent_path();cache=(dir/"halo6000-cache").string();}}
 auto begin=std::chrono::steady_clock::now();auto loaded=load_cached(model,cache);if(!loaded.ok()){fprintf(stderr,"cache: %s\n",loaded.error().message.c_str());return 2;}
 auto created=DeviceModel::create(loaded.value(),slots,context);if(!created.ok()){fprintf(stderr,"upload: %s\n",created.error().message.c_str());return 2;}auto m=std::move(created.value());m->matvec_lanes=row_lanes;
 if(row_lanes!=8&&row_lanes!=16&&row_lanes!=32){fprintf(stderr,"--row-lanes requires8/16/32\n");return 2;}
 const double load_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
 if(command=="load"){printf("{\"weights_bytes\":%zu,\"state_bytes\":%zu,\"load_seconds\":%.6f,\"cache\":\"%s\",\"native_forward_available\":true}\n",m->weight_bytes,m->state_bytes,load_s,loaded.value().provenance.cache_path.c_str());return 0;}
 if(steps>context){fprintf(stderr,"steps exceed context\n");return 2;}auto made=Workspace::create(context,command=="profile");if(!made.ok()){fprintf(stderr,"workspace: %s\n",made.error().message.c_str());return 2;}auto w=std::move(made.value());
 if(command=="dump-logits"){
  if(spec.empty()||output_dir.empty()){fprintf(stderr,"dump-logits requires --spec and --output-dir\n");return 2;}
  return report(dump_intrinsic(*m,w->view,loaded.value().provenance,spec,output_dir,grid))?0:2;
 }
 if(command=="serve"){
  auto tokenizer_loaded=Tokenizer::load(model);if(!tokenizer_loaded.ok()){fprintf(stderr,"tokenizer: %s\n",tokenizer_loaded.error().message.c_str());return 2;}
  std::shared_ptr<Tokenizer> tokenizer(std::move(tokenizer_loaded.value()));
  ServerConfig cfg;cfg.host=host;cfg.port=port;cfg.context_tokens=context;cfg.max_tokens=std::min(1024,context-1);
  auto server_created=HttpServer::create(cfg,tokenizer,[&](const PreparedChat&prompt,int32_t max_tokens){return generate_native(*m,w->view,*tokenizer,prompt,max_tokens,grid);});
  if(!server_created.ok()){fprintf(stderr,"server startup: %s\n",server_created.error().message.c_str());return 2;}auto server=std::move(server_created.value());auto bound=server->bind();if(!bound.ok()){fprintf(stderr,"server bind: %s\n",bound.error().message.c_str());return 2;}
  fprintf(stderr,"halo-6000 native server listening %s:%d; serialized GPU generation\n",host.c_str(),bound.value());return report(server->listen())?0:2;
 }
 if(command=="profile"){
  if(warm>=context){fprintf(stderr,"profile warm exceeds context\n");return 2;}
  for(int p=0;p<=warm;p++)if(!report(run_token(*m,w->view,100+(p*31+p*p*7)%20000,p,0,grid)))return 2;
  uint64_t times[646];if(!report(cuda_status(cudaMemcpy(times,w->view.timestamps,sizeof times,cudaMemcpyDeviceToHost),"read phase timestamps")))return 2;
  printf("{\"grid\":%d,\"row_lanes\":%d,\"position\":%d,\"timestamp_units\":\"ns\",\"timestamps\":[",grid,row_lanes,warm);for(int i=0;i<646;i++)printf("%s%llu",i?",":"",(unsigned long long)times[i]);puts("]}");return 0;
 }
 if(command=="bench"){
  if(warm+steps>context){fprintf(stderr,"warm+steps exceed context\n");return 2;}
  auto token=[](int p){return 100+(p*31+p*p*7)%20000;};
  for(int p=0;p<warm;p++)if(!report(run_token(*m,w->view,token(p),p,0,grid)))return 2;
  if(!report(cuda_status(cudaDeviceSynchronize(),"warm completion")))return 2;
  cudaEvent_t start,end;if(!report(cuda_status(cudaEventCreate(&start),"start event"))||!report(cuda_status(cudaEventCreate(&end),"end event")))return 2;
  auto wall_start=std::chrono::steady_clock::now();if(!report(cuda_status(cudaEventRecord(start),"start timing")))return 2;
  for(int p=warm;p<warm+steps;p++)if(!report(run_token(*m,w->view,token(p),p,0,grid)))return 2;
  if(!report(cuda_status(cudaEventRecord(end),"end timing"))||!report(cuda_status(cudaEventSynchronize(end),"timing completion")))return 2;
  double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall_start).count();float ms=0;if(!report(cuda_status(cudaEventElapsedTime(&ms,start,end),"elapsed timing")))return 2;
  cudaEventDestroy(start);cudaEventDestroy(end);printf("{\"backend\":\"halo6000_whole_token\",\"streams\":1,\"grid\":%d,\"row_lanes\":%d,\"warm\":%d,\"steps\":%d,\"gpu_ms\":%.6f,\"wall_ms\":%.6f,\"tokens_per_second\":%.6f,\"weights_bytes\":%zu,\"prefix_route\":\"all_serial_teacher_forcing\"}\n",grid,row_lanes,warm,steps,ms,wall_ms,steps*1000./ms,m->weight_bytes);return 0;
 }
 if(command=="forward-test"){
  if(tokens_file.empty()||output_dir.empty()){fprintf(stderr,"forward-test requires --tokens-file and --output-dir\n");return 2;}
  if(!layer_output_dir.empty()){if(!report(dump_layer0(*m,w->view,model,tokens_file,layer_output_dir,steps,grid))||!report(m->reset()))return 2;}
  if(!report(dump_forward(*m,w->view,model,tokens_file,output_dir,steps,grid)))return 2;printf("{\"steps\":%d,\"whole_token_dump\":true}\n",steps);return 0;
 }
 if(command=="debug-layer"){
  if(tokens_file.empty()||output_dir.empty()){fprintf(stderr,"debug-layer requires --tokens-file and --output-dir\n");return 2;}
  if(!report(dump_layer0(*m,w->view,model,tokens_file,output_dir,steps,grid)))return 2;printf("{\"layer\":0,\"steps\":%d,\"debug_dump\":true,\"whole_model\":false}\n",steps);return 0;
 }
 struct Frame{std::vector<float>hidden,state,history;};std::vector<Frame>reference;reference.reserve(steps);
 for(int p=0;p<steps;p++){
  if(!report(embed(*m,w->view,100+(p*31+p*p*7)%20000)))return 2;
  for(int phase=0;phase<=int(LayerPhase::Down);phase++)if(!report(run_gdn_phase(*m,w->view,0,0,LayerPhase(phase),grid)))return 2;
  reference.push_back({read_device(w->view.hidden,Hidden),read_device(m->state.gdn,ValueHeads*ValueDim*StateDim),read_device(m->state.conv,(ConvTaps-1)*ConvChannels)});
 }
 if(!report(m->reset()))return 2;size_t errors=0;double maxabs=0;
 for(int p=0;p<steps;p++){
  if(!report(embed(*m,w->view,100+(p*31+p*p*7)%20000))||!report(run_gdn_layer(*m,w->view,0,0,grid)))return 2;
  Frame got{read_device(w->view.hidden,Hidden),read_device(m->state.gdn,ValueHeads*ValueDim*StateDim),read_device(m->state.conv,(ConvTaps-1)*ConvChannels)};
  auto compare=[&](const std::vector<float>&a,const std::vector<float>&b){for(size_t i=0;i<a.size();i++){errors+=memcmp(&a[i],&b[i],4)!=0;maxabs=std::max(maxabs,double(std::abs(a[i]-b[i])));}};
  compare(reference[p].hidden,got.hidden);compare(reference[p].state,got.state);compare(reference[p].history,got.history);
 }
 printf("{\"guard\":\"own_split_vs_persistent_GDN_MoE_layer0\",\"steps\":%d,\"grid\":%d,\"weights_bytes\":%zu,\"state_bytes\":%zu,\"workspace_bytes\":%zu,\"bit_errors\":%zu,\"max_abs\":%.9g,\"fork_logits_checked\":false}\n",steps,grid,m->weight_bytes,m->state_bytes,w->bytes,errors,maxabs);return errors?1:0;
}
