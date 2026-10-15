#include "server.hpp"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace halo6000;
using json=nlohmann::json;
template<class T>T take(Result<T>r){if(!r.ok())throw std::runtime_error(r.error().message);return std::move(r.value());}
void require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
struct Running {
 std::unique_ptr<HttpServer>server;std::thread thread;Status result=success();int port;
 explicit Running(std::unique_ptr<HttpServer>s):server(std::move(s)),port(take(server->bind())){thread=std::thread([this]{result=server->listen();});}
 ~Running(){server->stop();if(thread.joinable())thread.join();}
};
httplib::Result post(httplib::Client&c,const char*path,const json&j){return c.Post(path,j.dump(),"application/json");}
json body(const httplib::Result&r,int status){require(bool(r),"HTTP transport failed");require(r->status==status,"HTTP status "+std::to_string(r->status)+": "+r->body);return json::parse(r->body);}
int main(int argc,char**argv){
 try{
  if(argc!=3)throw std::runtime_error("Usage: server-test MODEL fixtures.json");
  std::ifstream in(argv[2]);json fixture;in>>fixture;
  auto tok=std::shared_ptr<Tokenizer>(take(Tokenizer::load(argv[1])));
  auto raw=take(tok->tokenize(u8"<think>malformed 日本語",true));
  auto fragment=take(tok->tokenize(std::string(1,char(0xc3)),false));
  std::atomic<int>active{0},peak{0},calls{0};std::atomic<bool>broken{false},length{false},immediate_eog{false},utf8_fragment{false};
  Generator fake=[&](const PreparedChat&,int32_t cap)->Result<Completion>{
   ++calls;int a=++active;int old=peak.load();while(old<a&&!peak.compare_exchange_weak(old,a)){}
   // Test-only deterministic callback, not native execution or GPU timings.
   std::this_thread::sleep_for(std::chrono::milliseconds(8));
   --active;if(broken)return Error{ErrorCode::Cuda,"injected native failure"};
   Completion out;out.prompt_ms=2;out.predicted_ms=3;
   if(utf8_fragment){out.tokens=fragment;out.tokens.push_back(tok->eog_tokens().front());}
   else if(immediate_eog)out.tokens={tok->eog_tokens().front()};
   else if(cap==1)out.tokens={raw.front()};
   else if(length)out.tokens=std::vector<Token>(cap,raw.front());
   else{out.tokens=raw;out.tokens.push_back(tok->eog_tokens().front());}
   out.completion_tokens=out.tokens.size();return out;
  };
  ServerConfig cfg;cfg.port=0;
  require(!HttpServer::create(cfg,tok,{}).ok(),"Missing generator accepted");
  require(!HttpServer::create(cfg,{},fake).ok(),"Missing tokenizer accepted");
  auto empty=[](const PreparedChat&,int32_t)->Result<Completion>{return Completion{};};
  require(!HttpServer::create(cfg,tok,empty).ok(),"Fake successful empty readiness accepted");
  auto error=[](const PreparedChat&,int32_t)->Result<Completion>{return Error{ErrorCode::Cuda,"not loaded"};};
  require(!HttpServer::create(cfg,tok,error).ok(),"Failed readiness accepted");
  Running run(take(HttpServer::create(cfg,tok,fake)));
  httplib::Client client("127.0.0.1",run.port);client.set_read_timeout(10,0);
  body(client.Get("/health"),200);
  auto models=body(client.Get("/v1/models"),200);require(models["data"][0]["id"]=="evaluated-model","Wrong model");
  for(const auto&f:fixture["fixtures"]){
   auto rendered=body(post(client,"/apply-template",{{"messages",f["messages"]},{"add_generation_prompt",true},{"chat_template_kwargs",{{"enable_thinking",false}}}}),200);
   require(rendered["prompt"]==f["prompt"],"HTTP template parity failed");
  }
  json req={{"model","evaluated-model"},{"messages",fixture["fixtures"][0]["messages"]},{"temperature",0},{"top_p",1},{"seed",20261004},{"max_tokens",1024},{"chat_template_kwargs",{{"enable_thinking",false}}},{"cache_prompt",false}};
  auto first=body(post(client,"/v1/chat/completions",req),200);
  require(first["choices"][0]["message"]["content"]==u8"<think>malformed 日本語","Raw markup/Unicode changed");
  require(first["choices"][0]["finish_reason"]=="stop","EOG finish mismatch");
  require(first["usage"]["prompt_tokens"]==98,"Prompt usage mismatch");
  require(first["usage"]["completion_tokens"]==raw.size()+1,"EOG not counted");
  require(first["timings"]["predicted_n"]==raw.size()+1&&first["timings"]["prompt_ms"]==2&&first["timings"]["predicted_ms"]==3,"Timings changed or queue included");
  std::vector<std::future<json>>futures;
  for(int i=0;i<8;++i)futures.push_back(std::async(std::launch::async,[&]{httplib::Client c("127.0.0.1",run.port);c.set_read_timeout(10,0);return body(post(c,"/v1/chat/completions",req),200);}));
  for(auto&f:futures)require(f.get()["timings"]["predicted_ms"]==3,"Concurrent queue time included");
  require(peak==1,"Native callback ran concurrently");
  auto bad=req;bad["stream"]=true;body(post(client,"/v1/chat/completions",bad),400);
  bad=req;bad["temperature"]=0.1;body(post(client,"/v1/chat/completions",bad),400);
  bad=req;bad["cache_prompt"]=true;body(post(client,"/v1/chat/completions",bad),400);
  bad=req;bad["chat_template_kwargs"]["enable_thinking"]=true;body(post(client,"/v1/chat/completions",bad),400);
  bad=req;bad["messages"][0]["content"]=std::string(30000,'a');body(post(client,"/v1/chat/completions",bad),400);
  bad=req;bad["max_tokens"]=0;body(post(client,"/v1/chat/completions",bad),400);
  bad=req;bad["stop"]="x";body(post(client,"/v1/chat/completions",bad),400);
  bad=req;bad["model"]="unknown";body(post(client,"/v1/chat/completions",bad),400);
  body(client.Post("/v1/chat/completions","{broken","application/json"),400);
  bad=req;bad["max_tokens"]=raw.size()+1;
  auto at_cap=body(post(client,"/v1/chat/completions",bad),200);
  require(at_cap["choices"][0]["finish_reason"]=="stop","EOG must override length at final allowed token");
  immediate_eog=true;bad=req;bad["max_tokens"]=1;
  auto immediate=body(post(client,"/v1/chat/completions",bad),200);
  require(immediate["choices"][0]["message"]["content"]==""&&immediate["usage"]["completion_tokens"]==1&&immediate["choices"][0]["finish_reason"]=="stop","Legitimate immediate EOG miscounted");
  immediate_eog=false;
  utf8_fragment=true;auto replaced=body(post(client,"/v1/chat/completions",req),200);
  require(replaced["choices"][0]["message"]["content"]==std::string("\xef\xbf\xbd"),"Invalid UTF8 wire policy differs from oracle");
  utf8_fragment=false;
  length=true;bad=req;bad["max_tokens"]=2;
  auto capped=body(post(client,"/v1/chat/completions",bad),200);
  require(capped["choices"][0]["finish_reason"]=="length"&&capped["usage"]["completion_tokens"]==2,"Cap finish/count mismatch");
  auto metrics=client.Get("/metrics");require(bool(metrics)&&metrics->status==200,"Metrics failed");
  require(metrics->body.find("halo_completed_requests_total 13\n")!=std::string::npos,"Metrics completed count mismatch");
  const std::string metric_key="\nhalo_gpu_generated_seconds_total ";
  auto offset=metrics->body.find(metric_key);require(offset!=std::string::npos,"Missing GPU time metric");
  double seconds=std::stod(metrics->body.substr(offset+metric_key.size()));
  require(std::abs(seconds-0.039)<1e-12,"GPU metric included queue time");
  broken=true;auto failed=body(post(client,"/v1/chat/completions",req),500);require(failed.contains("error")&&!failed.contains("choices"),"Failure became successful empty answer");
  body(client.Get("/health"),503);body(post(client,"/v1/chat/completions",req),503);
  json output={{"test_callback","synthetic; no native inference or GPU timing claim"},{"http_template_fixtures",4},{"parallel_http_requests",8},{"max_concurrent_generator_calls",peak.load()},{"sampled_eog_counted",true},{"eog_hidden",true},{"raw_content_preserved",true},{"invalid_utf8_wire_replacement",true},{"length_cap",true},{"eog_overrides_length",true},{"immediate_eog_usage_one",true},{"queue_excluded_from_phase_metrics",true},{"startup_failure_before_bind",true},{"runtime_failure_health_503",true},{"invalid_request_checks",9}};
  std::cout<<output.dump(2)<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
