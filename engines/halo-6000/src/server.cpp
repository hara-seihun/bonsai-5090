#include "server.hpp"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <atomic>
#include <cmath>
#include <ctime>
#include <mutex>
#include <sstream>
#include <set>

namespace halo6000 {
namespace {
using json=nlohmann::json;
Error invalid(const std::string&s){return {ErrorCode::Invalid,s};}
void reply(httplib::Response&r,int status,const json&body){r.status=status;r.set_content(body.dump(-1,' ',false,json::error_handler_t::replace),"application/json; charset=utf-8");}
void fail(httplib::Response&r,int status,const std::string&s){reply(r,status,{{"error",{{"message",s},{"type",status>=500?"server_error":"invalid_request_error"}}}});}
Status keys(const json&j,const std::set<std::string>&allowed){
 if(!j.is_object())return invalid("Expected JSON object");
 for(auto it=j.begin();it!=j.end();++it)if(!allowed.count(it.key()))return Error{ErrorCode::Unsupported,"Unsupported field: "+it.key()};
 return success();
}
Status boolean(const json&j,const char*name,bool expected){
 if(j.contains(name)&&(!j[name].is_boolean()||j[name].get<bool>()!=expected))return invalid(std::string(name)+" must be "+(expected?"true":"false"));
 return success();
}
Status number(const json&j,const char*name,double expected){
 if(j.contains(name)&&(!j[name].is_number()||j[name].get<double>()!=expected))return invalid(std::string(name)+" must equal "+std::to_string(expected));
 return success();
}
Status thinking_off(const json&j){
 if(!j.contains("chat_template_kwargs"))return success();
 const auto&kwargs=j["chat_template_kwargs"];
 auto k=keys(kwargs,{"enable_thinking"});if(!k.ok())return k;
 return boolean(kwargs,"enable_thinking",false);
}
Result<std::vector<ChatMessage>> messages(const json&j){
 if(!j.contains("messages")||!j["messages"].is_array()||j["messages"].empty())return invalid("messages must be a nonempty array");
 std::vector<ChatMessage>out;
 for(const auto&m:j["messages"]){
  auto k=keys(m,{"role","content"});if(!k.ok())return k.error();
  if(!m.contains("role")||!m["role"].is_string()||!m.contains("content")||!m["content"].is_string())return invalid("Message role and content must be strings");
  out.push_back({m["role"].get<std::string>(),m["content"].get<std::string>()});
 }
 return out;
}
Status validate(const Tokenizer&t,const Completion&c,int cap){
 if(c.tokens.empty()||c.tokens.size()>size_t(cap)||c.completion_tokens!=int64_t(c.tokens.size()))return invalid("Generator returned inconsistent sampled token count");
 if(!std::isfinite(c.prompt_ms)||!std::isfinite(c.predicted_ms)||c.prompt_ms<0||c.predicted_ms<0)return invalid("Generator returned invalid GPU phase timings");
 for(size_t i=0;i<c.tokens.size();++i){
  if(c.tokens[i]<0||c.tokens[i]>=t.vocab_size())return invalid("Generator returned token outside vocabulary");
  if(t.is_eog(c.tokens[i])&&i+1!=c.tokens.size())return invalid("Generator returned tokens after EOG");
 }
 if(!t.is_eog(c.tokens.back())&&c.tokens.size()!=size_t(cap))return invalid("Generator stopped before EOG or requested token cap");
 return success();
}
Result<Completion> invoke(const Generator&g,const PreparedChat&p,int cap){
 try{return g(p,cap);}catch(const std::bad_alloc&){return Error{ErrorCode::Capacity,"Generator allocation failed"};}
 catch(const std::exception&e){return Error{ErrorCode::Cuda,std::string("Generator exception: ")+e.what()};}
 catch(...){return Error{ErrorCode::Cuda,"Generator threw a nonstandard exception"};}
}
}
struct HttpServer::Impl {
 ServerConfig cfg;std::shared_ptr<Tokenizer>tok;Generator gen;httplib::Server http;
 std::mutex execution,metrics;
 std::atomic<bool>ready{true};std::atomic<uint64_t>sequence{0};bool bound=false;
 uint64_t completed=0,failed=0,prompt_tokens=0,generated_tokens=0;
 double prompt_seconds=0,generated_seconds=0;
 void chat(const httplib::Request&req,httplib::Response&res){
  auto reject=[&](int code,const std::string&s){std::lock_guard<std::mutex>lock(metrics);++failed;fail(res,code,s);};
  try{
   auto j=json::parse(req.body);
   auto k=keys(j,{"model","messages","temperature","top_p","seed","max_tokens","chat_template_kwargs","cache_prompt","stream","n","presence_penalty","frequency_penalty"});
   if(!k.ok()){reject(400,k.error().message);return;}
   for(auto status:{boolean(j,"stream",false),boolean(j,"cache_prompt",false),number(j,"temperature",0),number(j,"top_p",1),number(j,"n",1),number(j,"presence_penalty",0),number(j,"frequency_penalty",0),thinking_off(j)})
    if(!status.ok()){reject(400,status.error().message);return;}
   if(j.contains("model")&&(!j["model"].is_string()||(j["model"]!=cfg.model&&j["model"]!="evaluated-model"))){reject(400,"Unknown model alias");return;}
   if(j.contains("seed")&&!j["seed"].is_number_integer()){reject(400,"seed must be an integer (greedy execution ignores RNG)");return;}
   if(!j.contains("max_tokens")||!j["max_tokens"].is_number_integer()||j["max_tokens"]<1||j["max_tokens"]>cfg.max_tokens){reject(400,"max_tokens outside configured limit");return;}
   int cap=j["max_tokens"].get<int>();
   auto msgs=messages(j);if(!msgs.ok()){reject(400,msgs.error().message);return;}
   // Serializes native state and vocabulary/template operations; waiting here
   // never contributes to callback GPU milliseconds or phase-rate metrics.
   std::lock_guard<std::mutex>lock(execution);
   if(!ready.load()){reject(503,"Native generator is not healthy; restart required");return;}
   auto prepared=tok->prepare_chat(msgs.value());if(!prepared.ok()){reject(400,prepared.error().message);return;}
   if(prepared.value().tokens.empty()||prepared.value().tokens.size()+size_t(cap)>size_t(cfg.context_tokens)){reject(400,"Prompt plus max_tokens exceeds context; context shifting is disabled");return;}
   auto generated=invoke(gen,prepared.value(),cap);
   if(!generated.ok()){ready=false;reject(500,generated.error().message);return;}
   auto&c=generated.value();auto checked=validate(*tok,c,cap);
   if(!checked.ok()){ready=false;reject(500,checked.error().message);return;}
   bool eog=tok->is_eog(c.tokens.back());
   auto visible=c.tokens;if(eog)visible.pop_back();
   auto text=tok->detokenize(visible,true);
   if(!text.ok()){ready=false;reject(500,text.error().message);return;}
   const size_t pn=prepared.value().tokens.size();
   reply(res,200,{{"id","chatcmpl-halo-"+std::to_string(++sequence)},{"object","chat.completion"},{"created",std::time(nullptr)},{"model",cfg.model},
    {"choices",json::array({{{"index",0},{"message",{{"role","assistant"},{"content",text.value()}}},{"finish_reason",eog?"stop":"length"}}})},
    {"usage",{{"prompt_tokens",pn},{"completion_tokens",c.completion_tokens},{"total_tokens",pn+c.completion_tokens}}},
    {"timings",{{"cache_n",0},{"prompt_n",pn},{"prompt_ms",c.prompt_ms},{"predicted_n",c.completion_tokens},{"predicted_ms",c.predicted_ms}}}});
   std::lock_guard<std::mutex>ml(metrics);++completed;prompt_tokens+=pn;generated_tokens+=c.completion_tokens;prompt_seconds+=c.prompt_ms/1000;generated_seconds+=c.predicted_ms/1000;
  }catch(const json::exception&e){reject(400,e.what());}
  catch(const std::exception&e){reject(500,e.what());}
 }
 void apply(const httplib::Request&req,httplib::Response&res){
  try{
   auto j=json::parse(req.body);auto k=keys(j,{"messages","chat_template_kwargs","add_generation_prompt"});
   if(!k.ok()){fail(res,400,k.error().message);return;}
   auto s=thinking_off(j);if(!s.ok()){fail(res,400,s.error().message);return;}
   if(j.contains("add_generation_prompt")&&!j["add_generation_prompt"].is_boolean()){fail(res,400,"add_generation_prompt must be boolean");return;}
   auto msgs=messages(j);if(!msgs.ok()){fail(res,400,msgs.error().message);return;}
   std::lock_guard<std::mutex>lock(execution);
   auto rendered=tok->render_chat(msgs.value(),j.value("add_generation_prompt",true));
   if(!rendered.ok()){fail(res,400,rendered.error().message);return;}
   reply(res,200,{{"prompt",rendered.value().prompt}});
  }catch(const json::exception&e){fail(res,400,e.what());}
  catch(const std::exception&e){fail(res,500,e.what());}
 }
 void routes(){
  http.new_task_queue=[](){return new httplib::ThreadPool(8);};
  http.set_payload_max_length(cfg.max_request_bytes);
  http.set_exception_handler([](const auto&,auto&res,std::exception_ptr){fail(res,500,"Unhandled HTTP handler failure");});
  http.Get("/health",[this](const auto&,auto&res){reply(res,ready?200:503,{{"status",ready?"ok":"error"},{"backend","halo6000"},{"gpu_batching",false}});});
  http.Get("/v1/models",[this](const auto&,auto&res){reply(res,200,{{"object","list"},{"data",json::array({{{"id",cfg.model},{"object","model"},{"owned_by","halo6000"}}})}});});
  http.Get("/metrics",[this](const auto&,auto&res){
   std::lock_guard<std::mutex>lock(metrics);std::ostringstream out;out.precision(17);
   out<<"# TYPE halo_ready gauge\nhalo_ready "<<ready.load()<<"\n"
      <<"# TYPE halo_completed_requests_total counter\nhalo_completed_requests_total "<<completed<<"\n"
      <<"# TYPE halo_failed_requests_total counter\nhalo_failed_requests_total "<<failed<<"\n"
      <<"# TYPE halo_prompt_tokens_total counter\nhalo_prompt_tokens_total "<<prompt_tokens<<"\n"
      <<"# TYPE halo_generated_tokens_total counter\nhalo_generated_tokens_total "<<generated_tokens<<"\n"
      <<"# TYPE halo_gpu_prompt_seconds_total counter\nhalo_gpu_prompt_seconds_total "<<prompt_seconds<<"\n"
      <<"# TYPE halo_gpu_generated_seconds_total counter\nhalo_gpu_generated_seconds_total "<<generated_seconds<<"\n";
   res.set_content(out.str(),"text/plain; version=0.0.4");
  });
  http.Post("/apply-template",[this](const auto&req,auto&res){apply(req,res);});
  http.Post("/v1/chat/completions",[this](const auto&req,auto&res){chat(req,res);});
 }
};
HttpServer::HttpServer(std::unique_ptr<Impl>p):impl_(std::move(p)){}
HttpServer::~HttpServer(){stop();}
Result<std::unique_ptr<HttpServer>> HttpServer::create(ServerConfig cfg,std::shared_ptr<Tokenizer>tok,Generator gen){
 try{
  if(!tok||!gen)return invalid("Loaded tokenizer and working native generator are required before serving");
  if(cfg.host.empty()||cfg.model.empty()||cfg.port<0||cfg.port>65535||cfg.context_tokens<2||cfg.max_tokens<1||cfg.max_tokens>=cfg.context_tokens||!cfg.max_request_bytes)return invalid("Invalid HTTP server configuration");
  auto probe=tok->prepare_chat({{"user","Hi"}});if(!probe.ok())return probe.error();
  if(probe.value().tokens.size()+1>size_t(cfg.context_tokens))return invalid("Context too small for startup execution probe");
  auto generated=invoke(gen,probe.value(),1);if(!generated.ok())return generated.error();
  auto checked=validate(*tok,generated.value(),1);if(!checked.ok())return checked.error();
  auto p=std::make_unique<Impl>();p->cfg=std::move(cfg);p->tok=std::move(tok);p->gen=std::move(gen);p->routes();
  return std::unique_ptr<HttpServer>(new HttpServer(std::move(p)));
 }catch(const std::bad_alloc&){return Error{ErrorCode::Capacity,"HTTP initialization allocation failed"};}
 catch(const std::exception&e){return invalid(e.what());}
}
Result<int> HttpServer::bind(){
 try{
  if(impl_->bound)return invalid("HTTP server is already bound");
  int port=impl_->cfg.port;
  if(port==0)port=impl_->http.bind_to_any_port(impl_->cfg.host);
  else if(!impl_->http.bind_to_port(impl_->cfg.host,port))port=-1;
  if(port<=0)return Error{ErrorCode::Io,"HTTP socket bind failed"};
  impl_->bound=true;return port;
 }catch(const std::exception&e){return Error{ErrorCode::Io,e.what()};}
}
Status HttpServer::listen(){
 try{
  if(!impl_->bound)return invalid("Bind HTTP server before listening");
  if(!impl_->http.listen_after_bind())return Error{ErrorCode::Io,"HTTP listen failed"};
  return success();
 }catch(const std::exception&e){return Error{ErrorCode::Io,e.what()};}
}
void HttpServer::stop(){if(impl_)impl_->http.stop();}
} // namespace halo6000
