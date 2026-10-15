#include "tokenizer.hpp"
#include "llama.h"
#include "chat.h"
#include <algorithm>
#include <climits>
#include <mutex>
#include <new>

namespace halo6000 {
namespace {
struct ModelDeleter {void operator()(llama_model*m)const{llama_model_free(m);}};
Error invalid(const std::string&s){return {ErrorCode::Invalid,s};}
Error capacity(){return {ErrorCode::Capacity,"Tokenizer input/output exceeds memory or int32 API capacity"};}
template<class T,class F> Result<T> guarded(F&&f){
 try{return f();}catch(const std::bad_alloc&){return capacity();}
 catch(const std::exception&e){return invalid(e.what());}
}
}
struct Tokenizer::Impl {
 std::unique_ptr<llama_model,ModelDeleter> model;
 const llama_vocab*vocab=nullptr;
 common_chat_templates_ptr templates;
 std::string source;
 std::vector<Token> eog;
 int32_t size=0;
};
Tokenizer::Tokenizer(std::unique_ptr<Impl>p):impl_(std::move(p)){}
Tokenizer::~Tokenizer()=default;
Result<std::unique_ptr<Tokenizer>> Tokenizer::load(const std::string&path){
 return guarded<std::unique_ptr<Tokenizer>>([&]()->Result<std::unique_ptr<Tokenizer>>{
  if(path.empty()||path.find('\0')!=std::string::npos)return invalid("Invalid GGUF path");
  // Process-global ggml initialization only; no inference context or GPU allocation.
  // Do not free global backends here: other vocabulary instances can coexist.
  static std::once_flag once;
  std::call_once(once,[]{llama_backend_init();});
  auto p=std::make_unique<Impl>();
  auto params=llama_model_default_params();params.vocab_only=true;params.n_gpu_layers=0;
  p->model.reset(llama_model_load_from_file(path.c_str(),params));
  if(!p->model)return Error{ErrorCode::Io,"Vocab-only GGUF load failed: "+path};
  p->vocab=llama_model_get_vocab(p->model.get());
  if(!p->vocab)return invalid("GGUF has no vocabulary");
  p->size=llama_vocab_n_tokens(p->vocab);
  const char*source=llama_model_chat_template(p->model.get(),nullptr);
  if(!source||!*source)return Error{ErrorCode::Unsupported,"GGUF has no embedded chat template"};
  p->source=source;
  p->templates=common_chat_templates_init(p->model.get(),"");
  if(!p->templates)return invalid("Embedded Jinja initialization failed");
  for(Token t=0;t<p->size;++t)if(llama_vocab_is_eog(p->vocab,t))p->eog.push_back(t);
  return std::unique_ptr<Tokenizer>(new Tokenizer(std::move(p)));
 });
}
Result<std::vector<Token>> Tokenizer::tokenize(const std::string&text,bool parse_special)const{
 return guarded<std::vector<Token>>([&]()->Result<std::vector<Token>>{
  if(text.size()>INT32_MAX)return capacity();
  int32_t n=llama_tokenize(impl_->vocab,text.data(),int32_t(text.size()),nullptr,0,false,parse_special);
  if(n==INT32_MIN)return capacity();
  if(n==0)return std::vector<Token>{};
  if(n>0)return invalid("Unexpected tokenizer sizing result");
  std::vector<Token>out(size_t(-n));
  n=llama_tokenize(impl_->vocab,text.data(),int32_t(text.size()),out.data(),int32_t(out.size()),false,parse_special);
  if(n<0||size_t(n)>out.size())return invalid("Tokenization failed after sizing");
  out.resize(n);return out;
 });
}
Result<std::string> Tokenizer::detokenize(const std::vector<Token>&tokens,bool special)const{
 return guarded<std::string>([&]()->Result<std::string>{
  if(tokens.size()>INT32_MAX)return capacity();
  for(Token t:tokens)if(t<0||t>=impl_->size)return invalid("Token ID outside vocabulary");
  int32_t n=llama_detokenize(impl_->vocab,tokens.data(),int32_t(tokens.size()),nullptr,0,false,special);
  if(n==INT32_MIN)return capacity();
  if(n==0)return std::string{};
  if(n>0)return invalid("Unexpected detokenizer sizing result");
  std::string out(size_t(-n),'\0');
  n=llama_detokenize(impl_->vocab,tokens.data(),int32_t(tokens.size()),out.data(),int32_t(out.size()),false,special);
  if(n<0||size_t(n)>out.size())return invalid("Detokenization failed after sizing");
  out.resize(n);return out;
 });
}
Result<std::string> Tokenizer::piece(Token token,bool special)const{
 return guarded<std::string>([&]()->Result<std::string>{
  if(token<0||token>=impl_->size)return invalid("Token ID outside vocabulary");
  int32_t n=llama_token_to_piece(impl_->vocab,token,nullptr,0,0,special);
  if(n==INT32_MIN)return capacity();
  if(n==0)return std::string{};
  if(n>0)return invalid("Unexpected token-piece sizing result");
  std::string out(size_t(-n),'\0');
  n=llama_token_to_piece(impl_->vocab,token,out.data(),int32_t(out.size()),0,special);
  if(n<0||size_t(n)>out.size())return invalid("Token-piece conversion failed after sizing");
  out.resize(n);return out;
 });
}
Result<RenderedChat> Tokenizer::render_chat(const std::vector<ChatMessage>&messages,bool add_generation_prompt)const{
 return guarded<RenderedChat>([&]()->Result<RenderedChat>{
  if(messages.empty())return invalid("Chat requires at least one message");
  common_chat_templates_inputs in;
  for(const auto&m:messages){
   if(m.role!="system"&&m.role!="developer"&&m.role!="user"&&m.role!="assistant")
    return Error{ErrorCode::Unsupported,"Unsupported text chat role: "+m.role};
   common_chat_msg msg;msg.role=m.role;msg.content=m.content;in.messages.push_back(std::move(msg));
  }
  in.use_jinja=true;in.add_generation_prompt=add_generation_prompt;
  in.enable_thinking=false;in.reasoning_format=COMMON_REASONING_FORMAT_NONE;in.force_pure_content=true;
  in.chat_template_kwargs["enable_thinking"]="false";
  auto out=common_chat_templates_apply(impl_->templates.get(),in);
  return RenderedChat{std::move(out.prompt),std::move(out.generation_prompt)};
 });
}
Result<PreparedChat> Tokenizer::prepare_chat(const std::vector<ChatMessage>&messages,bool add_generation_prompt)const{
 auto rendered=render_chat(messages,add_generation_prompt);if(!rendered.ok())return rendered.error();
 auto tokens=tokenize(rendered.value().prompt,true);if(!tokens.ok())return tokens.error();
 return PreparedChat{std::move(rendered.value()),std::move(tokens.value())};
}
bool Tokenizer::is_eog(Token token)const{return std::binary_search(impl_->eog.begin(),impl_->eog.end(),token);}
const std::vector<Token>&Tokenizer::eog_tokens()const{return impl_->eog;}
const std::string&Tokenizer::template_source()const{return impl_->source;}
int32_t Tokenizer::vocab_size()const{return impl_->size;}
} // namespace halo6000
