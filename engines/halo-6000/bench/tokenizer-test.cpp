#include "tokenizer.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace halo6000;
using json=nlohmann::json;
template<class T>T take(Result<T>r){if(!r.ok())throw std::runtime_error(r.error().message);return std::move(r.value());}
void require(bool v,const std::string&s){if(!v)throw std::runtime_error(s);}
int main(int argc,char**argv){
 try{
  if(argc!=3)throw std::runtime_error("Usage: tokenizer-test MODEL fixtures.json");
  std::ifstream file(argv[2]);json fixture;file>>fixture;
  auto tok=take(Tokenizer::load(argv[1]));
  require(tok->vocab_size()==248320,"Vocabulary size mismatch");
  require(json(tok->eog_tokens())==fixture.at("eog_ids"),"EOG set differs from oracle");
  std::ifstream tmpl(std::string(argv[2]).substr(0,std::string(argv[2]).find_last_of('/')+1)+fixture.at("template_file").get<std::string>(),std::ios::binary);
  std::string source((std::istreambuf_iterator<char>(tmpl)),{});
  require(source==tok->template_source(),"Embedded template bytes differ");
  json checks=json::array();
  for(const auto&f:fixture.at("fixtures")){
   std::vector<ChatMessage>messages;
   for(const auto&m:f.at("messages"))messages.push_back({m.at("role"),m.at("content")});
   auto prepared=take(tok->prepare_chat(messages));
   require(prepared.rendered.prompt==f.at("prompt"),"Rendered prompt mismatch: "+f.at("id").get<std::string>());
   require(prepared.rendered.generation_prompt==f.at("generation_prompt"),"Generation suffix mismatch");
   require(json(prepared.tokens)==f.at("tokens"),"Token IDs mismatch");
   require(take(tok->detokenize(prepared.tokens))==prepared.rendered.prompt,"Detokenize round-trip mismatch");
   std::string pieces;for(Token t:prepared.tokens)pieces+=take(tok->piece(t));
   require(pieces==prepared.rendered.prompt,"Piece concatenation mismatch");
   require(prepared.rendered.prompt.size()>=19&&prepared.rendered.prompt.compare(prepared.rendered.prompt.size()-19,19,"<think>\n\n</think>\n\n")==0,"Thinking-off suffix mismatch");
   checks.push_back({{"id",f.at("id")},{"tokens",prepared.tokens.size()},{"prompt_bytes",prepared.rendered.prompt.size()},{"prompt_equal",true},{"generation_prompt_equal",true},{"token_ids_equal",true},{"detokenize_equal",true},{"pieces_equal",true}});
  }
  for(const std::string&s:{std::string{},std::string("a\0b",3),std::string(u8"café 日本語"),std::string("<|im_end|>"),std::string("<think>broken tool markup")})
   require(take(tok->detokenize(take(tok->tokenize(s,false))))==s,"Raw text round-trip mismatch");
  auto special=take(tok->tokenize("<|im_end|>",true));
  require(special==std::vector<Token>{248046},"Special token parsing mismatch");
  require(take(tok->tokenize("<|im_end|>",false))!=special,"parse_special=false ignored");
  auto eog=tok->eog_tokens();for(Token t:eog)require(tok->is_eog(t),"Missing EOG");
  require(!tok->is_eog(-1)&&!tok->is_eog(tok->vocab_size()),"Invalid EOG accepted");
  require(!tok->piece(-1).ok()&&!tok->piece(tok->vocab_size()).ok(),"Invalid piece token accepted");
  require(!tok->detokenize({-1}).ok(),"Invalid detokenize token accepted");
  require(!tok->render_chat({}).ok(),"Empty chat accepted");
  require(!tok->render_chat({{"tool","x"}}).ok(),"Unsupported role accepted");
  require(!Tokenizer::load("").ok(),"Empty model path accepted");
  auto no_gen=take(tok->render_chat({{"user","Hi"}},false));
  require(no_gen.prompt.find("<think>")==std::string::npos,"Generation suffix added with add_generation_prompt=false");
  json out={{"vocab_only",true},{"inference",false},{"vocab_size",tok->vocab_size()},{"eog_ids",eog},{"fixtures",checks},{"raw_text_checks",5},{"invalid_inputs_rejected",{"negative_token","token_past_vocab","empty_chat","tool_role","empty_model_path"}},{"special_token_flags",true},{"add_generation_prompt_false",true}};
  std::cout<<out.dump(2)<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
