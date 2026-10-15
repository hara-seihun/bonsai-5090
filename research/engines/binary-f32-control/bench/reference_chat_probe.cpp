#include "llama.h"
#include "ggml-backend.h"
#include "sampling.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
int main(int argc,char** argv){
  if(argc!=4)return 2;
  llama_log_set([](ggml_log_level,const char*,void*){},nullptr);
  llama_backend_init();ggml_backend_load_all();
  auto mp=llama_model_default_params();mp.n_gpu_layers=0;
  auto* model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
  const auto* vocab=llama_model_get_vocab(model);
  std::string question="State the definition of the de Bruijn–Newman constant and the established lower bound in three sentences.";
  std::string prompt="<|im_start|>user\n"+question+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  std::ofstream(argv[2])<<prompt;
  const char* templ=llama_model_chat_template(model,nullptr);fprintf(stderr,"model_chat_template=%s\n",templ?templ:"UNSET");
  std::vector<llama_token> tokens(prompt.size()+16);
  int np=llama_tokenize(vocab,prompt.c_str(),prompt.size(),tokens.data(),tokens.size(),false,true);if(np<=0)return 4;tokens.resize(np);
  auto cp=llama_context_default_params();cp.n_ctx=1024;cp.n_batch=256;cp.n_ubatch=128;cp.n_threads=16;cp.n_threads_batch=16;
  auto* context=llama_init_from_model(model,cp);if(!context)return 5;
  auto batch=llama_batch_init(np,0,1);batch.n_tokens=np;
  for(int i=0;i<np;++i){batch.token[i]=tokens[i];batch.pos[i]=i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=i==np-1;}
  if(llama_decode(context,batch))return 6;
  const int nv=llama_vocab_n_tokens(vocab);
  {std::ofstream logits(std::string(argv[2])+".logits",std::ios::binary);logits.write(reinterpret_cast<const char*>(llama_get_logits_ith(context,-1)),nv*sizeof(float));}
  h5::sampling::Config config;std::string error;
  if(!h5::sampling::parse(nlohmann::ordered_json{{"temperature",.7},{"top_p",.8},{"top_k",20},{"presence_penalty",1.5},{"seed",42}},config,error))return 7;
  std::vector<unsigned char> seen(nv,0);std::mt19937_64 rng(config.seed);std::string text;
  for(int step=0;step<128;++step){
    auto result=h5::sampling::select(llama_get_logits_ith(context,-1),nv,config,seen,rng);if(!result.error.empty())return 8;
    int id=result.token;seen[id]=1;if(llama_vocab_is_eog(vocab,id))break;
    char piece[512];int size=llama_token_to_piece(vocab,id,piece,sizeof piece,0,false);if(size>0){text.append(piece,size);fwrite(piece,1,size,stdout);fflush(stdout);}
    auto one=llama_batch_get_one(&id,1);if(llama_decode(context,one))return 9;
  }
  std::ofstream(argv[3])<<text;
  llama_batch_free(batch);llama_free(context);llama_model_free(model);llama_backend_free();
}
