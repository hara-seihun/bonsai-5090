#include "llama.h"
#include "head_demand.h"
#include "model.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
extern "C" {
void* h5_create(const char*, int, int, int);
void h5_reset(void*, int);
void h5_keep(void*, int, int);
void h5_pretune(void*);
int h5_logits(void*, int, float*);
}
int main(int argc, char** argv) {
  if (argc != 2 && argc != 4) return 2;
  llama_log_set([](ggml_log_level,const char*,void*){},nullptr);
  llama_backend_init(); auto params=llama_model_default_params(); params.vocab_only=true;params.n_gpu_layers=0;
  auto* model=llama_model_load_from_file(argv[1],params); if(!model)return 3;
  const auto* vocab=llama_model_get_vocab(model);
  std::string prompt="<|im_start|>user\nGive one friendly greeting in a short sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  if(argc==4){std::ifstream file(argv[2]);if(!file)return 7;prompt.assign(std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>());}
  std::vector<int> tokens(256); int n=llama_tokenize(vocab,prompt.c_str(),prompt.size(),tokens.data(),tokens.size(),false,true);
  if(n<=0 || n>128)return 4; tokens.resize(n);
  auto* engine=h5_create(argv[1],1,8192,128);if(!engine)return 5;h5_pretune(engine);
  std::vector<int> slots(n,0),positions(n),output(n),rows{n-1};for(int i=0;i<n;++i)positions[i]=i;
  std::vector<float> reference(h5::VOCAB),current(h5::VOCAB);
  for(int trial=0;trial<5;++trial){
    h5_reset(engine,0);h5_keep(engine,0,0);
    if(h5::step_head_rows(engine,n,slots.data(),tokens.data(),positions.data(),output.data(),H5_HEAD_SELECTED,rows)||h5_logits(engine,0,current.data()))return 6;
    if(trial==0){reference=current;if(argc==4){std::ofstream logits(argv[3],std::ios::binary);logits.write(reinterpret_cast<const char*>(current.data()),current.size()*sizeof(float));}}
    double maximum=0,rms=0;int differing=0;
    for(int i=0;i<h5::VOCAB;++i){double d=double(current[i])-reference[i];maximum=std::max(maximum,std::abs(d));rms+=d*d;differing+=current[i]!=reference[i];}
    printf("trial=%d prompt_tokens=%d top1=%d logit_max_abs_delta=%.9g logit_rms_delta=%.9g differing_vocab_entries=%d\n",trial,n,output[n-1],maximum,std::sqrt(rms/h5::VOCAB),differing);fflush(stdout);
  }
  llama_model_free(model);llama_backend_free();
}
