#include "llama.h"
#include "head_demand.h"
#include "model.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
extern "C" {
void* h5_create(const char*,int,int,int);
void h5_reset(void*,int);
void h5_keep(void*,int,int);
void h5_pretune(void*);
int h5_logits(void*,int,float*);
}
static std::string text(const char* path){std::ifstream file(path);return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};}
int main(int argc,char** argv){
  if(argc!=6){fprintf(stderr,"usage: native-trajectory vocab|infer MODEL PROMPT TEACHER_TEXT_OR_IDS OUTPUT\n");return 2;}
  llama_log_set([](ggml_log_level,const char*,void*){},nullptr);llama_backend_init();
  auto params=llama_model_default_params();params.vocab_only=true;params.n_gpu_layers=0;
  auto* model=llama_model_load_from_file(argv[2],params);if(!model)return 3;
  auto* vocab=llama_model_get_vocab(model);
  auto encode=[&](const std::string& value){std::vector<int> ids(value.size()+16);int n=llama_tokenize(vocab,value.c_str(),value.size(),ids.data(),ids.size(),false,true);if(n<0){ids.resize(-n);n=llama_tokenize(vocab,value.c_str(),value.size(),ids.data(),ids.size(),false,true);}ids.resize(std::max(n,0));return ids;};
  if(std::string(argv[1])=="vocab"){
    auto ids=encode(text(argv[4]));if(ids.empty())return 4;ids.resize(std::min(size_t(256),ids.size()));
    std::ofstream file(argv[5],std::ios::binary);int count=ids.size();file.write(reinterpret_cast<char*>(&count),4);file.write(reinterpret_cast<const char*>(ids.data()),count*4);
    printf("prepared teacher_tokens=%d prompt_tokens=%zu\n",count,encode(text(argv[3])).size());return file?0:5;
  }
  if(std::string(argv[1])!="infer")return 2;
  auto prompt=encode(text(argv[3]));std::ifstream input(argv[4],std::ios::binary);int nt=0;input.read(reinterpret_cast<char*>(&nt),4);if(!input||nt<1||nt>256||prompt.empty())return 6;
  std::vector<int> teachers(nt);input.read(reinterpret_cast<char*>(teachers.data()),nt*4);if(!input)return 7;
  auto* engine=h5_create(argv[2],1,8192,128);if(!engine)return 8;h5_pretune(engine);h5_reset(engine,0);h5_keep(engine,0,0);
  std::vector<int> output(128),rows;int np=prompt.size();
  for(int begin=0;begin<np;begin+=128){int n=std::min(128,np-begin);std::vector<int> slots(n,0),pos(n);for(int i=0;i<n;++i)pos[i]=begin+i;rows.clear();if(begin+n==np)rows.push_back(n-1);if(h5::step_head_rows(engine,n,slots.data(),prompt.data()+begin,pos.data(),output.data(),H5_HEAD_SELECTED,rows))return 9;}
  std::vector<int> checkpoints;for(int p:{0,32,64,128,256})if(p<=nt)checkpoints.push_back(p);
  std::ofstream file(argv[5],std::ios::binary);int header[]{int(checkpoints.size()),h5::VOCAB,np,nt};file.write(reinterpret_cast<char*>(header),sizeof header);std::vector<float> logits(h5::VOCAB);
  auto save=[&](int position){if(h5_logits(engine,0,logits.data()))return false;file.write(reinterpret_cast<char*>(&position),4);file.write(reinterpret_cast<const char*>(logits.data()),logits.size()*4);printf("saved continuation_position=%d top1=%d\n",position,int(std::max_element(logits.begin(),logits.end())-logits.begin()));fflush(stdout);return bool(file);};
  if(!save(0))return 10;
  for(int i=0;i<nt;++i){int slot=0,pos=np+i,out=0;h5_keep(engine,0,0);if(h5::step_head_rows(engine,1,&slot,&teachers[i],&pos,&out,H5_HEAD_SELECTED,{0}))return 11;if(std::find(checkpoints.begin(),checkpoints.end(),i+1)!=checkpoints.end()&&!save(i+1))return 12;}
  llama_model_free(model);llama_backend_free();
}
