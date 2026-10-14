// Length experiment: same generation loop as bench.cpp, interleaved lengths, fixed cache capacity.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
extern "C" {
void* qc_create(const char*,int,int,int);
void qc_pretune(void*);
int qc_step(void*,int,const int*,const int*,const int*,int*);
void qc_reset(void*,int);
void qc_prof_report(void*);
void qc_explorer_profile(void*,int);
void qc_explorer_profile_json(void*);
}
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv){try{
 if(argc!=5)return 2;int R=std::stoi(argv[3]),capacity_steps=std::stoi(argv[4]);
 llama_log_set([](ggml_log_level,const char*,void*){},nullptr);auto lp=llama_model_default_params();lp.vocab_only=true;lp.n_gpu_layers=0;auto*m=llama_model_load_from_file(argv[1],lp);if(!m)throw std::runtime_error("tokenizer");auto*v=llama_model_get_vocab(m);
 std::ifstream file(argv[2],std::ios::binary);std::vector<std::vector<int>>prompts;int maxp=0;
 while(prompts.size()<size_t(R)){int n;if(!file.read((char*)&n,4))throw std::runtime_error("prompt count");std::string s(n,0);file.read(s.data(),n);llama_chat_message msg{"user",s.c_str()};std::vector<char>chat(n*2+4096);int nc=llama_chat_apply_template(llama_model_chat_template(m,nullptr),&msg,1,true,chat.data(),chat.size());if(nc<0||nc>(int)chat.size())throw std::runtime_error("chat template");s=std::string(chat.data(),nc)+"<think>\n\n</think>\n\n";std::vector<int>p(s.size()+16);int nt=llama_tokenize(v,s.data(),s.size(),p.data(),p.size(),true,true);if(nt<0)throw std::runtime_error("token buffer");p.resize(nt);maxp=std::max(maxp,nt);prompts.push_back(p);}
 int context=((maxp+capacity_steps+96+255)/256)*256;
 printf("{\"kind\":\"setup\",\"context_capacity\":%d,\"max_prompt_tokens\":%d}\n",context,maxp);fflush(stdout);
 void*engine=qc_create(argv[1],R,context,256);if(!engine)throw std::runtime_error("engine");qc_pretune(engine);
 // All weights, scales, graph tuning and tokenization are outside the generation timer.
 const std::vector<int> phases={R};
 for(int active:phases){R=active;
 std::string default_sequence;for(int pass=0,passes=std::getenv("EXPLORER_PASSES")?std::atoi(std::getenv("EXPLORER_PASSES")):3;pass<passes;pass++){if(pass)default_sequence+=",";default_sequence+=std::to_string(capacity_steps);}
 std::vector<int> lengths;std::stringstream seq(std::getenv("SWEEP_SEQUENCE")?std::getenv("SWEEP_SEQUENCE"):default_sequence.c_str());std::string item;while(std::getline(seq,item,',')){int n=std::stoi(item);if(n<1||n>capacity_steps)throw std::runtime_error("length exceeds capacity");lengths.push_back(n);}
 const int passes=lengths.size();
 for(int pass=0;pass<passes;pass++){
  int steps=lengths[pass];
  std::vector<int>done(R),next(R),pos(R),slot(R);for(int r=0;r<R;r++){qc_reset(engine,r);slot[r]=r;}
  bool more=true;while(more){std::vector<int>ss,tt,pp,oo;std::vector<int>last(R,-1);more=false;for(int r=0;r<R;r++){while(done[r]<(int)prompts[r].size()&&ss.size()<256){last[r]=ss.size();ss.push_back(r);tt.push_back(prompts[r][done[r]]);pp.push_back(done[r]++);}if(done[r]<(int)prompts[r].size())more=true;}oo.resize(ss.size());if(qc_step(engine,ss.size(),ss.data(),tt.data(),pp.data(),oo.data()))throw std::runtime_error("prefill");for(int r=0;r<R;r++)if(last[r]>=0)next[r]=oo[last[r]];}
  for(int r=0;r<R;r++)pos[r]=prompts[r].size();std::vector<int>out(R);std::vector<std::vector<int>>tokens(R);
  for(int warm=0;warm<8;warm++){if(qc_step(engine,R,slot.data(),next.data(),pos.data(),out.data()))throw std::runtime_error("warm");next=out;for(auto&p:pos)p++;}
  std::vector<double> checkpoints;checkpoints.reserve((steps+127)/128);
  double start=now();for(int i=0;i<steps;i++){if(qc_step(engine,R,slot.data(),next.data(),pos.data(),out.data()))throw std::runtime_error("decode");next=out;for(int r=0;r<R;r++){pos[r]++;tokens[r].push_back(out[r]);}if((i+1)%128==0||i+1==steps)checkpoints.push_back(now()-start);}
  double elapsed=now()-start;printf("{\"pass\":%d,\"streams\":%d,\"tokens\":%d,\"generation_s\":%.9f,\"generation_tps\":%.6f,\"warmup_tokens_per_stream\":8}\n",pass,R,R*steps,elapsed,R*steps/elapsed);fflush(stdout);
  for(size_t j=0;j<checkpoints.size();j++){int end=std::min(steps,int(j+1)*128),begin=int(j)*128;double previous=j?checkpoints[j-1]:0;printf("{\"kind\":\"interval\",\"pass\":%d,\"length\":%d,\"begin\":%d,\"end\":%d,\"seconds\":%.9f,\"tps\":%.6f,\"cumulative_tps\":%.6f}\n",pass,steps,begin,end,checkpoints[j]-previous,R*(end-begin)/(checkpoints[j]-previous),R*end/checkpoints[j]);}fflush(stdout);
  if(false) {
   qc_explorer_profile(engine,1);
   for(int i=0;i<16;i++){if(qc_step(engine,R,slot.data(),next.data(),pos.data(),out.data()))throw std::runtime_error("profile");next=out;for(auto&p:pos)p++;}
   qc_explorer_profile_json(engine);qc_explorer_profile(engine,0);
  }
  if(const char*answer=std::getenv("P12_ANSWERS")){std::ofstream a(answer,std::ios::app);for(int r=0;r<R;r++){a<<"{\"pass\":"<<pass<<",\"row\":"<<r<<",\"tokens\":[";for(size_t j=0;j<tokens[r].size();j++){if(j)a<<",";a<<tokens[r][j];}a<<"]}\n";}}
  fprintf(stderr,"tokens pass=%d",pass);for(int t:tokens[0])fprintf(stderr," %d",t);fprintf(stderr,"\n");
 }}
 if(std::getenv("H5_PROF"))qc_prof_report(engine);
 return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
