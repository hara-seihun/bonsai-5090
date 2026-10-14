#include "platform.h"
#include "llama.h"
#include "sampling_contract.h"
#include "sampling.h"
#include "head_demand.h"
#include <vector>
#include <cstring>
extern "C" {void*h5_create(const char*,int,int,int);int h5_step(void*,int,const int*,const int*,const int*,int*);int h5_step_logits(void*,int,const int*,const int*,const int*,int*,float*);int h5_logits(void*,int,float*);void h5_reset(void*,int);void h5_pretune(void*);long h5_graph_stats(void*,int);}
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
API void* qc_create(const char*p,int slots,int ctx,int rows){return h5_create(p,slots,ctx,rows);}
API int qc_step(void*e,int n,const int*s,const int*t,const int*p,int*out){return h5_step(e,n,s,t,p,out);}
API int qc_step_logits(void*e,int n,const int*s,const int*t,const int*p,int*out,float*l){return h5_step_logits(e,n,s,t,p,out,l);}
API int qc_logits(void*e,int row,float*l){return h5_logits(e,row,l);}
API void qc_reset(void*e,int s){h5_reset(e,s);}
API void qc_pretune(void*e){h5_pretune(e);}
API long qc_graph_stats(void*e,int w){return h5_graph_stats(e,w);}
API int qc_sample(void*e,int row,const H5SamplingConfig*c,uint64_t seed,uint64_t index,const unsigned char*seen,int*out){return h5_sample_target(e,c,seed,index,seen,nullptr,row,1,out);}
API int qc_sample_cpu(const float*l,int vocab,const H5SamplingConfig*c,uint64_t seed,uint64_t index,const unsigned char*seen){h5::sampling::Config x;x.temperature=c->temperature;x.top_p=c->top_p;x.min_p=c->min_p;x.presence_penalty=c->presence_penalty;x.repetition_penalty=c->repetition_penalty;x.top_k=c->top_k;std::vector<unsigned char>v(seen,seen+vocab);return h5::sampling::select(l,vocab,x,v,seed,index).token;}
API void*qc_tokenizer(const char*p){auto mp=llama_model_default_params();mp.vocab_only=true;return llama_model_load_from_file(p,mp);}
API int qc_tokenize(void*m,const char*s,int bytes,int*out,int cap){return llama_tokenize(llama_model_get_vocab((llama_model*)m),s,bytes,out,cap,true,true);}
API int qc_piece(void*m,int token,char*out,int cap){return llama_token_to_piece(llama_model_get_vocab((llama_model*)m),token,out,cap,0,true);}
API int qc_is_eog(void*m,int token){return llama_vocab_is_eog(llama_model_get_vocab((llama_model*)m),token);}
API int qc_vocab_size(void*m){return llama_vocab_n_tokens(llama_model_get_vocab((llama_model*)m));}

API int qc_prefill(void*e,int n,const int*s,const int*t,const int*p,int*out,int final){int row=n-1;return h5_step_head(e,n,s,t,p,out,H5_HEAD_SELECTED,final?1:0,final?&row:nullptr);}

extern "C" {void h5_prof_report(void*);void h5_explorer_profile(void*,int);void h5_explorer_profile_json(void*);}
API void qc_prof_report(void*e){h5_prof_report(e);}
API void qc_explorer_profile(void*e,int x){h5_explorer_profile(e,x);}
API void qc_explorer_profile_json(void*e){h5_explorer_profile_json(e);}
