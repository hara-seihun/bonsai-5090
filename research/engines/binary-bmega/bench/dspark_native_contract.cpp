#include "dspark_native.cuh"
#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void cuda_check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main(int argc,char**argv) {
  if(argc!=2 && argc!=4){fprintf(stderr,"usage: dspark-native-contract MODEL [TAPS_F32.bin ANCHOR_ID]\nTaps are exactly four rows of five concatenated 5120-wide post-block residuals. Without a file, the explicit fixture is deterministic synthetic taps.\n");return 1;}
  try {
    constexpr int D=5120,C=4,V=248320,Q=4,K=5*D;
    int anchor=198;
    std::vector<float> taps(C*K);
    if(argc==4){char*end=nullptr;long value=strtol(argv[3],&end,10);if(!end||*end||value<0||value>=V)throw std::runtime_error("invalid anchor");anchor=value;FILE*f=fopen(argv[2],"rb");if(!f)throw std::runtime_error("cannot open tap fixture");bool ok=fread(taps.data(),sizeof(float),taps.size(),f)==taps.size()&&fgetc(f)==EOF;fclose(f);if(!ok)throw std::runtime_error("tap fixture must have exactly four rows");}
    else for(size_t i=0;i<taps.size();++i)taps[i]=.1f*std::sin(float(i%104729)*.017f);
    for(float x:taps)if(!std::isfinite(x))throw std::runtime_error("nonfinite tap fixture");
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;
    llama_model*model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("reference model load");
    auto cp=llama_context_default_params();cp.n_ctx=64;cp.n_batch=32;cp.n_ubatch=32;cp.n_seq_max=1;cp.embeddings=false;
    llama_context*ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("reference context");
    llama_set_causal_attn(ctx,false);
    int32_t pos[C]={0,1,2,3};llama_set_dspark_ctx(ctx,taps.data(),C,K,pos);
    llama_batch batch=llama_batch_init(C+Q,0,1);batch.n_tokens=C+Q;
    for(int r=0;r<C+Q;r++){batch.token[r]=r<C?0:r==C?anchor:248319;batch.pos[r]=r;batch.n_seq_id[r]=1;batch.seq_id[r][0]=0;batch.logits[r]=r>=C;}
    if(llama_decode(ctx,batch))throw std::runtime_error("reference decode");
    std::vector<float> ref(Q*V);for(int r=0;r<Q;r++){const float*p=llama_get_logits_ith(ctx,C+r);if(!p)throw std::runtime_error("reference logits missing");std::copy_n(p,V,ref.data()+size_t(r)*V);}
    std::vector<float>a,b;if(!llama_model_dspark_get_markov(model,a,b))throw std::runtime_error("reference Markov weights missing");
    cudaStream_t stream;cuda_check(cudaStreamCreate(&stream));float*device_taps;cuda_check(cudaMalloc(&device_taps,taps.size()*4));cuda_check(cudaMemcpyAsync(device_taps,taps.data(),taps.size()*4,cudaMemcpyHostToDevice,stream));
    int native_ids[Q];
    {
      h5::NativeDspark native(argv[1],1,64,32,Q,stream);int slot=0,count=C,row=0,position=0,start=C;
      const auto& output_ids=native.output_ids();const int output_vocab=native.output_vocab();
      if(output_vocab<1||output_ids.size()!=size_t(output_vocab))throw std::runtime_error("native candidate shape");
      std::vector<float>got(size_t(Q)*output_vocab),selected_ref(got.size());
      int ref_ids[Q],previous=anchor;
      for(int r=0;r<Q;r++){double best=-INFINITY;int winner=-1;for(int i=0;i<output_vocab;i++){int t=output_ids[i];if(t<0||t>=V)throw std::runtime_error("native candidate id");selected_ref[size_t(r)*output_vocab+i]=ref[size_t(r)*V+t];if(t==248319)continue;double dot=0;for(int k=0;k<256;k++)dot+=double(a[size_t(previous)*256+k])*b[size_t(t)*256+k];double logit=ref[size_t(r)*V+t]+dot;if(logit>best||(logit==best&&t<winner)){best=logit;winner=t;}}if(winner<0)throw std::runtime_error("reference has no candidate");previous=ref_ids[r]=winner;}
      native.run(device_taps,C,1,&slot,&count,&row,&position,&anchor,&start,native_ids);
      if(native.query_rows()!=Q)throw std::runtime_error("native query shape");cuda_check(native.read_base_logits(got.data(),got.size()));
      double diff=0,ref2=0,got2=0,dot=0,maxerr=0;int matched=0;
      for(size_t i=0;i<got.size();i++){if(!std::isfinite(got[i])||!std::isfinite(selected_ref[i]))throw std::runtime_error("nonfinite logit");double e=double(got[i])-selected_ref[i];diff+=e*e;ref2+=double(selected_ref[i])*selected_ref[i];got2+=double(got[i])*got[i];dot+=double(selected_ref[i])*got[i];maxerr=std::max(maxerr,std::abs(e));}
      for(int r=0;r<Q;r++){matched+=native_ids[r]==ref_ids[r];fprintf(stderr,"row%d reference%d native%d learned-confidence%.9g\n",r,ref_ids[r],native_ids[r],native.confidence()[r]);}
      const double rrms=std::sqrt(diff/ref2),cos=dot/std::sqrt(ref2*got2);
      printf("{\"fixture\":\"%s\",\"query_rows\":4,\"output_vocab\":%d,\"proposals_matched\":%d,\"base_relative_rms\":%.17g,\"base_cosine\":%.17g,\"base_max_error\":%.17g}\n",argc==4?"file":"synthetic",output_vocab,matched,rrms,cos,maxerr);
      if(matched!=Q||rrms>=.01||cos<=.9999)throw std::runtime_error("DSpark reference comparison failed");
    }
    cuda_check(cudaFree(device_taps));cuda_check(cudaStreamDestroy(stream));llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
  }catch(const std::exception&e){fprintf(stderr,"DSparkContractError: %s\n",e.what());return 2;}
}
