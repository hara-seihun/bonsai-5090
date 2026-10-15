#include <vector>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <chrono>
#include <cstdlib>
#include <cuda_profiler_api.h>
extern "C" {
void* h5_create_draft(const char*,int,int,int,int);int h5_draft_load(void*,const char*);
int h5_draft(void*,int,const int*,const int*,const int*,const int*,const int*,const int*,int*);
int h5_draft_chain(void*,int,const int*,const int*,const int*,const int*,const int*,const int*,int*);
int h5_step(void*,int,const int*,const int*,const int*,int*);
void h5_reset(void*,int);void h5_capture(void*,int);void h5_keep(void*,int,int);void h5_accept(void*,int,int);
long h5_graph_stats(void*,int);void h5_pretune(void*);
}
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){try{
 if(argc<3)throw std::runtime_error("usage: p12-chain-check MODEL DRAFTER [route -1=compare,0=ordinary,1=chain] [rounds]");
 int route=argc>3?atoi(argv[3]):-1,rounds=argc>4?atoi(argv[4]):32;
 void*e=h5_create_draft(argv[1],1,1024,128,7);if(!e||h5_draft_load(e,argv[2]))throw std::runtime_error("load");h5_pretune(e);
 std::vector<int>reference;int differences=0;
 for(int pass=0;pass<(route<0?2:1);++pass){
  bool chain=route<0?pass:route;h5_reset(e,0);h5_keep(e,0,0);h5_capture(e,1);
  int ss[8]={},pos[8],tok[8],out[8];for(int i=0;i<8;++i){pos[i]=i;tok[i]=1000+17*i;}
  if(h5_step(e,8,ss,tok,pos,out))throw std::runtime_error("prefill");
  int zero=0,len=8,ing=8,ingpos=0,anchor=out[7],dr[7],emitted=0;double sec=0;
  for(int j=0;j<rounds+4;++j){
   if(j==4&&route>=0)cudaProfilerStart();
   auto t=Clock::now();auto draft=chain?h5_draft_chain:h5_draft;
   if(draft(e,1,&zero,&ing,&zero,&ingpos,&anchor,&len,dr))throw std::runtime_error("draft");
   h5_keep(e,0,1);tok[0]=anchor;for(int k=0;k<8;k++){pos[k]=len+k;if(k)tok[k]=dr[k-1];}
   if(h5_step(e,8,ss,tok,pos,out))throw std::runtime_error("verify/cache");
   int a=0;while(a<7&&dr[a]==out[a]){++a;if(dr[a-1]==248046||dr[a-1]==248044)break;}
   h5_accept(e,0,a+1);ing=a+1;ingpos=len;len+=a+1;anchor=out[a];
   if(route<0)for(int k=0;k<8;++k){if(!pass)reference.push_back(out[k]);else differences+=reference[j*8+k]!=out[k];}
   if(j>=4){sec+=std::chrono::duration<double>(Clock::now()-t).count();emitted+=a+1;}
  }
  if(route>=0)cudaProfilerStop();
  printf("route=%d rounds=%d seconds=%.9f round_ms=%.6f emitted=%d native_tps=%.3f graph_hits=%ld\n",chain,rounds,sec,sec*1000/rounds,emitted,emitted/sec,h5_graph_stats(e,0));fflush(stdout);
 }
 printf("mismatched_target_ids=%d\n",differences);return differences?2:0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
