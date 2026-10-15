#pragma once
#include "two_path_policy.h"
#include "state_format.h"
#include <random>

extern "C" {
int h5_clone_slot(void*,int,int,int);
int h5_draft_two(void*,int,const int*,const int*,const int*,const int*,const int*,const int*,int*,int*,int*,int);
}
static bool path_eos(int t) { return t==248046||t==248044; }
static bool path_prefill(void* e,int slot,const std::vector<int>& p,int& anchor) {
  h5_reset(e,slot);h5_keep(e,slot,0);
  for(int off=0;off<(int)p.size();off+=512) {
    int n=std::min(512,(int)p.size()-off);std::vector<int>s(n,slot),pos(n),out(n);
    for(int i=0;i<n;i++)pos[i]=off+i;
    if(h5_step(e,n,s.data(),p.data()+off,pos.data(),out.data()))return false;
    anchor=out.back();
  }
  return !p.empty();
}
static std::vector<int> path_greedy(void*e,const std::vector<int>&p,int n) {
  int t;if(!path_prefill(e,2,p,t))return {};
  std::vector<int> g{t};
  while((int)g.size()<n&&!path_eos(g.back())) {
    int s=2,pos=p.size()+g.size()-1,o;
    if(h5_step(e,1,&s,&g.back(),&pos,&o))return {};
    g.push_back(o);
  }
  return g;
}
static int run_pathbench(int argc,char**argv,void** cache=nullptr,const char* run="single",int warm_tokens=0) {
  if(argc<9){fprintf(stderr,"usage: h5 pathbench MODEL DRAFTER PROMPTS.bin OUT.bin PATHS DEPTH MAXNEW [GREEDY_GUARD]\n");return 1;}
  int paths=atoi(argv[6]),depth=atoi(argv[7]),maxnew=atoi(argv[8]);bool guard=argc>9&&atoi(argv[9]);
  if((paths!=1&&paths!=2)||(depth!=4&&depth!=8)||maxnew<2)return 1;
  int fork_span=std::min(depth-1,getenv("H5_TWO_FORK_SPAN")?atoi(getenv("H5_TWO_FORK_SPAN")):3);
  if(fork_span<1||fork_span>7)return 1;
  Tok tk(argv[2]);std::vector<std::vector<int>> prompts;
  FILE* pf=fopen(argv[4],"rb");if(!pf)return 1;
  int sz;while(fread(&sz,4,1,pf)==1){if(sz<0||sz>1000000){fclose(pf);return 1;}std::string s(sz,0);if(fread(s.data(),1,sz,pf)!=(size_t)sz){fclose(pf);return 1;}prompts.push_back(tk.enc(getenv("H5_PATH_RAW")?s:tk.chat(s)));}fclose(pf);
  if(prompts.empty())return 1;
  const int emit_cap=warm_tokens?std::min(maxnew,warm_tokens):maxnew;
  int maxp=0;for(auto&p:prompts)maxp=std::max(maxp,(int)p.size());
  const int ctx=(maxp+maxnew+64+31)/32*32;
  void*e=cache?*cache:nullptr;
  if(!e){e=h5_create(argv[2],guard?3:2,ctx,512);if(!e||h5_draft_load(e,argv[3]))return 1;if(cache)*cache=e;}
  FILE* of=fopen(argv[5],"wb"),*tf=fopen((std::string(argv[5])+".tokens").c_str(),"w");if(!of||!tf)return 1;
  FILE* trace=getenv("H5_PATH_TRACE")?fopen(getenv("H5_PATH_TRACE"),"w"):nullptr;
  if(getenv("H5_PATH_TRACE")&&!trace)return 1;
  long total_tokens=0,total_rounds=0;double total_s=0;
  for(size_t req=0;req<(warm_tokens?std::min(prompts.size(),(size_t)1):prompts.size());req++) {
    auto&p=prompts[req];std::vector<int> ref;
    h5_capture(e,0);if(guard){ref=path_greedy(e,p,emit_cap);if(ref.empty())return 2;}
    h5_capture(e,1);h5_reset(e,0);h5_reset(e,1);h5_keep(e,0,0);h5_keep(e,1,0);
    int active=0,len=p.size(),anchor=-1,dr[7],alt[7],fork=0;
    double pre0=now_s();
    for(int off=0;off<(int)p.size();off+=512) {
      int n=std::min(512,(int)p.size()-off),last=off+n==(int)p.size();
      std::vector<int>s(n,active),pos(n),out(n);for(int i=0;i<n;i++)pos[i]=off+i;
      if(h5_step(e,n,s.data(),p.data()+off,pos.data(),out.data()))return 2;
      int an=last?out.back():-1,ir=0,ip=off,start=len;
      int rc=paths==2?h5_draft_two(e,1,&active,&n,&ir,&ip,&an,&start,dr,alt,&fork,fork_span):h5_draft(e,1,&active,&n,&ir,&ip,&an,&start,dr);
      if(rc){fprintf(stderr,"prefill ingestion failed %d\n",rc);return 2;}if(last)anchor=an;
    }
    double pre_s=now_s()-pre0;
    std::vector<int> gen{anchor};long rounds=0,accepted=0,alt_selected=0,alt_rescues=0,clone_bytes=0;double clone_s=0,verify_s=0,draft_s=0,ctx_sum=0;
    double t0=now_s();
    while((int)gen.size()<emit_cap&&!path_eos(gen.back())) {
      h5_keep(e,active,1);int other=active^1;
      double t=now_s();if(paths==2&&h5_clone_slot(e,active,other,len)){fprintf(stderr,"clone failed\n");return 2;}clone_s+=now_s()-t;
      std::vector<int> rs(paths*depth),rt(paths*depth),rp(paths*depth),ro(paths*depth);
      for(int b=0;b<paths;b++)for(int k=0;k<depth;k++){int r=b*depth+k;rs[r]=b?other:active;rt[r]=k?(b?alt[k-1]:dr[k-1]):anchor;rp[r]=len+k;}
      t=now_s();if(h5_step(e,rs.size(),rs.data(),rt.data(),rp.data(),ro.data()))return 2;verify_s+=now_s()-t;
      int chosen=0,a=h5::accepted_prefix(rt.data(),ro.data(),depth);
      if(paths==2){auto c=h5::choose_two_path(rt.data(),ro.data(),depth,fork);if(c.error!=h5::PathError::None){fprintf(stderr,"shared causal prefix guard failed req%zu round%ld error%d\n",req,rounds,(int)c.error);return 3;}chosen=c.path;a=c.matched;}
      int before=gen.size(),row0=chosen*depth,ipos=len,old_active=active,base_a=h5::accepted_prefix(rt.data(),ro.data(),depth);
      if(chosen){active=other;alt_selected++;if(a>base_a)alt_rescues++;}
      h5::PathChoice choice{h5::PathError::None,chosen,a};
      int bonus=h5::selected_target(ro.data(),depth,fork,choice,a);
      for(int k=0;k<=a&&(int)gen.size()<emit_cap;k++){int tok=h5::selected_target(ro.data(),depth,fork,choice,k);gen.push_back(tok);if(path_eos(tok))break;}
      h5_accept(e,active,a+1);int ni=a+1;len+=ni;anchor=bonus;accepted+=a;ctx_sum+=ipos;rounds++;
      // Clone traffic is reported separately; wall decode includes it, including the blocking completion.
      if(paths==2) {
        const long state_es=getenv("H5_STATE")&&!strcmp(getenv("H5_STATE"),"f32")?4:2;
        long kv_es=getenv("H5_KV")&&strcmp(getenv("H5_KV"),"fp8")?2:1;
        const size_t recurrent_bytes = getenv("H5_STATE") && !strcmp(getenv("H5_STATE"), "q8")
          ? h5::gdn_state_slot_bytes(h5::StateFormat::Q8) : 48L*48*128*128*state_es;
        clone_bytes+=recurrent_bytes+48L*3*10240*4+48L*2*8*(2048+6144+96)*state_es+48L*8*10240*state_es;
        int copy_n=getenv("H5_CLONE_FULL_KV")&&atoi(getenv("H5_CLONE_FULL_KV"))?ctx:ipos;
        clone_bytes+=16L*4*256*(copy_n*kv_es+((copy_n+31)/32*32)*kv_es)+(kv_es==1?16L*4*copy_n*8:0)+2L*5*8*1056*128*2;
      }
      if(trace){fprintf(trace,"{\"request\":%zu,\"round\":%ld,\"position\":%d,\"fork\":%d,\"chosen\":%d,\"base_accept\":%d,\"accept\":%d,\"source_slot\":%d,\"selected_slot\":%d,\"ing_row0\":%d,\"ing_rows\":%d,\"emitted\":%zu}\n",req,rounds,ipos,fork,chosen,base_a,a,old_active,active,row0,ni,gen.size()-before);}
      if((int)gen.size()>=emit_cap||path_eos(gen.back()))break;
      t=now_s();int rc=paths==2?h5_draft_two(e,1,&active,&ni,&row0,&ipos,&anchor,&len,dr,alt,&fork,fork_span):h5_draft(e,1,&active,&ni,&row0,&ipos,&anchor,&len,dr);
      if(rc){fprintf(stderr,"selected feature ingestion failed %d\n",rc);return 3;}draft_s+=now_s()-t;
    }
    double elapsed=now_s()-t0;int same=0;
    if(guard){while(same<(int)gen.size()&&same<(int)ref.size()&&gen[same]==ref[same])same++;}
    bool exact=!guard||(same==(int)gen.size()&&gen.size()==ref.size());
    std::string text;for(int tok:gen){fprintf(tf,"%d ",tok);if(!path_eos(tok))text+=tk.dec(tok);}fprintf(tf,"\n");int bytes=text.size();fwrite(&bytes,4,1,of);fwrite(text.data(),1,text.size(),of);
    printf("{\"run\":\"%s\",\"request\":%zu,\"paths\":%d,\"depth\":%d,\"nodes\":%d,\"prompt_tokens\":%zu,\"generated\":%zu,\"decode_tokens\":%zu,\"rounds\":%ld,\"accepted_drafts\":%ld,\"emitted_per_round\":%.5f,\"alternate_selected\":%ld,\"alternate_rescues\":%ld,\"prefill_s\":%.6f,\"decode_s\":%.6f,\"clone_s\":%.6f,\"verify_s\":%.6f,\"draft_s\":%.6f,\"clone_bytes\":%ld,\"tok_s\":%.3f,\"mean_context\":%.2f,\"greedy_guard\":%s,\"greedy_matches\":%d,\"exact\":%s}\n",run,req,paths,depth,paths*depth,p.size(),gen.size(),gen.size()-1,rounds,accepted,rounds?(double)(gen.size()-1)/rounds:0.,alt_selected,alt_rescues,pre_s,elapsed,clone_s,verify_s,draft_s,clone_bytes,(gen.size()-1)/std::max(elapsed,1e-9),rounds?ctx_sum/rounds:0.,guard?"true":"false",same,guard?(exact?"true":"false"):"null");fflush(stdout);
    if(!exact){fprintf(stderr,"exact-greedy guard differs at token%d req%zu\n",same,req);fclose(of);fclose(tf);if(trace)fclose(trace);return 3;}
    total_tokens+=gen.size()-1;total_rounds+=rounds;total_s+=elapsed;
  }
  if(trace)fclose(trace);fclose(of);fclose(tf);
  printf("{\"run\":\"%s\",\"summary\":true,\"paths\":%d,\"depth\":%d,\"decode_tokens\":%ld,\"rounds\":%ld,\"decode_s\":%.6f,\"tok_s\":%.3f,\"emitted_per_round\":%.5f}\n",run,paths,depth,total_tokens,total_rounds,total_s,total_tokens/std::max(total_s,1e-9),(double)total_tokens/std::max(total_rounds,1L));return 0;
}
static int run_pathmatrix(int argc,char**argv) {
  if(argc<7){fprintf(stderr,"usage: h5 pathmatrix MODEL DRAFTER PROMPTS.bin OUT_PREFIX MAXNEW [GREEDY_GUARD]\n");return 1;}
  void* cache=nullptr;
  const int paths[]={1,2,2,1,2,2,1},depth[]={8,4,8,8,4,8,8};
  const char* run[]={"warm-base","warm-eight","warm-sixteen","base-before","eight","sixteen","base-after"};
  for(int c=0;c<7;c++){
    std::vector<std::string> args={argv[0],"pathbench",argv[2],argv[3],argv[4],
      std::string(argv[5])+"-"+run[c]+"-p"+std::to_string(paths[c])+"-d"+std::to_string(depth[c])+".bin",
      std::to_string(paths[c]),std::to_string(depth[c]),argv[6],argc>7?argv[7]:"0"};
    std::vector<char*> av;for(auto&s:args)av.push_back(s.data());
    int rc=run_pathbench(av.size(),av.data(),&cache,run[c],c<3?32:0);if(rc)return rc;
  }
  return 0;
}
static int run_pathguard(int argc,char**argv) {
  if(argc<7){fprintf(stderr,"usage: h5 pathguard MODEL PROMPT TOKENS DEPTH CORRUPT_PCT\n");return 1;}
  int n=atoi(argv[4]),depth=atoi(argv[5]),corrupt=atoi(argv[6]);if(n<2||(depth!=4&&depth!=8)||corrupt<0||corrupt>100)return 1;
  Tok tk(argv[2]);auto p=tk.enc(tk.chat(argv[3]));int ctx=(p.size()+n+96)/32*32;
  void*e=h5_create(argv[2],3,ctx,512);if(!e)return 2;
  auto ref=path_greedy(e,p,n);if(ref.empty())return 2;
  int anchor;if(!path_prefill(e,0,p,anchor))return 2;
  std::vector<int>gen{anchor};int active=0,len=p.size();long rounds=0,swaps=0;std::mt19937 rng(5090);
  const bool clone_back=getenv("H5_PATH_CLONE_BACK") && atoi(getenv("H5_PATH_CLONE_BACK"));
  while(gen.size()<ref.size()) {
    int other=active^1;h5_keep(e,active,1);if(h5_clone_slot(e,active,other,len))return 2;
    int fork=rounds%std::min(3,depth-1);std::vector<int>rs(2*depth),rt(2*depth),rp(2*depth),ro(2*depth);
    for(int k=0;k<depth;k++) {
      int idx=gen.size()-1+k,t=k?(idx<(int)ref.size()?ref[idx]:1000):gen.back();
      int a=t,b=t;
      if(k&&k<=fork&&rng()%100<(unsigned)corrupt)a=b=(t+17)%248000;
      if(k==fork+1){if(rounds%2)a=t,b=(t+17)%248000;else a=(t+17)%248000,b=t;}
      if(k>fork+1){if(rng()%100<(unsigned)corrupt)a=(t+17)%248000;if(rng()%100<(unsigned)corrupt)b=(t+29)%248000;}
      rs[k]=active;rs[depth+k]=other;rt[k]=a;rt[depth+k]=b;rp[k]=rp[depth+k]=len+k;
    }
    if(h5_step(e,2*depth,rs.data(),rt.data(),rp.data(),ro.data()))return 2;
    auto c=h5::choose_two_path(rt.data(),ro.data(),depth,fork);if(c.error!=h5::PathError::None){fprintf(stderr,"clone/replay shared guard error%d round%ld len%d fork%d active%d\n",(int)c.error,rounds,len,fork,active);for(int k=0;k<depth;k++)fprintf(stderr,"  row%d input%d/%d target%d/%d\n",k,rt[k],rt[depth+k],ro[k],ro[depth+k]);return 3;}
    if(c.path){active=other;swaps++;}
    for(int k=0;k<=c.matched&&gen.size()<ref.size();k++)gen.push_back(h5::selected_target(ro.data(),depth,fork,c,k));
    h5_accept(e,active,c.matched+1);len+=c.matched+1;rounds++;
    if(clone_back && active!=0){if(h5_clone_slot(e,active,0,len))return 2;active=0;}
  }
  int same=0;while(same<(int)gen.size()&&gen[same]==ref[same])same++;
  printf("{\"guard\":\"clone_pending_corrupt_replay\",\"depth\":%d,\"corrupt\":%d,\"tokens\":%zu,\"matches\":%d,\"rounds\":%ld,\"slot_swaps\":%ld,\"clone_back\":%s,\"pass\":%s}\n",depth,corrupt,gen.size(),same,rounds,swaps,clone_back?"true":"false",same==(int)gen.size()?"true":"false");
  return same==(int)gen.size()?0:3;
}
