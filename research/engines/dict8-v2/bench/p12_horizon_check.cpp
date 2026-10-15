#include <vector>
#include <array>
#include <map>
#include <cstdio>
#include <stdexcept>
extern "C" {
void* h5_create_draft(const char*,int,int,int,int);
int h5_draft_load(void*,const char*);
int h5_set_draft_count(void*,int);
int h5_draft(void*,int,const int*,const int*,const int*,const int*,const int*,const int*,int*);
int h5_step(void*,int,const int*,const int*,const int*,int*);
void h5_reset(void*,int);void h5_capture(void*,int);void h5_keep(void*,int,int);
long h5_graph_stats(void*,int);
}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: p12-horizon-check MODEL DRAFTER");
 void* e=h5_create_draft(argv[1],1,128,64,32);if(!e||h5_draft_load(e,argv[2]))throw std::runtime_error("load");
 if(!h5_set_draft_count(e,0)||!h5_set_draft_count(e,33))throw std::runtime_error("invalid horizon admitted");
 std::map<int,std::vector<int>> ref;
 h5_reset(e,0);h5_keep(e,0,0);h5_capture(e,1);
 int slot[8]={},pos[8],tok[8],out[8];for(int i=0;i<8;i++){pos[i]=i;tok[i]=1000+17*i;}
 if(h5_step(e,8,slot,tok,pos,out))throw std::runtime_error("prefix");
 int mismatches=0;
 for(int n:{15,15,15,15,3,7,31,16,3,7,31,16,15,3,7,31,16,15,3,7,31,16}){
  if(h5_set_draft_count(e,n))throw std::runtime_error("count");
  int zero=0,eight=8,anchor=out[7];std::array<int,40> drafts;drafts.fill(-1234567);
  if(h5_draft(e,1,&zero,&eight,&zero,&zero,&anchor,&eight,drafts.data()))throw std::runtime_error("draft");
  for(int i=n;i<40;i++)if(drafts[i]!=-1234567)throw std::runtime_error("draft output overrun");
  std::vector<int> got(drafts.begin(),drafts.begin()+n);
  if(ref.count(n)&&ref[n]!=got){mismatches++;printf("MISMATCH n=%d anchor=%d ref:",n,anchor);for(int x:ref[n])printf(" %d",x);printf(" got:");for(int x:got)printf(" %d",x);puts("");}
  ref[n]=got;printf("dynamic query %d anchor=%d sentinel OK\n",n,anchor);fflush(stdout);
 }
 if(h5_graph_stats(e,0)<1)throw std::runtime_error("no graph replay");
 printf("dynamic horizons capacity/sentinel pass; repeated mismatches=%d graph replays=%ld\n",mismatches,h5_graph_stats(e,0));return mismatches?1:0;
}catch(const std::exception&x){fprintf(stderr,"%s\n",x.what());return 1;}}
