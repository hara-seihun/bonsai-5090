// Design search (CPU, discovery only): byte transforms whose e2m1 nibble windows contain an affine 3^5 grid.
// 5 windows = pass1 {lo,hi} of tau1(b), pass2 {lo,hi} of tau2(b), one window of tau3(b).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <map>
#include <array>
#include <cstdlib>
#include <chrono>
#include <algorithm>
static const int MAG[8]={0,1,2,3,4,6,8,12}; // e2m1 *2 (integers)
int e2(int nib){int v=MAG[nib&7];return (nib&8)?-v:v;}
struct Tr{int k,x,a;};
int apply(Tr t,int b){int r=((b<<t.k)|(b>>(8-t.k)))&255; if(t.k==0)r=b; return (r^t.x)&t.a;}
int main(int argc,char**argv){
  int part=atoi(argv[1]),parts=atoi(argv[2]);
  std::vector<Tr> L; int xs[]={0,0x11,0x22,0x44,0x66,0x33,0x55,0x77}; int as[]={0xff,0x77,0xbb,0xdd,0xee,0x33,0x99,0x66,0x55,0xaa,0x37,0x73,0x3b,0xb3};
  for(int k=0;k<8;k++)for(int x:xs)for(int a:as)L.push_back({k,x&a,a});
  // dedupe by window function pair
  std::map<std::vector<int>,int> seen; std::vector<Tr> U;
  for(auto t:L){std::vector<int> f;for(int b=0;b<256;b++){int r=apply(t,b);f.push_back(e2(r&15)*64+e2(r>>4));} if(!seen.count(f)){seen[f]=1;U.push_back(t);}}
  fprintf(stderr,"transforms %zu\n",U.size());
  auto t0=std::chrono::steady_clock::now(); long sets=0,found=0;
  Tr id{0,0,0xff};
  for(size_t i=0;i<U.size();i++){ if((int)(i%parts)!=part)continue;
   for(size_t j=0;j<U.size();j++)for(int half=0;half<2;half++){
    sets++;
    // points
    std::array<int,5> P[256]; 
    for(int b=0;b<256;b++){int r1=apply(id,b),r2=apply(U[i],b),r3=apply(U[j],b);P[b]={e2(r1&15),e2(r1>>4),e2(r2&15),e2(r2>>4),half?e2(r3>>4):e2(r3&15)};}
    std::map<std::array<int,5>,int> idx; for(int b=0;b<256;b++)idx[P[b]]=b;
    if(idx.size()<243)continue;
    // try centers
    for(auto &cp:idx){auto o=cp.first; std::vector<std::array<int,5>> G;
      for(auto &q:idx){std::array<int,5> g,m;bool nz=false;for(int d=0;d<5;d++){g[d]=q.first[d]-o[d];m[d]=o[d]-g[d];if(g[d])nz=true;} if(!nz)continue; if(idx.count(m)&&q.first>m)G.push_back(g);}
      if(G.size()<5)continue;
      // backtrack choose 5 generators forming a full grid
      std::vector<std::array<int,5>> pts{o}; std::vector<int> chosen;
      std::function<bool(int,std::vector<std::array<int,5>>&)> rec=[&](int s,std::vector<std::array<int,5>>& cur)->bool{
        if(chosen.size()==5){return true;}
        for(int c=s;c<(int)G.size();c++){std::vector<std::array<int,5>> nx; bool ok=true;
          for(auto &p:cur){for(int sg=-1;sg<=1;sg+=2){std::array<int,5> q;for(int d=0;d<5;d++)q[d]=p[d]+sg*G[c][d]; if(!idx.count(q)){ok=false;break;} nx.push_back(q);} if(!ok)break;}
          if(!ok)continue; nx.insert(nx.end(),cur.begin(),cur.end()); std::sort(nx.begin(),nx.end()); if(std::unique(nx.begin(),nx.end())!=nx.end())continue;
          chosen.push_back(c); if(rec(c+1,nx))return true; chosen.pop_back();}
        return false;};
      if(rec(0,pts)){found++; printf("FOUND t2={%d,%d,%d} t3={%d,%d,%d} half=%d center=",U[i].k,U[i].x,U[i].a,U[j].k,U[j].x,U[j].a,half);for(int d=0;d<5;d++)printf("%d,",o[d]);printf(" gens=");for(int c:chosen){for(int d=0;d<5;d++)printf("%d,",G[c][d]);printf(" ");}printf("\n");fflush(stdout);break;}
    }
    if(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>48){fprintf(stderr,"timeout i=%zu sets=%ld found=%ld\n",i,sets,found);return 0;}
  }}
  fprintf(stderr,"done sets=%ld found=%ld\n",sets,found);
}
