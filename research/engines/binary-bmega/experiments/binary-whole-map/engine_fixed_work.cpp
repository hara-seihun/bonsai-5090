#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" {
void* h5_create(const char*,int,int,int);
int h5_step(void*,int,const int*,const int*,const int*,int*);
void h5_keep(void*,int,int);
double h5_last_step_s(void*);
}
int main(int argc,char** argv) {
  if(argc!=3) { fprintf(stderr,"usage: engine-fixed-work MODEL ROWS\n"); return 2; }
  int rows=atoi(argv[2]); if(rows!=16 && rows!=32) return 2;
  auto* engine=h5_create(argv[1],rows,512,rows); if(!engine) return 3;
  std::vector<int> slots(rows),tokens(rows,198),positions(rows),output(rows);
  for(int i=0;i<rows;++i) slots[i]=i;
  std::vector<double> gpu,wall;
  for(int p=0;p<64;++p) {
    std::fill(positions.begin(),positions.end(),p);
    for(int i=0;i<rows;++i) h5_keep(engine,i,0);
    auto before=std::chrono::steady_clock::now();
    if(h5_step(engine,rows,slots.data(),tokens.data(),positions.data(),output.data())) return 4;
    auto after=std::chrono::steady_clock::now();
    if(p>=32) { gpu.push_back(h5_last_step_s(engine)); wall.push_back(std::chrono::duration<double>(after-before).count()); }
  }
  printf("{\"rows\":%d,\"token\":198,\"prefix\":32,\"steps\":32,\"context_allocated\":512,\"gpu_seconds\":[",rows);
  for(size_t i=0;i<gpu.size();++i) printf("%s%.9g",i?",":"",gpu[i]);
  printf("],\"wall_seconds\":["); for(size_t i=0;i<wall.size();++i) printf("%s%.9g",i?",":"",wall[i]);
  std::sort(gpu.begin(),gpu.end()); std::sort(wall.begin(),wall.end());
  printf("],\"gpu_median_tok_s\":%.9g,\"wall_median_tok_s\":%.9g}\n",rows/((gpu[15]+gpu[16])/2),rows/((wall[15]+wall[16])/2));
}
