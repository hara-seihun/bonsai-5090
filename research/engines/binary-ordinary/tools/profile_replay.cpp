// Replay actual prompt batches/state commits. Outputs are retained but never select work.
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include "profile_trace.h"
extern "C" {
void* h5_create(const char*,int,int,int);
void h5_pretune(void*); void h5_capture(void*,int);
void h5_reset(void*,int); void h5_keep(void*,int,int); void h5_accept(void*,int,int);
int h5_step(void*,int,const int*,const int*,const int*,int*);
long h5_graph_stats(void*,int);
}
struct Event {std::string op;std::vector<int> v;};
int main(int argc,char**argv) { try {
 if(argc!=4 && argc!=5) throw std::runtime_error("usage: profile-replay MODEL EVENT_FILE OUTPUT_PREFIX [--decode-only|--ordinary]");
 bool decode_only=argc==5 && std::string(argv[4])=="--decode-only";
 bool ordinary=argc==5 && std::string(argv[4])=="--ordinary";
 if(argc==5 && !decode_only && !ordinary)throw std::runtime_error("unknown replay option");
 bool measuring=false;
 auto begin_capture=[&](){if(cudaProfilerStart()!=cudaSuccess)throw std::runtime_error("profiler start failed");
#ifdef H5_PDLCLOCK
 h5_pdl_begin_kernels();h5_pdl_begin_fp4();h5_pdl_begin_draft();
#endif
 }; 
 std::ifstream input(argv[2]); if(!input) throw std::runtime_error("event file missing");
 std::vector<Event> events; std::string line;
 while(std::getline(input,line)) {std::istringstream s(line);Event e;s>>e.op;int x;while(s>>x)e.v.push_back(x);if(!s.eof())throw std::runtime_error("malformed replay integer");
  if(e.op=="STEP") {if(e.v.empty() || e.v[0]<=0 || e.v[0]>520 || e.v.size()!=1+3*e.v[0])throw std::runtime_error("invalid STEP");}
  else if(e.op=="RESET") {if(e.v.size()!=1 || e.v[0]!=0)throw std::runtime_error("invalid RESET");}
  else if(e.op=="KEEP" || e.op=="ACCEPT") {if(e.v.size()!=2 || e.v[0]!=0 || e.v[1]<0 || e.v[1]>8 || (e.op=="KEEP" && e.v[1]>1))throw std::runtime_error("invalid state event");}
  else throw std::runtime_error("unknown replay event"); events.push_back(std::move(e));
 }
 if(events.empty() || events[0].op!="RESET")throw std::runtime_error("replay must begin with reset");
 void*engine=h5_create(argv[1],1,4096,520);if(!engine)throw std::runtime_error("model load failed"); h5_pretune(engine);h5_capture(engine,ordinary?0:1);
 auto run=[&](std::ostream*o){unsigned round=0;for(auto&e:events){auto&v=e.v;
  if(e.op=="RESET")h5_reset(engine,v[0]);else if(e.op=="KEEP")h5_keep(engine,v[0],v[1]);else if(e.op=="ACCEPT")h5_accept(engine,v[0],v[1]);else {
   int n=v[0];std::vector<int>slots(n),toks(n),pos(n),out(n);for(int i=0;i<n;i++){slots[i]=v[1+3*i];toks[i]=v[2+3*i];pos[i]=v[3+3*i];if(slots[i]!=0 || pos[i]<0 || pos[i]>=4096)throw std::runtime_error("invalid row");}
   if(measuring && decode_only && round==1)begin_capture();
   nvtxRangePushA(round==0?"replay-prefill":"replay-target");auto start=std::chrono::steady_clock::now();int rc=h5_step(engine,n,slots.data(),toks.data(),pos.data(),out.data());auto end=std::chrono::steady_clock::now();nvtxRangePop();if(rc)throw std::runtime_error("target step failed");
   if(o){*o<<round<<","<<n<<","<<std::chrono::duration<double>(end-start).count();for(int t:out)*o<<","<<t;*o<<"\n";}round++;
  }
 }};
 run(nullptr);run(nullptr);run(nullptr);
 std::ofstream output(std::string(argv[3])+".csv");if(!output)throw std::runtime_error("output missing");
 auto check=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
 measuring=true;if(!decode_only)begin_capture();
 auto start=std::chrono::steady_clock::now();run(&output);auto end=std::chrono::steady_clock::now();
#ifdef H5_PDLCLOCK
 h5_pdl_end_kernels((std::string(argv[3])+"-kernels.csv").c_str());h5_pdl_end_fp4((std::string(argv[3])+"-fp4.csv").c_str());h5_pdl_end_draft((std::string(argv[3])+"-draft.csv").c_str());
#endif
 check(cudaProfilerStop());output.close();if(!output)throw std::runtime_error("output write failed");
 std::cout<<"{\"total_s\":"<<std::chrono::duration<double>(end-start).count()<<",\"graph_replays\":"<<h5_graph_stats(engine,0)<<",\"graph_captures\":"<<h5_graph_stats(engine,1)<<"}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
