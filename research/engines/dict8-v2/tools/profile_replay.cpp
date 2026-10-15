// Replay actual prompt batches/state commits. Outputs are retained but never select work.
#if defined(H5_SMLIT) && !defined(H5_REPLAY_CPU_ONLY)
#include "smlit.cuh"
extern "C" void h5_lit_set_kernels(h5::lit::Buf*);extern "C" void h5_lit_set_fp4(h5::lit::Buf*);
extern "C" void h5_lit_set_draft(h5::lit::Buf*);extern "C" void h5_lit_set_gmem(h5::lit::Buf*);extern "C" void h5_lit_set_sk(h5::lit::Buf*);
#endif
extern "C" void h5_set_prewait_cap(int);
extern "C" void h5_set_launch_carveout(int);
extern "C" void h5_set_prep_wide(int);
extern "C" void h5_set_fuse_gu(int);
extern "C" void h5_set_variant_kernels(unsigned);extern "C" void h5_set_variant_fp4(unsigned);extern "C" void h5_set_variant_draft(unsigned);
#include <charconv>
#include <climits>
#include <set>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef H5_REPLAY_CPU_ONLY
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include "profile_trace.h"
#endif
#include "graph_probe.h"
#include "head_demand.h"
#include <iomanip>
extern "C" {
void* h5_create(const char*,int,int,int);
void h5_pretune(void*); void h5_capture(void*,int);
void h5_reset(void*,int); void h5_keep(void*,int,int); void h5_accept(void*,int,int);
int h5_step(void*,int,const int*,const int*,const int*,int*);
long h5_graph_stats(void*,int);
}
struct Event {std::string op;std::vector<int> v;bool prefill=false;};
static int replay_integer(const std::string& text) {
 int value=0;const auto rc=std::from_chars(text.data(),text.data()+text.size(),value);
 if(rc.ec!=std::errc{} || rc.ptr!=text.data()+text.size())throw std::runtime_error("invalid replay integer: "+text);
 return value;
}
int main(int argc,char**argv) { try {
 if(argc<4) throw std::runtime_error("usage: profile-replay MODEL EVENT_FILE OUTPUT_PREFIX [--decode-only] [--ordinary] [--graph-upload launch|capture --graph-timing off|wall-thread] [--graph-nodes native|kernel-memory] [--passes N] [--head-rows all|demand] [--slots N] [--ctx N] [--dry-run]");
 bool decode_only=false,ordinary=false,passes_set=false,slots_set=false,ctx_set=false,dry_run=false;
 // Omitted capacities retain the established one-slot replay contract.
 int upload=-1,timing=-1,nodes=-1,passes=1,slots=1,ctx=4096;
 constexpr int rowcap=520;
 H5HeadMode head_mode=H5_HEAD_SELECTED; bool head_mode_set=false;
 for(int i=4;i<argc;i++) {
  std::string a=argv[i];
  if(a=="--decode-only" && !decode_only)decode_only=true;
  else if(a=="--ordinary" && !ordinary)ordinary=true;
  else if(a=="--graph-upload" && upload==-1 && i+1<argc){std::string v=argv[++i];if(v=="launch")upload=H5_GRAPH_UPLOAD_AT_LAUNCH;else if(v=="capture")upload=H5_GRAPH_UPLOAD_AT_CAPTURE;else throw std::runtime_error("invalid graph upload policy");}
  else if(a=="--graph-timing" && timing==-1 && i+1<argc){std::string v=argv[++i];if(v=="off")timing=H5_GRAPH_TIMING_OFF;else if(v=="wall-thread")timing=H5_GRAPH_TIMING_WALL_THREAD;else throw std::runtime_error("invalid graph timing mode");}
  else if(a=="--graph-nodes" && nodes==-1 && i+1<argc){std::string v=argv[++i];if(v=="native")nodes=H5_GRAPH_NODES_NATIVE;else if(v=="kernel-memory")nodes=H5_GRAPH_NODES_KERNEL_MEMORY;else throw std::runtime_error("invalid graph nodes mode");}
  else if(a=="--head-rows" && !head_mode_set && i+1<argc){head_mode_set=true;if(h5::parse_head_mode(argv[++i],head_mode)!=H5_HEAD_OK)throw std::runtime_error("invalid head rows policy");}
  else if(a=="--passes" && !passes_set && i+1<argc){passes_set=true;passes=replay_integer(argv[++i]);if(passes<1||passes>100)throw std::runtime_error("invalid passes");}
  else if(a=="--slots" && !slots_set && i+1<argc){slots_set=true;slots=replay_integer(argv[++i]);if(slots<1)throw std::runtime_error("invalid slots");}
  else if(a=="--ctx" && !ctx_set && i+1<argc){ctx_set=true;ctx=replay_integer(argv[++i]);if(ctx<1||ctx>INT_MAX-31)throw std::runtime_error("invalid ctx");}
  else if(a=="--dry-run" && !dry_run)dry_run=true;
  else throw std::runtime_error("unknown or repeated replay option: "+a);
 }
 if((upload==-1)!=(timing==-1))throw std::runtime_error("graph upload and timing must both be explicit");
 if(nodes!=-1 && upload==-1)throw std::runtime_error("graph nodes require explicit upload and timing");
 std::ifstream input(argv[2]); if(!input) throw std::runtime_error("event file missing");
 std::vector<Event> events; std::string line;std::set<int> reset_slots;size_t line_number=0,steps=0,prefills=0;
 while(std::getline(input,line)) {++line_number;try {std::istringstream s(line);Event e;s>>e.op;std::string word;while(s>>word)e.v.push_back(replay_integer(word));
  if(e.op=="STEP") {if(e.v.empty() || e.v[0]<=0 || e.v[0]>rowcap || e.v.size()!=size_t(1+3*e.v[0]))throw std::runtime_error("invalid STEP");}
  else if(e.op=="HEADSTEP") {
   if(e.v.empty() || e.v[0]<=0 || e.v[0]>rowcap)throw std::runtime_error("invalid HEADSTEP rows");
   const int end=1+3*e.v[0];
   if((int)e.v.size()<=end || e.v[end]<0 || e.v[end]>e.v[0] || (int)e.v.size()!=end+1+e.v[end])throw std::runtime_error("invalid HEADSTEP demand");
   if(h5::HeadDemand::selected(e.v[end],e.v[end]?e.v.data()+end+1:nullptr).validate(e.v[0],false)!=H5_HEAD_OK)throw std::runtime_error("invalid HEADSTEP selection");
  }
  else if(e.op=="RESET") {if(e.v.size()!=1)throw std::runtime_error("invalid RESET");}
  else if(e.op=="KEEP" || e.op=="ACCEPT") {if(e.v.size()!=2 || e.v[1]<0 || e.v[1]>8 || (e.op=="KEEP" && e.v[1]>1))throw std::runtime_error("invalid state event");}
  else throw std::runtime_error("unknown replay event");
  auto check_slot=[&](int slot,bool reset){
   if(slot<0 || slot>=slots)throw std::runtime_error("slot out of range: "+std::to_string(slot));
   if(!reset && !reset_slots.count(slot))throw std::runtime_error("slot used before RESET: "+std::to_string(slot));
  };
  if(e.op=="STEP" || e.op=="HEADSTEP") {
   for(int i=0;i<e.v[0];++i){check_slot(e.v[1+3*i],false);const int pos=e.v[3+3*i];
    if(pos<0 || pos>=ctx)throw std::runtime_error("position out of range: "+std::to_string(pos));
    if(pos==0)e.prefill=true;
   }
   ++steps;if(e.prefill)++prefills;
  }else{check_slot(e.v[0],e.op=="RESET");if(e.op=="RESET")reset_slots.insert(e.v[0]);}
  events.push_back(std::move(e));
 }catch(const std::exception& e){throw std::runtime_error("event line "+std::to_string(line_number)+": "+e.what());}}
 if(!input.eof())throw std::runtime_error("event file read failed");
 if(events.empty() || events[0].op!="RESET")throw std::runtime_error("replay must begin with reset");
 if(decode_only && steps==prefills)throw std::runtime_error("decode-only replay has no decode steps");
 std::cerr<<"replay_config slots="<<slots<<" ctx="<<ctx<<" rowcap="<<rowcap<<" events="<<events.size()<<" steps="<<steps<<" prefills="<<prefills<<"\n";
 if(dry_run){std::cout<<"{\"validated\":true,\"slots\":"<<slots<<",\"ctx\":"<<ctx<<",\"rowcap\":"<<rowcap<<",\"events\":"<<events.size()<<",\"steps\":"<<steps<<",\"prefills\":"<<prefills<<",\"decode_steps\":"<<steps-prefills<<"}\n";return 0;}
#ifdef H5_REPLAY_CPU_ONLY
 throw std::runtime_error("CPU-only replay requires --dry-run");
#else
 bool measuring=false,profiler_started=false;
 auto begin_capture=[&](){if(cudaProfilerStart()!=cudaSuccess)throw std::runtime_error("profiler start failed");profiler_started=true;
#ifdef H5_PDLCLOCK
 h5_pdl_begin_kernels();h5_pdl_begin_fp4();h5_pdl_begin_draft();
#endif
 };
 struct EnvArm {std::string name,value[2];};std::vector<EnvArm> env_arms;
 auto select_env=[&](int arm){for(const auto& e:env_arms)setenv(e.name.c_str(),e.value[arm].c_str(),1);};
 if(const char* v=std::getenv("H5_AB_ENV")){
  if(std::getenv("H5_AB_PLAN_B")||std::getenv("H5_AB_CARVEOUT")||std::getenv("H5_AB_PREPWIDE")||std::getenv("H5_AB_FUSEGU"))throw std::runtime_error("H5_AB_ENV excludes other engine A/B");
  std::istringstream entries(v);std::string s;
  while(std::getline(entries,s,';')){const auto eq=s.find('='),bar=s.find('|');
   if(eq==std::string::npos||eq==0||bar==std::string::npos||bar<eq||s.find('|',bar+1)!=std::string::npos)throw std::runtime_error("H5_AB_ENV must be NAME=A|B[;NAME=C|D...]");
   EnvArm e{s.substr(0,eq),{s.substr(eq+1,bar-eq-1),s.substr(bar+1)}};
   for(const auto& old:env_arms)if(old.name==e.name)throw std::runtime_error("repeated H5_AB_ENV key");
   env_arms.push_back(std::move(e));}
  if(env_arms.empty())throw std::runtime_error("empty H5_AB_ENV");select_env(0);}
 // H5_AB_PLAN_B=PATH: second engine instance created with that plan (H5_PLAN_READ) and its own
 // weights/graphs; measured passes alternate engines in ABBA order. Engine A uses the inherited plan.
 void*engine_a=h5_create(argv[1],slots,ctx,rowcap);if(!engine_a)throw std::runtime_error("model load failed");
 void*engine_b=nullptr;
 // H5_AB_CARVEOUT=A,B: two engines with the same plan whose graphs capture launch carveout A / B (-1 = unset).
 int co_a=-2,co_b=-2,pw_a=-1,pw_b=-1,fu_a=-1,fu_b=-1;
 if(const char* c=std::getenv("H5_AB_FUSEGU")){if(std::sscanf(c,"%d,%d",&fu_a,&fu_b)!=2||fu_a<0||fu_a>1||fu_b<0||fu_b>1)throw std::runtime_error("H5_AB_FUSEGU must be A,B in 0..1");
  if(std::getenv("H5_AB_PLAN_B")||std::getenv("H5_AB_CARVEOUT")||std::getenv("H5_AB_PREPWIDE"))throw std::runtime_error("H5_AB_FUSEGU excludes other engine A/B");
  setenv("H5_AB_PLAN_B",std::getenv("H5_PLAN_READ")?std::getenv("H5_PLAN_READ"):"",1);}
 if(const char* c=std::getenv("H5_AB_PREPWIDE")){if(std::sscanf(c,"%d,%d",&pw_a,&pw_b)!=2||pw_a<0||pw_a>2||pw_b<0||pw_b>2)throw std::runtime_error("H5_AB_PREPWIDE must be A,B in 0..2");
  if(std::getenv("H5_AB_PLAN_B")||std::getenv("H5_AB_CARVEOUT"))throw std::runtime_error("H5_AB_PREPWIDE excludes other engine A/B");
  setenv("H5_AB_PLAN_B",std::getenv("H5_PLAN_READ")?std::getenv("H5_PLAN_READ"):"",1);}
 if(const char* c=std::getenv("H5_AB_CARVEOUT")){if(std::sscanf(c,"%d,%d",&co_a,&co_b)!=2||co_a<-1||co_a>100||co_b<-1||co_b>100)throw std::runtime_error("H5_AB_CARVEOUT must be A,B in -1..100");
  if(std::getenv("H5_AB_PLAN_B")||std::getenv("H5_CARVEOUT"))throw std::runtime_error("H5_AB_CARVEOUT excludes H5_AB_PLAN_B and H5_CARVEOUT");
  setenv("H5_AB_PLAN_B",std::getenv("H5_PLAN_READ")?std::getenv("H5_PLAN_READ"):"",1);}
 if(const char* pb=std::getenv("H5_AB_PLAN_B")){std::string keep=std::getenv("H5_PLAN_READ")?std::getenv("H5_PLAN_READ"):"";if(keep.empty())throw std::runtime_error("H5_AB_PLAN_B requires H5_PLAN_READ for engine A");
  setenv("H5_PLAN_READ",pb,1);engine_b=h5_create(argv[1],slots,ctx,rowcap);if(!engine_b)throw std::runtime_error("engine B load failed");h5_pretune(engine_b);setenv("H5_PLAN_READ",keep.c_str(),1);}
 if(!env_arms.empty()){select_env(1);engine_b=h5_create(argv[1],slots,ctx,rowcap);if(!engine_b)throw std::runtime_error("engine B load failed");h5_pretune(engine_b);select_env(0);}
 void*engine=engine_a;
 if(upload!=-1){auto rc=h5_graph_probe_configure(engine,(H5GraphUpload)upload,(H5GraphTiming)timing);if(rc!=H5_GRAPH_PROBE_OK)throw std::runtime_error("graph probe configuration error "+std::to_string(rc));}
 if(nodes!=-1){auto rc=h5_graph_nodes_configure(engine,(H5GraphNodes)nodes);if(rc!=H5_GRAPH_PROBE_OK)throw std::runtime_error("graph nodes configuration error "+std::to_string(rc));}
 if(co_a>-2)h5_set_launch_carveout(co_a);
 if(pw_a>=0)h5_set_prep_wide(pw_a);
 if(fu_a>=0)h5_set_fuse_gu(fu_a);
 h5_pretune(engine_a);h5_capture(engine_a,ordinary?0:1);
 if(engine_b){if(upload!=-1||nodes!=-1)throw std::runtime_error("H5_AB_PLAN_B does not combine with graph probe options");h5_capture(engine_b,ordinary?0:1);}
 std::ofstream probe_output;
 int pass=-1;
#ifdef H5_SMLIT
 // Explicit window: H5_SMLIT_ROUNDS=FIRST:END records measured pass 0 rounds [FIRST,END).
 const char* lit_env=std::getenv("H5_SMLIT_ROUNDS");if(!lit_env)throw std::runtime_error("H5_SMLIT build requires H5_SMLIT_ROUNDS=FIRST:END");
 unsigned lit_first=0,lit_end=0;if(std::sscanf(lit_env,"%u:%u",&lit_first,&lit_end)!=2||lit_end<=lit_first)throw std::runtime_error("invalid H5_SMLIT_ROUNDS");
 const unsigned long long lit_cap=(unsigned long long)(lit_end-lit_first)*1500000ULL;
 h5::lit::Buf* lit_dev=nullptr;if(cudaMalloc(&lit_dev,sizeof(h5::lit::Buf)+lit_cap*sizeof(h5::lit::Rec))!=cudaSuccess)throw std::runtime_error("smlit allocation failed");
 {h5::lit::Buf head{};head.count=0;head.cap=lit_cap;if(cudaMemcpy(lit_dev,&head,16,cudaMemcpyHostToDevice)!=cudaSuccess)throw std::runtime_error("smlit init failed");}
 std::vector<unsigned long long> lit_bounds;
 auto lit_set=[&](h5::lit::Buf*b){h5_lit_set_kernels(b);h5_lit_set_fp4(b);h5_lit_set_draft(b);h5_lit_set_gmem(b);h5_lit_set_sk(b);};
#endif
 auto run=[&](std::ostream*o){select_env(engine==engine_b?1:0);unsigned round=0;for(auto&e:events){auto&v=e.v;
  if(e.op=="RESET")h5_reset(engine,v[0]);else if(e.op=="KEEP")h5_keep(engine,v[0],v[1]);else if(e.op=="ACCEPT")h5_accept(engine,v[0],v[1]);else {
   int n=v[0];std::vector<int>slots(n),toks(n),pos(n),out(n);for(int i=0;i<n;i++){slots[i]=v[1+3*i];toks[i]=v[2+3*i];pos[i]=v[3+3*i];}
   if(measuring && decode_only && pass==0 && !e.prefill && !profiler_started)begin_capture();
   #ifdef H5_SMLIT
   if(measuring && pass==0 && round==lit_first)lit_set(lit_dev);
#endif
   nvtxRangePushA(e.prefill?"replay-prefill":"replay-target");auto start=std::chrono::steady_clock::now();int rc;
   if(e.op=="STEP" || head_mode==H5_HEAD_ALL) rc=h5_step(engine,n,slots.data(),toks.data(),pos.data(),out.data());
   else {int end=1+3*n;rc=h5_step_head(engine,n,slots.data(),toks.data(),pos.data(),out.data(),H5_HEAD_SELECTED,v[end],v[end]?v.data()+end+1:nullptr);}auto end=std::chrono::steady_clock::now();nvtxRangePop();if(rc)throw std::runtime_error("target step failed: "+std::to_string(rc));
#ifdef H5_SMLIT
   if(measuring && pass==0 && round>=lit_first && round<lit_end){unsigned long long c=0;if(cudaMemcpy(&c,lit_dev,8,cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("smlit count read failed");lit_bounds.push_back(c);if(round+1==lit_end)lit_set(nullptr);}
#endif
   if(o){*o<<round<<","<<n<<","<<std::chrono::duration<double>(end-start).count();for(int t:out)*o<<","<<t;*o<<"\n";}
   if(measuring && timing==H5_GRAPH_TIMING_WALL_THREAD){
    H5GraphSample sample;auto status=h5_graph_probe_read(engine,&sample);if(status!=H5_GRAPH_PROBE_OK)throw std::runtime_error("graph probe read error "+std::to_string(status));
    probe_output<<pass<<","<<round<<","<<n<<","<<(int)sample.route;
    for(int x:sample.shape)probe_output<<","<<x;
    probe_output<<","<<sample.nodes<<","<<sample.roots<<","<<sample.prepare_ns<<","<<sample.submit_ns<<","<<sample.launch_wall_ns<<","<<sample.launch_thread_ns<<","<<sample.sync_ns<<","<<sample.finish_ns<<","<<sample.total_ns<<","<<sample.upload_wall_ns<<","<<sample.rewrite_ns<<","<<sample.memory_copies<<","<<sample.memory_clears<<","<<sample.copy_bytes<<","<<sample.clear_bytes<<","<<sample.host_read_bytes<<","<<sample.host_write_bytes<<"\n";
   }
   round++;
  }
 }};
 run(nullptr);run(nullptr);run(nullptr);
 if(engine_b){if(co_b>-2)h5_set_launch_carveout(co_b);if(pw_b>=0)h5_set_prep_wide(pw_b);if(fu_b>=0)h5_set_fuse_gu(fu_b);engine=engine_b;run(nullptr);run(nullptr);run(nullptr);engine=engine_a;}
 std::ofstream output(std::string(argv[3])+".csv");if(!output)throw std::runtime_error("output missing");output<<std::setprecision(17);
 if(timing==H5_GRAPH_TIMING_WALL_THREAD){probe_output.open(std::string(argv[3])+"-host.csv");if(!probe_output)throw std::runtime_error("probe output missing");probe_output<<"pass,round,rows,route,kind,a,b,c,d,e,f,g,nodes,roots,prepare_ns,submit_ns,launch_wall_ns,launch_thread_ns,sync_ns,finish_ns,total_ns,upload_wall_ns,rewrite_ns,memory_copies,memory_clears,copy_bytes,clear_bytes,host_read_bytes,host_write_bytes\n";}
 auto check=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
 measuring=true;if(!decode_only)begin_capture();
 long warm_hits=h5_graph_stats(engine,0),warm_caps=h5_graph_stats(engine,1);
 // In-process A/B: H5_AB_PREWAIT=A,B alternates the GEMM pre-wait ring cap per pass (ABBA order).
 // Each CSV row gains no column; arm order is printed so the analyzer can split passes.
 // H5_AB_VARIANT=A,B (hex bitmasks): bit0 GDN-norm gate loads before wait, bit1 batched conv rows.
 std::vector<int> ab; std::vector<unsigned> abv2;
 if(const char* abv=std::getenv("H5_AB_PREWAIT")){int x=0,y=0;if(std::sscanf(abv,"%d,%d",&x,&y)!=2||x<0||y<0)throw std::runtime_error("H5_AB_PREWAIT must be A,B");ab={x,y};}
 if(const char* v=std::getenv("H5_AB_VARIANT")){unsigned x=0,y=0;if(std::sscanf(v,"%x,%x",&x,&y)!=2)throw std::runtime_error("H5_AB_VARIANT must be A,B hex");abv2={x,y};}
 if((!ab.empty())+(!abv2.empty())+(engine_b!=nullptr)>1)throw std::runtime_error("one A/B knob at a time");
 std::string arms;
 auto start=std::chrono::steady_clock::now();for(pass=0;pass<passes;pass++){int arm=((pass&3)==0||(pass&3)==3)?0:1;if(!ab.empty()){h5_set_prewait_cap(ab[arm]);arms+=char('A'+arm);}if(engine_b){engine=arm?engine_b:engine_a;arms+=char('A'+arm);}if(!abv2.empty()){h5_set_variant_kernels(abv2[arm]);h5_set_variant_fp4(abv2[arm]);h5_set_variant_draft(abv2[arm]);arms+=char('A'+arm);}run(&output);}auto end=std::chrono::steady_clock::now();
 if(!arms.empty())std::cerr<<"ab_arms "<<arms<<"\n";
#ifdef H5_PDLCLOCK
 h5_pdl_end_kernels((std::string(argv[3])+"-kernels.csv").c_str());h5_pdl_end_fp4((std::string(argv[3])+"-fp4.csv").c_str());h5_pdl_end_draft((std::string(argv[3])+"-draft.csv").c_str());
#endif
 check(cudaProfilerStop());
#ifdef H5_SMLIT
 {if(lit_bounds.size()!=lit_end-lit_first)throw std::runtime_error("smlit window not reached");
  unsigned long long n=lit_bounds.back();if(n>lit_cap)throw std::runtime_error("smlit overflow: "+std::to_string(n));
  std::vector<h5::lit::Rec> recs(n);if(n)check(cudaMemcpy(recs.data(),lit_dev->recs,n*sizeof(h5::lit::Rec),cudaMemcpyDeviceToHost));
  std::ofstream f(std::string(argv[3])+"-smlit.bin",std::ios::binary);unsigned long long hdr[3]={0x544c4d53ULL,(unsigned long long)lit_first,(unsigned long long)lit_bounds.size()};
  f.write((const char*)hdr,sizeof(hdr));f.write((const char*)lit_bounds.data(),lit_bounds.size()*8);f.write((const char*)recs.data(),n*sizeof(h5::lit::Rec));if(!f)throw std::runtime_error("smlit write failed");}
#endif
 output.close();if(!output)throw std::runtime_error("output write failed");
 if(timing==H5_GRAPH_TIMING_WALL_THREAD){probe_output.close();if(!probe_output)throw std::runtime_error("probe output write failed");}
 std::cout<<std::setprecision(17)<<"{\"total_s\":"<<std::chrono::duration<double>(end-start).count()<<",\"passes\":"<<passes<<",\"slots\":"<<slots<<",\"ctx\":"<<ctx<<",\"rowcap\":"<<rowcap<<",\"head_rows\":\""<<(head_mode==H5_HEAD_ALL?"all":"demand")<<"\",\"graph_upload\":"<<upload<<",\"graph_timing\":"<<timing<<",\"graph_nodes\":"<<nodes<<",\"graph_replays\":"<<h5_graph_stats(engine,0)-warm_hits<<",\"graph_captures\":"<<h5_graph_stats(engine,1)-warm_caps<<"}\n";
 return 0;
#endif
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
