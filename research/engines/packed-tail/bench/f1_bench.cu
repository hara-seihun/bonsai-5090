// Complete normalized-input -> FFN contribution discriminator. All GPU work is ours.
// F1 arm tok32/64 names refer to GU ownership; down independently uses FW8/TOK64,
// or TOK16/splitK16 at1-8rows. Baseline independently autotunes its two projections.
// No GEMM library. Baseline reuses current halo FP4 streaming kernels and prep unchanged.
#include "../src/f1.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
namespace h5 {
int fp4_ncfg(); bool fp4_cfg_wg2(int); const char* fp4_cfg_name(int,int);
void launch_prep(const PrepArgs&,int,cudaStream_t);
void launch_gemm_fp4_silu(int,const uint8_t*,const uint8_t*,const float*,int,int,const void*,const void*,int,float*,int,cudaStream_t,const float*,int);
void launch_gemm_fp4_cfg(int,int,const uint8_t*,const uint8_t*,const float*,int,int,const void*,const void*,int,float*,int,bool,cudaStream_t,const float*,int);
}
#define CU(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));exit(2);}}while(0)
constexpr int D=5120,F=17408,MAXR=512;
std::vector<unsigned char> file(const std::string& path){std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f){fprintf(stderr,"missing %s\n",path.c_str());exit(2);}size_t n=f.tellg();std::vector<unsigned char> b(n);f.seekg(0);f.read((char*)b.data(),n);return b;}
template<class T>T* alloc(size_t n){T* p;CU(cudaMalloc(&p,n*sizeof(T)));CU(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T>T* load(const std::string& p,size_t expected){auto b=file(p);if(b.size()!=expected){fprintf(stderr,"bad size %s: %zu != %zu\n",p.c_str(),b.size(),expected);exit(2);}T* d=alloc<T>((b.size()+sizeof(T)-1)/sizeof(T));CU(cudaMemcpy(d,b.data(),b.size(),cudaMemcpyHostToDevice));return d;}
struct Arm{std::string name;std::function<void()> run;cudaGraphExec_t graph=nullptr;std::vector<double> us;};
float time_calls(const std::function<void()>& f,cudaStream_t st,int n=4){cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));CU(cudaEventRecord(a,st));for(int i=0;i<n;i++)f();CU(cudaEventRecord(b,st));CU(cudaEventSynchronize(b));CU(cudaGetLastError());float ms;CU(cudaEventElapsedTime(&ms,a,b));CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));return ms*1000/n;}
double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
__global__ void perturb(unsigned* p,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=p[i]*1664525u+1013904223u;}
int main(int argc,char**argv){
 if(argc<2){fprintf(stderr,"usage: f1-bench FIXTURE_DIR [repeats=50] [rounds=10] [graphs=1] [cold=0] [rows=0(all)]\n");return 1;}
 std::string dir=argv[1];int repeats=argc>2?atoi(argv[2]):50,rounds=argc>3?atoi(argv[3]):10;bool graphs=argc>4?atoi(argv[4]):1,cold=argc>5?atoi(argv[5]):0;int only=argc>6?atoi(argv[6]):0;
 cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));printf("{\"device\":\"%s\",\"sm\":%d,\"smem_optin\":%zu,\"graphs\":%d,\"cold\":%d}\n",prop.name,prop.major*10+prop.minor,prop.sharedMemPerBlockOptin,graphs,cold);
 cudaStream_t st;CU(cudaStreamCreate(&st));
 float* X=load<float>(dir+"/input.bin",MAXR*D*4),*SI=load<float>(dir+"/sign_in.bin",D*4),*SH=load<float>(dir+"/sign_hid.bin",F*4);
 uint8_t* W=load<uint8_t>(dir+"/gu.bin",(size_t)2*F*D/4),*S=load<uint8_t>(dir+"/gu_scale.bin",(size_t)2*F*D/128*2);
 uint8_t* WF=load<uint8_t>(dir+"/gu_f1.bin",(size_t)2*F*D/4),*SF=load<uint8_t>(dir+"/gu_f1_scale.bin",(size_t)2*F*D/128*2);
 uint8_t* WD=load<uint8_t>(dir+"/down.bin",(size_t)D*F/4),*SD=load<uint8_t>(dir+"/down_scale.bin",(size_t)D*F/128*2),*WE=load<uint8_t>(dir+"/down_expanded.bin",(size_t)D*F/2);
 auto cf=file(dir+"/fixed.bin");float scales[3];if(cf.size()!=12)return 2;memcpy(scales,cf.data(),12);
 uint8_t* A=alloc<uint8_t>(MAXR*D/2),*AS=alloc<uint8_t>(MAXR*D/16),*B=alloc<uint8_t>(MAXR*F/2),*BS=alloc<uint8_t>(MAXR*F/16);float* AG=alloc<float>(MAXR),*BG=alloc<float>(MAXR),*H=alloc<float>(MAXR*F),*Y=alloc<float>(MAXR*D);
 // Explicit matched current prep on normalized capture, omitting common RMSNorm/residual add.
 auto original_prep=[&](float* x,float* sg,int K,int R,uint8_t* q,uint8_t* sb,float* g){h5::PrepArgs a{};a.x=x;a.ldx=K;a.signs=sg;a.n=K;a.flags=h5::P_SIGN|h5::P_HADAMARD|h5::P_QUANT;a.q4=q;a.s4=sb;a.g4=g;h5::launch_prep(a,R,st);};
 unsigned* trash=cold?alloc<unsigned>(32*1024*1024):nullptr;
 for(int R:{1,8,512}){
   if(only && R!=only)continue;
   original_prep(X,SI,D,R,A,AS,AG);
   int gu=3,dc=0,ks=1;float best=1e30;
   for(int ci=0;ci<h5::fp4_ncfg();ci++)if(h5::fp4_cfg_wg2(ci)){
     auto fn=[&,ci]{h5::launch_gemm_fp4_silu(ci,A,AS,AG,R,D,W,S,2*F,H,F,st,nullptr,0);};fn();CU(cudaStreamSynchronize(st));float us=time_calls(fn,st);if(us<best){best=us;gu=ci;}
   }
   h5::launch_gemm_fp4_silu(gu,A,AS,AG,R,D,W,S,2*F,H,F,st,nullptr,0);original_prep(H,SH,F,R,B,BS,BG);CU(cudaStreamSynchronize(st));best=1e30;
   for(int ci=0;ci<h5::fp4_ncfg();ci++)for(int split:{1,4,16}){
     auto fn=[&,ci,split]{h5::launch_gemm_fp4_cfg(ci,split,B,BS,BG,R,F,WD,SD,D,Y,D,false,st,nullptr,0);};fn();CU(cudaStreamSynchronize(st));float us=time_calls(fn,st);if(us<best){best=us;dc=ci;ks=split;}
   }
   printf("{\"rows\":%d,\"baseline_gu\":\"%s\",",R,h5::fp4_cfg_name(gu,1));printf("\"baseline_down\":\"%s\"}\n",h5::fp4_cfg_name(dc,ks));
   auto chain=[&](int variant,int tok){
     if(variant==0)original_prep(X,SI,D,R,A,AS,AG);else f1::prep(X,SI,D,R,scales[0],A,AS,AG,st);
     if(variant<2){h5::launch_gemm_fp4_silu(gu,A,AS,AG,R,D,W,S,2*F,H,F,st,nullptr,0);if(variant==0)original_prep(H,SH,F,R,B,BS,BG);else f1::prep(H,SH,F,R,scales[1],B,BS,BG,st);h5::launch_gemm_fp4_cfg(dc,ks,B,BS,BG,R,F,WD,SD,D,Y,D,false,st,nullptr,0);}
     else{f1::gu_fused(tok,A,AS,AG,R,D,WF,SF,2*F,SH,scales[2],B,BS,BG,st);f1::down(64,variant==2,B,BS,BG,R,F,variant==2?WD:WE,SD,D,Y,st);}
   };
   std::vector<Arm> arms;
   for(int v=0;v<4;v++)for(int tok:{16,32,64}){if((v<2 && tok!=32) || (R>8 && tok==16))continue;std::string name=v==0?"baseline":v==1?"fixed_fullH":v==2?"F1_compact":"F1_expanded";if(v>=2)name+="_tok"+std::to_string(tok);arms.push_back({name,[&,v,tok]{chain(v,tok);}});}
   std::vector<float> compact;
   for(auto& arm:arms){
     arm.run();CU(cudaStreamSynchronize(st));CU(cudaGetLastError());
     std::vector<float> y(std::min(R,8)*D);CU(cudaMemcpy(y.data(),Y,y.size()*4,cudaMemcpyDeviceToHost));
     auto bytes=file(dir+"/ref_"+(arm.name=="baseline"?"baseline":arm.name=="fixed_fullH"?"fixed":"f1")+".bin");double e=0,m=0;bool finite=true;
     for(size_t i=0;i<y.size();i++){float ref;memcpy(&ref,bytes.data()+i*4,4);e+=(double)(y[i]-ref)*(y[i]-ref);m+=(double)ref*ref;finite&=std::isfinite(y[i]);}
     double error=sqrt(e/m);printf("{\"rows\":%d,\"arm\":\"%s\",\"vs_cpu_rel_rms\":%.9g,\"finite\":%d}\n",R,arm.name.c_str(),error,finite);
     if(!finite || error>.015){fprintf(stderr,"numerical lowering failed; refusing to time %s\n",arm.name.c_str());return 3;}
     if(arm.name.rfind("F1_compact",0)==0)compact=y;
     if(arm.name.rfind("F1_expanded",0)==0){double d=0;for(size_t i=0;i<y.size();i++)d=std::max(d,(double)fabs(y[i]-compact[i]));printf("{\"rows\":%d,\"arm\":\"%s\",\"compact_expanded_max_abs\":%.9g}\n",R,arm.name.c_str(),d);if(d>1e-4)return 3;}
     // Paid phase diagnostic. These component times are not summed as a speedup claim.
     int kind=arm.name=="baseline"?0:arm.name=="fixed_fullH"?1:arm.name.rfind("F1_compact",0)==0?2:3;
     int tok=arm.name.find("tok64")!=std::string::npos?64:arm.name.find("tok16")!=std::string::npos?16:32;
     std::vector<std::pair<std::string,std::function<void()>>> phases;
     phases.push_back({"input_prep",[&]{if(kind==0)original_prep(X,SI,D,R,A,AS,AG);else f1::prep(X,SI,D,R,scales[0],A,AS,AG,st);}});
     phases.push_back({"gate_up",[&]{if(kind<2)h5::launch_gemm_fp4_silu(gu,A,AS,AG,R,D,W,S,2*F,H,F,st,nullptr,0);else f1::gu_fused(tok,A,AS,AG,R,D,WF,SF,2*F,SH,scales[2],B,BS,BG,st);}});
     if(kind<2)phases.push_back({"hidden_prep",[&]{if(kind==0)original_prep(H,SH,F,R,B,BS,BG);else f1::prep(H,SH,F,R,scales[1],B,BS,BG,st);}});
     phases.push_back({"down",[&]{if(kind<2)h5::launch_gemm_fp4_cfg(dc,ks,B,BS,BG,R,F,WD,SD,D,Y,D,false,st,nullptr,0);else f1::down(64,kind==2,B,BS,BG,R,F,kind==2?WD:WE,SD,D,Y,st);}});
     for(auto& phase:phases)printf("{\"rows\":%d,\"arm\":\"%s\",\"phase\":\"%s\",\"component_us\":%.6f}\n",R,arm.name.c_str(),phase.first.c_str(),time_calls(phase.second,st,10));
     // Initialize attributes/table outside capture, then capture whole chain.
     if(graphs){cudaGraph_t g;CU(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));arm.run();CU(cudaStreamEndCapture(st,&g));CU(cudaGraphInstantiate(&arm.graph,g,0));CU(cudaGraphDestroy(g));}
   }
   auto execute=[&](Arm& a){if(graphs)CU(cudaGraphLaunch(a.graph,st));else a.run();};
   for(int i=0;i<10;i++)for(auto& a:arms)execute(a);CU(cudaStreamSynchronize(st));
   for(int round=0;round<rounds;round++)for(size_t j=0;j<arms.size();j++){
     auto& a=arms[(j+round)%arms.size()];
     double us=0;
     if(cold) {
       // Evict before, outside the event bracket. No subtraction of independently timed work.
       cudaEvent_t start,stop;CU(cudaEventCreate(&start));CU(cudaEventCreate(&stop));
       for(int i=0;i<repeats;i++) {
         perturb<<<(32*1024*1024+255)/256,256,0,st>>>(trash,32*1024*1024);
         CU(cudaEventRecord(start,st));execute(a);CU(cudaEventRecord(stop,st));CU(cudaEventSynchronize(stop));
         float ms;CU(cudaEventElapsedTime(&ms,start,stop));us+=ms*1000;
       }
       CU(cudaEventDestroy(start));CU(cudaEventDestroy(stop));us/=repeats;
     } else {auto fn=[&]{execute(a);};us=time_calls(fn,st,repeats);}
     a.us.push_back(us);
     printf("{\"rows\":%d,\"arm\":\"%s\",\"round\":%d,\"chain_us\":%.6f}\n",R,a.name.c_str(),round,us);
   }
   double base=median(arms[0].us);
   for(auto& a:arms){printf("{\"rows\":%d,\"arm\":\"%s\",\"median_us\":%.6f,\"ratio_to_baseline\":%.6f}\n",R,a.name.c_str(),median(a.us),median(a.us)/base);if(a.graph)CU(cudaGraphExecDestroy(a.graph));}
 }
 CU(cudaStreamSynchronize(st));return 0;
}
