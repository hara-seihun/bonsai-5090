#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
namespace h5 {
int fp4_ncfg();bool fp4_cfg_wg2(int);
cudaError_t scale_carrier_prepare(const void*,float*,float4*,int,int,cudaStream_t);
cudaError_t scale_carrier_resources(int,int,cudaFuncAttributes*,int*);
cudaError_t launch_scale_carrier(int,int,int,const uint8_t*,const uint8_t*,const float*,int,int,const void*,int,float*,int,const float*,cudaStream_t);
void launch_gemm_fp4_cfg(int,int,const uint8_t*,const uint8_t*,const float*,int,int,const void*,const void*,int,float*,int,bool,cudaStream_t,const float*,int);
void launch_gemm_fp4_silu(int,const uint8_t*,const uint8_t*,const float*,int,int,const void*,const void*,int,float*,int,cudaStream_t,const float*,int);
}
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>static T* upload(const std::vector<T>&v){T*p;check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
static std::vector<uint8_t> file(const std::string&path,size_t n){std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(n))throw std::runtime_error("fixture size/missing: "+path);std::vector<uint8_t>b(n);f.seekg(0);f.read((char*)b.data(),n);if(!f)throw std::runtime_error("fixture read failed");return b;}
static void save(const std::filesystem::path&path,const void*data,size_t n){FILE*f=fopen(path.c_str(),"wbx");if(!f||fwrite(data,1,n,f)!=n||fclose(f))throw std::runtime_error("snapshot write failed: "+path.string());}
int main(int argc,char**argv){try{
 if(argc!=10){fprintf(stderr,"usage: scale-carrier R N K cfg split fused(0|1) FIXTURE_DIR|constant|varying ROUNDS NEW_OUTPUT_DIR\n");return 2;}
 int x[6];for(int i=0;i<6;i++){char*end;long v=strtol(argv[i+1],&end,10);if(!*argv[i+1]||*end||v<0||v>1000000)return 2;x[i]=v;}
 const int R=x[0],N=x[1],K=x[2],cfg=x[3],split=x[4],fused=x[5],B=K/128,ldo=fused?N/2:N,mode=fused?3:split>1?2:0;
 char*end;int rounds=strtol(argv[8],&end,10);if(*end||rounds<2||rounds>100)return 2;
 if((R!=1&&R!=2&&R!=8)||N<=0||N%32||K<=0||K%128||cfg>=h5::fp4_ncfg()||split<1||split>B||fused>1||(fused&&(split!=1||N%64||!h5::fp4_cfg_wg2(cfg))))return 2;
 if(!getenv("H5_PDL")||(strcmp(getenv("H5_PDL"),"0")&&strcmp(getenv("H5_PDL"),"1")))return 2;
 std::filesystem::path dir=argv[9];if(std::filesystem::exists(dir))throw std::runtime_error("output directory exists");std::filesystem::create_directories(dir);
 std::string fixture=argv[7];uint32_t rng=17;std::vector<uint8_t>w(size_t(N/32)*B*832);std::vector<__half>s(size_t(N)*B);
 if(fixture=="constant"||fixture=="varying"){
  for(auto&v:w){rng=rng*1664525u+1013904223u;v=(rng>>16)%243;}
  for(int kb=0;kb<B;kb++)for(int row=0;row<N;row++){
   int idx=(row/32)*32+(row&7)*4+((row>>3)&3);float v=.02f+float(row%13)*.001f;
   if(fixture=="varying")v*=.5f+float((kb*17+row*3)%29)/28.f;
   s[size_t(kb)*N+idx]=__float2half(v);
  }
 }else{w=file(fixture+"/weights.bin",w.size());auto raw=file(fixture+"/scales.bin",s.size()*2);memcpy(s.data(),raw.data(),raw.size());}
 std::vector<uint8_t>a(size_t(64)*K/2);for(auto&v:a){rng=rng*1664525u+1013904223u;v=rng>>16;}
 auto W=upload(w);auto S=upload(s);auto A=upload(a);auto AS=upload(std::vector<uint8_t>(size_t(64)*K/16,0x38));auto AG=upload(std::vector<float>(64,1.f));auto Y=upload(std::vector<float>(size_t(R)*ldo));auto mean=upload(std::vector<float>(N));auto stats=upload(std::vector<float4>(N));
 cudaStream_t st;check(cudaStreamCreate(&st));cudaEvent_t begin,stop;check(cudaEventCreate(&begin));check(cudaEventCreate(&stop));
 auto prepare=[&](){check(h5::scale_carrier_prepare(S,mean,stats,N,K,st));};
 check(cudaEventRecord(begin,st));prepare();check(cudaEventRecord(stop,st));check(cudaEventSynchronize(stop));float ms;check(cudaEventElapsedTime(&ms,begin,stop));
 double initial_prepare_us=ms*1000;std::vector<float4>hst(N);check(cudaMemcpy(hst.data(),stats,N*sizeof(float4),cudaMemcpyDeviceToHost));
 double scale_err=0,scale_mag=0,mean_check=0;
 for(int row=0;row<N;row++){
  double sum=0;int idx=(row/32)*32+(row&7)*4+((row>>3)&3);
  for(int kb=0;kb<B;kb++){double v=__half2float(s[size_t(kb)*N+idx]);sum+=v;scale_err+=(v-hst[row].x)*(v-hst[row].x);scale_mag+=v*v;}
  mean_check=std::max(mean_check,fabs(sum/B-hst[row].x));if(!std::isfinite(hst[row].x)||mean_check>1e-6)throw std::runtime_error("row mean mismatch/nonfinite");
 }
 save(dir/"row-stats.bin",hst.data(),N*sizeof(float4));
 printf("{\"fixture\":\"%s\",\"rows\":%d,\"N\":%d,\"K\":%d,\"cfg\":%d,\"split\":%d,\"initial_prepare_us\":%.9g,\"prepare_scale_read_bytes\":%zu,\"prepared_bytes\":%d,\"mean_max_abs_error\":%.9g,\"scale_rel_rms\":%.9g}\n",fixture.c_str(),R,N,K,cfg,split,initial_prepare_us,s.size()*2,N*20,mean_check,sqrt(scale_err/scale_mag));
 auto launch=[&](int arm){
  if(arm==2)prepare();
  if(arm==0){if(fused)h5::launch_gemm_fp4_silu(cfg,A,AS,AG,R,K,W,S,N,Y,ldo,st,nullptr,3);else h5::launch_gemm_fp4_cfg(cfg,split,A,AS,AG,R,K,W,S,N,Y,ldo,false,st,nullptr,3);}
  else{if(split>1)check(cudaMemsetAsync(Y,0,size_t(R)*ldo*4,st));check(h5::launch_scale_carrier(cfg,split,mode,A,AS,AG,R,K,W,N,Y,ldo,mean,st));}
  check(cudaGetLastError());
 };
 std::vector<float>ref(size_t(R)*ldo),out(ref.size());
 launch(0);check(cudaStreamSynchronize(st));check(cudaMemcpy(ref.data(),Y,ref.size()*4,cudaMemcpyDeviceToHost));save(dir/"control.bin",ref.data(),ref.size()*4);
 launch(1);check(cudaStreamSynchronize(st));check(cudaMemcpy(out.data(),Y,out.size()*4,cudaMemcpyDeviceToHost));save(dir/"carrier.bin",out.data(),out.size()*4);
 double e2=0,mag=0,abs=0;for(size_t i=0;i<out.size();i++){if(!std::isfinite(out[i])||!std::isfinite(ref[i]))throw std::runtime_error("nonfinite output");double d=double(out[i])-ref[i];e2+=d*d;mag+=double(ref[i])*ref[i];abs=std::max(abs,fabs(d));}
 printf("{\"output_max_abs\":%.9g,\"output_rel_rms\":%.9g,\"mean_rms\":%.9g,\"fused\":%d}\n",abs,sqrt(e2/std::max(mag,1e-30)),sqrt(e2/out.size()),fused);
 cudaFuncAttributes attrs;int blocks;check(h5::scale_carrier_resources(cfg,mode,&attrs,&blocks));printf("{\"carrier_regs\":%d,\"carrier_static_shared\":%zu,\"carrier_local_bytes\":%zu,\"carrier_resident_blocks\":%d}\n",attrs.numRegs,attrs.sharedSizeBytes,attrs.localSizeBytes,blocks);
 cudaGraphExec_t graph[3];for(int arm=0;arm<3;arm++){cudaGraph_t g;check(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));for(int i=0;i<16;i++)launch(arm);check(cudaStreamEndCapture(st,&g));check(cudaGraphInstantiate(&graph[arm],g,0));check(cudaGraphDestroy(g));}
 for(int arm=0;arm<3;arm++)for(int i=0;i<3;i++)check(cudaGraphLaunch(graph[arm],st));check(cudaStreamSynchronize(st));
 std::vector<double>timing[3];
 for(int round=0;round<rounds;round++)for(int slot=0;slot<4;slot++){
  int order[4]={0,1,1,0};int arm=round%2?order[3-slot]:order[slot];
  check(cudaEventRecord(begin,st));for(int r=0;r<8;r++)check(cudaGraphLaunch(graph[arm],st));check(cudaEventRecord(stop,st));check(cudaEventSynchronize(stop));check(cudaEventElapsedTime(&ms,begin,stop));timing[arm].push_back(ms*1000/128);printf("{\"round\":%d,\"slot\":%d,\"arm\":%d,\"chain_us\":%.9g}\n",round,slot,arm,ms*1000/128);
 }
 for(int round=0;round<rounds;round++){check(cudaEventRecord(begin,st));for(int r=0;r<8;r++)check(cudaGraphLaunch(graph[2],st));check(cudaEventRecord(stop,st));check(cudaEventSynchronize(stop));check(cudaEventElapsedTime(&ms,begin,stop));timing[2].push_back(ms*1000/128);printf("{\"round\":%d,\"arm\":2,\"refresh_chain_us\":%.9g}\n",round,ms*1000/128);}
 double median[3];for(int i=0;i<3;i++){std::sort(timing[i].begin(),timing[i].end());median[i]=timing[i][timing[i].size()/2];}
 printf("{\"control_us\":%.9g,\"carrier_cached_us\":%.9g,\"carrier_refresh_us\":%.9g,\"cached_reduction_pct\":%.9g,\"refresh_reduction_pct\":%.9g,\"output_rel_rms\":%.9g,\"output_max_abs\":%.9g,\"initial_prepare_us\":%.9g}\n",median[0],median[1],median[2],100*(1-median[1]/median[0]),100*(1-median[2]/median[0]),sqrt(e2/std::max(mag,1e-30)),abs,initial_prepare_us);
 for(auto&g:graph)check(cudaGraphExecDestroy(g));check(cudaEventDestroy(begin));check(cudaEventDestroy(stop));check(cudaStreamDestroy(st));
 check(cudaFree(W));check(cudaFree(S));check(cudaFree(A));check(cudaFree(AS));check(cudaFree(AG));check(cudaFree(Y));check(cudaFree(mean));check(cudaFree(stats));return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
