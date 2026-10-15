#include "kernels.cuh"
#include "carrier_wf0_pack.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
namespace h5 {void launch_prep(const PrepArgs&,int,cudaStream_t);}
#define ENTRY(n) extern "C" cudaError_t n(int,int,int,const uint8_t*,const uint8_t*,const float*,int,int,const void*,int,float*,int,const float*,cudaStream_t,cudaFuncAttributes*,int*,int*)
ENTRY(carrier_dense5);ENTRY(carrier_twobit);
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>static T*upload(const std::vector<T>&v){T*p;check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
static std::vector<uint8_t>load(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(n))throw std::runtime_error("fixture size/missing: "+p);std::vector<uint8_t>b(n);f.seekg(0);f.read((char*)b.data(),n);if(!f)throw std::runtime_error("fixture read failed");return b;}
static void save(const std::filesystem::path&p,const std::vector<float>&v){FILE*f=fopen(p.c_str(),"wbx");if(!f||fwrite(v.data(),4,v.size(),f)!=v.size()||fclose(f))throw std::runtime_error("snapshot write failed");}
__global__ void delayed_input(float*X,size_t n){h5::pdl_wait();__nanosleep(20000);for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)X[i]=float(int(i%23)-11)*.0625f;__syncthreads();h5::pdl_launch();}
int main(int argc,char**argv){try{
 if(argc!=11){fprintf(stderr,"usage: carrier-wf0 R N K cfg split mode FIXTURE_DIR|varying ROUNDS NEW_OUTPUT_DIR delayed(0|1)\n");return 2;}
 int x[6];for(int i=0;i<6;i++){char*e;long v=strtol(argv[i+1],&e,10);if(!*argv[i+1]||*e||v<0||v>1000000)return 2;x[i]=v;}
 int R=x[0],N=x[1],K=x[2],cfg=x[3],split=x[4],mode=x[5],B=K/128,ldo=mode==3?N/2:N;char*e;int rounds=strtol(argv[8],&e,10);if(*e||rounds<2||rounds>100||R<1||R>8||N<1||N%32||K<1||K%1024||mode>3)return 2;
 if(strcmp(argv[10],"0")&&strcmp(argv[10],"1"))return 2;bool delayed=!strcmp(argv[10],"1");
 if(!getenv("H5_PDL")||(strcmp(getenv("H5_PDL"),"0")&&strcmp(getenv("H5_PDL"),"1")))return 2;
 if(!getenv("H5_PREP_CHUNK")||strcmp(getenv("H5_PREP_CHUNK"),"0"))return 2;
 std::filesystem::path dir=argv[9];if(std::filesystem::exists(dir))throw std::runtime_error("output already exists");std::filesystem::create_directories(dir);
 std::string fixture=argv[7];std::vector<uint8_t>w(size_t(N/32)*B*832);std::vector<__half>s(size_t(N)*B);
 if(fixture=="varying"){
  for(size_t i=0;i<w.size();i++)w[i]=i%256;
  for(int kb=0;kb<B;kb++)for(int row=0;row<N;row++){int idx=(row/32)*32+(row&7)*4+((row>>3)&3);s[size_t(kb)*N+idx]=__float2half((.02f+(row%13)*.001f)*(.5f+float((kb*17+row*3)%29)/28.f));}
 }else{w=load(fixture+"/weights.bin",w.size());auto b=load(fixture+"/scales.bin",s.size()*2);memcpy(s.data(),b.data(),b.size());}
 auto begin_pack=std::chrono::steady_clock::now();auto native=carrier_pack_wf0(w);double pack_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin_pack).count();
 printf("{\"load_time_repack_ms\":%.9g,\"dense_payload_bytes\":%zu,\"native_payload_bytes\":%zu,\"extra_bytes\":%zu,\"cpu_repack_read_bytes\":%zu,\"cpu_repack_write_bytes\":%zu}\n",pack_ms,w.size(),native.size(),native.size()-w.size(),w.size(),native.size());
 // Same once-only row-mean cache as the control, FP32 sequential K summation.
 std::vector<float>mu(N);for(int row=0;row<N;row++){int idx=(row/32)*32+(row&7)*4+((row>>3)&3);float sum=0;for(int kb=0;kb<B;kb++)sum+=__half2float(s[size_t(kb)*N+idx]);mu[row]=sum/B;if(!std::isfinite(mu[row]))throw std::runtime_error("nonfinite carrier");}
 auto W3=upload(w),W0=upload(native),A=upload(std::vector<uint8_t>(size_t(64)*K/2)),AS=upload(std::vector<uint8_t>(size_t(64)*K/16));auto AG=upload(std::vector<float>(R)),mean=upload(mu),Y=upload(std::vector<float>(size_t(R)*ldo));
 std::vector<float>input(size_t(R)*K),norm(K),signs(K);for(size_t i=0;i<input.size();i++)input[i]=float(int(i%23)-11)*.0625f;for(int i=0;i<K;i++){norm[i]=.8f+(i%11)*.03125f;signs[i]=i%3?1.f:-1.f;}
 auto X=upload(input),NW=upload(norm),SG=upload(signs);
 h5::PrepArgs prep{};prep.x=X;prep.norm_w=NW;prep.signs=SG;prep.n=K;prep.ldx=K;prep.ldo=K;prep.flags=h5::P_NORM|h5::P_SIGN|h5::P_HADAMARD|h5::P_QUANT;prep.q4=A;prep.s4=AS;prep.g4=AG;if(mode==2){prep.clear_a=Y;prep.clear_a_stride=ldo;}
 cudaStream_t st;check(cudaStreamCreate(&st));decltype(&carrier_dense5)fn[]={carrier_dense5,carrier_twobit};const void*weights[]={W3,W0};std::vector<float>ref(size_t(R)*ldo),out(ref.size());
 auto launch=[&](int arm,bool paidprep){
  if(paidprep){if(delayed)h5::pdl(delayed_input,dim3(16),dim3(128),0,st,X,input.size());h5::launch_prep(prep,R,st);}
  if(mode==2&&!paidprep)check(cudaMemsetAsync(Y,0,out.size()*4,st));if(mode==1)check(cudaMemsetAsync(Y,0x3d,out.size()*4,st));check(fn[arm](cfg,mode,split,A,AS,AG,R,K,weights[arm],N,Y,ldo,mean,st,nullptr,nullptr,nullptr));};
 for(int arm=0;arm<2;arm++){
  cudaFuncAttributes at;int blocks,bytes;check(fn[arm](cfg,mode,split,A,AS,AG,R,K,weights[arm],N,Y,ldo,mean,st,&at,&blocks,&bytes));printf("{\"arm\":%d,\"regs\":%d,\"dynamic_shared_bytes\":%d,\"local_bytes\":%zu,\"resident_blocks\":%d}\n",arm,at.numRegs,bytes,at.localSizeBytes,blocks);
  launch(arm,true);check(cudaStreamSynchronize(st));check(cudaMemcpy(out.data(),Y,out.size()*4,cudaMemcpyDeviceToHost));save(dir/("arm"+std::to_string(arm)+".bin"),out);if(!arm)ref=out;
  double ab=0,err=0,mag=0;for(size_t i=0;i<out.size();i++){if(!std::isfinite(out[i]))throw std::runtime_error("nonfinite output");double d=double(out[i])-ref[i];ab=std::max(ab,fabs(d));err+=d*d;mag+=double(ref[i])*ref[i];}bool equal=!memcmp(out.data(),ref.data(),out.size()*4);double rel=sqrt(err/std::max(mag,1e-30));printf("{\"arm\":%d,\"byte_equal\":%s,\"max_abs\":%.9g,\"rel_rms\":%.9g}\n",arm,equal?"true":"false",ab,rel);if((mode!=2&&!equal)||(mode==2&&rel>1e-5))throw std::runtime_error("WF0 changed cached-carrier math");
 }
 cudaGraphExec_t graph[2][2];for(int paid=0;paid<2;paid++)for(int arm=0;arm<2;arm++){cudaGraph_t g;check(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));for(int i=0;i<16;i++)launch(arm,paid);check(cudaStreamEndCapture(st,&g));check(cudaGraphInstantiate(&graph[paid][arm],g,0));check(cudaGraphDestroy(g));for(int i=0;i<3;i++)check(cudaGraphLaunch(graph[paid][arm],st));}check(cudaStreamSynchronize(st));
 cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));double med[2][2];for(int paid=0;paid<2;paid++){std::vector<double>t[2];for(int round=0;round<rounds;round++)for(int slot=0;slot<4;slot++){int order[]={0,1,1,0};int arm=order[slot];float ms;check(cudaEventRecord(begin,st));for(int i=0;i<8;i++)check(cudaGraphLaunch(graph[paid][arm],st));check(cudaEventRecord(end,st));check(cudaEventSynchronize(end));check(cudaEventElapsedTime(&ms,begin,end));t[arm].push_back(ms*1000/128);printf("{\"paid_prep\":%d,\"round\":%d,\"slot\":%d,\"arm\":%d,\"chain_us\":%.9g}\n",paid,round,slot,arm,ms*1000/128);}for(int arm=0;arm<2;arm++){std::sort(t[arm].begin(),t[arm].end());med[paid][arm]=t[arm][t[arm].size()/2];}}
 printf("{\"gemm_control_us\":%.9g,\"gemm_wf0_us\":%.9g,\"gemm_gain_pct\":%.9g,\"prep_gemm_control_us\":%.9g,\"prep_gemm_wf0_us\":%.9g,\"paid_gain_pct\":%.9g,\"repack_ms\":%.9g}\n",med[0][0],med[0][1],100*(1-med[0][1]/med[0][0]),med[1][0],med[1][1],100*(1-med[1][1]/med[1][0]),pack_ms);
 for(auto&gs:graph)for(auto g:gs)check(cudaGraphExecDestroy(g));check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaStreamDestroy(st));for(void*p:{(void*)W3,(void*)W0,(void*)A,(void*)AS,(void*)AG,(void*)mean,(void*)Y,(void*)X,(void*)NW,(void*)SG})check(cudaFree(p));return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
