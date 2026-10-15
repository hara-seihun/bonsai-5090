#include "head.cuh"
#include <cooperative_groups.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
using namespace halo6000;
namespace cg=cooperative_groups;
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s line%d\n",#x,cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int N=248320,K=2048;
__global__ void quant_input(const float*x,ActivationBuffer a){quantize_phase(x,a);}
__global__ void baseline(MatrixView w,ActivationView a,float*out,int repeats){auto grid=cg::this_grid();for(int r=0;r<repeats;r++){matvec_phase(w,a,nullptr,out);grid.sync();}}
template<int L,bool P,bool C=false>
__global__ void resident(MatrixView w,HeadPreparedView p,ActivationView a,float*out,int repeats){auto grid=cg::this_grid();for(int r=0;r<repeats;r++){head_phase_impl<L,P,C>(w,p,a,out);grid.sync();}}
__global__ void flush_cache(uint32_t*p,size_t n){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]+=1;}
__global__ void verify_prepared(const uint8_t*src,HeadPreparedView p,int*bad){const int R=32/p.lanes;for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(N)*512;i+=size_t(gridDim.x)*blockDim.x){int r=i%R,g=i/R%16,t=i/R/16%32,row=int(i/R/16/32)*R+r;int q=load_i8x4(reinterpret_cast<const int8_t*>(src)+(size_t(row)*16+g)*136+t*4);if(p.words[i]!=q)atomicAdd(bad,1);}for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(N)*64;i+=size_t(gridDim.x)*blockDim.x){int r=i%R,g=i/R%16,c=i/R/16%4,row=int(i/R/16/4)*R+r;auto s=*reinterpret_cast<const uint16_t*>(src+(size_t(row)*16+g)*136+128+c*2);if(p.scales[i]!=s)atomicAdd(bad,1);}}
std::vector<uint8_t> read_bytes(const std::string&p,size_t expected){std::ifstream f(p,std::ios::binary);if(!f){fprintf(stderr,"cannot read %s\n",p.c_str());exit(1);}f.seekg(0,std::ios::end);auto n=size_t(f.tellg());if(n!=expected){fprintf(stderr,"wrong bytes %s %zu !=%zu\n",p.c_str(),n,expected);exit(1);}f.seekg(0);std::vector<uint8_t>v(n);f.read(reinterpret_cast<char*>(v.data()),n);if(!f)exit(1);return v;}
float hf(uint16_t h){uint32_t sign=uint32_t(h&32768)<<16,e=(h>>10)&31,m=h&1023,b;if(e==31)b=sign|0x7f800000|(m<<13);else if(e)b=sign|((e+112)<<23)|(m<<13);else if(!m)b=sign;else{int exponent=-14;while(!(m&1024)){m<<=1;--exponent;}b=sign|uint32_t(exponent+127)<<23|((m&1023)<<13);}float f;memcpy(&f,&b,4);return f;}
float cpu_row(const uint8_t*w,const int8_t*x,const float*ds,int row){float v[16];for(int g=0;g<16;g++){v[g]=0;const auto*q=reinterpret_cast<const int8_t*>(w+(size_t(row)*16+g)*136);for(int c=0;c<4;c++){int isum=0;for(int t=0;t<32;t++)isum+=int(q[c*32+t])*int(x[g*128+c*32+t]);uint16_t sc;memcpy(&sc,q+128+c*2,2);float scale=hf(sc)*ds[g*4+c];v[g]=std::fma(scale,float(isum),v[g]);}v[g]=0.f+v[g];}for(int d=8;d;d/=2)for(int r=0;r<d;r++)v[r]+=v[r+d];return v[0];}
void guard(const std::vector<float>&oracle,const std::vector<float>&actual,const char*name,int input){size_t bad=0;float maxabs=0;for(int i=0;i<N;i++){if(memcmp(&oracle[i],&actual[i],4))++bad;maxabs=fmaxf(maxabs,fabsf(oracle[i]-actual[i]));}printf("{\"guard\":\"%s\",\"input\":%d,\"bit_mismatches\":%zu,\"max_abs\":%.9g}\n",name,input,bad,maxabs);if(bad)exit(2);}
struct Timing{double cold,hot;};
// A non-default stream is required for capture (the helpers below use it too).
cudaStream_t S;
Timing time_on_stream(std::function<void(int)>launch,uint32_t*flush){cudaGraph_t graphs[2];cudaGraphExec_t execs[2];for(int j=0;j<2;j++){CU(cudaStreamBeginCapture(S,cudaStreamCaptureModeGlobal));launch(j?8:1);CU(cudaStreamEndCapture(S,&graphs[j]));CU(cudaGraphInstantiate(&execs[j],graphs[j],nullptr,nullptr,0));}cudaEvent_t start,end;CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));double med[2];for(int j=0;j<2;j++){for(int w=0;w<3;w++)CU(cudaGraphLaunch(execs[j],S));CU(cudaStreamSynchronize(S));std::vector<double>samples;for(int s=0;s<9;s++){if(j==0)flush_cache<<<188,256,0,S>>>(flush,(1ULL<<30)/4);CU(cudaEventRecord(start,S));CU(cudaGraphLaunch(execs[j],S));CU(cudaEventRecord(end,S));CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,start,end));samples.push_back(ms*1000/(j?8:1));}std::sort(samples.begin(),samples.end());med[j]=samples[4];}for(int j=0;j<2;j++){CU(cudaGraphExecDestroy(execs[j]));CU(cudaGraphDestroy(graphs[j]));}CU(cudaEventDestroy(start));CU(cudaEventDestroy(end));return {med[0],med[1]};}
void print_price(const char*name,const char*launch,Timing t,int regs){printf("{\"timing\":\"%s\",\"launch\":\"%s\",\"cold_us\":%.6f,\"hot_us\":%.6f,\"registers\":%d,\"cold_GBs\":%.6f}\n",name,launch,t.cold,t.hot,regs,double(size_t(N)*16*136)/t.cold/1000);}
void coop_base(MatrixView w,ActivationView a,float*out,int reps,cudaStream_t stream){void*args[]={&w,&a,&out,&reps};CU(cudaLaunchCooperativeKernel((void*)baseline,188,256,args,0,stream));}
template<int L,bool P,bool C=false>void coop(MatrixView w,HeadPreparedView p,ActivationView a,float*out,int reps,cudaStream_t stream){void*args[]={&w,&p,&a,&out,&reps};CU(cudaLaunchCooperativeKernel((void*)resident<L,P,C>,188,256,args,0,stream));}
template<int L,bool P,bool C=false>
void variant(MatrixView w,HeadPreparedView p,ActivationBuffer a,float*out,const std::vector<float>&inputs,const std::vector<std::vector<float>>&refs,float*d_input,uint32_t*flush){
 char name[64];snprintf(name,sizeof(name),"%s-L%d%s",P?"prepared":"grouped128",L,C?"-chunk":"");
 if constexpr(P){p.lanes=L;cudaEvent_t begin,end;CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));CU(cudaEventRecord(begin,S));CU(prepare_head_q8<L>(w,p,S));CU(cudaEventRecord(end,S));CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,begin,end));CU(cudaEventDestroy(begin));CU(cudaEventDestroy(end));printf("{\"prepare\":\"%s\",\"one_time_ms\":%.6f,\"prepared_bytes\":%zu}\n",name,double(ms),head_prepared_word_bytes<L>(N)+head_prepared_scale_bytes<L>(N));int*bad;CU(cudaMalloc(&bad,4));CU(cudaMemsetAsync(bad,0,4,S));verify_prepared<<<188,256,0,S>>>(static_cast<const uint8_t*>(w.data),p,bad);int count;CU(cudaMemcpyAsync(&count,bad,4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));CU(cudaFree(bad));printf("{\"permutation\":\"%s\",\"byte_mismatches\":%d}\n",name,count);if(count)exit(2);}
 std::vector<float>got(N);
 for(int i=0;i<int(refs.size());i++){CU(cudaMemcpyAsync(d_input,inputs.data()+i*K,K*4,cudaMemcpyHostToDevice,S));quant_input<<<188,256,0,S>>>(d_input,a);coop<L,P,C>(w,p,a.view(),out,1,S);CU(cudaMemcpyAsync(got.data(),out,N*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));guard(refs[i],got,name,i);if constexpr(P)CU(launch_head_prepared<L>(p,a.view(),out,S));else CU(launch_head_q8<L>(w,a.view(),out,S));CU(cudaMemcpyAsync(got.data(),out,N*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));char wide_name[80];snprintf(wide_name,sizeof(wide_name),"%s-wide",name);guard(refs[i],got,wide_name,i);}
 CU(cudaMemcpyAsync(d_input,inputs.data(),K*4,cudaMemcpyHostToDevice,S));quant_input<<<188,256,0,S>>>(d_input,a);
 cudaFuncAttributes fa;CU(cudaFuncGetAttributes(&fa,resident<L,P,C>));int occupancy;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,resident<L,P,C>,256,0));printf("{\"resource\":\"%s\",\"resident_blocks_per_SM\":%d,\"registers\":%d,\"local_bytes\":%zu}\n",name,occupancy,fa.numRegs,fa.localSizeBytes);
 auto resident_t=time_on_stream([&](int r){coop<L,P,C>(w,p,a.view(),out,r,S);},flush);print_price(name,"resident188",resident_t,fa.numRegs);
 if constexpr(!C){CU(cudaFuncGetAttributes(&fa,head_kernel<L,P>));auto wide_t=time_on_stream([&](int r){for(int j=0;j<r;j++){if constexpr(P)CU(launch_head_prepared<L>(p,a.view(),out,S));else CU(launch_head_q8<L>(w,a.view(),out,S));}},flush);print_price(name,"wide-graph",wide_t,fa.numRegs);}
}
int main(int argc,char**argv){if(argc!=2){fprintf(stderr,"usage: head FIXTURE_DIRECTORY\n");return 1;}setvbuf(stdout,nullptr,_IOLBF,0);cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));printf("{\"device\":\"%s\",\"SMs\":%d,\"head_bytes\":%zu,\"actual_inputs\":16}\n",prop.name,prop.multiProcessorCount,size_t(N)*16*136);if(prop.multiProcessorCount<188)return 1;CU(cudaStreamCreate(&S));auto raw=read_bytes(std::string(argv[1])+"/head.q8",size_t(N)*16*136);auto acts=read_bytes(std::string(argv[1])+"/head-input.f32",16*K*4);std::vector<float>inputs(20*K);memcpy(inputs.data(),acts.data(),acts.size());for(int j=0;j<K;j++){inputs[16*K+j]=0;inputs[17*K+j]=j&1?1e-8f:-1e-8f;inputs[18*K+j]=((j*37)%255-127)*.03125f;inputs[19*K+j]=j%32==0?3.25f:0;}
 uint8_t*d_w;float*d_input,*out,*d_scale;int8_t*d_q;int*prep;uint16_t*prep_scale;uint32_t*flush;
 CU(cudaMalloc(&d_w,raw.size()));CU(cudaMemcpy(d_w,raw.data(),raw.size(),cudaMemcpyHostToDevice));CU(cudaMalloc(&d_input,K*4));CU(cudaMalloc(&out,N*4));CU(cudaMalloc(&d_q,K));CU(cudaMalloc(&d_scale,64*4));CU(cudaMalloc(&prep,size_t(N)*K));CU(cudaMalloc(&prep_scale,size_t(N)*64*2));CU(cudaMalloc(&flush,1ULL<<30));CU(cudaMemset(flush,0,1ULL<<30));
 MatrixView w;w.data=d_w;w.n=N;w.k=K;w.format=WeightFormat::Q8Grouped128;w.bytes=raw.size();ActivationBuffer a{d_q,d_scale,K,1,32,0,K,64};HeadPreparedView p{prep,prep_scale,N,K,0};
 std::vector<std::vector<float>>refs(20,std::vector<float>(N));std::array<int8_t,K>qs;std::array<float,64>ds;size_t cpu_bad=0;
 for(int i=0;i<20;i++){CU(cudaMemcpyAsync(d_input,inputs.data()+i*K,K*4,cudaMemcpyHostToDevice,S));quant_input<<<188,256,0,S>>>(d_input,a);coop_base(w,a.view(),out,1,S);CU(cudaMemcpyAsync(refs[i].data(),out,N*4,cudaMemcpyDeviceToHost,S));CU(cudaMemcpyAsync(qs.data(),d_q,K,cudaMemcpyDeviceToHost,S));CU(cudaMemcpyAsync(ds.data(),d_scale,64*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));for(int j=0;j<257;j++){int row=j==256?N-1:j*967%N;float expected=cpu_row(raw.data(),qs.data(),ds.data(),row);if(memcmp(&expected,&refs[i][row],4)){if(cpu_bad<4)fprintf(stderr,"CPU mismatch input%d row%d %.9g %.9g\n",i,row,expected,refs[i][row]);cpu_bad++;}}}
 printf("{\"cpu_independent_int32_and_float_tree\":true,\"checked_rows\":5140,\"bit_mismatches\":%zu}\n",cpu_bad);if(cpu_bad)return 2;
 CU(cudaMemcpyAsync(d_input,inputs.data(),K*4,cudaMemcpyHostToDevice,S));quant_input<<<188,256,0,S>>>(d_input,a);cudaFuncAttributes fa;CU(cudaFuncGetAttributes(&fa,baseline));print_price("native-warp32","resident188",time_on_stream([&](int r){coop_base(w,a.view(),out,r,S);},flush),fa.numRegs);
 variant<4,false>(w,p,a,out,inputs,refs,d_input,flush);variant<8,false>(w,p,a,out,inputs,refs,d_input,flush);variant<16,false>(w,p,a,out,inputs,refs,d_input,flush);variant<32,false>(w,p,a,out,inputs,refs,d_input,flush);
 variant<4,true>(w,p,a,out,inputs,refs,d_input,flush);variant<8,true>(w,p,a,out,inputs,refs,d_input,flush);variant<16,true>(w,p,a,out,inputs,refs,d_input,flush);variant<32,true>(w,p,a,out,inputs,refs,d_input,flush);
 variant<4,false,true>(w,p,a,out,inputs,refs,d_input,flush);variant<8,false,true>(w,p,a,out,inputs,refs,d_input,flush);variant<16,false,true>(w,p,a,out,inputs,refs,d_input,flush);variant<32,false,true>(w,p,a,out,inputs,refs,d_input,flush);
 variant<4,true,true>(w,p,a,out,inputs,refs,d_input,flush);variant<8,true,true>(w,p,a,out,inputs,refs,d_input,flush);variant<16,true,true>(w,p,a,out,inputs,refs,d_input,flush);variant<32,true,true>(w,p,a,out,inputs,refs,d_input,flush);
 CU(cudaDeviceSynchronize());printf("{\"passed\":true,\"candidate_inputs\":320,\"rows_each\":%d,\"same_cached_bytes\":true}\n",N);
 CU(cudaFree(flush));CU(cudaFree(prep_scale));CU(cudaFree(prep));CU(cudaFree(d_scale));CU(cudaFree(d_q));CU(cudaFree(out));CU(cudaFree(d_input));CU(cudaFree(d_w));CU(cudaStreamDestroy(S));return 0;
}
