#include "matvec.cuh"
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
using namespace halo6000;
namespace cg=cooperative_groups;
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s line%d\n",#x,cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
__global__ void prepare(const float*x,ActivationBuffer a){quantize_phase(x,a);}
__global__ void flush_cache(uint32_t*p,size_t n){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]+=1;}
template<int L>__device__ __forceinline__ void phase(MatrixView w,ActivationView a,const float*x,float*out){if constexpr(L==0)matvec_phase(w,a,x,out);else matvec_exact_phase<L>(w,a,x,out);}
template<int L>__global__ void wide(MatrixView w,ActivationView a,const float*x,float*out){phase<L>(w,a,x,out);}
template<int L>__global__ void resident(MatrixView w,ActivationView a,const float*x,float*out,int repeats){auto grid=cg::this_grid();for(int i=0;i<repeats;i++){phase<L>(w,a,x,out);grid.sync();}}
struct Case{std::string key,weights,inputs;int format,n,k;size_t bytes;};
std::vector<uint8_t> bytes(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||size_t(f.tellg())!=n){fprintf(stderr,"wrong fixture %s expected%zu\n",p.c_str(),n);exit(1);}f.seekg(0);std::vector<uint8_t>v(n);f.read(reinterpret_cast<char*>(v.data()),n);if(!f)exit(1);return v;}
cudaStream_t S;
template<int L>void launch(MatrixView w,ActivationView a,const float*x,float*out,int repeats,int block,int grid,bool coop){if(coop){void*args[]={&w,&a,&x,&out,&repeats};CU(cudaLaunchCooperativeKernel((void*)resident<L>,grid,block,args,0,S));}else for(int r=0;r<repeats;r++)wide<L><<<grid,block,0,S>>>(w,a,x,out);CU(cudaGetLastError());}
void guard(const Case&c,int L,int input,const char*mode,int block,const std::vector<float>&expected,const std::vector<float>&got){size_t bad=0;float maxabs=0;for(int i=0;i<c.n;i++){if(memcmp(&expected[i],&got[i],4))++bad;maxabs=fmaxf(maxabs,fabsf(expected[i]-got[i]));}printf("{\"guard\":\"%s\",\"lanes\":%d,\"input\":%d,\"mode\":\"%s\",\"block\":%d,\"rows\":%d,\"bit_mismatches\":%zu,\"max_abs\":%.9g}\n",c.key.c_str(),L,input,mode,block,c.n,bad,maxabs);if(bad){for(int i=0;i<c.n;i++)if(memcmp(&expected[i],&got[i],4)){uint32_t e,g;memcpy(&e,&expected[i],4);memcpy(&g,&got[i],4);fprintf(stderr,"row%d expected %.9g %08x got %.9g %08x\n",i,expected[i],e,got[i],g);break;}exit(2);}}
struct Price{double cold,hot;};
Price price(std::function<void(int)>launch,uint32_t*flush,int hot_reps){cudaGraph_t graphs[2];cudaGraphExec_t execs[2];for(int j=0;j<2;j++){CU(cudaStreamBeginCapture(S,cudaStreamCaptureModeGlobal));launch(j?hot_reps:1);CU(cudaStreamEndCapture(S,&graphs[j]));CU(cudaGraphInstantiate(&execs[j],graphs[j],nullptr,nullptr,0));}cudaEvent_t start,end;CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));double median[2];for(int j=0;j<2;j++){for(int w=0;w<3;w++)CU(cudaGraphLaunch(execs[j],S));CU(cudaStreamSynchronize(S));std::vector<double>samples;for(int s=0;s<9;s++){if(j==0)flush_cache<<<188,256,0,S>>>(flush,(1ULL<<30)/4);CU(cudaEventRecord(start,S));CU(cudaGraphLaunch(execs[j],S));CU(cudaEventRecord(end,S));CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,start,end));samples.push_back(ms*1000/(j?hot_reps:1));}std::sort(samples.begin(),samples.end());median[j]=samples[4];}for(int j=0;j<2;j++){CU(cudaGraphExecDestroy(execs[j]));CU(cudaGraphDestroy(graphs[j]));}CU(cudaEventDestroy(start));CU(cudaEventDestroy(end));return {median[0],median[1]};}
template<int L>void variant(const Case&c,MatrixView w,ActivationBuffer a,float*x,float*out,const std::vector<float>&inputs,const std::vector<std::vector<float>>&refs,uint32_t*flush){
 constexpr int Width=L?L:32;std::vector<float>got(c.n);std::vector<int>blocks={256,128};if(c.format==0)blocks.push_back(32);
 for(int i=0;i<int(refs.size());i++){CU(cudaMemcpyAsync(x,inputs.data()+size_t(i)*c.k,c.k*4,cudaMemcpyHostToDevice,S));prepare<<<188,256,0,S>>>(x,a);if constexpr(L!=0){launch<L>(w,a.view(),x,out,1,256,188,true);CU(cudaMemcpyAsync(got.data(),out,c.n*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));guard(c,L,i,"resident188",256,refs[i],got);}for(int block:blocks){const int grid=(c.n+block/Width-1)/(block/Width);launch<L>(w,a.view(),x,out,1,block,grid,false);CU(cudaMemcpyAsync(got.data(),out,c.n*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));guard(c,L,i,"wide",block,refs[i],got);}}
 CU(cudaMemcpyAsync(x,inputs.data(),c.k*4,cudaMemcpyHostToDevice,S));prepare<<<188,256,0,S>>>(x,a);int hot_reps=c.n>100000?8:32;
 cudaFuncAttributes fa;CU(cudaFuncGetAttributes(&fa,resident<L>));int occupancy;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,resident<L>,256,0));printf("{\"resource\":\"%s\",\"lanes\":%d,\"registers\":%d,\"local_bytes\":%zu,\"resident_blocks_per_SM\":%d}\n",c.key.c_str(),L,fa.numRegs,fa.localSizeBytes,occupancy);
 Price r=price([&](int reps){launch<L>(w,a.view(),x,out,reps,256,188,true);},flush,hot_reps);printf("{\"timing\":\"%s\",\"lanes\":%d,\"mode\":\"resident188\",\"block\":256,\"grid\":188,\"cold_us\":%.6f,\"hot_us\":%.6f,\"hot_repeats\":%d}\n",c.key.c_str(),L,r.cold,r.hot,hot_reps);
 for(int block:blocks){int grid=(c.n+block/Width-1)/(block/Width);Price r=price([&](int reps){launch<L>(w,a.view(),x,out,reps,block,grid,false);},flush,hot_reps);printf("{\"timing\":\"%s\",\"lanes\":%d,\"mode\":\"wide\",\"block\":%d,\"grid\":%d,\"cold_us\":%.6f,\"hot_us\":%.6f,\"hot_repeats\":%d}\n",c.key.c_str(),L,block,grid,r.cold,r.hot,hot_reps);}
}
int main(int argc,char**argv){if(argc!=2){fprintf(stderr,"usage: row-exact FIXTURE_DIRECTORY\n");return 1;}setvbuf(stdout,nullptr,_IOLBF,0);std::ifstream manifest(std::string(argv[1])+"/cases.tsv");if(!manifest)return 1;std::vector<Case>cases;std::string line;while(std::getline(manifest,line)){std::istringstream s(line);Case c;s>>c.key>>c.format>>c.n>>c.k>>c.bytes>>c.weights>>c.inputs;if(!s)return 1;cases.push_back(c);}if(cases.size()!=8)return 1;cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));printf("{\"device\":\"%s\",\"SMs\":%d,\"cases\":8,\"actual_inputs\":16,\"stress_inputs\":6}\n",prop.name,prop.multiProcessorCount);if(prop.multiProcessorCount<188)return 1;CU(cudaStreamCreate(&S));uint32_t*flush;CU(cudaMalloc(&flush,1ULL<<30));CU(cudaMemsetAsync(flush,0,1ULL<<30,S));
 for(const Case&c:cases){auto raw=bytes(c.weights,c.bytes),actual=bytes(c.inputs,size_t(16)*c.k*4);std::vector<float>inputs(size_t(22)*c.k);memcpy(inputs.data(),actual.data(),actual.size());for(int i=0;i<c.k;i++){inputs[size_t(16)*c.k+i]=0;inputs[size_t(17)*c.k+i]=std::copysign(0.f,i&1?-1.f:1.f);inputs[size_t(18)*c.k+i]=i&1?1e-8f:-1e-8f;inputs[size_t(19)*c.k+i]=float((i*37)%255-127)*.03125f;inputs[size_t(20)*c.k+i]=i%c.k==c.k/2?3.25f:0;inputs[size_t(21)*c.k+i]=std::ldexp(float(((i*73)%31)-15),(i%19)-9);}
  uint8_t*d_weights;float*x,*out,*ds;int8_t*qs;CU(cudaMalloc(&d_weights,c.bytes));CU(cudaMemcpyAsync(d_weights,raw.data(),c.bytes,cudaMemcpyHostToDevice,S));CU(cudaMalloc(&x,c.k*4));CU(cudaMalloc(&out,c.n*4));CU(cudaMalloc(&ds,c.k/32*4));CU(cudaMalloc(&qs,c.k));MatrixView w;w.data=d_weights;w.n=c.n;w.k=c.k;w.format=WeightFormat(c.format);w.row_stride=c.format==0?c.k*4:0;w.bytes=c.bytes;ActivationBuffer a{qs,ds,c.k,1,32,0,c.k,c.k/32};std::vector<std::vector<float>>refs(22,std::vector<float>(c.n));
  printf("{\"shape\":\"%s\",\"N\":%d,\"K\":%d,\"format\":%d,\"bytes\":%zu}\n",c.key.c_str(),c.n,c.k,c.format,c.bytes);
  for(int i=0;i<22;i++){CU(cudaMemcpyAsync(x,inputs.data()+size_t(i)*c.k,c.k*4,cudaMemcpyHostToDevice,S));prepare<<<188,256,0,S>>>(x,a);launch<0>(w,a.view(),x,out,1,256,188,true);CU(cudaMemcpyAsync(refs[i].data(),out,c.n*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));}
  variant<0>(c,w,a,x,out,inputs,refs,flush);variant<4>(c,w,a,x,out,inputs,refs,flush);variant<8>(c,w,a,x,out,inputs,refs,flush);variant<16>(c,w,a,x,out,inputs,refs,flush);variant<32>(c,w,a,x,out,inputs,refs,flush);
  CU(cudaFree(qs));CU(cudaFree(ds));CU(cudaFree(out));CU(cudaFree(x));CU(cudaFree(d_weights));
 }
 CU(cudaFree(flush));CU(cudaStreamDestroy(S));printf("{\"passed\":true,\"cases\":8,\"inputs_per_case\":22,\"exact_header_edited\":false}\n");return 0;
}
