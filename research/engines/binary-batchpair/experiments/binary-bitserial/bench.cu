#include "bitserial.cuh"
#include "reference_int8.cuh"
#include "packed1.cuh"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#define CUDA(c) do{auto e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
__global__ void evict(unsigned* p,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]+=1;}
__global__ void reference(const uint8_t* raw,const int8_t* q,const float* sx,float* out,int n,int k) {
  int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,t=blockIdx.y;if(row>=n)return;
  float total=0;for(int g=0;g<k/128;g++) {
    const uint8_t* b=raw+(size_t(row)*(k/128)+g)*18;int dot=0;
    for(int j=lane;j<128;j+=32){int a=q[(size_t(t)*(k/128)+g)*128+j];dot+=((b[2+j/8]>>(j%8))&1)?a:-a;}
    for(int d=16;d;d/=2)dot+=__shfl_down_sync(0xffffffff,dot,d);
    total=fmaf(float(dot),__half2float(*reinterpret_cast<const __half*>(b))*sx[size_t(t)*(k/128)+g],total);
  }
  if(lane==0)out[size_t(t)*n+row]=total;
}
struct Graph{cudaGraph_t g;cudaGraphExec_t exec;};
template<class F> Graph capture(cudaStream_t st,F f){Graph g;CUDA(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));f();CUDA(cudaStreamEndCapture(st,&g.g));CUDA(cudaGraphInstantiate(&g.exec,g.g,0));return g;}
float timed(Graph g,cudaStream_t st,unsigned* trash,size_t count,bool cold) {
  cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));std::vector<float> samples;
  for(int z=0;z<18;z++){if(cold)evict<<<(count+255)/256,256,0,st>>>(trash,count);
    CUDA(cudaEventRecord(a,st));CUDA(cudaGraphLaunch(g.exec,st));CUDA(cudaEventRecord(b,st));CUDA(cudaEventSynchronize(b));float ms;CUDA(cudaEventElapsedTime(&ms,a,b));if(z>=3)samples.push_back(ms*1000);}
  std::sort(samples.begin(),samples.end());CUDA(cudaEventDestroy(a));CUDA(cudaEventDestroy(b));return samples[samples.size()/2];
}
int run(int argc,char** argv) {
  if(argc<2)return 1;h5::Gguf model;model.open(argv[1]);
  if(argc==2){for(auto& t:model.tensors)if(t.type==41&&t.ne.size()==2)printf("%s %lu %lu\n",t.name.c_str(),t.ne[1],t.ne[0]);return 0;}
  if(argc!=6){fprintf(stderr,"usage: bench MODEL TENSOR R N_ROWS(0=full) PREFIX\n");return 1;}
  std::vector<uint8_t> joined;h5::GTensor tensor;
  std::vector<std::string> names;
  std::string requested=argv[2];
  if(requested=="alpha_beta")names={"blk.0.ssm_alpha.weight","blk.0.ssm_beta.weight"};
  else if(requested=="gdn_in")names={"blk.0.attn_qkv.weight","blk.0.attn_gate.weight"};
  else if(requested=="attention_in")names={"blk.3.attn_q.weight","blk.3.attn_k.weight","blk.3.attn_v.weight"};
  else if(requested=="gu")names={"blk.0.ffn_gate.weight","blk.0.ffn_up.weight"};
  else names={requested};
  tensor=model.t(names[0]);
  if(names.size()>1) {
    joined.assign(tensor.data,tensor.data+tensor.bytes);
    for(size_t i=1;i<names.size();i++){auto& t=model.t(names[i]);if(t.type!=tensor.type||t.ne[0]!=tensor.ne[0])return 1;joined.insert(joined.end(),t.data,t.data+t.bytes);tensor.ne[1]+=t.ne[1];}
    tensor.data=joined.data();tensor.bytes=joined.size();tensor.name=requested;
  }
  int r=atoi(argv[3]),limit=atoi(argv[4]);
  if(tensor.type!=41||tensor.ne.size()!=2||r<1||r>16||limit<0)return 1;
  int k=tensor.ne[0],n=limit?std::min(size_t(limit),size_t(tensor.ne[1])):tensor.ne[1];if(n%32||k%128||n<=0)return 1;
  const uint8_t* host=tensor.data;size_t blocks=size_t(n)*(k/128);cudaStream_t st;CUDA(cudaStreamCreate(&st));
  uint8_t* raw;uint32_t* signs;__half* scales;float *x,*y,*ref,*tables,*partial;uint4* wi=nullptr;float* si=nullptr;
  CUDA(cudaMalloc(&raw,blocks*18));CUDA(cudaMalloc(&signs,blocks*16));CUDA(cudaMalloc(&scales,blocks*2));
  CUDA(cudaMalloc(&x,size_t(r)*k*4));CUDA(cudaMalloc(&y,size_t(r)*n*4));CUDA(cudaMalloc(&ref,size_t(r)*n*4));
  CUDA(cudaMalloc(&tables,size_t(std::min(r,8))*k*16));CUDA(cudaMalloc(&partial,size_t(r)*n*(k/128)*4));
  bitserial::Carrier c{};CUDA(cudaMalloc(&c.q,size_t(r)*k));CUDA(cudaMalloc(&c.planes,size_t(r)*k));
  CUDA(cudaMalloc(&c.scales,size_t(r)*(k/128)*4));CUDA(cudaMalloc(&c.dc,size_t(r)*(k/128)*4));CUDA(cudaMalloc(&c.status,size_t(r)*(k/128)*4));
  size_t trashn=128*1024*1024/4;unsigned* trash;CUDA(cudaMalloc(&trash,trashn*4));CUDA(cudaMemsetAsync(trash,0,trashn*4,st));
  std::vector<float> hx(size_t(r)*k);for(size_t i=0;i<hx.size();i++)hx[i]=(int(mix(unsigned(i)+999)%65536)-32768)/32761.f;
  CUDA(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));CUDA(cudaMemcpyAsync(raw,host,blocks*18,cudaMemcpyHostToDevice,st));
  CUDA(bitserial::prepare(raw,signs,scales,n,k,st));CUDA(cudaStreamSynchronize(st));
  auto start=std::chrono::steady_clock::now();CUDA(int8_reference::prepare(host,blocks*18,n,k,&wi,&si));
  double int8prep=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(bitserial::produce(bitserial::Route::Popcount,x,k,r,c,st));
  reference<<<dim3((n+3)/4,r),128,0,st>>>(raw,c.q,c.scales,ref,n,k);CUDA(cudaStreamSynchronize(st));
  std::vector<int8_t> hq(size_t(r)*k);std::vector<float> hs(size_t(r)*(k/128));std::vector<uint32_t> hp(size_t(r)*k/4),hd(size_t(r)*(k/128)),flags(hd.size());
  CUDA(cudaMemcpy(hq.data(),c.q,hq.size(),cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(hs.data(),c.scales,hs.size()*4,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(hp.data(),c.planes,hp.size()*4,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(hd.data(),c.dc,hd.size()*4,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(flags.data(),c.status,flags.size()*4,cudaMemcpyDeviceToHost));
  for(size_t g=0;g<hs.size();g++) {if(flags[g]){fprintf(stderr,"invalid inputstatus %u\n",flags[g]);return 3;}int dc=0;
    for(int j=0;j<128;j++)dc+=hq[g*128+j];if(dc!=int(hd[g]))return 3;
    for(int p=0;p<8;p++)for(int w=0;w<4;w++){unsigned bits=0;for(int j=0;j<32;j++)bits|=((unsigned(uint8_t(hq[g*128+w*32+j]))>>p)&1)<<j;if(bits!=hp[g*32+p*4+w])return 3;}}
  std::vector<float> href(size_t(r)*n),hy(href.size()),hf32(href.size());CUDA(cudaMemcpy(href.data(),ref,href.size()*4,cudaMemcpyDeviceToHost));
  FILE* out=fopen((std::string(argv[5])+".jsonl").c_str(),"w");if(!out)return 2;
  auto exact=[&](const std::vector<float>& result){double ae=0;for(int t=0;t<r;t++)for(int s=0;s<std::min(n,64);s++){
    int row=size_t(s)*n/std::min(n,64);double total=0;
    for(int g=0;g<k/128;g++){auto* b=host+(size_t(row)*(k/128)+g)*18;int dot=0;for(int j=0;j<128;j++){int a=hq[(size_t(t)*(k/128)+g)*128+j];dot+=((b[2+j/8]>>(j%8))&1)?a:-a;}
      total+=double(dot)*__half2float(*reinterpret_cast<const __half*>(b))*hs[size_t(t)*(k/128)+g];}
    ae=std::max(ae,std::abs(total-result[size_t(t)*n+row]));}return ae;};
  auto record=[&](const char* route,int chunk,int threads,Graph graph,bool quantized){
    float warm=timed(graph,st,trash,trashn,false),cold=timed(graph,st,trash,trashn,true);CUDA(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
    double maxerr=0,se=0,ss=0,qerr=0,qs=0;for(size_t i=0;i<hy.size();i++){double d=hy[i]-href[i];maxerr=std::max(maxerr,std::abs(d));se+=d*d;ss+=double(href[i])*href[i];d=hy[i]-hf32[i];qerr=std::max(qerr,std::abs(d));qs+=d*d;}
    if(quantized&&(!std::isfinite(maxerr)||sqrt(se/ss)>1e-5)){fprintf(stderr,"numericalfailure %s rms%.9g\n",route,sqrt(se/ss));exit(3);}
    fprintf(out,"{\"tensor\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"route\":\"%s\",\"chunk\":%d,\"threads\":%d,\"warm_us\":%.6f,\"cold_us\":%.6f,\"max_error_a8_f32\":%.9g,\"relative_rms_a8\":%.9g,\"max_error_a8_f64_sample\":%.9g,\"max_error_vs_f32\":%.9g,\"relative_rms_vs_f32\":%.9g,\"int8_prepare_ms\":%.6f}\n",argv[2],n,k,r,route,chunk,threads,warm,cold,maxerr,sqrt(se/ss),exact(hy),qerr,sqrt(qs/ss),int8prep);fflush(out);
    printf("%s N%d K%d R%d chunk%d threads%d cold%.3fus rmsA8%.3g\n",route,n,k,r,chunk,threads,cold,sqrt(se/ss));CUDA(cudaGraphExecDestroy(graph.exec));CUDA(cudaGraphDestroy(graph.g));
  };
  int rr=std::min(r,8);packed1::Plan base{};
  if(n==65536&&k==5120)base=packed1::Plan{n,k,rr,rr==1?40:16,256,packed1::Route::Lut4};
  else CUDA(packed1::select(n,k,rr,&base));
  base.route=packed1::Route::Lut4;
  auto full=[&](){for(int t=0;t<r;t+=rr){auto p=base;p.r=std::min(rr,r-t);CUDA(packed1::project(p,signs,scales,x+size_t(t)*k,y+size_t(t)*n,tables,partial,1,0,st));}};
  full();CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hf32.data(),y,hf32.size()*4,cudaMemcpyDeviceToHost));record(r==16?"lut4_f32_2x8":r>8?"lut4_f32_tiled8":"lut4_f32",base.groups_per_split,base.threads,capture(st,full),false);
  CUDA(int8_reference::project(c.q,c.scales,wi,si,y,n,k,r,1,0,st));CUDA(cudaStreamSynchronize(st));
  auto original=[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(int8_reference::project(c.q,c.scales,wi,si,y,n,k,r,1,0,st));};
  record("original_int8",0,256,capture(st,original),true);
  for(auto route:{bitserial::Route::Popcount,bitserial::Route::Dp4a01}) {
    std::vector<int> chunks={1,4,16,k/128};std::sort(chunks.begin(),chunks.end());chunks.erase(std::unique(chunks.begin(),chunks.end()),chunks.end());
    for(int chunk:chunks)for(int threads:{128,256}){bitserial::Plan p{n,k,r,chunk,threads,route};
      auto f=[&](){CUDA(bitserial::project(p,signs,scales,x,c,y,partial,1,0,st));};record(route==bitserial::Route::Popcount?"popcount":"dp4a01",chunk,threads,capture(st,f),true);}
  }
  fclose(out);return 0;
}
int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"modelerror: %s\n",e.what());return 2;}}
