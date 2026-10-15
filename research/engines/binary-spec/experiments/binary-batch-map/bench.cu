#include "batch_map.cuh"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#define CK(c) do{auto e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
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
template<class F> Graph capture(cudaStream_t st,F f){Graph g;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));f();CK(cudaStreamEndCapture(st,&g.g));CK(cudaGraphInstantiate(&g.exec,g.g,0));return g;}
float timed(Graph g,cudaStream_t st,unsigned* trash,size_t count,bool cold) {
  cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));std::vector<float> samples;
  for(int z=0;z<12;z++){if(cold)evict<<<(count+255)/256,256,0,st>>>(trash,count);
    CK(cudaEventRecord(a,st));CK(cudaGraphLaunch(g.exec,st));CK(cudaEventRecord(b,st));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));if(z>=3)samples.push_back(ms*1000);}
  std::sort(samples.begin(),samples.end());CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return samples[samples.size()/2];
}
void destroy(Graph g){CK(cudaGraphExecDestroy(g.exec));CK(cudaGraphDestroy(g.g));}
int run(int argc,char** argv) {
  if(argc!=6){fprintf(stderr,"usage: bench MODEL TENSOR R N_ROWS(0=full) PREFIX\n");return 1;}
  h5::Gguf model;model.open(argv[1]);std::string requested=argv[2];std::vector<std::string> names;
  if(requested=="gu")names={"blk.0.ffn_gate.weight","blk.0.ffn_up.weight"};else names={requested};
  auto tensor=model.t(names[0]);std::vector<uint8_t> joined;
  if(names.size()>1){joined.assign(tensor.data,tensor.data+tensor.bytes);for(size_t i=1;i<names.size();i++){
    auto& t=model.t(names[i]);if(t.type!=tensor.type||t.ne[0]!=tensor.ne[0])return 1;
    joined.insert(joined.end(),t.data,t.data+t.bytes);tensor.ne[1]+=t.ne[1];}tensor.data=joined.data();}
  int r=atoi(argv[3]),limit=atoi(argv[4]);if(r!=2&&r!=4&&r!=5&&r!=8&&r!=16&&r!=32)return 1;
  if(tensor.type!=41||tensor.ne.size()!=2||limit<0)return 1;int k=tensor.ne[0],n=limit?std::min(size_t(limit),size_t(tensor.ne[1])):tensor.ne[1];if(n<=0||k%128)return 1;
  int groups=k/128,rr=std::min(r,16);size_t blocks=size_t(n)*groups;cudaStream_t st;CK(cudaStreamCreate(&st));
  uint8_t *raw,*counts;uint32_t *signs,*packed;__half* scales;float *x,*y,*ref,*partial;
  CK(cudaMalloc(&raw,blocks*18));CK(cudaMalloc(&signs,blocks*16));CK(cudaMalloc(&scales,blocks*2));CK(cudaMalloc(&counts,blocks));
  CK(cudaMalloc(&packed,size_t((r+1)/2)*k*4));CK(cudaMalloc(&x,size_t(r)*k*4));CK(cudaMalloc(&y,size_t(r)*n*4));CK(cudaMalloc(&ref,size_t(r)*n*4));CK(cudaMalloc(&partial,size_t(rr)*n*groups*4));
  bitserial::Carrier c{},mc{};
  CK(cudaMalloc(&c.q,size_t(r)*k));
  for(auto* m:{&c,&mc}){CK(cudaMalloc(&m->scales,size_t(r)*groups*4));CK(cudaMalloc(&m->dc,size_t(r)*groups*4));CK(cudaMalloc(&m->status,size_t(r)*groups*4));}
  batch_map::Carrier bc{packed,mc,k,r};size_t trashn=128*1024*1024/4;unsigned* trash;CK(cudaMalloc(&trash,trashn*4));CK(cudaMemsetAsync(trash,0,trashn*4,st));
  std::vector<float> hx(size_t(r)*k);for(size_t i=0;i<hx.size();i++)hx[i]=(int(mix(unsigned(i)+999)%65536)-32768)/32761.f;
  CK(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));CK(cudaMemcpyAsync(raw,tensor.data,blocks*18,cudaMemcpyHostToDevice,st));
  CK(bitserial::prepare(raw,signs,scales,n,k,st));CK(batch_map::prepare_counts(signs,counts,n,k,st));
  CK(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CK(batch_map::produce(x,bc,st));
  reference<<<dim3((n+3)/4,r),128,0,st>>>(raw,c.q,c.scales,ref,n,k);CK(cudaStreamSynchronize(st));
  std::vector<int8_t> hq(size_t(r)*k);std::vector<unsigned> hp(size_t((r+1)/2)*k);std::vector<float> hs(size_t(r)*groups),hms(hs.size());
  std::vector<int> hd(hs.size()),hmd(hs.size());std::vector<unsigned> flags(hs.size());
  CK(cudaMemcpy(hq.data(),c.q,hq.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(hp.data(),packed,hp.size()*4,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hs.data(),c.scales,hs.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hms.data(),mc.scales,hms.size()*4,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hd.data(),c.dc,hd.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hmd.data(),mc.dc,hmd.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(flags.data(),mc.status,flags.size()*4,cudaMemcpyDeviceToHost));
  if(hs!=hms||hd!=hmd||std::any_of(flags.begin(),flags.end(),[](unsigned v){return v!=0;})){fprintf(stderr,"producer metadata mismatch\n");return 3;}
  for(int p=0;p<(r+1)/2;p++)for(int j=0;j<k;j++)for(int h=0;h<2;h++) {
    int t=p*2+h;unsigned want=t<r?unsigned(int(hq[size_t(t)*k+j])+128):0;
    if(((hp[size_t(p)*k+j]>>(h*16))&65535)!=want){fprintf(stderr,"packed producer mismatch\n");return 3;}}
  std::vector<float> href(size_t(r)*n),hy(href.size());CK(cudaMemcpy(href.data(),ref,href.size()*4,cudaMemcpyDeviceToHost));
  FILE* out=fopen((std::string(argv[5])+".jsonl").c_str(),"w");if(!out)return 2;
  auto record=[&](const char* route,int chunk,int threads,auto prod,auto consume) {
    auto full=capture(st,[&](){prod();consume();});float warm=timed(full,st,trash,trashn,false),cold=timed(full,st,trash,trashn,true);
    CK(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));double maxerr=0,se=0,ss=0;
    for(size_t i=0;i<hy.size();i++){if(!std::isfinite(hy[i]))return false;double d=hy[i]-href[i];maxerr=std::max(maxerr,std::abs(d));se+=d*d;ss+=double(href[i])*href[i];}
    if(sqrt(se/ss)>1e-5)return false;
    auto pg=capture(st,prod),cg=capture(st,consume);float pu=timed(pg,st,trash,trashn,false),cu=timed(cg,st,trash,trashn,true);
    fprintf(out,"{\"tensor\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"route\":\"%s\",\"chunk\":%d,\"threads\":%d,\"producer_us\":%.6f,\"consumer_cold_us\":%.6f,\"warm_us\":%.6f,\"cold_us\":%.6f,\"max_error_a8\":%.9g,\"relative_rms_a8\":%.9g,\"outputs_checked\":%zu}\n",argv[2],n,k,r,route,chunk,threads,pu,cu,warm,cold,maxerr,sqrt(se/ss),hy.size());fflush(out);
    printf("%s N%d K%d R%d chunk%d threads%d cold%.3fus rms%.3g\n",route,n,k,r,chunk,threads,cold,sqrt(se/ss));destroy(full);destroy(pg);destroy(cg);return true;
  };
  bitserial::Plan dp;CK(bitserial::select(n,k,rr,&dp));
  auto dp_prod=[&](){CK(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));};
  auto dp_consume=[&](){for(int t=0;t<r;t+=rr){auto cc=c;cc.q+=size_t(t)*k;cc.scales+=size_t(t)*groups;cc.dc+=size_t(t)*groups;cc.status+=size_t(t)*groups;CK(bitserial::project_carrier(dp,signs,scales,cc,y+size_t(t)*n,partial,1,0,st));}};
  if(!record("dp4a_selected",dp.groups_per_split,dp.threads,dp_prod,dp_consume))return 3;
  std::vector<int> chunks={1,4,16,groups};std::sort(chunks.begin(),chunks.end());chunks.erase(std::unique(chunks.begin(),chunks.end()),chunks.end());
  for(int chunk:chunks)for(int threads:{128,256}){
    auto prod=[&](){CK(batch_map::produce(x,bc,st));};auto consume=[&](){for(int t=0;t<r;t+=rr){batch_map::Plan p{n,k,rr,t,chunk,threads};CK(batch_map::project(p,signs,scales,counts,bc,y+size_t(t)*n,partial,1,0,st));}};
    if(!record("imad_pair",chunk,threads,prod,consume))return 3;
  }
  fclose(out);return 0;
}
int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"modelerror: %s\n",e.what());return 2;}}
