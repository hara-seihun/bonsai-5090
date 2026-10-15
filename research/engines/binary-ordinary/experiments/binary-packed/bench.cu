#include "packed1.cuh"
#include "tuning.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#include <chrono>
#define CUDA(c) do { auto e=(c); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} } while(0)
__host__ __device__ unsigned mix(unsigned x) { x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16); }
__host__ __device__ float scale_value(size_t i) { return (1+(mix(unsigned(i)^11)%127))/128.0f; }
__global__ void init_weights(uint8_t* w,size_t blocks) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i>=blocks)return;
  *reinterpret_cast<__half*>(w+i*18)=__float2half(scale_value(i));
  for(int j=0;j<4;j++){unsigned b=mix(unsigned(i)*4+j+17);for(int t=0;t<4;t++)w[i*18+2+j*4+t]=b>>(8*t);}
}
__global__ void evict(unsigned* v,size_t n) { size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)v[i]+=1; }
__global__ void reference(const uint8_t* w,const float* x,float* y,int n,int k,int r) {
  int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,token=blockIdx.y;
  if(row>=n)return; float sum=0;
  for(int g=0;g<k/128;g++) {
    const uint8_t* b=w+(size_t(row)*(k/128)+g)*18;float dot=0;
    for(int j=lane;j<128;j+=32) dot+=((b[2+j/8]>>(j%8))&1)?x[size_t(token)*k+g*128+j]:-x[size_t(token)*k+g*128+j];
    for(int s=16;s>0;s/=2)dot+=__shfl_down_sync(0xffffffff,dot,s);
    sum=fmaf(dot,__half2float(*reinterpret_cast<const __half*>(b)),sum);
  }
  if(lane==0)y[size_t(token)*n+row]=sum;
}
struct Graph {cudaGraph_t graph;cudaGraphExec_t exec;};
Graph capture(packed1::Plan p,uint32_t* signs,__half* scales,float* x,float* y,float* tables,float* partial,cudaStream_t stream,float a,float b,const std::string& mode) {
  Graph g;CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  if(mode=="lut8"||mode=="ilp") {
    packed1::tuning::Plan tp{p.n,p.k,p.r,p.groups_per_split,p.threads,mode=="lut8"?packed1::tuning::Route::Lut8:packed1::tuning::Route::Lut4Ilp};
    CUDA(packed1::tuning::project(tp,signs,scales,x,y,tables,partial,a,b,stream));
  } else CUDA(packed1::project(p,signs,scales,x,y,tables,partial,a,b,stream));
  CUDA(cudaStreamEndCapture(stream,&g.graph));CUDA(cudaGraphInstantiate(&g.exec,g.graph,0));return g;
}
float timed(Graph g,cudaStream_t st,unsigned* trash,size_t trashn,bool cold,int iters) {
  cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));std::vector<float> times;
  for(int i=0;i<iters+3;i++) {
    if(cold)evict<<<(trashn+255)/256,256,0,st>>>(trash,trashn);
    CUDA(cudaEventRecord(a,st));CUDA(cudaGraphLaunch(g.exec,st));CUDA(cudaEventRecord(b,st));CUDA(cudaEventSynchronize(b));
    float ms;CUDA(cudaEventElapsedTime(&ms,a,b));if(i>=3)times.push_back(ms*1000);
  }
  std::sort(times.begin(),times.end());CUDA(cudaEventDestroy(a));CUDA(cudaEventDestroy(b));return times[times.size()/2];
}
int main(int argc,char**argv) {
  if(argc!=6){fprintf(stderr,"usage: bench N K R route(lut|sign|warp|sweep|selected|lut8|ilp) output-prefix\n");return 1;}
  int n=atoi(argv[1]),k=atoi(argv[2]),r=atoi(argv[3]);std::string route=argv[4];
  if(n<=0||k<=0||k%128||r<1||r>8||(route!="lut"&&route!="sign"&&route!="warp"&&route!="sweep"&&route!="selected"&&route!="lut8"&&route!="ilp"))return 1;
  cudaDeviceProp prop;CUDA(cudaGetDeviceProperties(&prop,0));fprintf(stderr,"%s SM%d CUDA runtime%d\n",prop.name,prop.multiProcessorCount,CUDART_VERSION);
  cudaStream_t st;CUDA(cudaStreamCreate(&st));size_t blocks=size_t(n)*(k/128);
  uint8_t* raw;uint32_t* signs;__half* scales;float *x,*y,*tables,*partial,*ref;
  CUDA(cudaMalloc(&raw,blocks*18));CUDA(cudaMalloc(&signs,blocks*16));CUDA(cudaMalloc(&scales,blocks*2));
  CUDA(cudaMalloc(&x,size_t(r)*k*4));CUDA(cudaMalloc(&y,size_t(r)*n*4));CUDA(cudaMalloc(&ref,size_t(r)*n*4));
  CUDA(cudaMalloc(&tables,size_t(r)*k*(route=="lut8"?64:16)));
  if(route=="lut8"||route=="ilp")CUDA(packed1::tuning::configure());CUDA(cudaMalloc(&partial,size_t(r)*n*(k/128)*4));
  size_t trashn=128*1024*1024/4;unsigned* trash;CUDA(cudaMalloc(&trash,trashn*4));CUDA(cudaMemsetAsync(trash,0,trashn*4,st));
  std::vector<float> hx(size_t(r)*k);for(size_t i=0;i<hx.size();i++)hx[i]=(int(mix(unsigned(i)+999)%65536)-32768)/32761.f;
  CUDA(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));CUDA(cudaMemsetAsync(y,0,size_t(r)*n*4,st));
  init_weights<<<(blocks+255)/256,256,0,st>>>(raw,blocks);
  auto t0=std::chrono::steady_clock::now();CUDA(packed1::prepare(raw,signs,scales,n,k,st));CUDA(cudaStreamSynchronize(st));
  double prepms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
  reference<<<dim3((n+3)/4,r),128,0,st>>>(raw,x,ref,n,k,r);CUDA(cudaStreamSynchronize(st));
  std::vector<float> href(size_t(r)*n),hy(href.size());CUDA(cudaMemcpy(href.data(),ref,href.size()*4,cudaMemcpyDeviceToHost));
  FILE* f=fopen((std::string(argv[5])+".jsonl").c_str(),"w");if(!f)return 2;
  float best=1e30;packed1::Plan bestp{};
  packed1::Plan selected{};if(route=="selected")CUDA(packed1::select(n,k,r,&selected));
  for(auto mode:{packed1::Route::Lut4,packed1::Route::Sign,packed1::Route::WarpSign}) {
    if((route=="lut8"||route=="ilp")&&mode!=packed1::Route::Lut4)continue;
    if(route=="selected"&&mode!=selected.route)continue;
    if(route=="lut"&&mode!=packed1::Route::Lut4)continue;if(route=="sign"&&mode!=packed1::Route::Sign)continue;if(route=="warp"&&mode!=packed1::Route::WarpSign)continue;
    std::vector<int> chunks={1,2,4,8,16,k/128};std::sort(chunks.begin(),chunks.end());chunks.erase(std::unique(chunks.begin(),chunks.end()),chunks.end());
    for(int chunk:chunks)for(int threads:{64,128,256}) {
      if(route=="selected"&&(chunk!=selected.groups_per_split||threads!=selected.threads))continue;
      packed1::Plan p{n,k,r,chunk,threads,mode};Graph g=capture(p,signs,scales,x,y,tables,partial,st,1,0,route);
      float warm=timed(g,st,trash,trashn,false,route=="selected"?51:15),cold=timed(g,st,trash,trashn,true,route=="selected"?51:9);
      CUDA(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));double ae=0,se=0,sr=0;for(size_t i=0;i<hy.size();i++){ae=std::max(ae,std::abs(double(hy[i])-href[i]));se+=std::pow(double(hy[i])-href[i],2);sr+=double(href[i])*href[i];}
      double exacterr=0;for(int token=0;token<r;token++)for(int z=0;z<std::min(n,64);z++) {
        int row=int(size_t(z)*n/std::min(n,64));double total=0;
        for(int group=0;group<k/128;group++) {size_t bi=size_t(row)*(k/128)+group;double dot=0;
          for(int j=0;j<128;j++){unsigned word=mix(unsigned(bi)*4+j/32+17);dot+=((word>>(j%32))&1)?hx[size_t(token)*k+group*128+j]:-hx[size_t(token)*k+group*128+j];}
          total+=dot*scale_value(bi);
        }
        exacterr=std::max(exacterr,std::abs(total-hy[size_t(token)*n+row]));
      }
      const char* m=(route=="lut8"||route=="ilp")?route.c_str():mode==packed1::Route::Lut4?"lut4":mode==packed1::Route::Sign?"sign":"warp";
      fprintf(f,"{\"n\":%d,\"k\":%d,\"r\":%d,\"route\":\"%s\",\"chunk\":%d,\"threads\":%d,\"warm_us\":%.6f,\"cold_us\":%.6f,\"max_error_f32\":%.9g,\"relative_rms_f32\":%.9g,\"max_error_f64_sample\":%.9g,\"prepare_ms\":%.6f}\n",n,k,r,m,chunk,threads,warm,cold,ae,sqrt(se/sr),exacterr,prepms);fflush(f);
      if(cold<best){best=cold;bestp=p;}
      CUDA(cudaGraphExecDestroy(g.exec));CUDA(cudaGraphDestroy(g.graph));
    }
  }
  fclose(f);printf("best N%d K%d R%d %s chunk%d threads%d %.3fus cold\n",n,k,r,(route=="lut8"||route=="ilp")?route.c_str():bestp.route==packed1::Route::Lut4?"lut4":bestp.route==packed1::Route::Sign?"sign":"warp",bestp.groups_per_split,bestp.threads,best);
  Graph g=capture(bestp,signs,scales,x,y,tables,partial,st,.75f,.25f,route);
  std::vector<float> initial(href.size());for(size_t i=0;i<initial.size();i++)initial[i]=(int(mix(unsigned(i)+111)%65536)-32768)/32761.f;
  CUDA(cudaMemcpyAsync(y,initial.data(),initial.size()*4,cudaMemcpyHostToDevice,st));CUDA(cudaGraphLaunch(g.exec,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
  double ae=0;for(size_t i=0;i<hy.size();i++)ae=std::max(ae,std::abs(double(hy[i])-(.75*href[i]+.25*initial[i])));
  printf("alpha .75 beta .25 independent initial Y maxerr %.9g\n",ae);
  return 0;
}
