#include "bitserial.cuh"
#include "gguf.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#define CK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
constexpr int N=5120,K=17408,G=K/128;

// Four rows share the original K128 scale boundaries. A key is class(3b)+orientation(1b).
__global__ void prepare_keys(const uint8_t* raw,uint8_t* keys,__half* scales){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=(N/4)*G*64)return;
  int byte=i%64,g=(i/64)%G,tile=i/(64*G);unsigned packed=0;
  #pragma unroll
  for(int h=0;h<2;h++){
    int j=byte*2+h;unsigned s=0;
    #pragma unroll
    for(int row=0;row<4;row++){
      const uint8_t* b=raw+(size_t(tile*4+row)*G+g)*18;
      s|=((b[2+j/8]>>(j%8))&1)<<row;
      if(byte==0&&h==0)scales[(size_t(tile)*G+g)*4+row]=*reinterpret_cast<const __half*>(b);
    }
    unsigned orientation=s&1,code=(s>>1)^(orientation?7:0);
    packed|=(code|(orientation<<3))<<(h*4);
  }
  keys[i]=packed;
}

// Each warp owns a 4-output tile. Native match/reduce aggregates equal keys before row reductions.
__global__ void class_projection(const uint8_t* __restrict__ keys,const __half* __restrict__ scales,
    bitserial::Carrier in,float* partial,int r,int chunk,int ns){
  int lane=threadIdx.x%32,warp=threadIdx.x/32,tile=blockIdx.x*4+warp,t=blockIdx.z;
  if(tile>=N/4)return;
  __shared__ int bins[4][8];float sums[4]={};
  int first=blockIdx.y*chunk,last=min(first+chunk,G);
  for(int g=first;g<last;g++){
    if(lane<8)bins[warp][lane]=0;
    __syncwarp();
    #pragma unroll
    for(int h=0;h<4;h++){
      int j=lane+32*h;
      unsigned packed=keys[(size_t(tile)*G+g)*64+j/2];
      unsigned key=(packed>>((j%2)*4))&15,code=key&7;
      int q=in.q[(size_t(t)*G+g)*128+j],oriented=(key&8)?q:-q;
      unsigned group=__match_any_sync(0xffffffff,code);
      int aggregate=__reduce_add_sync(group,oriented);
      if(lane==__ffs(group)-1)atomicAdd(&bins[warp][code],aggregate);
    }
    __syncwarp();
    int aggregate=lane<8?bins[warp][lane]:0;
    float sx=in.scales[size_t(t)*G+g];
    bool bad=in.status[size_t(t)*G+g]!=bitserial::InputStatus::Valid;
    #pragma unroll
    for(int row=0;row<4;row++){
      int v=row==0?aggregate:((lane>>(row-1))&1)?-aggregate:aggregate;
      int dot=__reduce_add_sync(0xffffffff,v);
      if(lane==0)sums[row]=bad?__int_as_float(0x7fc00000):fmaf(float(dot),sx*__half2float(scales[(size_t(tile)*G+g)*4+row]),sums[row]);
    }
    __syncwarp();
  }
  if(lane==0){
    #pragma unroll
    for(int row=0;row<4;row++)partial[(size_t(t)*ns+blockIdx.y)*N+tile*4+row]=sums[row];
  }
}
__device__ __forceinline__ int signed_coeff(unsigned nibble_bits){
  unsigned v=nibble_bits&0x1111;
  v=(v|(v<<8))&0x00ff00ff;
  v=(v|(v<<4))&0x01010101;
  return int(~(v*254u));
}
template<int TR> __global__ void nibble_projection(const uint8_t* __restrict__ keys,const __half* __restrict__ scales,
    bitserial::Carrier in,float* partial,int r,int chunk,int ns){
  int lane=threadIdx.x%32,warp=threadIdx.x/32,tile=blockIdx.x*4+warp,first_t=blockIdx.z*TR;
  if(tile>=N/4)return;float sums[TR][4]={};
  int first=blockIdx.y*chunk,last=min(first+chunk,G);
  for(int g=first;g<last;g++){
    unsigned key=reinterpret_cast<const uint16_t*>(keys)[(size_t(tile)*G+g)*32+lane];
    int coeff[4]={signed_coeff(key>>3),signed_coeff((key>>3)^key),signed_coeff((key>>3)^(key>>1)),signed_coeff((key>>3)^(key>>2))};
    #pragma unroll
    for(int tt=0;tt<TR;tt++){
      int t=first_t+tt;if(t>=r)continue;
      int q=reinterpret_cast<const int*>(in.q)[(size_t(t)*G+g)*32+lane];
      float sx=in.scales[size_t(t)*G+g];bool bad=in.status[size_t(t)*G+g]!=bitserial::InputStatus::Valid;
      #pragma unroll
      for(int row=0;row<4;row++){
        int dot=__reduce_add_sync(0xffffffff,__dp4a(q,coeff[row],0));
        if(lane==0)sums[tt][row]=bad?__int_as_float(0x7fc00000):fmaf(float(dot),sx*__half2float(scales[(size_t(tile)*G+g)*4+row]),sums[tt][row]);
      }
    }
  }
  if(lane==0){
    #pragma unroll
    for(int tt=0;tt<TR;tt++){
      int t=first_t+tt;if(t>=r)continue;
      #pragma unroll
      for(int row=0;row<4;row++)partial[(size_t(t)*ns+blockIdx.y)*N+tile*4+row]=sums[tt][row];
    }
  }
}
__global__ void continuation(const float* partial,float* y,int ns){
  int row=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(row>=N)return;
  float sum=0;for(int s=0;s<ns;s++)sum+=partial[(size_t(t)*ns+s)*N+row];
  y[size_t(t)*N+row]=sum;
}
__global__ void raw_reference(const uint8_t* raw,bitserial::Carrier in,float* y){
  int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,t=blockIdx.y;
  if(row>=N)return;float sum=0;
  for(int g=0;g<G;g++){
    const uint8_t* b=raw+(size_t(row)*G+g)*18;int dot=0;
    for(int j=lane;j<128;j+=32){int q=in.q[(size_t(t)*G+g)*128+j];dot+=((b[2+j/8]>>(j%8))&1)?q:-q;}
    dot=__reduce_add_sync(0xffffffff,dot);
    if(lane==0)sum=fmaf(float(dot),__half2float(*reinterpret_cast<const __half*>(b))*in.scales[size_t(t)*G+g],sum);
  }
  if(lane==0)y[size_t(t)*N+row]=sum;
}
__global__ void evict(unsigned* p,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]+=1;}
struct Graph{cudaGraph_t graph;cudaGraphExec_t exec;};
template<class F> Graph capture(cudaStream_t st,F f){Graph a;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));f();CK(cudaStreamEndCapture(st,&a.graph));CK(cudaGraphInstantiate(&a.exec,a.graph,0));return a;}
float time_graph(Graph a,cudaStream_t st,unsigned* trash,size_t count,bool cold){
  cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));std::vector<float> samples;
  for(int z=0;z<12;z++){
    if(cold)evict<<<(count+255)/256,256,0,st>>>(trash,count);
    CK(cudaEventRecord(begin,st));CK(cudaGraphLaunch(a.exec,st));CK(cudaEventRecord(end,st));CK(cudaEventSynchronize(end));
    float ms;CK(cudaEventElapsedTime(&ms,begin,end));if(z>=3)samples.push_back(ms*1000);
  }
  std::sort(samples.begin(),samples.end());CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));return samples[samples.size()/2];
}
unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
struct Memory{
  std::vector<void*> owned;
  template<class T>T* alloc(size_t n){T* p;CK(cudaMalloc(&p,n*sizeof(T)));owned.push_back(p);return p;}
  ~Memory(){for(auto p:owned)cudaFree(p);}
};
bitserial::Carrier slice(bitserial::Carrier c,int t){c.q+=size_t(t)*K;c.scales+=size_t(t)*G;c.dc+=size_t(t)*G;c.status+=size_t(t)*G;return c;}
int run(int argc,char**argv){
  if(argc!=4){fprintf(stderr,"usage: down_classes MODEL R(1|16|32) OUTPUT.jsonl\n");return 1;}
  int r=std::stoi(argv[2]);if(r!=1&&r!=16&&r!=32)return 1;
  h5::Gguf model;model.open(argv[1]);const auto& tensor=model.t("blk.0.ffn_down.weight");
  if(tensor.type!=41||tensor.ne.size()!=2||tensor.ne[0]!=K||tensor.ne[1]!=N)return 1;
  Memory m;cudaStream_t st;CK(cudaStreamCreate(&st));
  auto* raw=m.alloc<uint8_t>(tensor.bytes);auto* keys=m.alloc<uint8_t>(size_t(N)*K/8);
  auto* scales=m.alloc<__half>(size_t(N)*G);auto* signs=m.alloc<uint32_t>(size_t(N)*G*4);
  auto* serial_scales=m.alloc<__half>(size_t(N)*G);
  auto* x=m.alloc<float>(size_t(r)*K);auto* y=m.alloc<float>(size_t(r)*N);auto* ref=m.alloc<float>(size_t(r)*N);
  auto* partial=m.alloc<float>(size_t(r)*N*G);auto* baseline_partial=m.alloc<float>(size_t(r)*N*G);
  bitserial::Carrier c{};c.q=m.alloc<int8_t>(size_t(r)*K);c.scales=m.alloc<float>(size_t(r)*G);
  c.dc=m.alloc<int32_t>(size_t(r)*G);c.status=m.alloc<bitserial::InputStatus>(size_t(r)*G);
  size_t trashn=128*1024*1024/4;auto* trash=m.alloc<unsigned>(trashn);CK(cudaMemsetAsync(trash,0,trashn*4,st));
  std::vector<float> hx(size_t(r)*K);for(size_t i=0;i<hx.size();i++)hx[i]=(int(mix(unsigned(i)+999)%65536)-32768)/32761.f;
  CK(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));CK(cudaMemcpyAsync(raw,tensor.data,tensor.bytes,cudaMemcpyHostToDevice,st));
  prepare_keys<<<(size_t(N/4)*G*64+255)/256,256,0,st>>>(raw,keys,scales);CK(cudaGetLastError());
  CK(bitserial::prepare(raw,signs,serial_scales,N,K,st));CK(bitserial::produce(bitserial::Route::Dp4a01,x,K,r,c,st));
  raw_reference<<<dim3(N/4,r),128,0,st>>>(raw,c,ref);CK(cudaGetLastError());CK(cudaStreamSynchronize(st));
  std::vector<float> href(size_t(r)*N),hy(href.size());CK(cudaMemcpy(href.data(),ref,href.size()*4,cudaMemcpyDeviceToHost));
  std::vector<unsigned> status(size_t(r)*G);CK(cudaMemcpy(status.data(),c.status,status.size()*4,cudaMemcpyDeviceToHost));
  for(auto s:status)if(s){fprintf(stderr,"invalid producer status %u\n",s);return 3;}
  FILE* out=fopen(argv[3],"w");if(!out)return 2;
  fprintf(out,"{\"kind\":\"scope\",\"tensor\":\"blk.0.ffn_down.weight\",\"r\":%d,\"n\":%d,\"k\":%d,\"activations\":\"synthetic_uniform_fp32_COST_ONLY\",\"oracle\":\"original_Q1_K128_A8_integer_dots_FP32_original_FP16_scale_continuation\",\"key_bytes\":%zu,\"original_scale_bytes\":%zu,\"keys_layout\":\"tile4_k128_64bytes_two_class_orientation_nibbles_per_byte\",\"carrier_q_bytes\":%zu,\"carrier_scale_dc_status_bytes\":%zu,\"no_new_quantization\":true,\"alpha\":1,\"beta\":0}\n",r,N,K,size_t(N)*K/8,size_t(N)*G*2,size_t(r)*K,size_t(r)*G*12);
  auto record=[&](const char* route,int chunk,bool inclusive,auto f){
    f();CK(cudaGetLastError());CK(cudaStreamSynchronize(st));CK(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
    double se=0,ss=0,maxabs=0;size_t exact=0;for(size_t i=0;i<hy.size();i++){double e=double(hy[i])-href[i];se+=e*e;ss+=double(href[i])*href[i];maxabs=std::max(maxabs,std::abs(e));exact+=hy[i]==href[i];}
    if(chunk==G&&exact!=hy.size()){fprintf(stderr,"same-order exact-A8 failure %s %zu/%zu\n",route,exact,hy.size());exit(3);}
    if(!std::isfinite(se)||sqrt(se/ss)>1e-5){fprintf(stderr,"numerical failure %s %.9g\n",route,sqrt(se/ss));exit(3);}
    if(chunk==G)fprintf(out,"{\"kind\":\"same_order_exact_a8\",\"route\":\"%s\",\"r\":%d,\"inclusive_producer\":%s,\"equal_fp32_values\":%zu,\"outputs\":%zu,\"maxabs\":%.9g,\"integer_dot_bound\":16256}\n",route,r,inclusive?"true":"false",exact,hy.size(),maxabs);
    Graph graph=capture(st,f);float warm=time_graph(graph,st,trash,trashn,false),cold=time_graph(graph,st,trash,trashn,true);
    fprintf(out,"{\"kind\":\"timing\",\"route\":\"%s\",\"r\":%d,\"chunk\":%d,\"inclusive_producer\":%s,\"warm_us\":%.6f,\"cold_us\":%.6f,\"maxabs_a8\":%.9g,\"relative_rms_a8\":%.9g,\"partial_bytes\":%zu,\"native\":\"%s\"}\n",route,r,chunk,inclusive?"true":"false",warm,cold,maxabs,sqrt(se/ss),size_t(r)*N*((G+chunk-1)/chunk)*4,std::string(route)=="class4"?"match_any+reduce_add+warp_private_shared_atomic_add":std::string(route)=="bitserial_dp4a01"?"dp4a01":"nibble_class_orientation_to_signed_dp4a+reduce_add");fflush(out);
    CK(cudaGraphExecDestroy(graph.exec));CK(cudaGraphDestroy(graph.graph));
  };
  for(int chunk:{1,4,16,136})for(bool inclusive:{false,true}){
    auto classes=[&](){if(inclusive)CK(bitserial::produce(bitserial::Route::Dp4a01,x,K,r,c,st));int ns=(G+chunk-1)/chunk;
      class_projection<<<dim3((N/4+3)/4,ns,r),128,0,st>>>(keys,scales,c,partial,r,chunk,ns);CK(cudaGetLastError());
      continuation<<<dim3((N+255)/256,r),256,0,st>>>(partial,y,ns);CK(cudaGetLastError());};
    record("class4",chunk,inclusive,classes);
    for(int tr:{1,4}){
      auto nibble=[&](){if(inclusive)CK(bitserial::produce(bitserial::Route::Dp4a01,x,K,r,c,st));int ns=(G+chunk-1)/chunk;
        dim3 grid((N/4+3)/4,ns,(r+tr-1)/tr);
        if(tr==1)nibble_projection<1><<<grid,128,0,st>>>(keys,scales,c,partial,r,chunk,ns);
        else nibble_projection<4><<<grid,128,0,st>>>(keys,scales,c,partial,r,chunk,ns);
        CK(cudaGetLastError());continuation<<<dim3((N+255)/256,r),256,0,st>>>(partial,y,ns);CK(cudaGetLastError());};
      record(tr==1?"nibble4_dp4a_t1":"nibble4_dp4a_t4",chunk,inclusive,nibble);
    }
    auto baseline=[&](){if(inclusive)CK(bitserial::produce(bitserial::Route::Dp4a01,x,K,r,c,st));
      for(int t=0;t<r;t+=16){bitserial::Plan p{N,K,std::min(16,r-t),chunk,128,bitserial::Route::Dp4a01};
        CK(bitserial::project_carrier(p,signs,serial_scales,slice(c,t),y+size_t(t)*N,baseline_partial,1,0,st));}};
    record("bitserial_dp4a01",chunk,inclusive,baseline);
  }
  fclose(out);CK(cudaStreamDestroy(st));return 0;
}
int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"inputerror: %s\n",e.what());return 2;}}
