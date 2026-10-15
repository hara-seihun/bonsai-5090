#include "matmul-tiles.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
using namespace halo6000;
#define CU(x) do{auto rc=(x);if(rc!=cudaSuccess){fprintf(stderr,"line%d %s: %s\n",__LINE__,#x,cudaGetErrorString(rc));exit(1);}}while(0)
cudaStream_t stream;
struct Buffers{std::vector<void*> values;template<class T>T* alloc(size_t n){T*p;CU(cudaMalloc(&p,n*sizeof(T)));values.push_back(p);return p;}~Buffers(){for(void*p:values)cudaFree(p);}};
template<class T>std::vector<T> read(const std::string&path,size_t n){std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||size_t(f.tellg())!=n*sizeof(T)){fprintf(stderr,"bad fixture %s\n",path.c_str());exit(1);}f.seekg(0);std::vector<T>v(n);f.read((char*)v.data(),n*sizeof(T));if(!f)exit(1);return v;}
struct Case{std::string key,cache,qs,scales;int format,n,k,experts;uint64_t bytes;};
__global__ void dense_integer_reference(const int8_t*q,PreparedActivationView a,int n,int experts,int first,int begin,int rows,int32_t*out){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,total=uint64_t(a.columns)*experts*rows*(a.source.k/32);
 if(i>=total)return;int kt=a.source.k/32,g=i%kt,r=i/kt%rows+begin,e=i/(uint64_t(kt)*rows)%experts,c=i/(uint64_t(kt)*rows*experts);
 int ie=a.input_experts==1?0:e,ar=a.row_indices?a.row_indices[ie*a.columns+c]:c*a.input_experts+ie,value=0;
 if(ar>=0){const int8_t*w=q+(uint64_t(first+e)*n+r)*a.source.k+g*32;const int8_t*x=a.source.qs+ar*a.source.q_stride+g*32;
  #pragma unroll
  for(int k=0;k<8;k++)value=__dp4a(*reinterpret_cast<const int*>(w+k*4),*reinterpret_cast<const int*>(x+k*4),value);
 }out[i]=value;
}
__global__ void evict(uint32_t*p,size_t n){for(size_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(gridDim.x)*blockDim.x)p[i]++;}

template<WeightFormat F,int G,int T,bool Guard=false>void launch(PreparedMatmulView w,PreparedActivationView a,float*out,int block,bool resident,int first,int count,MatmulIntegerGuard guard={}){
 int ct=((a.columns+7)/8+T-1)/T,grid=resident?188:(w.source.n/16)*ct*count;
 if constexpr(T==1)matmul_q8_splitk<F,G,Guard><<<grid,block,0,stream>>>(w,a,out,first,count,guard);
 else matmul_q8_splitk_tiles<F,G,T,Guard><<<grid,block,T*G*128*sizeof(float),stream>>>(w,a,out,first,count,guard);
 CU(cudaGetLastError());
}
struct Price{double cold,hot;};
Price price(const std::function<void()>&fn,uint32_t*flush){
 cudaGraph_t graph;cudaGraphExec_t exec;CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));fn();CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));cudaEvent_t start,end;CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));double result[2];
 for(int hot=0;hot<2;hot++){for(int j=0;j<2;j++)CU(cudaGraphLaunch(exec,stream));CU(cudaStreamSynchronize(stream));std::vector<double>samples;
  for(int j=0;j<7;j++){if(!hot)evict<<<188,256,0,stream>>>(flush,(256ULL<<20)/4);CU(cudaEventRecord(start,stream));CU(cudaGraphLaunch(exec,stream));CU(cudaEventRecord(end,stream));CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,start,end));samples.push_back(ms*1000);}
  std::sort(samples.begin(),samples.end());result[hot]=samples[3];
 }
 CU(cudaEventDestroy(start));CU(cudaEventDestroy(end));CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));return{result[0],result[1]};
}
size_t bits(const std::vector<float>&a,const std::vector<float>&b){size_t count=0;for(size_t i=0;i<a.size();i++)count+=memcmp(a.data()+i,b.data()+i,4)!=0;return count;}

template<WeightFormat F,int G,int T>void variant(const Case&c,PreparedMatmulView w,PreparedActivationView a,float*out,const int8_t*dense,int first,int count,const std::vector<float>&reference,uint32_t*flush,bool integer){
 if constexpr(T!=1){CU((configure_matmul_q8_splitk_tiles<F,G,T>()));CU((configure_matmul_q8_splitk_tiles<F,G,T,true>()));}
 Buffers buffers;auto bad=buffers.alloc<unsigned long long>(1);int chunk=std::min(c.n,4096),kt=c.k/32;
 if(integer){auto iref=buffers.alloc<int32_t>(uint64_t(a.columns)*count*chunk*kt);CU(cudaMemsetAsync(bad,0,8,stream));
  for(int begin=0;begin<c.n;begin+=chunk){int rows=std::min(chunk,c.n-begin);uint64_t n=uint64_t(a.columns)*count*rows*kt;
   dense_integer_reference<<<(n+255)/256,256,0,stream>>>(dense,a,c.n,count,first,begin,rows,iref);CU(cudaGetLastError());
   launch<F,G,T,true>(w,a,out,128,false,first,count,{iref,bad,begin,rows});
  }
  unsigned long long different;CU(cudaMemcpyAsync(&different,bad,8,cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));
  printf("{\"guard\":\"all_K32_integer_partials\",\"case\":\"%s\",\"columns\":%d,\"first_expert\":%d,\"experts\":%d,\"tiles\":%d,\"partials\":%llu,\"different\":%llu}\n",c.key.c_str(),a.columns,first,count,T,(unsigned long long)reference.size()*kt,different);if(different)exit(2);
 }
 std::vector<float>got(reference.size());
 for(int block:{128,256})for(bool resident:{false,true}){
  cudaFuncAttributes attr;int occupancy;
  if constexpr(T==1){CU(cudaFuncGetAttributes(&attr,matmul_q8_splitk<F,G>));CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,matmul_q8_splitk<F,G>,block,0));}
  else{CU(cudaFuncGetAttributes(&attr,matmul_q8_splitk_tiles<F,G,T>));CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,matmul_q8_splitk_tiles<F,G,T>,block,T*G*128*sizeof(float)));}
  auto fn=[&]{launch<F,G,T>(w,a,out,block,resident,first,count);};
  CU(cudaMemsetAsync(out,0xff,got.size()*4,stream));fn();CU(cudaMemcpyAsync(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));size_t direct=bits(reference,got);if(direct){fprintf(stderr,"float mismatch %s T%d block%d resident%d %zu\n",c.key.c_str(),T,block,resident,direct);exit(2);}
  Price p=price(fn,flush);CU(cudaMemcpyAsync(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));size_t replay=bits(reference,got);if(replay)exit(2);
  printf("{\"case\":\"%s\",\"format\":%d,\"N\":%d,\"K\":%d,\"columns\":%d,\"input_experts\":%d,\"first_expert\":%d,\"experts\":%d,\"tiles\":%d,\"block\":%d,\"resident188\":%s,\"outputs\":%llu,\"direct_bit_differences\":%zu,\"graph_bit_differences\":%zu,\"registers\":%d,\"local_bytes\":%zu,\"shared_bytes\":%zu,\"active_CTAs_per_SM\":%d,\"cold_us\":%.6f,\"hot_us\":%.6f}\n",c.key.c_str(),c.format,c.n,c.k,a.columns,a.input_experts,first,count,T,block,resident?"true":"false",(unsigned long long)got.size(),direct,replay,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes+(T==1?0:T*G*128*sizeof(float)),occupancy,p.cold,p.hot);
 }
}

template<WeightFormat F,int G>void run(const Case&c,int columns,bool subset){
 Buffers buffers;auto raw=read<uint8_t>(c.cache,c.bytes);auto dense=read<int8_t>(c.qs,uint64_t(c.n)*c.k*c.experts);
 auto source=buffers.alloc<uint8_t>(raw.size());auto dq=buffers.alloc<int8_t>(dense.size());CU(cudaMemcpyAsync(source,raw.data(),raw.size(),cudaMemcpyHostToDevice,stream));CU(cudaMemcpyAsync(dq,dense.data(),dense.size(),cudaMemcpyHostToDevice,stream));
 MatrixView matrix;matrix.data=source;matrix.n=c.n;matrix.k=c.k;matrix.experts=c.experts;matrix.format=F;matrix.bytes=c.bytes;matrix.scale_group=F==WeightFormat::Pane2?128:32;matrix.expert_stride=c.bytes/c.experts;matrix.row_stride=F==WeightFormat::Pane2?0:uint64_t(c.k/128)*136;
 auto wb=matmul_weight_bytes(matrix);auto wi=buffers.alloc<uint32_t>(wb.operands/4);auto ws=buffers.alloc<uint16_t>(wb.scales/2);uint64_t threads=uint64_t(c.n/16)*(c.k/32)*c.experts*32;
 prepare_matmul_weights<F><<<(threads+255)/256,256,0,stream>>>(matrix,wi,ws);CU(cudaGetLastError());PreparedMatmulView w{matrix,wi,ws};
 int inputs=c.experts>1&&c.k==512?c.experts:1,arows=columns*inputs,kt=c.k/32;
 std::vector<int8_t>hq(uint64_t(arows)*c.k);std::vector<float>hs(uint64_t(arows)*kt);std::vector<int32_t>indices(uint64_t(columns)*inputs);
 for(size_t i=0;i<hq.size();i++)hq[i]=int((i*73+i/31*19)%255)-127;
 for(size_t i=0;i<hs.size();i++)hs[i]=__half2float(__float2half(float(1+i%29)*.001731f));
 for(int e=0;e<inputs;e++)for(int col=0;col<columns;col++)indices[e*columns+col]=((columns-1-col)*inputs+e);
 for(int e=0;e<inputs;e++)indices[e*columns+columns-1]=-1;
 auto q=buffers.alloc<int8_t>(hq.size());auto s=buffers.alloc<float>(hs.size());auto ri=buffers.alloc<int32_t>(indices.size());CU(cudaMemcpyAsync(q,hq.data(),hq.size(),cudaMemcpyHostToDevice,stream));CU(cudaMemcpyAsync(s,hs.data(),hs.size()*4,cudaMemcpyHostToDevice,stream));CU(cudaMemcpyAsync(ri,indices.data(),indices.size()*4,cudaMemcpyHostToDevice,stream));
 PreparedActivationView a;a.source={q,s,c.k,arows,32,0,c.k,kt};a.columns=columns;a.input_experts=inputs;a.row_indices=ri;uint64_t bbytes=matmul_activation_bytes(columns,c.k,inputs);auto bi=buffers.alloc<uint32_t>(bbytes/4);a.operands=bi;prepare_matmul_activations<<<(bbytes/4+255)/256,256,0,stream>>>(a,bi);CU(cudaGetLastError());
 int first=subset?1:0,count=subset?3:c.experts;if(first+count>c.experts)exit(1);
 if(subset&&inputs>1){a.operands+=uint64_t(first)*((columns+7)/8)*kt*64;a.row_indices+=first*columns;}
 if(!exact_q8_matmul(w,a))exit(1);
 auto out=buffers.alloc<float>(uint64_t(columns)*count*c.n);std::vector<float>reference(uint64_t(columns)*count*c.n);launch<F,G,1>(w,a,out,128,false,first,count);CU(cudaMemcpyAsync(reference.data(),out,reference.size()*4,cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));
 auto flush=buffers.alloc<uint32_t>((256ULL<<20)/4);CU(cudaMemsetAsync(flush,0,256ULL<<20,stream));
 variant<F,G,1>(c,w,a,out,dq,first,count,reference,flush,false);
 variant<F,G,2>(c,w,a,out,dq,first,count,reference,flush,true);
 variant<F,G,4>(c,w,a,out,dq,first,count,reference,flush,true);
 printf("{\"passed\":true,\"case\":\"%s\",\"columns\":%d,\"subset\":%s}\n",c.key.c_str(),columns,subset?"true":"false");
}
template<WeightFormat F>void dispatch(const Case&c,int columns,bool subset){switch(c.k){case 512:run<F,4>(c,columns,subset);break;case 2048:run<F,16>(c,columns,subset);break;case 4096:run<F,32>(c,columns,subset);break;default:exit(1);}}
int main(int argc,char**argv){if(argc<4||argc>5){fprintf(stderr,"usage: matmul-tiles FIXTURE_DIR CASE COLUMNS [subset]\n");return 1;}setvbuf(stdout,nullptr,_IOLBF,0);CU(cudaStreamCreate(&stream));cudaDeviceProp p;CU(cudaGetDeviceProperties(&p,0));printf("{\"device\":\"%s\",\"SMs\":%d,\"shared_optin_limit\":%zu,\"fixture\":\"real weights; synthetic gathered INT8 and effective FP16 group32 scales\"}\n",p.name,p.multiProcessorCount,p.sharedMemPerBlockOptin);
 std::ifstream f(std::string(argv[1])+"/cases.tsv");std::string line;int columns=atoi(argv[3]);if(columns<=0)return 1;
 while(std::getline(f,line)){Case c;std::istringstream parser(line);parser>>c.key>>c.format>>c.n>>c.k>>c.experts>>c.bytes>>c.cache>>c.qs>>c.scales;if(!parser)return 1;if(c.key!=argv[2])continue;if(c.format==1)dispatch<WeightFormat::Pane2>(c,columns,argc==5);else if(c.format==2)dispatch<WeightFormat::Q8Grouped128>(c,columns,argc==5);else return 1;CU(cudaStreamDestroy(stream));return 0;}return 1;
}
