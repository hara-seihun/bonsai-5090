#include "matmul.cuh"
#include "device.cuh"
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
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s line%d\n",#x,cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
cudaStream_t S;
template<class T>T* alloc(size_t n){T*p;CU(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>std::vector<T> read(const std::string&p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||size_t(f.tellg())!=count*sizeof(T)){fprintf(stderr,"bad fixture %s size%zu\n",p.c_str(),count*sizeof(T));exit(1);}f.seekg(0);std::vector<T>v(count);f.read((char*)v.data(),v.size()*sizeof(T));if(!f)exit(1);return v;}
struct Case{std::string key,cache,qs,scales;int format,n,k,experts;uint64_t bytes;};
__global__ void flush_cache(uint32_t*p,size_t n){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]++;}
// Independent dense oracle: no PANE unpacker, fragment mapper, or source scale
// helper is shared with the implementation under test.
__global__ void integer_reference(const int8_t*q,PreparedActivationView a,int n,int experts,int begin,int rows,int32_t*out){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,total=uint64_t(a.columns)*experts*rows*(a.source.k/32);
 if(i>=total)return;int kt=a.source.k/32,g=i%kt,r=i/kt%rows+begin,e=i/(uint64_t(kt)*rows)%experts,c=i/(uint64_t(kt)*rows*experts);
 int ar=mm_activation_row(a,c,a.input_experts==1?0:e),v=0;
 if(ar>=0){const int8_t*w=q+(uint64_t(e)*n+r)*a.source.k+g*32;const int8_t*x=a.source.qs+ar*a.source.q_stride+g*32;
  #pragma unroll
  for(int j=0;j<8;j++)v=__dp4a(*reinterpret_cast<const int*>(w+j*4),*reinterpret_cast<const int*>(x+j*4),v);
 }out[i]=v;
}
template<WeightFormat F>__global__ void image_reference(PreparedMatmulView w,const int8_t*q,const uint16_t*s,unsigned long long*bad){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,tile=i/32;int lane=i%32,kt=w.source.k/32;
 if(tile>=uint64_t(w.source.n/16)*kt*w.source.experts)return;
 int e=tile/(uint64_t(w.source.n/16)*kt),m=tile/kt%(w.source.n/16),g=tile%kt;constexpr int words=F==WeightFormat::Pane2?32:128;
 uint32_t operands[4];mm_operand<F>(w.operands+tile*words,lane,operands);
 #pragma unroll
 for(int j=0;j<4;j++){
  int r=m*16+lane/4+(j%2?8:0),col=g*32+(lane%4)*4+(j>=2?16:0);
  uint64_t index=(uint64_t(e)*w.source.n+r)*w.source.k+col;
  if(operands[j]!=*reinterpret_cast<const uint32_t*>(q+index))atomicAdd(bad,1ull);
  int sg=F==WeightFormat::Pane2?g/4:g;
  uint16_t actual=w.weight_scales[(uint64_t(e)*w.source.n+r)*(F==WeightFormat::Pane2?w.source.k/128:kt)+sg];
  if(actual!=s[(uint64_t(e)*w.source.n+r)*kt+g])atomicAdd(bad,1ull);
 }
}
__global__ void scalar_reference(MatrixView w,PreparedActivationView a,float*out,int experts){
 int r=(blockIdx.x*blockDim.x+threadIdx.x)/32,e=blockIdx.z,c=blockIdx.y;if(r>=w.n)return;int ar=mm_activation_row(a,c,a.input_experts==1?0:e);
 float v=ar<0?0.f:quantized_row(w,a.source,r,e,ar);if(threadIdx.x%32==0)out[(uint64_t(c)*experts+e)*w.n+r]=v;
}
template<WeightFormat F,int T,bool G=false>void launch(PreparedMatmulView w,PreparedActivationView a,float*out,MatmulIntegerGuard guard={}){matmul_q8<F,T,G><<<dim3((w.source.n/16+3)/4,((a.columns+7)/8+T-1)/T,w.source.experts),128,0,S>>>(w,a,out,0,w.source.experts,guard);CU(cudaGetLastError());}
template<WeightFormat F,int G,bool Guard=false>void split_launch(PreparedMatmulView w,PreparedActivationView a,float*out,int block,bool resident,MatmulIntegerGuard guard={},int first=0,int count=-1,bool compact=false){if(count<0)count=w.source.experts;int grid=resident?188:(w.source.n/16)*((a.columns+7)/8)*count;if(compact)matmul_q8_s1<F,G,Guard><<<grid,block,0,S>>>(w,a,out,first,count,guard);else matmul_q8_splitk<F,G,Guard><<<grid,block,0,S>>>(w,a,out,first,count,guard);CU(cudaGetLastError());}
template<WeightFormat F,bool Guard=false>void split(PreparedMatmulView w,PreparedActivationView a,float*out,int block,bool resident,MatmulIntegerGuard guard={},int first=0,int count=-1,bool compact=false){switch(w.source.k){case 512:split_launch<F,4,Guard>(w,a,out,block,resident,guard,first,count,compact);break;case 2048:split_launch<F,16,Guard>(w,a,out,block,resident,guard,first,count,compact);break;case 4096:split_launch<F,32,Guard>(w,a,out,block,resident,guard,first,count,compact);break;default:exit(1);}}
struct Diff{size_t bits=0;double maxabs=0,maxrelative=0,normrelative=0;};
Diff diff(const std::vector<float>&ref,const std::vector<float>&got){Diff d;double norm=0;for(size_t i=0;i<ref.size();i++){if(!std::isfinite(got[i])||!std::isfinite(ref[i])){fprintf(stderr,"nonfinite output%zu\n",i);exit(2);}d.bits+=memcmp(&ref[i],&got[i],4)!=0;double delta=fabs(double(ref[i])-got[i]);d.maxabs=std::max(d.maxabs,delta);norm=std::max(norm,fabs(double(ref[i])));if(fabs(ref[i])>1e-3)d.maxrelative=std::max(d.maxrelative,delta/fabs(ref[i]));}d.normrelative=d.maxabs/std::max(norm,1e-30);return d;}
void report(const char*kind,const Case&c,int rows,int inputs,int t,const Diff&d){printf("{\"guard\":\"%s\",\"case\":\"%s\",\"columns\":%d,\"input_experts\":%d,\"tiles\":%d,\"outputs\":%llu,\"bit_differences\":%zu,\"max_abs\":%.9g,\"max_relative_ref_gt_1e_3\":%.9g,\"relative_to_max_output\":%.9g}\n",kind,c.key.c_str(),rows,inputs,t,(unsigned long long)rows*c.n*c.experts,d.bits,d.maxabs,d.maxrelative,d.normrelative);}
struct Price{double cold,hot;};
Price price(std::function<void()>f,uint32_t*flush){cudaGraph_t graph;cudaGraphExec_t exec;CU(cudaStreamBeginCapture(S,cudaStreamCaptureModeGlobal));f();CU(cudaStreamEndCapture(S,&graph));CU(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));cudaEvent_t start,end;CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));double result[2];for(int hot=0;hot<2;hot++){for(int j=0;j<3;j++)CU(cudaGraphLaunch(exec,S));CU(cudaStreamSynchronize(S));std::vector<double>samples;for(int i=0;i<11;i++){if(!hot)flush_cache<<<188,256,0,S>>>(flush,(256ULL<<20)/4);CU(cudaEventRecord(start,S));CU(cudaGraphLaunch(exec,S));CU(cudaEventRecord(end,S));CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,start,end));samples.push_back(ms*1000);}std::sort(samples.begin(),samples.end());result[hot]=samples[5];}CU(cudaEventDestroy(start));CU(cudaEventDestroy(end));CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));return{result[0],result[1]};}
void rates(const char*kind,const Case&c,int rows,int inputs,int tiles,const Price&p){printf("{\"timing\":\"%s\",\"case\":\"%s\",\"columns\":%d,\"input_experts\":%d,\"tiles\":%d,\"cold_us\":%.6f,\"hot_us\":%.6f}\n",kind,c.key.c_str(),rows,inputs,tiles,p.cold,p.hot);}
template<WeightFormat F>void run(const Case&c,uint32_t*flush){
 auto raw=read<uint8_t>(c.cache,c.bytes);auto hq=read<int8_t>(c.qs,uint64_t(c.n)*c.k*c.experts);auto hs=read<uint16_t>(c.scales,uint64_t(c.n)*(c.k/32)*c.experts);
 uint8_t*source=alloc<uint8_t>(raw.size());int8_t*dq=alloc<int8_t>(hq.size());uint16_t*ds=alloc<uint16_t>(hs.size());CU(cudaMemcpyAsync(source,raw.data(),raw.size(),cudaMemcpyHostToDevice,S));CU(cudaMemcpyAsync(dq,hq.data(),hq.size(),cudaMemcpyHostToDevice,S));CU(cudaMemcpyAsync(ds,hs.data(),hs.size()*2,cudaMemcpyHostToDevice,S));
 MatrixView w;w.data=source;w.n=c.n;w.k=c.k;w.experts=c.experts;w.format=F;w.bytes=c.bytes;w.scale_group=F==WeightFormat::Pane2?128:32;w.expert_stride=c.bytes/c.experts;w.row_stride=F==WeightFormat::Pane2?0:uint64_t(c.k/128)*136;
 auto wb=matmul_weight_bytes(w);uint32_t*wi=alloc<uint32_t>(wb.operands/4);uint16_t*ws=alloc<uint16_t>(wb.scales/2);uint64_t image_threads=uint64_t(c.n/16)*(c.k/32)*c.experts*32;
 prepare_matmul_weights<F><<<(image_threads+255)/256,256,0,S>>>(w,wi,ws);CU(cudaGetLastError());PreparedMatmulView pm{w,wi,ws};unsigned long long*bad=alloc<unsigned long long>(1);CU(cudaMemsetAsync(bad,0,8,S));image_reference<F><<<(image_threads+255)/256,256,0,S>>>(pm,dq,ds,bad);unsigned long long mismatch;CU(cudaMemcpyAsync(&mismatch,bad,8,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));printf("{\"guard\":\"all_weight_bytes_and_original_scales\",\"case\":\"%s\",\"weights\":%llu,\"different\":%llu,\"prepared_bytes\":%llu}\n",c.key.c_str(),(unsigned long long)hq.size(),mismatch,(unsigned long long)(wb.operands+wb.scales));if(mismatch)exit(2);
 // Eight independent down producers, shared producer for all gate/up experts.
 int inputs=c.experts>1&&c.k==512?c.experts:1;
 for(int columns:{1,8,13,32,128,512}){
  int arows=columns*inputs,kt=c.k/32;std::vector<int8_t>aq(uint64_t(arows)*c.k);std::vector<float>as(uint64_t(arows)*kt);std::vector<int32_t>indices(uint64_t(columns)*inputs);
  for(int e=0;e<inputs;e++)for(int col=0;col<columns;col++)indices[e*columns+col]=((col*5)%columns)*inputs+e;
  // A nontrivial gathered order plus one zero sentinel, including full-row tails.
  if(columns>1)indices[columns-1]=-1;
  for(size_t i=0;i<aq.size();i++)aq[i]=int((i*73+i/31*19)%255)-127;
  for(size_t i=0;i<as.size();i++){half h=__float2half(float(1+i%29)*.001731f);as[i]=__half2float(h);}
  int8_t*xq=alloc<int8_t>(aq.size());float*xs=alloc<float>(as.size());int32_t*ri=alloc<int32_t>(indices.size());CU(cudaMemcpyAsync(xq,aq.data(),aq.size(),cudaMemcpyHostToDevice,S));CU(cudaMemcpyAsync(xs,as.data(),as.size()*4,cudaMemcpyHostToDevice,S));CU(cudaMemcpyAsync(ri,indices.data(),indices.size()*4,cudaMemcpyHostToDevice,S));
  PreparedActivationView a;a.source={xq,xs,c.k,arows,32,0,c.k,kt};a.columns=columns;a.input_experts=inputs;a.row_indices=ri;uint64_t bbytes=matmul_activation_bytes(columns,c.k,inputs);uint32_t*bi=alloc<uint32_t>(bbytes/4);a.operands=bi;auto prep=[&]{prepare_matmul_activations<<<(bbytes/4+255)/256,256,0,S>>>(a,bi);CU(cudaGetLastError());};prep();
  size_t outputs=uint64_t(columns)*c.experts*c.n;float*out=alloc<float>(outputs),*ref=alloc<float>(outputs);std::vector<float>got(outputs),reference(outputs);
  scalar_reference<<<dim3((c.n+3)/4,columns,c.experts),128,0,S>>>(w,a,ref,c.experts);CU(cudaMemcpyAsync(reference.data(),ref,outputs*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));
  int chunk=std::min(c.n,4096);uint64_t ints=uint64_t(columns)*c.experts*chunk*kt;int32_t*iref=alloc<int32_t>(ints);CU(cudaMemsetAsync(bad,0,8,S));
  for(int begin=0;begin<c.n;begin+=chunk){int rows=std::min(chunk,c.n-begin);uint64_t count=uint64_t(columns)*c.experts*rows*kt;integer_reference<<<(count+255)/256,256,0,S>>>(dq,a,c.n,c.experts,begin,rows,iref);launch<F,4,true>(pm,a,out,{iref,bad,begin,rows});}
  CU(cudaMemcpyAsync(&mismatch,bad,8,cudaMemcpyDeviceToHost,S));CU(cudaMemcpyAsync(got.data(),out,outputs*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));printf("{\"guard\":\"all_K32_integer_partials\",\"case\":\"%s\",\"columns\":%d,\"input_experts\":%d,\"partials\":%llu,\"different\":%llu}\n",c.key.c_str(),columns,inputs,(unsigned long long)outputs*kt,mismatch);if(mismatch)exit(2);
  report("scalar_float_tree_difference",c,columns,inputs,4,diff(reference,got));
  // Independent FP64 dequant dot at twelve positions, exact quantized input.
  double max64=0,rel64=0;
  for(int j=0;j<12;j++){int col=j*(columns-1)/11,e=j%c.experts,r=(uint64_t(j)*7919)%c.n,ar=indices[(inputs==1?0:e)*columns+col];double value=0;
   if(ar>=0)for(int g=0;g<kt;g++){int sum=0;for(int k=0;k<32;k++)sum+=int(hq[(uint64_t(e)*c.n+r)*c.k+g*32+k])*int(aq[uint64_t(ar)*c.k+g*32+k]);uint16_t h=hs[(uint64_t(e)*c.n+r)*kt+g];double scale=__half2float(*reinterpret_cast<half*>(&h));value+=sum*scale*double(as[uint64_t(ar)*kt+g]);}
   double delta=fabs(value-got[(uint64_t(col)*c.experts+e)*c.n+r]);max64=std::max(max64,delta);rel64=std::max(rel64,delta/std::max(1.,fabs(value)));}
  printf("{\"guard\":\"selected_FP64_dequant\",\"case\":\"%s\",\"columns\":%d,\"positions\":12,\"max_abs\":%.9g,\"max_relative_floor1\":%.9g}\n",c.key.c_str(),columns,max64,rel64);if(rel64>3e-5)exit(2);
  std::vector<float>base=got;
  CU(cudaMemsetAsync(bad,0,8,S));
  for(int begin=0;begin<c.n;begin+=chunk){int rows=std::min(chunk,c.n-begin);uint64_t count=uint64_t(columns)*c.experts*rows*kt;integer_reference<<<(count+255)/256,256,0,S>>>(dq,a,c.n,c.experts,begin,rows,iref);split<F,true>(pm,a,out,128,false,{iref,bad,begin,rows});}
  CU(cudaMemcpyAsync(&mismatch,bad,8,cudaMemcpyDeviceToHost,S));CU(cudaMemcpyAsync(got.data(),out,outputs*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));printf("{\"guard\":\"splitk_all_K32_integer_partials\",\"case\":\"%s\",\"columns\":%d,\"partials\":%llu,\"different\":%llu}\n",c.key.c_str(),columns,(unsigned long long)outputs*kt,mismatch);if(mismatch)exit(2);
  Diff sd=diff(reference,got);report("splitk_scalar_bit_identity",c,columns,inputs,0,sd);if(sd.bits)exit(2);
  for(int block:{128,256})for(bool resident:{false,true}){
   auto fn=[&]{split<F>(pm,a,out,block,resident);};fn();CU(cudaMemcpyAsync(got.data(),out,outputs*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));Diff d=diff(reference,got);report(resident?"splitk_resident188_bit_identity":"splitk_wide_bit_identity",c,columns,inputs,block,d);if(d.bits)exit(2);
   rates(resident?"mma_splitk_resident188":"mma_splitk_wide",c,columns,inputs,block,price(fn,flush));
  }
  if(columns==1){
   CU(cudaMemsetAsync(bad,0,8,S));
   for(int begin=0;begin<c.n;begin+=chunk){int rows=std::min(chunk,c.n-begin);uint64_t count=uint64_t(columns)*c.experts*rows*kt;integer_reference<<<(count+255)/256,256,0,S>>>(dq,a,c.n,c.experts,begin,rows,iref);split<F,true>(pm,a,out,128,false,{iref,bad,begin,rows},0,-1,true);}
   CU(cudaMemcpyAsync(&mismatch,bad,8,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));printf("{\"guard\":\"compact_s1_all_K32_integer_partials\",\"case\":\"%s\",\"partials\":%llu,\"different\":%llu}\n",c.key.c_str(),(unsigned long long)outputs*kt,mismatch);if(mismatch)exit(2);
   for(int block:{128,256})for(bool resident:{false,true}){
    auto fn=[&]{split<F>(pm,a,out,block,resident,{},0,-1,true);};fn();CU(cudaMemcpyAsync(got.data(),out,outputs*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));Diff d=diff(reference,got);report(resident?"compact_s1_resident188_bit_identity":"compact_s1_wide_bit_identity",c,columns,inputs,block,d);if(d.bits)exit(2);
    rates(resident?"mma_s1_compact_resident188":"mma_s1_compact_wide",c,columns,inputs,block,price(fn,flush));
   }
  }
  auto variant=[&](int t,std::function<void()>fn){fn();CU(cudaMemcpyAsync(got.data(),out,outputs*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));Diff d=diff(base,got);report("tile_identity",c,columns,inputs,t,d);if(d.bits)exit(2);rates("mma_q8",c,columns,inputs,t,price(fn,flush));};
  variant(1,[&]{launch<F,1>(pm,a,out);});variant(4,[&]{launch<F,4>(pm,a,out);});variant(16,[&]{launch<F,16>(pm,a,out);});rates("prepared_B_only_no_quantizer",c,columns,inputs,0,price(prep,flush));
  if(columns!=13)rates("scalar_warp32",c,columns,inputs,0,price([&]{scalar_reference<<<dim3((c.n+3)/4,columns,c.experts),128,0,S>>>(w,a,ref,c.experts);CU(cudaGetLastError());},flush));
  if(c.experts>1&&columns==13){
   constexpr int first=1,count=3;PreparedActivationView sub=a;
   if(inputs>1){sub.operands+=uint64_t(first)*((columns+7)/8)*kt*64;sub.row_indices+=first*columns;}
   Case subset=c;subset.experts=count;std::vector<float>expect(uint64_t(columns)*count*c.n),actual(expect.size());
   for(int col=0;col<columns;col++)for(int e=0;e<count;e++)std::copy_n(reference.data()+(uint64_t(col)*c.experts+first+e)*c.n,c.n,expect.data()+(uint64_t(col)*count+e)*c.n);
   for(bool resident:{false,true}){split<F>(pm,sub,out,256,resident,{},first,count);CU(cudaMemcpyAsync(actual.data(),out,actual.size()*4,cudaMemcpyDeviceToHost,S));CU(cudaStreamSynchronize(S));Diff d=diff(expect,actual);report(resident?"nonzero_first_expert_resident_bit_identity":"nonzero_first_expert_wide_bit_identity",subset,columns,inputs,256,d);if(d.bits)exit(2);}
  }
  CU(cudaFree(iref));CU(cudaFree(ref));CU(cudaFree(out));CU(cudaFree(bi));CU(cudaFree(ri));CU(cudaFree(xs));CU(cudaFree(xq));
 }
 CU(cudaFree(bad));CU(cudaFree(ws));CU(cudaFree(wi));CU(cudaFree(ds));CU(cudaFree(dq));CU(cudaFree(source));
}
int main(int argc,char**argv){if(argc<2||argc>3){fprintf(stderr,"usage: matmul FIXTURE_DIR [CASE]\n");return 1;}setvbuf(stdout,nullptr,_IOLBF,0);CU(cudaStreamCreate(&S));cudaDeviceProp p;CU(cudaGetDeviceProperties(&p,0));printf("{\"device\":\"%s\",\"SMs\":%d,\"weights\":\"real GGUF\",\"activations\":\"synthetic INT8 + non-dyadic effective FP16 scales\"}\n",p.name,p.multiProcessorCount);uint32_t*flush=alloc<uint32_t>((256ULL<<20)/4);CU(cudaMemsetAsync(flush,0,256ULL<<20,S));std::ifstream f(std::string(argv[1])+"/cases.tsv");if(!f)return 1;std::string line;int cases=0;while(std::getline(f,line)){Case c;std::istringstream s(line);s>>c.key>>c.format>>c.n>>c.k>>c.experts>>c.bytes>>c.cache>>c.qs>>c.scales;if(!s)return 1;if(argc==3&&c.key!=argv[2])continue;if(c.format==1)run<WeightFormat::Pane2>(c,flush);else if(c.format==2)run<WeightFormat::Q8Grouped128>(c,flush);else return 1;cases++;}CU(cudaFree(flush));CU(cudaStreamDestroy(S));printf("{\"passed\":true,\"cases\":%d}\n",cases);return cases?0:1;}
