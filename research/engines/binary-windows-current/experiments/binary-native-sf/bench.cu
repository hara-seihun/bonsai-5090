#define H5_WARP_MICROSCOPE_STANDALONE
#include "../../src/fp4.cu"
#include "native.cuh"
#include "../binary-bitserial/bitserial.cuh"
namespace binary_native_sf { cudaError_t pair_control(int,int,int,int,int,Plane,Plane,const uint8_t*,const uint16_t*,float*,float*,cudaStream_t); }
#include "../../src/gguf.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <chrono>
#define CUDA(c) do { auto er=(c); if(er!=cudaSuccess){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(er));exit(2);} } while(0)
namespace h5 {
void launch_argmax(const float*,int,int*,int,cudaStream_t);
void launch_prep(const PrepArgs&,int,cudaStream_t);
}
using namespace h5;
using binary_native_sf::Plane;
using binary_native_sf::Plan;
using binary_native_sf::Schedule;
unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
template<class T>T* alloc(size_t n){T* p;CUDA(cudaMalloc(&p,n*sizeof(T)));CUDA(cudaMemset(p,0,n*sizeof(T)));return p;}
__global__ void evict(unsigned* p,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]+=1;}
__global__ void reduce_ordered(const float* p,float* y,int n,int r,int splits,bool add){
  pdl_wait();pdl_launch();size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,plane=size_t(n)*r;if(i>=plane)return;
  float v=p[i];for(int s=1;s<splits;s++)v=__fadd_rn(v,p[size_t(s)*plane+i]);if(add)v=__fadd_rn(y[i],v);y[i]=v;
}
__global__ void silu_il(const float* y,float* h,int n,int r){pdl_wait();pdl_launch();size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*r)return;
  int col=i%n,t=i/n;size_t b=size_t(t)*n*2+(col/32)*64+col%32;float g=y[b];h[i]=g/(1.f+__expf(-g))*y[b+32];}
__global__ void add_residual(const float* y,const float* residual,float* h,int n,int r){pdl_wait();pdl_launch();size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(n)*r)h[i]=__fadd_rn(y[i],residual[i]);}
struct Graph { cudaGraph_t g;cudaGraphExec_t e; };
Graph capture(cudaStream_t st,const std::function<void()>& f){Graph g;CUDA(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));f();CUDA(cudaStreamEndCapture(st,&g.g));CUDA(cudaGraphInstantiate(&g.e,g.g,0));return g;}
float time_graph(Graph g,cudaStream_t st,unsigned* trash,size_t count,bool cold){cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));std::vector<float> times;
  for(int i=0;i<13;i++){if(cold)evict<<<(count+255)/256,256,0,st>>>(trash,count);CUDA(cudaEventRecord(a,st));CUDA(cudaGraphLaunch(g.e,st));CUDA(cudaEventRecord(b,st));CUDA(cudaEventSynchronize(b));float ms;CUDA(cudaEventElapsedTime(&ms,a,b));if(i>=3)times.push_back(ms*1000);}
  std::sort(times.begin(),times.end());CUDA(cudaEventDestroy(a));CUDA(cudaEventDestroy(b));return times[times.size()/2];}
void destroy(Graph g){CUDA(cudaGraphExecDestroy(g.e));CUDA(cudaGraphDestroy(g.g));}
template<int FW,int WG,int TW,int MODE>void launch_baseline(Plane a,int n,int k,int r,int splits,const uint8_t* w,const uint16_t* s,float* y,cudaStream_t st){
  using G=WsGeom<FW,WG,TW,8>;auto f=k_gemm_fp4<FW,WG,TW,MODE,8>;CUDA(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM));
  pdl(f,dim3((r+TW-1)/TW,(n/32+G::NG-1)/G::NG,splits),dim3((FW+1)*32),G::SMEM,st,a.q,a.sf,a.g,r,k,w,(const uint8_t*)s,n,y,n,(const float*)nullptr);CUDA(cudaGetLastError());}
template<int MODE>void baseline_cfg(int cfg,Plane a,int n,int k,int r,int splits,const uint8_t* w,const uint16_t* s,float* y,cudaStream_t st){
  switch(cfg){
    case 0:launch_baseline<2,2,8,MODE>(a,n,k,r,splits,w,s,y,st);break;
    case 1:launch_baseline<4,2,8,MODE>(a,n,k,r,splits,w,s,y,st);break;
    case 2:launch_baseline<8,1,8,MODE>(a,n,k,r,splits,w,s,y,st);break;
    case 3:launch_baseline<4,2,16,MODE>(a,n,k,r,splits,w,s,y,st);break;
    case 4:launch_baseline<8,1,32,MODE>(a,n,k,r,splits,w,s,y,st);break;
    case 5:launch_baseline<4,2,32,MODE>(a,n,k,r,splits,w,s,y,st);break;
    case 6:launch_baseline<8,2,16,MODE>(a,n,k,r,splits,w,s,y,st);break;
    default:exit(2);
  }
}
float fp4_value(unsigned q){unsigned m=q&7;float f=m<4?.5f*m:m==4?2.f:m==5?3.f:m==6?4.f:6.f;return q&8?-f:f;}
struct HostPlane{std::vector<uint8_t> q,sf;std::vector<float> g;};
HostPlane download(Plane p,int k,int r){HostPlane h;h.q.resize(size_t(64)*k/2);h.sf.resize(size_t(64)*k/16);h.g.resize(r);CUDA(cudaMemcpy(h.q.data(),p.q,h.q.size(),cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(h.sf.data(),p.sf,h.sf.size(),cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(h.g.data(),p.g,r*4,cudaMemcpyDeviceToHost));return h;}
double decode(const HostPlane& h,int t,int j,int k){size_t rb=size_t(j/128)*64+t;int o=j%128,seg=o/32;unsigned b=h.q[rb*64+((seg^((t>>1)&3))<<4)+(o%32)/2];unsigned code=(b>>(4*(o%2)))&15;
  float sf=__half2float(__half(__nv_cvt_fp8_to_halfraw(h.sf[rb*8+o/16],__NV_E4M3)));return double(fp4_value(code))*sf*h.g[t];}
int run(int argc,char** argv){
  if(argc!=5){fprintf(stderr,"usage: bench MODEL gu|down|head R OUTPUT.jsonl\n");return 1;}
  int r=atoi(argv[3]);std::string kind=argv[2];if((r!=1&&r!=8&&r!=16&&r!=32)||(kind!="gu"&&kind!="down"&&kind!="head"))return 1;
  Gguf model;model.open(argv[1]);std::string name=kind=="gu"?"blk.0.ffn_gate.weight":kind=="down"?"blk.0.ffn_down.weight":"output.weight";
  const auto& t=model.t(name);int k=t.ne[0],n=t.ne[1];if(t.type!=41||k%1024||n%32)return 1;if(kind=="gu")n*=2;
  cudaStream_t st;CUDA(cudaStreamCreate(&st));
  auto* w=alloc<uint8_t>(size_t(n)*k/8);auto* s=alloc<uint16_t>(size_t(n)*k/128);
  auto* raw=alloc<uint8_t>(t.bytes);CUDA(cudaMemcpy(raw,t.data,t.bytes,cudaMemcpyHostToDevice));
  CUDA(launch_repack_q1_fp4(raw,k,t.ne[1],w,s,n/32,kind=="gu"?2:1,0,true,st));
  if(kind=="gu"){const auto& u=model.t("blk.0.ffn_up.weight");if(u.type!=41||u.ne!=t.ne)return 1;CUDA(cudaMemcpyAsync(raw,u.data,u.bytes,cudaMemcpyHostToDevice,st));CUDA(launch_repack_q1_fp4(raw,k,u.ne[1],w,s,n/32,2,1,true,st));}
  CUDA(cudaStreamSynchronize(st));CUDA(cudaFree(raw));
  uint8_t* ns[2]={alloc<uint8_t>(binary_native_sf::scale_bytes(n,k,1)),alloc<uint8_t>(binary_native_sf::scale_bytes(n,k,2))};
  float* rs=alloc<float>(n);auto* scale_status=alloc<binary_native_sf::ScaleStatus>(n);
  for(int wp=1;wp<=2;wp++)for(int cfg=0;cfg<7;cfg++)CUDA(binary_native_sf::configure(Schedule(cfg),wp));
  double scale_rms[2]={},scale_max_rel[2]={};
  std::vector<uint16_t> hs(size_t(n)*k/128);std::vector<float> hg(n);std::vector<uint8_t> host_ns[2];
  CUDA(cudaMemcpy(hs.data(),s,hs.size()*2,cudaMemcpyDeviceToHost));
  for(int wp=1;wp<=2;wp++){
    CUDA(binary_native_sf::prepare(s,ns[wp-1],rs,scale_status,n,k,wp,st));CUDA(cudaStreamSynchronize(st));
    std::vector<binary_native_sf::ScaleStatus> statuses(n);CUDA(cudaMemcpy(statuses.data(),scale_status,n*sizeof(statuses[0]),cudaMemcpyDeviceToHost));
    for(auto status:statuses)if(status!=binary_native_sf::ScaleStatus::Valid){fprintf(stderr,"nonfinite original weight scale\n");return 3;}
    std::vector<uint8_t> hnsc(binary_native_sf::scale_bytes(n,k,wp));
    CUDA(cudaMemcpy(hnsc.data(),ns[wp-1],hnsc.size(),cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(hg.data(),rs,n*4,cudaMemcpyDeviceToHost));
    double sq=0,power=0;
    for(int kb=0;kb<k/128;kb++)for(int row=0;row<n;row++){
      int packed=(row/32)*32+(row%8)*4+(row%32)/8;size_t chunk=size_t(kb)*(n/32)+row/32;
      double v=0;for(int z=0;z<wp;z++)v+=__half2float(__half(__nv_cvt_fp8_to_halfraw(hnsc[chunk*32*wp+32*z+packed%32],__NV_E4M3)));
      v*=hg[row];double original=__half2float(*reinterpret_cast<const __half*>(&hs[size_t(kb)*n+packed]));double d=v-original;
      sq+=d*d;power+=original*original;if(original!=0)scale_max_rel[wp-1]=std::max(scale_max_rel[wp-1],std::abs(d/original));
    }
    scale_rms[wp-1]=sqrt(sq/std::max(power,1e-30));host_ns[wp-1]=std::move(hnsc);
  }
  size_t inputs=size_t(r)*k,outputs=size_t(r)*n;
  auto* x=alloc<float>(inputs);auto* x2=alloc<float>(inputs);auto* xf=alloc<float>(inputs);auto* norm=alloc<float>(std::max(k,n));
  auto* y=alloc<float>(outputs);auto* ref=alloc<float>(outputs);auto* partial=alloc<float>(outputs*16);auto* h=alloc<float>(outputs);auto* hf=alloc<float>(outputs);
  auto* res=alloc<float>(outputs);auto* token=alloc<int>(r);
  Plane a{alloc<uint8_t>(size_t(64)*k/2),alloc<uint8_t>(size_t(64)*k/16),alloc<float>(64)};
  Plane e{alloc<uint8_t>(size_t(64)*k/2),alloc<uint8_t>(size_t(64)*k/16),alloc<float>(64)};
  int cn=kind=="gu"?n/2:n;
  Plane ca{alloc<uint8_t>(size_t(64)*cn/2),alloc<uint8_t>(size_t(64)*cn/16),alloc<float>(64)};
  Plane ce{alloc<uint8_t>(size_t(64)*cn/2),alloc<uint8_t>(size_t(64)*cn/16),alloc<float>(64)};
  std::vector<float> hx(inputs),hn(std::max(k,n),1.f),hr(outputs),hu(inputs);
  for(size_t i=0;i<inputs;i++){hx[i]=(int(mix(i+999)%65536)-32768)/32761.f;hu[i]=(int(mix(i+412)%65536)-32768)/32761.f;}
  for(size_t i=0;i<outputs;i++)hr[i]=(int(mix(i+7)%65536)-32768)/32761.f;
  CUDA(cudaMemcpy(x,hx.data(),inputs*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(x2,hu.data(),inputs*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(res,hr.data(),outputs*4,cudaMemcpyHostToDevice));
  {const auto& nw=model.t(kind=="gu"?"blk.0.post_attention_norm.weight":kind=="down"?"blk.1.attn_norm.weight":"output_norm.weight");int width=kind=="down"?cn:k;if(nw.type!=0||nw.ne[0]!=size_t(width))return 1;std::copy((const float*)nw.data,(const float*)nw.data+width,hn.begin());}
  CUDA(cudaMemcpy(norm,hn.data(),hn.size()*4,cudaMemcpyHostToDevice));
  auto producer=[&](bool dp=false){PrepArgs pa{};pa.x=x;pa.x2=kind=="down"?x2:nullptr;pa.ldx=k;pa.n=k;pa.flags=(dp?0:P_QUANT)|P_STORE_F32|(kind=="down"?P_SILU_MUL:P_NORM);pa.norm_w=norm;pa.out_f32=xf;pa.ldo=k;pa.q4=(uint8_t*)a.q;pa.s4=(uint8_t*)a.sf;pa.g4=(float*)a.g;launch_prep(pa,r,st);if(dp)return;
    PrepArgs pe{};pe.x=xf;pe.ldx=k;pe.n=k;pe.flags=P_QUANT|P_FP4_RESIDUAL;pe.q4=(uint8_t*)e.q;pe.s4=(uint8_t*)e.sf;pe.g4=(float*)e.g;pe.ref_q4=a.q;pe.ref_s4=a.sf;pe.ref_g4=a.g;launch_prep(pe,r,st);CUDA(cudaGetLastError());};
  auto continuation=[&](){
    if(kind=="head"){launch_argmax(y,n,token,r,st);CUDA(cudaGetLastError());return;}
    if(kind=="gu")pdl(silu_il,dim3((size_t(cn)*r+255)/256),dim3(256),0,st,(const float*)y,h,cn,r);
    else pdl(add_residual,dim3((outputs+255)/256),dim3(256),0,st,(const float*)y,(const float*)res,h,n,r);
    PrepArgs pa{};pa.x=h;pa.ldx=cn;pa.n=cn;pa.flags=P_QUANT|P_STORE_F32|(kind=="down"?P_NORM:0);pa.norm_w=norm;pa.out_f32=hf;pa.ldo=cn;pa.q4=(uint8_t*)ca.q;pa.s4=(uint8_t*)ca.sf;pa.g4=(float*)ca.g;launch_prep(pa,r,st);
    PrepArgs pe{};pe.x=hf;pe.ldx=cn;pe.n=cn;pe.flags=P_QUANT|P_FP4_RESIDUAL;pe.q4=(uint8_t*)ce.q;pe.s4=(uint8_t*)ce.sf;pe.g4=(float*)ce.g;pe.ref_q4=ca.q;pe.ref_s4=ca.sf;pe.ref_g4=ca.g;launch_prep(pe,r,st);CUDA(cudaGetLastError());
  };
  auto* dw=alloc<uint32_t>(size_t(n)*k/32);auto* ds=alloc<__half>(size_t(n)*k/128);
  {
    std::vector<uint8_t> raw_dp(size_t(n)*k/128*18);size_t rowbytes=size_t(k)/128*18;
    for(int row=0;row<n;row++){
      int original=row;const uint8_t* source=t.data;
      if(kind=="gu"){int group=row/32;original=(group/2)*32+row%32;source=model.t(group&1?"blk.0.ffn_up.weight":"blk.0.ffn_gate.weight").data;}
      std::copy(source+size_t(original)*rowbytes,source+size_t(original+1)*rowbytes,raw_dp.data()+size_t(row)*rowbytes);
    }
    auto* rawdev=alloc<uint8_t>(raw_dp.size());CUDA(cudaMemcpy(rawdev,raw_dp.data(),raw_dp.size(),cudaMemcpyHostToDevice));CUDA(bitserial::prepare(rawdev,dw,ds,n,k,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaFree(rawdev));
  }
  bitserial::Carrier dc{alloc<int8_t>(inputs),nullptr,alloc<float>(inputs/128),alloc<int32_t>(inputs/128),alloc<bitserial::InputStatus>(inputs/128),nullptr};
  std::vector<bitserial::Plan> dpplans;std::vector<float*> dppartials;
  for(int tok=0;tok<r;tok+=16){bitserial::Plan dp;CUDA(bitserial::select(n,k,std::min(16,r-tok),&dp));dpplans.push_back(dp);dppartials.push_back(alloc<float>(bitserial::partial_bytes(dp)/4));}
  auto dp_project=[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,xf,k,r,dc,st));int tok=0;
    for(size_t z=0;z<dpplans.size();z++){auto d=dc;d.q+=size_t(tok)*k;d.scales+=size_t(tok)*k/128;d.dc+=size_t(tok)*k/128;d.status+=size_t(tok)*k/128;CUDA(bitserial::project_carrier(dpplans[z],dw,ds,d,y+size_t(tok)*n,dppartials[z],1,0,st));tok+=dpplans[z].r;}
  };
  auto base_project=[&](int cfg,int splits){baseline_cfg<0>(cfg,a,n,k,r,splits,w,s,splits==1?y:partial,st);
    if(splits>1)pdl(reduce_ordered,dim3((outputs+255)/256),dim3(256),0,st,(const float*)partial,y,n,r,splits,false);
    if(splits==1)baseline_cfg<1>(cfg,e,n,k,r,1,w,s,y,st);
    else{baseline_cfg<0>(cfg,e,n,k,r,splits,w,s,partial,st);pdl(reduce_ordered,dim3((outputs+255)/256),dim3(256),0,st,(const float*)partial,y,n,r,splits,true);}
  };
  producer();base_project(4,1);CUDA(cudaStreamSynchronize(st));std::vector<float> href(outputs),hy(outputs);CUDA(cudaMemcpy(href.data(),y,outputs*4,cudaMemcpyDeviceToHost));
  auto ha=download(a,k,r),he=download(e,k,r);double sample_error=0,sample_power=0;
  for(int tok=0;tok<r;tok++)for(int ii=0;ii<16;ii++){int col=ii*n/16;int original=col;const uint8_t* weights=t.data;
    if(kind=="gu"){int group=col/32;bool up=group&1;original=(group/2)*32+col%32;weights=model.t(up?"blk.0.ffn_up.weight":"blk.0.ffn_gate.weight").data;}
    double sum=0;for(int kb=0;kb<k/128;kb++){auto* block=weights+(size_t(original)*(k/128)+kb)*18;double dot=0;for(int j=0;j<128;j++){double av=decode(ha,tok,kb*128+j,k)+decode(he,tok,kb*128+j,k);dot+=((block[2+j/8]>>(j%8))&1)?av:-av;}sum+=dot*__half2float(*(const __half*)block);}
    double diff=sum-href[size_t(tok)*n+col];sample_error=std::max(sample_error,std::abs(diff));sample_power=std::max(sample_power,std::abs(sum));}
  if(sample_error>1e-4*std::max(1.,sample_power)){fprintf(stderr,"baseline scalar mismatch %.9g power %.9g\n",sample_error,sample_power);return 3;}
  size_t trashn=128*1024*1024/4;auto* trash=alloc<unsigned>(trashn);
  FILE* file=fopen(argv[4],"w");if(!file)return 2;
  for(int route=0;route<5;route++)for(int cfg=(r<=8?0:3);cfg<(r<=8?4:7);cfg++)for(int splits:{1,4,8,16}){
    if(splits>k/128||k/128/splits<2||(r==32&&splits==8)||(route==4&&(cfg!=(r<=8?0:3)||splits!=1)))continue;
    bool native=route==1||route==2;Plan plan{Schedule(cfg),n,k,r,splits,route};
    auto full=[&](){producer(route==4);if(native)CUDA(binary_native_sf::project(plan,a,e,w,ns[route-1],rs,y,n,partial,outputs*16,st,false));else if(route==3)CUDA(binary_native_sf::pair_control(cfg,n,k,r,splits,a,e,w,s,y,partial,st));else if(route==4)dp_project();else base_project(cfg,splits);continuation();};
    Graph graph=capture(st,full);float warm=time_graph(graph,st,trash,trashn,false),cold=time_graph(graph,st,trash,trashn,true);
    CUDA(cudaMemcpy(hy.data(),y,outputs*4,cudaMemcpyDeviceToHost));double maximum=0,sq=0,power=0;
    for(size_t i=0;i<outputs;i++){double d=double(hy[i])-href[i];if(!std::isfinite(d))return 3;maximum=std::max(maximum,std::abs(d));sq+=d*d;power+=double(href[i])*href[i];}
    double native_sample_error=0,native_sample_power=0;
    if(native)for(int tok=0;tok<r;tok++)for(int ii=0;ii<16;ii++){
      int col=ii*n/16,original=col;const uint8_t* weights=t.data;
      if(kind=="gu"){int group=col/32;original=(group/2)*32+col%32;weights=model.t(group&1?"blk.0.ffn_up.weight":"blk.0.ffn_gate.weight").data;}
      int packed=(col%8)*4+(col%32)/8;double sum=0;
      for(int kb=0;kb<k/128;kb++){
        size_t chunk=size_t(kb)*(n/32)+col/32;double ws=0;
        for(int z=0;z<route;z++)ws+=__half2float(__half(__nv_cvt_fp8_to_halfraw(host_ns[route-1][chunk*32*route+32*z+packed],__NV_E4M3)));
        ws*=hg[col];auto* block=weights+(size_t(original)*(k/128)+kb)*18;double dot=0;
        for(int j=0;j<128;j++){double av=decode(ha,tok,kb*128+j,k)+decode(he,tok,kb*128+j,k);dot+=((block[2+j/8]>>(j%8))&1)?av:-av;}sum+=dot*ws;
      }
      native_sample_error=std::max(native_sample_error,std::abs(sum-hy[size_t(tok)*n+col]));native_sample_power=std::max(native_sample_power,std::abs(sum));
    }
    if(native_sample_error>1e-4*std::max(1.,native_sample_power)){fprintf(stderr,"native scalar mismatch route%d cfg%d split%d error%.9g power%.9g\\n",route,cfg,splits,native_sample_error,native_sample_power);return 3;}
    double rms=sqrt(sq/std::max(power,1e-30));if(route==0&&rms>1e-5){fprintf(stderr,"numericalfailure route%d cfg%d split%d rms%.9g\n",route,cfg,splits,rms);return 3;}
    if(native){
      std::vector<float> added(outputs);CUDA(cudaMemcpyAsync(y,res,outputs*4,cudaMemcpyDeviceToDevice,st));
      CUDA(binary_native_sf::project(plan,a,e,w,ns[route-1],rs,y,n,partial,outputs*16,st,true));CUDA(cudaStreamSynchronize(st));
      CUDA(cudaMemcpy(added.data(),y,outputs*4,cudaMemcpyDeviceToHost));
      for(size_t i=0;i<outputs;i++)if(added[i]!=float(hy[i]+hr[i])){fprintf(stderr,"add contract mismatch cfg%d split%d at%zu\n",cfg,splits,i);return 3;}
    }
    size_t resident=size_t(n)*k/8+(native?binary_native_sf::scale_bytes(n,k,route)+size_t(n)*4:size_t(n)*k/64);
    fprintf(file,"{\"kind\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"route\":\"%s\",\"cfg\":%d,\"splits\":%d,\"inclusive_warm_us\":%.6f,\"inclusive_cold_us\":%.6f,\"max_error_ordered\":%.9g,\"relative_rms_ordered\":%.9g,\"baseline_scalar_sample_max\":%.9g,\"scale_relative_rms\":%.9g,\"scale_max_relative\":%.9g,\"weight_resident_bytes\":%zu,\"native_scalar_sample_max\":%.9g}\n",kind.c_str(),n,k,r,route==0?"ordered_two_pass":route==1?"native_sf1":route==2?"native_sf2":route==3?"joint_pair":"dp4a",cfg,splits,warm,cold,maximum,rms,sample_error,native?scale_rms[route-1]:0,native?scale_max_rel[route-1]:0,resident,native_sample_error);fflush(file);destroy(graph);
  }
  fclose(file);printf("complete %s N%d K%d R%d\n",kind.c_str(),n,k,r);return 0;
}
int main(int argc,char** argv){try{return run(argc,argv);}catch(const std::exception& e){fprintf(stderr,"error: %s\n",e.what());return 2;}}
