#include "whole_ffn.cuh"
#include "../binary-fp4-pair/pair.cuh"
#include "../binary-composition/composition.cuh"
#include "../binary-a8-fp4/digits.cuh"
#include "../../src/kernels.cuh"
#include "../../src/gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define CUDA(x) do{auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(err));exit(2);}}while(0)
namespace h5 {void launch_argmax(const float*,int,int*,int,cudaStream_t);}
unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
struct Memory{std::vector<void*> all;template<class T>T* alloc(size_t n){T* p;CUDA(cudaMalloc(&p,std::max(size_t(1),n)*sizeof(T)));all.push_back(p);CUDA(cudaMemset(p,0,std::max(size_t(1),n)*sizeof(T)));return p;}~Memory(){for(auto p:all)cudaFree(p);}};
bitserial::Carrier carrier(Memory& m,int k,int r){return {m.alloc<int8_t>(size_t(r)*k),nullptr,m.alloc<float>(size_t(r)*k/128),m.alloc<int32_t>(size_t(r)*k/128),m.alloc<bitserial::InputStatus>(size_t(r)*k/128),m.alloc<bitserial::InputStatus>(1)};}
bitserial::Carrier view(bitserial::Carrier c,int k,int t){c.q+=size_t(t)*k;c.scales+=size_t(t)*k/128;c.dc+=size_t(t)*k/128;c.status+=size_t(t)*k/128;return c;}
struct Dp{int k,n,r;std::vector<bitserial::Plan> p;std::vector<float*> partial;Dp(Memory& m,int N,int K,int R):k(K),n(N),r(R){for(int t=0;t<r;t+=16){bitserial::Plan plan;CUDA(bitserial::select(n,k,std::min(16,r-t),&plan));p.push_back(plan);partial.push_back(m.alloc<float>(bitserial::partial_bytes(plan)/4));}}void run(const uint32_t* w,const __half* s,bitserial::Carrier c,float* y,cudaStream_t st){int t=0;for(size_t i=0;i<p.size();i++){CUDA(bitserial::project_carrier(p[i],w,s,view(c,k,t),y+size_t(t)*n,partial[i],1,0,st));t+=p[i].r;}}};
__global__ void raw_reference(const uint8_t* raw,bitserial::Carrier c,float* y,const float* initial,int n,int k,int r,int stride,float alpha,float beta){int col=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,t=blockIdx.y;if(col>=n||t>=r)return;float sum=0;for(int g=0;g<k/128;g++){const uint8_t* b=raw+(size_t(col)*k/128+g)*18;int dot=0;for(int j=lane;j<128;j+=32){int q=c.q[(size_t(t)*k/128+g)*128+j];dot+=((b[2+j/8]>>(j%8))&1)?q:-q;}for(int d=16;d;d/=2)dot+=__shfl_down_sync(0xffffffff,dot,d);float scale=__half2float(*(const __half*)b)*c.scales[size_t(t)*k/128+g];sum=fmaf(float(dot),scale,sum);}if(lane==0)y[size_t(t)*stride+col]=beta==0?alpha*sum:fmaf(alpha,sum,beta*initial[size_t(t)*stride+col]);}
__global__ void next(const float* y,const float* residual,float* h,int n,int r,bool gu){h5::pdl_wait();h5::pdl_launch();int cn=gu?n/2:n;size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(r)*cn)return;int t=i/cn,j=i%cn;if(gu){float g=y[size_t(t)*n+j];h[i]=g/(1.f+__expf(-g))*y[size_t(t)*n+cn+j];}else h[i]=y[i]+residual[i];}
__global__ void evict(unsigned* p,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]++;}
struct Graph{cudaGraph_t g;cudaGraphExec_t e;};
template<class F>Graph capture(cudaStream_t st,F f){Graph g;CUDA(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));f();CUDA(cudaStreamEndCapture(st,&g.g));CUDA(cudaGraphInstantiate(&g.e,g.g,0));return g;}
float timed(Graph g,cudaStream_t st,unsigned* trash,size_t n,bool cold){cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));std::vector<float> ts;for(int i=0;i<9;i++){if(cold)evict<<<(n+255)/256,256,0,st>>>(trash,n);CUDA(cudaEventRecord(a,st));CUDA(cudaGraphLaunch(g.e,st));CUDA(cudaEventRecord(b,st));CUDA(cudaEventSynchronize(b));float ms;CUDA(cudaEventElapsedTime(&ms,a,b));if(i>=2)ts.push_back(ms*1000);}std::sort(ts.begin(),ts.end());CUDA(cudaEventDestroy(a));CUDA(cudaEventDestroy(b));return ts[3];}
void destroy(Graph g){CUDA(cudaGraphExecDestroy(g.e));CUDA(cudaGraphDestroy(g.g));}
void check(const std::vector<float>& x,const std::vector<float>& y,double* maximum,double* relative){double sq=0,ss=0;*maximum=0;for(size_t i=0;i<x.size();i++){if(!std::isfinite(x[i])||!std::isfinite(y[i]))exit(3);double d=double(x[i])-y[i];*maximum=std::max(*maximum,std::abs(d));sq+=d*d;ss+=double(y[i])*y[i];}*relative=sqrt(sq/std::max(ss,1e-30));if(*relative>1e-2){fprintf(stderr,"rms %.9g max %.9g\n",*relative,*maximum);exit(3);}}



namespace h5 {
void launch_prep(const PrepArgs&,int,cudaStream_t);
cudaError_t launch_gemm_fp4_deterministic(int,int,const uint8_t*,const uint8_t*,const float*,int,int,const void*,const void*,int,float*,int,bool,cudaStream_t,const float*,int,bool,float*,size_t);
}
__global__ void joint_silu(const float* gu,float* h,int width,int rows){
  h5::pdl_wait();h5::pdl_launch();size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(width)*rows)return;
  int row=i/width,j=i%width;size_t at=size_t(row)*width*2+(j/32)*64+j%32;
  float g=gu[at];h[i]=g/(1.f+__expf(-g))*gu[at+32];
}
using PairPlane=binary_fp4_pair::Plane;
struct PairArena {
  int k;PairPlane primary,residual;
  PairArena(Memory& m,int K):k(K),primary{m.alloc<uint8_t>(size_t(64)*K/2),m.alloc<uint8_t>(size_t(64)*K/16),m.alloc<float>(64)},residual{m.alloc<uint8_t>(size_t(64)*K/2),m.alloc<uint8_t>(size_t(64)*K/16),m.alloc<float>(64)}{}
};
void planes(const float* x,float* f32,int k,int r,PairArena& a,cudaStream_t st){
  h5::PrepArgs p{};p.x=x;p.ldx=k;p.n=k;p.flags=h5::P_QUANT|h5::P_STORE_F32;p.out_f32=f32;p.ldo=k;p.q4=(uint8_t*)a.primary.q;p.s4=(uint8_t*)a.primary.sf;p.g4=(float*)a.primary.g;h5::launch_prep(p,r,st);CUDA(cudaGetLastError());
  h5::PrepArgs e{};e.x=f32;e.ldx=k;e.n=k;e.flags=h5::P_QUANT|h5::P_FP4_RESIDUAL;e.q4=(uint8_t*)a.residual.q;e.s4=(uint8_t*)a.residual.sf;e.g4=(float*)a.residual.g;e.ref_q4=a.primary.q;e.ref_s4=a.primary.sf;e.ref_g4=a.primary.g;h5::launch_prep(e,r,st);CUDA(cudaGetLastError());
}
void ordered(PairArena& a,int n,int r,int splits,const void* w,const void* scales,float* y,float* partial,size_t capacity,cudaStream_t st){
  for(bool residual:{false,true}){PairPlane p=residual?a.residual:a.primary;CUDA(h5::launch_gemm_fp4_deterministic(3,splits,p.q,p.sf,p.g,r,a.k,w,scales,n,y,n,residual,st,nullptr,8,false,partial,capacity));}
}
int joint_control(int argc,char** argv){
  if(argc!=3){fprintf(stderr,"whole_ffn_joint_control MODEL OUTPUT\n");return 1;}
  h5::Gguf model;model.open(argv[1]);auto g=model.t("blk.0.ffn_gate.weight"),u=model.t("blk.0.ffn_up.weight"),d=model.t("blk.0.ffn_down.weight");int k=g.ne[0],h=g.ne[1],n=d.ne[1];
  if(k!=5120||h!=17408||n!=5120||g.type!=41||u.type!=41||d.type!=41)return 1;
  Memory m;cudaStream_t st;CUDA(cudaStreamCreate(&st));FILE* f=fopen(argv[2],"w");if(!f)return 2;
  std::vector<uint8_t> joined(g.data,g.data+g.bytes);joined.insert(joined.end(),u.data,u.data+u.bytes);
  auto* raw=m.alloc<uint8_t>(joined.size());auto* draw=m.alloc<uint8_t>(d.bytes);CUDA(cudaMemcpy(raw,joined.data(),joined.size(),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(draw,d.data,d.bytes,cudaMemcpyHostToDevice));
  size_t gb=size_t(h)*2*k/128,db=size_t(n)*h/128;
  auto* gw=m.alloc<uint8_t>(gb*16);auto* gs=m.alloc<uint16_t>(gb);auto* dw=m.alloc<uint8_t>(db*16);auto* ds=m.alloc<uint16_t>(db);
  CUDA(h5::launch_repack_q1_fp4(raw,k,h,gw,gs,2*h/32,2,0,true,st));CUDA(h5::launch_repack_q1_fp4(raw+g.bytes,k,h,gw,gs,2*h/32,2,1,true,st));CUDA(h5::launch_repack_q1_fp4(draw,h,n,dw,ds,n/32,1,0,true,st));
  auto* aqgw=m.alloc<uint32_t>(gb*4);auto* aqgs=m.alloc<__half>(gb);auto* aqdw=m.alloc<uint32_t>(db*4);auto* aqds=m.alloc<__half>(db);auto* cgw=m.alloc<uint32_t>(gb*4);auto* cgs=m.alloc<__half>(gb);
  CUDA(bitserial::prepare(raw,aqgw,aqgs,2*h,k,st));CUDA(bitserial::prepare(draw,aqdw,aqds,n,h,st));CUDA(whole_ffn::prepare(raw,h,k,cgw,cgs,st));
  for(int cfg:{0,1}){CUDA(signed_fp4::configure(signed_fp4::Schedule(cfg)));CUDA(whole_ffn::configure(signed_fp4::Schedule(cfg)));CUDA(a8_fp4::configure(a8_fp4::Schedule(cfg)));}
  auto* trash=m.alloc<unsigned>(128*1024*1024/4);
  for(int r:{16,32}){
    auto* x=m.alloc<float>(size_t(r)*k);auto* xf=m.alloc<float>(size_t(r)*k);auto* gu=m.alloc<float>(size_t(r)*2*h);auto* hid=m.alloc<float>(size_t(r)*h);auto* hf=m.alloc<float>(size_t(r)*h);auto* y=m.alloc<float>(size_t(r)*n);auto* gp=m.alloc<float>(size_t(r)*2*h*16);auto* dp=m.alloc<float>(size_t(r)*n*16);
    std::vector<float> hx(size_t(r)*k);for(size_t i=0;i<hx.size();i++)hx[i]=float(int(mix(i+999)%65536)-32768)/32761;CUDA(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
    PairArena input(m,k),hidden(m,h),tile_input(m,k),tile_hidden(m,h);std::vector<float> ordered_reference(size_t(r)*n),a8_reference(ordered_reference.size()),actual(ordered_reference.size());
    auto c=carrier(m,k,r),nc=carrier(m,h,r);signed_fp4::Digits id{m.alloc<uint8_t>(signed_fp4::digit_bytes(k,r)),signed_fp4::digit_bytes(k,r),signed_fp4::Encoding::ExactA8},nd{m.alloc<uint8_t>(signed_fp4::digit_bytes(h,r)),signed_fp4::digit_bytes(h,r),signed_fp4::Encoding::ExactA8};
    a8_fp4::Digits i4{m.alloc<uint8_t>(a8_fp4::digit_bytes(k,r)),a8_fp4::digit_bytes(k,r)},n4{m.alloc<uint8_t>(a8_fp4::digit_bytes(h,r)),a8_fp4::digit_bytes(h,r)};
    Dp dd(m,n,h,r);auto* cp=m.alloc<float>(size_t(2*h)*16*((k/128+3)/4));
    auto ordinary_full=[&](){planes(x,xf,k,r,input,st);ordered(input,2*h,r,1,gw,gs,gu,gp,size_t(r)*2*h*16,st);h5::pdl(joint_silu,dim3((size_t(h)*r+255)/256),dim3(256),0,st,(const float*)gu,hid,h,r);planes(hid,hf,h,r,hidden,st);ordered(hidden,n,r,8,dw,ds,y,dp,size_t(r)*n*16,st);};
    auto joint_full=[&](int down_splits){for(int offset=0;offset<r;offset+=16){int rows=std::min(16,r-offset);planes(x+size_t(offset)*k,xf+size_t(offset)*k,k,rows,tile_input,st);
      CUDA(binary_fp4_pair::project({binary_fp4_pair::Schedule::F4G2T16,2*h,k,rows,1},tile_input.primary,tile_input.residual,gw,gs,gu+size_t(offset)*2*h,2*h,gp,size_t(r)*2*h*16,st,false));
      h5::pdl(joint_silu,dim3((size_t(h)*rows+255)/256),dim3(256),0,st,(const float*)(gu+size_t(offset)*2*h),hid+size_t(offset)*h,h,rows);planes(hid+size_t(offset)*h,hf+size_t(offset)*h,h,rows,tile_hidden,st);
      CUDA(binary_fp4_pair::project({binary_fp4_pair::Schedule::F4G2T8,n,h,rows,down_splits},tile_hidden.primary,tile_hidden.residual,dw,ds,y+size_t(offset)*n,n,dp,size_t(r)*n*16,st,false));
    }};
    auto compound=[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));auto next=nc;next.q=nullptr;whole_ffn::Scratch scratch{id,next,nd,dp,size_t(r)*n*16};signed_fp4::Plan down{n,h,r,r==16?16:8,signed_fp4::Schedule::F4G2T16,signed_fp4::WeightLayout::Packed1};CUDA(whole_ffn::project({h,k,r,r==16?signed_fp4::Schedule::F4G2T8:signed_fp4::Schedule::F4G2T16},down,{cgw,cgs,aqdw,aqds},c,scratch,y,n,1,0,st));};
    auto four=[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));CUDA(a8_fp4::encode(c,k,r,i4,st));a8_fp4::Plan g{2*h,k,r,r==16?4:1,a8_fp4::Schedule::F4G2T16,a8_fp4::WeightLayout::Packed1};CUDA(a8_fp4::project(g,aqgw,aqgs,c,i4,gu,2*h,gp,size_t(r)*2*h*16,1,0,st));h5::pdl(next,dim3((size_t(h)*r+255)/256),dim3(256),0,st,(const float*)gu,(const float*)gu,hid,2*h,r,true);CUDA(bitserial::produce(bitserial::Route::Dp4a01,hid,h,r,nc,st));CUDA(a8_fp4::encode(nc,h,r,n4,st));a8_fp4::Plan down{n,h,r,r==16?16:8,a8_fp4::Schedule::F4G2T16,a8_fp4::WeightLayout::Packed1};CUDA(a8_fp4::project(down,aqdw,aqds,nc,n4,y,n,dp,size_t(r)*n*16,1,0,st));};
    auto dp4a=[&](){CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));for(int offset=0;offset<r;offset+=16)CUDA(composition::fused({h,k,std::min(16,r-offset),8,16},aqgw,aqgs,view(c,k,offset),view(nc,h,offset),cp,st));dd.run(aqdw,aqds,nc,y,st);};
    ordinary_full();CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(ordered_reference.data(),y,actual.size()*4,cudaMemcpyDeviceToHost));compound();CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(a8_reference.data(),y,actual.size()*4,cudaMemcpyDeviceToHost));
    auto record=[&](const char* route,int dsplits,bool native_pair,bool is_a8,auto body){Graph graph=capture(st,body);float warm=timed(graph,st,trash,128*1024*1024/4,false),cold=timed(graph,st,trash,128*1024*1024/4,true);CUDA(cudaMemcpy(actual.data(),y,actual.size()*4,cudaMemcpyDeviceToHost));
      double max=0,sq=0,power=0,qmax=0,qsq=0,qpower=0;for(size_t i=0;i<actual.size();i++){if(!std::isfinite(actual[i]))exit(3);double d=double(actual[i])-ordered_reference[i];max=std::max(max,std::abs(d));sq+=d*d;power+=double(ordered_reference[i])*ordered_reference[i];double q=double(actual[i])-a8_reference[i];qmax=std::max(qmax,std::abs(q));qsq+=q*q;qpower+=double(a8_reference[i])*a8_reference[i];}
      double rel=std::sqrt(sq/std::max(power,1e-30)),qrel=std::sqrt(qsq/std::max(qpower,1e-30));if(native_pair&&rel>1e-4){fprintf(stderr,"joint own2pass fullFFN rms %.9g max%.9g\n",rel,max);exit(3);}
      fprintf(f,"{\"kind\":\"complete_joint_control\",\"route\":\"%s\",\"r\":%d,\"boundary\":\"common_f32_postnorm_full_producers\",\"joint_gu_cfg\":3,\"joint_gu_splits\":1,\"joint_down_cfg\":1,\"selected_down_splits\":%d,\"ordinary_fp4_cfg\":3,\"uses_joint\":%s,\"quantization_class\":\"%s\",\"warm_us\":%.6f,\"cold_us\":%.6f,\"max_error_own_ordered_two_pass\":%.9g,\"relative_rms_own_ordered_two_pass\":%.9g,\"max_difference_a8_compound\":%.9g,\"relative_rms_difference_a8_compound\":%.9g,\"same_a8_exactness_claim\":false}\n",route,r,dsplits,native_pair?"true":"false",is_a8?"original_k128_a8":"original_primary_plus_residual_nvfp4",warm,cold,max,rel,qmax,qrel);fflush(f);destroy(graph);
    };
    record(r==16?"ordered_two_pass_reference_r16":"ordered_two_pass_r32_reference",8,false,false,ordinary_full);
    for(int split:{8,16})record(r==16?"joint_pair_current_r16":"joint_pair_explicit_two_r16_tiles",split,true,false,[&](){joint_full(split);});
    record("compound_exact3_a8",r==16?16:8,false,true,compound);record("four_digits_a8",r==16?16:8,false,true,four);record("fused_dp4a_a8",4,false,true,dp4a);
  }
  fclose(f);puts("complete joint FFN window passed R16/R32");return 0;
}
int main(int argc,char** argv){try{return joint_control(argc,argv);}catch(const std::exception& e){fprintf(stderr,"error: %s\n",e.what());return 2;}}
