// Compare the no-extra-boundary row kernel against row, duplicate and prepass routes.
#include "../src/kernels.cu"
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstring>
static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> static T* alloc(size_t count) { T* p; ck(cudaMalloc(&p,count*sizeof(T))); return p; }
static size_t different(const std::vector<float>& a,const std::vector<float>& b) {
  size_t n=0; for(size_t i=0;i<a.size();i++) if(memcmp(&a[i],&b[i],sizeof(float))) n++; return n;
}
int main(int argc,char** argv) { try {
  if(argc!=2) throw std::runtime_error("usage: prep-row-bench FIXED_G4");
  char* end; errno=0; float fixed=strtof(argv[1],&end);
  if(errno || end==argv[1] || *end || !std::isfinite(fixed) || fixed<=0 || !std::isfinite(1.f/fixed))
    throw std::runtime_error("invalid fixed global scale");
  cudaStream_t st; ck(cudaStreamCreate(&st));
  std::cout<<"rows,n,rowmap,out_norm,mode,chain_us,norm_different,prepared_different,norm_max_abs,prepared_max_abs\n";
  for(int rows:{1,8}) for(int n:{5120,6144,17408}) for(bool mapped:{false,true}) for(bool store_norm:{false,true}) {
    std::vector<void*> owned;
    auto floats=[&](const std::vector<float>& v) { auto p=alloc<float>(v.size()); owned.push_back(p); ck(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice)); return p; };
    std::mt19937 rng(9321+n); std::normal_distribution<float> normal(0,1);
    std::vector<float> x(rows*n),nw(n),signs(n);
    for(float& v:x) v=normal(rng);
    for(float& v:nw) v=.8f+.4f*std::fabs(normal(rng));
    for(float& v:signs) v=(rng()&1)?1.f:-1.f;
    h5::PrepArgs a{}; a.x=floats(x); a.norm_w=floats(nw); a.signs=floats(signs); a.n=n; a.ldx=n; a.ldo=n;
    a.flags=h5::P_NORM|h5::P_SIGN|h5::P_HADAMARD|h5::P_QUANT|(store_norm?h5::P_OUT_NORM:0);
    a.q4=alloc<uint8_t>(64*n/2); owned.push_back(a.q4);
    a.s4=alloc<uint8_t>(64*n/16); owned.push_back(a.s4);
    a.g4=alloc<float>(rows); owned.push_back(a.g4);
    a.rms_inv=alloc<float>(rows); owned.push_back(a.rms_inv); a.rms_rows=rows;
    a.out_norm=alloc<float>(rows*n); owned.push_back(a.out_norm);
    a.out_f32=alloc<float>(rows*n); owned.push_back(a.out_f32);
    if(mapped) { std::vector<int> map(rows); for(int i=0;i<rows;i++)map[i]=rows-i-1;
      int* p=alloc<int>(rows); owned.push_back(p); ck(cudaMemcpy(p,map.data(),rows*4,cudaMemcpyHostToDevice)); a.rowmap=p;
    }
    auto launch=[&](int mode) {
      if(mode==0) h5::pdl(h5::k_prep2,dim3(rows),dim3(32*n/1024),0,st,a);
      else if(mode==1) h5::pdl(h5::k_prep_chunk<false>,dim3(rows,n/1024),dim3(32),0,st,a,fixed);
      else if(mode==2) h5::launch_prep_rms(a,rows,fixed,st);
      else if(mode==4) h5::launch_prep_row_norm(a,rows,fixed,st);
      else h5::pdl(h5::k_prep_rms,dim3(rows),dim3(32*n/1024),0,st,a);
      ck(cudaGetLastError());
    };
    std::vector<float> refnorm(rows*n),reff32(rows*n);
    for(int mode=0;mode<5;mode++) {
      cudaGraph_t graph; cudaGraphExec_t exec;
      ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
      for(int i=0;i<256;i++)launch(mode);
      ck(cudaStreamEndCapture(st,&graph)); ck(cudaGraphInstantiate(&exec,graph,0));
      ck(cudaGraphLaunch(exec,st)); ck(cudaStreamSynchronize(st));
      cudaEvent_t start,stop; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
      std::vector<float> samples;
      for(int rep=0;rep<7;rep++) { ck(cudaEventRecord(start,st)); ck(cudaGraphLaunch(exec,st)); ck(cudaEventRecord(stop,st)); ck(cudaEventSynchronize(stop)); float ms; ck(cudaEventElapsedTime(&ms,start,stop)); samples.push_back(ms*1000/256); }
      std::sort(samples.begin(),samples.end());
      size_t nd=0,pd=0; float nm=0,pm=0;
      if(mode!=3) {
        a.flags|=h5::P_STORE_F32; launch(mode); ck(cudaStreamSynchronize(st)); a.flags&=~h5::P_STORE_F32;
        std::vector<float> norm(rows*n),prepared(rows*n);
        if(store_norm) ck(cudaMemcpy(norm.data(),a.out_norm,rows*n*4,cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(prepared.data(),a.out_f32,rows*n*4,cudaMemcpyDeviceToHost));
        if(!mode) {refnorm=norm; reff32=prepared;}
        for(size_t i=0;i<norm.size();i++) {
          if(!std::isfinite(norm[i])||!std::isfinite(prepared[i]))throw std::runtime_error("nonfinite output");
          nm=std::max(nm,std::fabs(norm[i]-refnorm[i])); pm=std::max(pm,std::fabs(prepared[i]-reff32[i]));
        }
        nd=different(norm,refnorm); pd=different(prepared,reff32);
        if(mode==2 && (nd||pd)) throw std::runtime_error("RMS prepass changed prequantization computation");
      }
      std::cout<<rows<<','<<n<<','<<mapped<<','<<store_norm<<','<<mode<<','<<std::setprecision(9)<<samples[3]<<','<<nd<<','<<pd<<','<<nm<<','<<pm<<'\n';
      ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop)); ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
    }
    for(void* p:owned)ck(cudaFree(p));
  }
  ck(cudaStreamDestroy(st)); return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
