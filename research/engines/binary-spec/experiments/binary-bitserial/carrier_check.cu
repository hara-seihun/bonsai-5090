#include "bitserial.cuh"
#include <vector>
#include <cmath>
#include <cstdio>
#include <limits>
#include <algorithm>
#define CUDA(c) do{auto e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){
  constexpr int N=32,K=128,MAXR=512;
  std::vector<uint8_t> raw(N*18);for(int n=0;n<N;n++){__half s=__float2half((n%8+1)/8.f);raw[n*18]=reinterpret_cast<uint8_t*>(&s)[0];raw[n*18+1]=reinterpret_cast<uint8_t*>(&s)[1];for(int j=0;j<16;j++)raw[n*18+2+j]=(n*71+j*33)&255;}
  std::vector<float> hx(MAXR*K),hy(MAXR*N),initial(MAXR*N),hs(MAXR);std::vector<int8_t> hq(MAXR*K);
  for(size_t i=0;i<hx.size();i++)hx[i]=int((i*71)%255)-127;
  for(size_t i=0;i<initial.size();i++)initial[i]=(int(i%23)-11)/7.f;
  cudaStream_t st;CUDA(cudaStreamCreate(&st));float *x,*y,*partial;uint8_t* d_raw;uint32_t* signs;__half* scales;
  CUDA(cudaMalloc(&x,hx.size()*4));CUDA(cudaMalloc(&y,hy.size()*4));CUDA(cudaMalloc(&partial,hy.size()*4));CUDA(cudaMalloc(&d_raw,raw.size()));CUDA(cudaMalloc(&signs,N*16));CUDA(cudaMalloc(&scales,N*2));
  bitserial::Carrier c{};CUDA(cudaMalloc(&c.q,MAXR*K));CUDA(cudaMalloc(&c.planes,MAXR*K));CUDA(cudaMalloc(&c.scales,MAXR*4));CUDA(cudaMalloc(&c.dc,MAXR*4));CUDA(cudaMalloc(&c.status,MAXR*4));CUDA(cudaMalloc(&c.error,4));
  CUDA(cudaMemcpyAsync(d_raw,raw.data(),raw.size(),cudaMemcpyHostToDevice,st));CUDA(bitserial::prepare(d_raw,signs,scales,N,K,st));CUDA(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));CUDA(cudaMemsetAsync(c.error,0,4,st));
  int compared=0;
  for(int r=1;r<=16;r++)for(auto route:{bitserial::Route::Popcount,bitserial::Route::Dp4a01}) {
    CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,K,r,c,st));CUDA(bitserial::produce(route,x,K,r,c,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hq.data(),c.q,r*K,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(hs.data(),c.scales,r*4,cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpyAsync(y,initial.data(),r*N*4,cudaMemcpyHostToDevice,st));bitserial::Plan p;CUDA(bitserial::select(N,K,r,route,&p));CUDA(bitserial::project_carrier(p,signs,scales,c,y,partial,.75f,.25f,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hy.data(),y,r*N*4,cudaMemcpyDeviceToHost));
    for(int t=0;t<r;t++)for(int n=0;n<N;n++){int dot=0;for(int j=0;j<K;j++)dot+=((raw[n*18+2+j/8]>>(j%8))&1)?int(hq[t*K+j]):-int(hq[t*K+j]);float s=(n%8+1)/8.f;float expect=std::fma(.75f,std::fma(float(dot),s*hs[t],0.f),.25f*initial[t*N+n]);if(expect!=hy[t*N+n]){fprintf(stderr,"map/continuation mismatch R%d row%d\n",r,n);return 3;}compared++;}
  }
  CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,K,MAXR,c,st));CUDA(cudaStreamSynchronize(st));
  std::vector<bitserial::InputStatus> statuses(MAXR);CUDA(cudaMemcpy(statuses.data(),c.status,MAXR*4,cudaMemcpyDeviceToHost));for(auto s:statuses)if(s!=bitserial::InputStatus::Valid)return 3;
  float bad=std::numeric_limits<float>::quiet_NaN();CUDA(cudaMemcpyAsync(x,&bad,4,cudaMemcpyHostToDevice,st));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,K,MAXR,c,st));
  CUDA(cudaMemcpyAsync(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice,st));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,K,MAXR,c,st));CUDA(cudaStreamSynchronize(st));
  bitserial::InputStatus flag,group;CUDA(cudaMemcpy(&flag,c.error,4,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(&group,c.status,4,cudaMemcpyDeviceToHost));if(flag!=bitserial::InputStatus::NonFinite||group!=bitserial::InputStatus::Valid)return 3;
  CUDA(cudaMemsetAsync(c.error,0,4,st));std::vector<float> tiny(K,1e-35f);CUDA(cudaMemcpyAsync(x,tiny.data(),K*4,cudaMemcpyHostToDevice,st));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,K,MAXR,c,st));
  bitserial::Plan p;CUDA(bitserial::select(N,K,1,&p));CUDA(bitserial::project_carrier(p,signs,scales,c,y,partial,1,0,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(&flag,c.error,4,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(hy.data(),y,N*4,cudaMemcpyDeviceToHost));if(flag!=bitserial::InputStatus::ScaleRange)return 3;for(int n=0;n<N;n++)if(!std::isnan(hy[n]))return 3;
  CUDA(cudaMemsetAsync(x,0,hx.size()*4,st));CUDA(cudaMemsetAsync(c.error,0,4,st));CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,K,MAXR,c,st));CUDA(bitserial::project_carrier(p,signs,scales,c,y,partial,1,0,st));CUDA(cudaStreamSynchronize(st));CUDA(cudaMemcpy(hy.data(),y,N*4,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(&flag,c.error,4,cudaMemcpyDeviceToHost));if(flag!=bitserial::InputStatus::Valid)return 3;for(int n=0;n<N;n++)if(hy[n]!=0)return 3;
  puts("all R1..16 native maps and alpha/beta continuation bit-exact: 8704 outputs");
  printf("checked %d outputs; R512 producer, persistent nonfinite error, typed scale-range error/NaN, and zero-carrier path pass\n",compared);
  return compared==8704?0:3;
}
