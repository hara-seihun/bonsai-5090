#include "batch_map.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#define CK(c) do{auto e=(c);if(e!=cudaSuccess){fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(e));return 2;}}while(0)
int main(){constexpr int N=17,K=128,R=512;
  float *x,*y;unsigned *signs,*packed;__half* ws;uint8_t* counts;bitserial::Carrier meta{},ref{};
  CK(cudaMalloc(&x,R*K*4));CK(cudaMalloc(&y,16*N*4));CK(cudaMalloc(&signs,N*16));CK(cudaMalloc(&ws,N*2));CK(cudaMalloc(&counts,N));CK(cudaMalloc(&packed,R*K*2));
  for(auto* c:{&meta,&ref}){CK(cudaMalloc(&c->scales,R*4));CK(cudaMalloc(&c->dc,R*4));CK(cudaMalloc(&c->status,R*4));}
  CK(cudaMalloc(&meta.error,4));CK(cudaMemset(meta.error,0,4));CK(cudaMalloc(&ref.q,R*K));
  std::vector<float> hx(R*K);for(int t=0;t<R;t++)for(int j=0;j<K;j++)hx[t*K+j]=t==0?0:float(((t*17+j*31)%255)-127)/64.f;
  std::vector<unsigned> hw(N*4);for(int w=0;w<4;w++)for(int row=0;row<N;row++)hw[w*N+row]=row==0?0:row==1?0xffffffffU:0x9e3779b9U*unsigned(row+w*7);
  std::vector<__half> hs(N,__float2half(1));CK(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(signs,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(ws,hs.data(),hs.size()*2,cudaMemcpyHostToDevice));
  batch_map::Carrier c{packed,meta,K,R};CK(batch_map::prepare_counts(signs,counts,N,K,0));CK(batch_map::produce(x,c,0));CK(bitserial::produce(bitserial::Route::Dp4a01,x,K,R,ref,0));
  std::vector<int8_t> hq(R*K);std::vector<float> sx(R);CK(cudaMemcpy(hq.data(),ref.q,hq.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(sx.data(),ref.scales,R*4,cudaMemcpyDeviceToHost));
  std::vector<float> initial(16*N),hy(16*N);for(size_t i=0;i<initial.size();i++)initial[i]=float(int(i%17)-8)/16;
  int checked=0;
  for(int first:{0,16,496})for(int r=1;r<=16;r++){
    CK(cudaMemcpy(y,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));batch_map::Plan p{N,K,r,first,1,128};CK(batch_map::project(p,signs,ws,counts,c,y,nullptr,.75f,.25f,0));CK(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
    for(int t=0;t<r;t++)for(int row=0;row<N;row++){int dot=0;for(int j=0;j<K;j++){int q=hq[(first+t)*K+j];dot+=((hw[(j/32)*N+row]>>(j%32))&1)?q:-q;}
      float sum=fmaf(float(dot),sx[first+t],0);float want=fmaf(.75f,sum,.25f*initial[t*N+row]);if(hy[t*N+row]!=want){fprintf(stderr,"output mismatch first%d R%d t%d row%d\n",first,r,t,row);return 3;}checked++;}
  }
  batch_map::Plan p{N,K,1,1,1,128};if(batch_map::project(p,signs,ws,counts,c,y,nullptr,1,0,0)!=cudaErrorInvalidValue)return 3;
  auto badc=c;badc.rows=0;if(batch_map::produce(x,badc,0)!=cudaErrorInvalidValue)return 3;
  hx[0]=std::numeric_limits<float>::infinity();for(int j=0;j<K;j++)hx[K+j]=1e-35f;
  CK(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));CK(batch_map::produce(x,c,0));p={N,K,2,0,1,128};CK(batch_map::project(p,signs,ws,counts,c,y,nullptr,1,0,0));
  std::vector<unsigned> status(R);unsigned error;CK(cudaMemcpy(status.data(),meta.status,R*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&error,meta.error,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
  if(status[0]!=1||status[1]!=2||error!=2)return 3;for(int i=0;i<2*N;i++)if(!std::isnan(hy[i]))return 3;
  for(float& v:hx)v=0;CK(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));CK(batch_map::produce(x,c,0));CK(cudaMemcpy(&error,meta.error,4,cudaMemcpyDeviceToHost));if(error!=2)return 3;
  printf("R512 fused producer; every tileR1..16 at first0,16,496: %d outputs bit-exact, originalscale+alpha/beta; zero valid; nonfinite/scale-range statuses+NaN; persistenterror; oddoffset/emptycarrier rejected\n",checked);return 0;
}
