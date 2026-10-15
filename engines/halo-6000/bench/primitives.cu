#include "device.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace halo6000;
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}
template<class T>static T*upload(const std::vector<T>&x){T*p;ck(cudaMalloc(&p,x.size()*sizeof(T)));ck(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
__global__ void prep(const float*x,ActivationBuffer a){quantize_phase(x,a);}
__global__ void mv(MatrixView w,ActivationView a,const float*x,float*y){matvec_phase(w,a,x,y);}
__global__ void norm(const float*x,const float*g,float*y){__shared__ float s[32];norm_phase<2048>(x,g,y,s);}
__global__ void chain(MatrixView w,ActivationBuffer a,const float*x,const float*g,float*y,float*z){
 __shared__ float s[32];auto grid=cooperative_groups::this_grid();
 if(blockIdx.x==0)norm_phase<2048>(x,g,y,s);
 grid.sync();quantize_phase(y,a);grid.sync();matvec_phase(w,a.view(),y,z);
}
static int trit(int row,int i){return ((row*17+i*7+i/16)%3)-1;}
static std::vector<uint8_t>pane(int n,int k){
 const size_t groups=size_t(n)*k/128;std::vector<uint8_t>v(((groups+15)/16)*544);
 for(int row=0;row<n;row++)for(int b=0;b<k/128;b++){
  size_t g=size_t(row)*(k/128)+b;
  for(int w=0;w<8;w++){
   uint32_t word=0;
   for(int e=0;e<16;e++){int t=trit(row,b*128+w*16+e);unsigned c=t<0?5:t>0?1:0;word|=c<<(4*(e&7)+(e>>3));}
   memcpy(v.data()+(g/16)*544+(g%16)*32+w*4,&word,4);
  }
  uint16_t h=uint16_t(0x2400-((b+row)&1)*0x400);memcpy(v.data()+(g/16)*544+512+(g%16)*2,&h,2);
 }
 return v;
}
int main(){
 constexpr int k=2048,n=512,threads=256,blocks=32;
 std::vector<float>x(k),gain(k,1);for(int i=0;i<k;i++)x[i]=float((i*73)%255-127)/32.f;
 float*dx=upload(x),*dg=upload(gain),*dy,*dz,*dcontrol;ck(cudaMalloc(&dy,k*4));ck(cudaMalloc(&dz,n*4));ck(cudaMalloc(&dcontrol,n*4));
 ActivationBuffer a{};a.k=k;a.rows=1;a.group=32;a.q_stride=k;a.scale_stride=k/32;ck(cudaMalloc(&a.qs,k));ck(cudaMalloc(&a.scale,k/32*4));
 prep<<<blocks,threads>>>(dx,a);ck(cudaGetLastError());std::vector<int8_t>q(k);std::vector<float>d(k/32);ck(cudaMemcpy(q.data(),a.qs,k,cudaMemcpyDeviceToHost));ck(cudaMemcpy(d.data(),a.scale,k/32*4,cudaMemcpyDeviceToHost));
 int quant_errors=0;for(int b=0;b<k/32;b++){float am=0;for(int t=0;t<32;t++)am=std::max(am,std::abs(x[b*32+t]));float ds=am/127;for(int t=0;t<32;t++)quant_errors+=q[b*32+t]!=(am==0?0:int(std::round(x[b*32+t]/ds)));}
 auto packed=pane(n,k);auto*dp=upload(packed);MatrixView w{};w.data=dp;w.format=WeightFormat::Pane2;w.n=n;w.k=k;w.bytes=packed.size();w.scale_group=128;
 mv<<<blocks,threads>>>(w,a.view(),dx,dz);ck(cudaGetLastError());std::vector<float>got(n);ck(cudaMemcpy(got.data(),dz,n*4,cudaMemcpyDeviceToHost));
 double maxerr=0;for(int row=0;row<n;row++){
  double ref=0;for(int b=0;b<k/128;b++){float ws=((row+b)&1)?1.f/128:1.f/64;for(int c=0;c<4;c++){int sum=0;for(int t=0;t<32;t++)sum+=trit(row,b*128+c*32+t)*int(q[b*128+c*32+t]);ref+=double(ws)*d[b*4+c]*sum;}}
  maxerr=std::max(maxerr,std::abs(double(got[row])-ref));
 }
 norm<<<1,threads>>>(dx,dg,dy);prep<<<blocks,threads>>>(dy,a);mv<<<blocks,threads>>>(w,a.view(),dy,dcontrol);ck(cudaGetLastError());
 MatrixView wc=w;void*args[]={&wc,&a,&dx,&dg,&dy,&dz};int occupancy=0;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,chain,threads,0));ck(cudaLaunchCooperativeKernel((void*)chain,blocks,threads,args));
 std::vector<float>control(n);ck(cudaMemcpy(control.data(),dcontrol,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(got.data(),dz,n*4,cudaMemcpyDeviceToHost));int chain_errors=0;for(int i=0;i<n;i++)chain_errors+=memcmp(&control[i],&got[i],4)!=0;
 std::vector<float>normalized(k);ck(cudaMemcpy(normalized.data(),dy,k*4,cudaMemcpyDeviceToHost));double normsum=0;for(float v:x)normsum+=double(v)*v;double inv=1/std::sqrt(normsum/k+NormEpsilon),normerr=0;for(int i=0;i<k;i++)normerr=std::max(normerr,std::abs(normalized[i]-x[i]*inv));
 printf("{\"quant_byte_errors\":%d,\"pane_max_abs_vs_double\":%.9g,\"norm_max_abs_vs_double\":%.9g,\"persistent_vs_control_bit_errors\":%d,\"persistent_blocks_per_SM\":%d,\"note\":\"synthetic operand and publication guard; NOT full model logits or timing\"}\n",quant_errors,maxerr,normerr,chain_errors,occupancy);
 for(void*p:{(void*)dx,(void*)dg,(void*)dy,(void*)dz,(void*)dcontrol,(void*)a.qs,(void*)a.scale,(void*)dp})ck(cudaFree(p));
 return quant_errors||chain_errors||maxerr>1e-4||normerr>1e-5?1:0;
}
