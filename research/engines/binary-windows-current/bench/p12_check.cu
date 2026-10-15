#include "bmega_observers.cuh"
#include "joint_five.cuh"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
template<class T> T* upload(const std::vector<T>& h){T* d=nullptr;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template<class T>std::vector<T> download(T* d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static uint32_t codes(int q){unsigned a=unsigned(q)&3,b=(unsigned(q)>>2)&3;int c=q>>4;unsigned cm=c<0?-c:c,bcode=b;if(cm==5||cm==7){int delta=(b&1)?1:-1;c+=delta;bcode=(0x95b4u>>(4*b))&15;}cm=c<0?-c:c;unsigned cc=(cm<=4?cm:cm/2+2)|(c<0?8u:0u);return a|(bcode<<8)|(cc<<16);}
static int digit(int code){static const int values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};return values[code];}
__global__ void observer(const uint8_t* ch,const uint2* act,const float* scale,float* out,int route){int lane=threadIdx.x;float a,b;bmega::chunk_dot(ch,act,scale,lane,a,b,route);if(!(lane&3)){out[lane>>2]=a;out[(lane>>2)+8]=b;}}
int main(){try{
 CK(p12_joint_five::initialize());size_t failures=0,encoder_checks=0,observer_checks=0,projection_checks=0;
 {
  int k=65536;std::vector<int8_t> q(2*k);for(int i=0;i<k;i++){q[i]=int8_t((i&255)-128);q[k+i]=int8_t((i>>8)-128);}auto* dq=upload(q);std::vector<uint8_t> bytes(p12_joint_five::bytes(k,2));auto* d=upload(bytes);bitserial::Carrier c{};c.q=dq;
  CK(p12_joint_five::encode(c,k,2,d,bytes.size(),0));auto h=download(d,bytes.size());
  for(int i=0;i<k;i++){int v[5];for(int p=0;p<5;p++)v[p]=digit((h[size_t(p)*(k/2)+i/2]>>((i&1)*4))&15);failures+=v[0]+11*v[1]+11*v[4]!=q[i];failures+=15*v[2]+19*v[3]+v[4]!=q[k+i];encoder_checks++;}cudaFree(dq);cudaFree(d);
 }
 for(int trial=0;trial<16;trial++){
  std::vector<uint32_t> ch(2304/4,0);std::vector<uint2> act(16*12);std::vector<float> scales(8);std::vector<int> signs(16*1024),q(1024);std::vector<float> ws(16*8),expected(16);
  for(int k=0;k<1024;k++)q[k]=trial==0?-128:trial==1?127:int((uint32_t(k*2053+trial*721)*2246822519u)>>24)-128;
  for(int r=0;r<16;r++)for(int k=0;k<1024;k++)signs[r*1024+k]=((uint32_t(r*3559+k*1217+trial*17)*2654435761u)>>31)?1:-1;
  for(int b=0;b<8;b++){scales[b]=float((b+trial)%11+1)/127.f;for(int r=0;r<16;r++)ws[r*8+b]=std::ldexp(1.f+float((r+b)%7)/8.f,-3);for(int g=0;g<8;g++){__half2 h=__floats2half2_rn(ws[g*8+b],ws[(g+8)*8+b]);std::memcpy(&ch[512+b*8+g],&h,4);}}
  for(int s=0;s<16;s++)for(int lane=0;lane<32;lane++){int g=lane>>2,t=lane&3;uint32_t w=0;for(int r=0;r<4;r++)for(int j=0;j<8;j++){int row=g+8*(r&1),k=s*64+(r>>1)*32+t*8+j;if(signs[row*1024+k]<0)w|=1u<<(4*j+3-r);}ch[s*32+lane]=w;}
  for(int s=0;s<16;s++)for(int t=0;t<4;t++)for(int p=0;p<3;p++){uint2 a{};for(int j=0;j<8;j++){a.x|=((codes(q[s*64+t*8+j])>>(p*8))&15)<<(4*j);a.y|=((codes(q[s*64+32+t*8+j])>>(p*8))&15)<<(4*j);}act[s*12+p*4+t]=a;}
  for(int r=0;r<16;r++){float acc=0;for(int b=0;b<8;b++){int dot=0;for(int j=0;j<128;j++)dot+=signs[r*1024+b*128+j]*q[b*128+j];float scale=ws[r*8+b]*scales[b];acc=std::fma(float(dot),scale,acc);}expected[r]=acc;}
  auto* dc=upload(ch);auto* da=upload(act);auto* ds=upload(scales);auto* out=upload(expected);
  for(int route=0;route<4;route++){observer<<<1,32>>>(reinterpret_cast<uint8_t*>(dc),da,ds,out,route);CK(cudaGetLastError());auto actual=download(out,16);for(int r=0;r<16;r++){failures+=std::memcmp(&actual[r],&expected[r],4)!=0;observer_checks++;}}
  cudaFree(dc);cudaFree(da);cudaFree(ds);cudaFree(out);
 }
 for(int trial=0;trial<8;trial++)for(int rows:{16,17,32}){
  int n=32,k=1024,blocks=k/128;std::vector<int8_t> q(rows*k);std::vector<uint32_t> w(blocks*4*n);std::vector<__half> ws(blocks*n);std::vector<float> scales(rows*blocks),expected(rows*n);std::vector<bitserial::InputStatus> status(rows*blocks,bitserial::InputStatus::Valid);
  for(int token=0;token<rows;token++)for(int j=0;j<k;j++)q[token*k+j]=trial==0?-128:trial==1?127:int((uint32_t(j*2053+token*1237+trial*721)*2246822519u)>>24)-128;
  for(int b=0;b<blocks;b++)for(int r=0;r<n;r++){ws[b*n+r]=__float2half(std::ldexp(1.f+float((r+b)%7)/8.f,-3));for(int strip=0;strip<4;strip++){uint32_t bits=0;for(int j=0;j<32;j++)if(((uint32_t(r*3559+(b*128+strip*32+j)*1217+trial*17)*2654435761u)>>31))bits|=1u<<j;w[(b*4+strip)*n+r]=bits;}}
  for(int token=0;token<rows;token++)for(int b=0;b<blocks;b++)scales[token*blocks+b]=float((b+token+trial)%11+1)/127.f;
  for(int token=0;token<rows;token++)for(int r=0;r<n;r++){float acc=0;for(int b=0;b<blocks;b++){int dot=0;for(int j=0;j<128;j++){int sign=((w[(b*4+j/32)*n+r]>>(j%32))&1)?1:-1;dot+=sign*q[token*k+b*128+j];}float scale=__half2float(ws[b*n+r])*scales[token*blocks+b];acc=std::fma(float(dot),scale,acc);}expected[token*n+r]=acc;}
  auto* dq=upload(q);auto* dw=upload(w);auto* dws=upload(ws);auto* ds=upload(scales);auto* dst=upload(status);auto* out=upload(expected);std::vector<uint8_t> scratch(p12_joint_five::bytes(k,rows));auto* dd=upload(scratch);bitserial::Carrier c{};c.q=dq;c.scales=ds;c.status=dst;
  CK(p12_joint_five::encode(c,k,rows,dd,scratch.size(),0));CK(p12_joint_five::project(reinterpret_cast<uint8_t*>(dw),dws,c,dd,n,k,rows,out,n,1,0,0));auto actual=download(out,expected.size());for(size_t i=0;i<expected.size();i++){failures+=std::memcmp(&actual[i],&expected[i],4)!=0;projection_checks++;}
  cudaFree(dq);cudaFree(dw);cudaFree(dws);cudaFree(ds);cudaFree(dst);cudaFree(out);cudaFree(dd);
 }
 printf("{\"encoder_pairs\":%zu,\"observer_outputs\":%zu,\"projection_outputs\":%zu,\"bit_mismatches\":%zu}\n",encoder_checks,observer_checks,projection_checks,failures);return failures?1:0;
}catch(const std::exception& e){fprintf(stderr,"P12 native check: %s\n",e.what());return 2;}}
