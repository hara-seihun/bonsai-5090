#include "../binary-fp4-pair/pair.cu"
#include "native.cuh"
namespace binary_native_sf {
cudaError_t pair_control(int cfg,int n,int k,int r,int splits,Plane a,Plane e,const uint8_t* w,const uint16_t* s,float* y,float* partial,cudaStream_t st){
  namespace bp=binary_fp4_pair;
  bp::Plan p{bp::Schedule(cfg),n,k,r,splits};bp::Plane ap{a.q,a.sf,a.g},ep{e.q,e.sf,e.g};
  float* dst=splits==1?y:partial;cudaError_t error;
#define CASE(ID,FW,WG,TW) case ID:error=bp::launch<FW,WG,TW>(p,ap,ep,w,(const uint8_t*)s,dst,n,st,false);break
  switch(cfg){CASE(0,2,2,8);CASE(1,4,2,8);CASE(2,8,1,8);CASE(3,4,2,16);CASE(4,8,1,32);CASE(5,4,2,32);CASE(6,8,2,16);default:return cudaErrorInvalidValue;}
#undef CASE
  if(error!=cudaSuccess||splits==1)return error;
  h5::pdl(bp::fold,dim3((size_t(r)*n+255)/256),dim3(256),0,st,(const float*)partial,y,n,r,n,splits,false);return cudaGetLastError();
}
}
