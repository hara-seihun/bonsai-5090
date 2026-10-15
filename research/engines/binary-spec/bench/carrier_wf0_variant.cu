#define H5_WARP_MICROSCOPE_STANDALONE
#include "../src/fp4.cu"
#ifndef CARRIER_WF
#error CARRIER_WF must explicitly select 0 or 3
#endif
#ifndef CARRIER_ENTRY
#error CARRIER_ENTRY must name the immutable component
#endif
namespace h5 {
template<int FW,int WG,int TOK,int PIPE,int MODE>
cudaError_t carrier_native(int split,const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const void*W,int N,float*Y,int ldo,const float*mean,cudaStream_t st,cudaFuncAttributes*attrs,int*blocks,int*bytes){
 using G=WsGeom<FW,WG,TOK,CARRIER_WF,PIPE>;
 auto f=k_gemm_fp4<FW,WG,TOK,MODE,CARRIER_WF,PIPE,false,true>;
 cudaError_t e=cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,G::SMEM);if(e!=cudaSuccess)return e;
 if(attrs){*bytes=G::SMEM;e=cudaFuncGetAttributes(attrs,f);if(e!=cudaSuccess)return e;return cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks,f,(FW+1)*32,G::SMEM);}
 pdl(f,dim3((R+TOK-1)/TOK,(N/32+FW*WG-1)/(FW*WG),split),dim3((FW+1)*32),G::SMEM,st,A,AS,AG,R,K,(const uint8_t*)W,(const uint8_t*)nullptr,N,Y,ldo,mean);return cudaGetLastError();
}
template<int FW,int WG,int TOK,int PIPE>
cudaError_t carrier_mode(int mode,int split,const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const void*W,int N,float*Y,int ldo,const float*mean,cudaStream_t st,cudaFuncAttributes*attrs,int*blocks,int*bytes){
#define GO(m) return carrier_native<FW,WG,TOK,PIPE,m>(split,A,AS,AG,R,K,W,N,Y,ldo,mean,st,attrs,blocks,bytes)
 switch(mode){case 0:GO(0);case 1:GO(1);case 2:GO(2);case 3:if constexpr(WG==2){GO(3);}else return cudaErrorInvalidValue;default:return cudaErrorInvalidValue;}
#undef GO
}
}
extern "C" cudaError_t CARRIER_ENTRY(int cfg,int mode,int split,const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const void*W,int N,float*Y,int ldo,const float*mean,cudaStream_t st,cudaFuncAttributes*attrs,int*blocks,int*bytes){
 if(R<=0||K<=0||K%128||N<=0||N%32||split<1||split>K/128||mode<0||mode>3||(mode==3&&(N%64||split!=1))||(split>1&&mode!=2)||(split==1&&mode==2)||ldo<(mode==3?N/2:N)||(attrs&&(!blocks||!bytes)))return cudaErrorInvalidValue;
#define GO(f,w,t,p) return h5::carrier_mode<f,w,t,p>(mode,split,A,AS,AG,R,K,W,N,Y,ldo,mean,st,attrs,blocks,bytes)
 switch(cfg){case 12:GO(8,1,16,0);case 19:GO(4,2,8,0);case 23:GO(4,2,8,4);case 26:GO(8,1,8,4);case 29:GO(8,2,8,2);default:return cudaErrorInvalidValue;}
#undef GO
}
