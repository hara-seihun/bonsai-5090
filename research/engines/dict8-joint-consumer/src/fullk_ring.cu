#define H5_FULLK_RING_ONLY
#include "fp4.cu"

namespace h5 { namespace fullk_ring_backend {
template<bool MIXED> __global__ __launch_bounds__(288) void k_fullk_h1(const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means,const uint32_t*book,const float*signs){
 TailArgs ta{};ta.mul=signs;
 gemm_fp4_body<8,2,32,3,3,0,false,true,false,false,TAIL_NONE,true,MIXED,true,true>(A,AS,AG,R,K,W,nullptr,N,out,ldo,means,FuseArgs{},ta,book);
}
template<bool MIXED> cudaError_t dispatch_fullk_h1(const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means,const uint32_t*book,const float*signs,cudaStream_t st){
 auto fn=k_fullk_h1<MIXED>;constexpr int bytes=WsGeom<8,2,32,3,0>::SMEM;auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e!=cudaSuccess)return e;
 pdl(fn,dim3((R+31)/32,(N/32+15)/16),dim3(288),bytes,st,A,AS,AG,R,K,W,N,out,ldo,means,book,signs);return cudaGetLastError();
}
}
cudaError_t launch_fullk_h1(int cfg,FullKGuMode mode,const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means,const uint32_t*book,const float*signs,cudaStream_t st){
 if(!A||!AS||!AG||!W||!out||!means||!book||!signs||R!=64||K!=D||N!=GU||ldo!=FF||cfg!=13)return cudaErrorInvalidValue;
 if(mode==FullKGuMode::H1Mixed)return fullk_ring_backend::dispatch_fullk_h1<true>(A,AS,AG,R,K,W,N,out,ldo,means,book,signs,st);
 if(mode==FullKGuMode::H1Plain)return fullk_ring_backend::dispatch_fullk_h1<false>(A,AS,AG,R,K,W,N,out,ldo,means,book,signs,st);
 return cudaErrorInvalidValue;
}
}
