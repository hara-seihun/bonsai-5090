#include "dict8.cuh"
#include "kernels.cuh"
namespace h5 {
__global__ void dict8_pair_check(const float* a,const float* b,const float* x,const float* y,uint32_t* s) {
 pdl_wait();
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<64*34816) { atomicAdd(s,1u);if(__float_as_uint(a[i])!=__float_as_uint(b[i]))atomicAdd(s+1,1u);if(!isfinite(a[i])||!isfinite(b[i]))atomicAdd(s+10,1u); }
 if(i<64*17408) { atomicAdd(s+2,1u);if(__float_as_uint(x[i])!=__float_as_uint(y[i]))atomicAdd(s+3,1u);if(!isfinite(x[i])||!isfinite(y[i]))atomicAdd(s+11,1u); }
}
__global__ void dict8_quant_check(const uint8_t* q,const uint8_t* qr,const uint8_t* b,const uint8_t* br,const float* g,const float* gr,uint32_t* s) {
 pdl_wait();
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<64*17408/2) { atomicAdd(s+4,1u);if(q[i]!=qr[i])atomicAdd(s+5,1u); }
 if(i<64*17408/16) { atomicAdd(s+6,1u);if(b[i]!=br[i])atomicAdd(s+7,1u); }
 if(i<64) { atomicAdd(s+8,1u);if(__float_as_uint(g[i])!=__float_as_uint(gr[i]))atomicAdd(s+9,1u); }
}
cudaError_t compare_dict8_pairs(const float* a,const float* b,const float* x,const float* y,uint32_t* s,cudaStream_t st) {
 if(!a||!b||!x||!y||!s)return cudaErrorInvalidValue;
 dict8_pair_check<<<(64*34816+255)/256,256,0,st>>>(a,b,x,y,s);
 return cudaGetLastError();
}
cudaError_t compare_dict8_quant(const uint8_t* q,const uint8_t* qr,const uint8_t* b,const uint8_t* br,const float* g,const float* gr,uint32_t* s,cudaStream_t st) {
 if(!q||!qr||!b||!br||!g||!gr||!s)return cudaErrorInvalidValue;
 dict8_quant_check<<<(64*17408/2+255)/256,256,0,st>>>(q,qr,b,br,g,gr,s);
 return cudaGetLastError();
}
}
