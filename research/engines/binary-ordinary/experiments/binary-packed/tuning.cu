#include "tuning.cuh"
namespace packed1::tuning {
static int splits(const Plan& p) { return (p.k/128+p.groups_per_split-1)/p.groups_per_split; }
cudaError_t validate(const Plan& p) {
  if(p.n<=0 || p.k<=0 || p.k%128 || p.r<1 || p.r>8 || p.groups_per_split<=0 ||
     (p.threads!=64 && p.threads!=128 && p.threads!=256) ||
     (p.route!=Route::Lut8 && p.route!=Route::Lut4Ilp))return cudaErrorInvalidValue;
  return cudaSuccess;
}
size_t table_bytes(const Plan& p) { return size_t(p.r)*p.k*(p.route==Route::Lut8?64:16); }
size_t partial_bytes(const Plan& p) { return splits(p)>1?size_t(p.r)*p.n*splits(p)*4:0; }
template<bool EIGHT> __global__ void producer(const float* x,float* table,int k,int r) {
  constexpr int patterns=EIGHT?128:16, width=EIGHT?8:4;
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(r)*(k/width)*patterns)return;
  int pattern=i%patterns;size_t first=(i/patterns)*width;
  float v=((pattern&1)?x[first]:-x[first])+((pattern&2)?x[first+1]:-x[first+1]);
  v+=((pattern&4)?x[first+2]:-x[first+2]);v+=((pattern&8)?x[first+3]:-x[first+3]);
  if constexpr(EIGHT) {
    float u=((pattern&16)?x[first+4]:-x[first+4])+((pattern&32)?x[first+5]:-x[first+5]);
    u+=((pattern&64)?x[first+6]:-x[first+6]);u-=x[first+7];v+=u;
  }
  table[i]=v;
}
template<int R,bool EIGHT> __global__ void projection(const uint32_t* __restrict__ signs,const __half* __restrict__ scales,
                     const float* __restrict__ table,float* y,float* partial,int n,int groups,int chunk,int ns,float alpha,float beta) {
  constexpr int entries=EIGHT?2048:512, patterns=EIGHT?128:16, parts=EIGHT?4:8;
  extern __shared__ float shared[];
  int row=blockIdx.x*blockDim.x+threadIdx.x,split=blockIdx.y;
  int first=split*chunk,last=min(first+chunk,groups);float sum[R]={};
  for(int g=first;g<last;g++) {
    for(int i=threadIdx.x;i<R*entries;i+=blockDim.x)shared[i]=table[(size_t(i/entries)*groups+g)*entries+i%entries];
    __syncthreads();
    if(row<n) {
      float dot[R][4]={};
      #pragma unroll
      for(int w=0;w<4;w++) {
        unsigned word=signs[(size_t(g)*4+w)*n+row];
        #pragma unroll
        for(int j=0;j<parts;j++) {
          unsigned bits=EIGHT?((word>>(j*8))&255):((word>>(j*4))&15);
          unsigned complement=EIGHT?(0u-(bits>>7)):0;
          unsigned address=(w*parts+j)*patterns+((bits^complement)&(patterns-1));
          #pragma unroll
          for(int t=0;t<R;t++) {
            float value=shared[t*entries+address];
            if constexpr(EIGHT)value=__uint_as_float(__float_as_uint(value)^(complement&0x80000000u));
            dot[t][w]+=value;
          }
        }
      }
      float scale=__half2float(scales[size_t(g)*n+row]);
      #pragma unroll
      for(int t=0;t<R;t++)sum[t]=fmaf((dot[t][0]+dot[t][1])+(dot[t][2]+dot[t][3]),scale,sum[t]);
    }
    __syncthreads();
  }
  if(row<n) {
    #pragma unroll
    for(int t=0;t<R;t++) {
      if(ns>1)partial[(size_t(t)*ns+split)*n+row]=sum[t];
      else {size_t i=size_t(t)*n+row;y[i]=beta==0?alpha*sum[t]:fmaf(alpha,sum[t],beta*y[i]);}
    }
  }
}
__global__ void continuation(const float* partial,float* y,int n,int ns,float alpha,float beta) {
  int row=blockIdx.x*blockDim.x+threadIdx.x,token=blockIdx.y;if(row>=n)return;
  float sum=0;for(int s=0;s<ns;s++)sum+=partial[(size_t(token)*ns+s)*n+row];
  size_t i=size_t(token)*n+row;y[i]=beta==0?alpha*sum:fmaf(alpha,sum,beta*y[i]);
}
cudaError_t configure() {
  cudaError_t e=cudaFuncSetAttribute(projection<7,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
  if(e!=cudaSuccess)return e;
  return cudaFuncSetAttribute(projection<8,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
}
template<int R> static void launch(const Plan& p,const uint32_t* signs,const __half* scales,const float* table,float* y,float* partial,float alpha,float beta,cudaStream_t st) {
  dim3 grid((p.n+p.threads-1)/p.threads,splits(p));
  if(p.route==Route::Lut8)projection<R,true><<<grid,p.threads,R*8192,st>>>(signs,scales,table,y,partial,p.n,p.k/128,p.groups_per_split,splits(p),alpha,beta);
  else projection<R,false><<<grid,p.threads,R*2048,st>>>(signs,scales,table,y,partial,p.n,p.k/128,p.groups_per_split,splits(p),alpha,beta);
}
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* scales,const float* x,float* y,float* table,float* partial,float alpha,float beta,cudaStream_t st) {
  cudaError_t e=validate(p);if(e!=cudaSuccess)return e;
  if(!signs||!scales||!x||!y||!table||(splits(p)>1&&!partial))return cudaErrorInvalidValue;
  if(p.route==Route::Lut8)producer<true><<<(size_t(p.r)*(p.k/8)*128+255)/256,256,0,st>>>(x,table,p.k,p.r);
  else producer<false><<<(size_t(p.r)*(p.k/4)*16+255)/256,256,0,st>>>(x,table,p.k,p.r);
  e=cudaGetLastError();if(e!=cudaSuccess)return e;
  switch(p.r) {
    case 1: launch<1>(p,signs,scales,table,y,partial,alpha,beta,st);break;
    case 2: launch<2>(p,signs,scales,table,y,partial,alpha,beta,st);break;
    case 3: launch<3>(p,signs,scales,table,y,partial,alpha,beta,st);break;
    case 4: launch<4>(p,signs,scales,table,y,partial,alpha,beta,st);break;
    case 5: launch<5>(p,signs,scales,table,y,partial,alpha,beta,st);break;
    case 6: launch<6>(p,signs,scales,table,y,partial,alpha,beta,st);break;
    case 7: launch<7>(p,signs,scales,table,y,partial,alpha,beta,st);break;
    case 8: launch<8>(p,signs,scales,table,y,partial,alpha,beta,st);break;
  }
  e=cudaGetLastError();if(e!=cudaSuccess)return e;
  if(splits(p)>1) {continuation<<<dim3((p.n+255)/256,p.r),256,0,st>>>(partial,y,p.n,splits(p),alpha,beta);e=cudaGetLastError();}
  return e;
}
}
