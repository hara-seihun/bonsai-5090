#include "packed1.cuh"
#include <algorithm>
namespace packed1 {
size_t sign_bytes(int n, int k) { return size_t(n)*(k/128)*16; }
size_t scale_bytes(int n, int k) { return size_t(n)*(k/128)*2; }
size_t table_bytes(const Plan& p) { return p.route == Route::Lut4 ? size_t(p.r)*p.k*16 : 0; }
static int splits(const Plan& p) { return (p.k/128+p.groups_per_split-1)/p.groups_per_split; }
size_t partial_bytes(const Plan& p) { return splits(p)>1 ? size_t(p.r)*p.n*splits(p)*4 : 0; }
cudaError_t validate(const Plan& p) {
  if(p.n<=0 || p.k<=0 || p.k%128 || p.r<1 || p.r>8 || p.groups_per_split<=0 ||
     (p.threads!=64 && p.threads!=128 && p.threads!=256) ||
     (p.route!=Route::Lut4 && p.route!=Route::Sign && p.route!=Route::WarpSign)) return cudaErrorInvalidValue;
  return cudaSuccess;
}
cudaError_t select(int n,int k,int r,Plan* out) {
  if(!out || r<1 || r>8) return cudaErrorInvalidValue;
  int chunk,threads;Route route;
  if(n==96 && k==5120) {
    constexpr int chunks[8]={2,2,2,1,4,2,4,4};
    constexpr int widths[8]={256,64,64,256,64,128,128,256};
    route=Route::WarpSign;chunk=chunks[r-1];threads=widths[r-1];
  } else if(n==5120 && k==5120) {
    route=r==1?Route::Sign:Route::Lut4;chunk=r==8?2:1;threads=r==1?64:r==3?128:256;
  } else if(n==5120 && k==6144) {
    route=Route::Lut4;chunk=2;threads=r==1?64:256;
  } else if(n==5120 && k==17408) {
    route=Route::Lut4;chunk=(r==1||r==3||r==4)?4:2;threads=r==1?128:256;
  } else if((n==14336||n==16384||n==17408) && k==5120) {
    route=Route::Lut4;chunk=r<=2?1:2;threads=r==1?(n==16384?64:256):r==2?128:256;
  } else if(n==34816 && k==5120) {
    route=Route::Lut4;chunk=2;threads=r==1?64:256;
  } else if(n==248320 && k==5120) {
    route=Route::Lut4;chunk=r==1?40:r==3?8:16;threads=256;
  } else return cudaErrorInvalidValue;
  *out=Plan{n,k,r,chunk,threads,route};return cudaSuccess;
}
__global__ void transpose(const uint8_t* raw, uint32_t* signs, __half* scales, int n, int groups) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(n)*groups) return;
  int row=i%n, group=i/n;
  const uint8_t* b=raw+(size_t(row)*groups+group)*18;
  scales[i]=*reinterpret_cast<const __half*>(b);
  #pragma unroll
  for(int w=0;w<4;w++) {
    const uint8_t* q=b+2+w*4;
    signs[(size_t(group)*4+w)*n+row]=uint32_t(q[0])|(uint32_t(q[1])<<8)|(uint32_t(q[2])<<16)|(uint32_t(q[3])<<24);
  }
}
cudaError_t prepare(const uint8_t* raw,uint32_t* signs,__half* scales,int n,int k,cudaStream_t st) {
  if(!raw || !signs || !scales || n<=0 || k<=0 || k%128) return cudaErrorInvalidValue;
  transpose<<<(size_t(n)*(k/128)+255)/256,256,0,st>>>(raw,signs,scales,n,k/128);
  return cudaGetLastError();
}
__global__ void produce(const float* x,float* table,int k,int r) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(r)*(k/4)*16) return;
  int pattern=i%16; size_t quartet=i/16;
  float4 a=reinterpret_cast<const float4*>(x)[quartet];
  float v=((pattern&1)?a.x:-a.x)+((pattern&2)?a.y:-a.y);
  v+=((pattern&4)?a.z:-a.z); v+=((pattern&8)?a.w:-a.w);
  table[i]=v;
}
__global__ void projection(const uint32_t* __restrict__ signs,const __half* __restrict__ scales,
                const float* __restrict__ x,const float* __restrict__ tables,float* y,float* partial,
                int n,int groups,int chunk,int ns,float alpha,float beta) {
  extern __shared__ float shared[];
  int row=blockIdx.x*blockDim.x+threadIdx.x, split=blockIdx.y, token=blockIdx.z;
  int first=split*chunk,last=min(first+chunk,groups);
  float sum=0;
  for(int g=first;g<last;g++) {
    if(threadIdx.x<128) shared[threadIdx.x]=x[(size_t(token)*groups+g)*128+threadIdx.x];
    if(blockDim.x==64) shared[threadIdx.x+64]=x[(size_t(token)*groups+g)*128+threadIdx.x+64];
    __syncthreads();
    if(row<n) {
      float dot=0;
      #pragma unroll
      for(int w=0;w<4;w++) {
        uint32_t bits=signs[(size_t(g)*4+w)*n+row];
        #pragma unroll
        for(int j=0;j<32;j++) {
          unsigned a=__float_as_uint(shared[w*32+j]);
          dot+=__uint_as_float(a^(((~bits>>j)&1u)<<31));
        }
      }
      sum=fmaf(dot,__half2float(scales[size_t(g)*n+row]),sum);
    }
    __syncthreads();
  }
  if(row<n) {
    if(ns>1) partial[(size_t(token)*ns+split)*n+row]=sum;
    else { size_t i=size_t(token)*n+row; y[i]=beta==0 ? alpha*sum : fmaf(alpha,sum,beta*y[i]); }
  }
}
template<int R> __global__ void lut_multi(const uint32_t* __restrict__ signs,const __half* __restrict__ scales,
                const float* __restrict__ tables,float* y,float* partial,
                int n,int groups,int chunk,int ns,float alpha,float beta) {
  extern __shared__ float shared[];
  int row=blockIdx.x*blockDim.x+threadIdx.x,split=blockIdx.y;
  int first=split*chunk,last=min(first+chunk,groups);float sum[R]={};
  for(int g=first;g<last;g++) {
    for(int t=threadIdx.x;t<R*512;t+=blockDim.x)shared[t]=tables[(size_t(t/512)*groups+g)*512+t%512];
    __syncthreads();
    if(row<n) {
      float dot[R]={};
      #pragma unroll
      for(int w=0;w<4;w++) {
        unsigned bits=signs[(size_t(g)*4+w)*n+row];
        #pragma unroll
        for(int j=0;j<8;j++) {
          int address=(w*8+j)*16+((bits>>(4*j))&15);
          #pragma unroll
          for(int t=0;t<R;t++)dot[t]+=shared[t*512+address];
        }
      }
      float scale=__half2float(scales[size_t(g)*n+row]);
      #pragma unroll
      for(int t=0;t<R;t++)sum[t]=fmaf(dot[t],scale,sum[t]);
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
template<int R> static void multi_launch(const Plan& p,const uint32_t* signs,const __half* scales,const float* tables,float* y,float* partial,float alpha,float beta,cudaStream_t st) {
  lut_multi<R><<<dim3((p.n+p.threads-1)/p.threads,splits(p)),p.threads,R*2048,st>>>(signs,scales,tables,y,partial,p.n,p.k/128,p.groups_per_split,splits(p),alpha,beta);
}
__global__ void warp_projection(const uint32_t* __restrict__ signs,const __half* __restrict__ scales,
                const float* __restrict__ x,float* y,float* partial,
                int n,int groups,int chunk,int ns,float alpha,float beta) {
  int lane=threadIdx.x%32, row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
  int split=blockIdx.y,token=blockIdx.z;
  if(row>=n)return;
  int first=split*chunk,last=min(first+chunk,groups);float sum=0;
  for(int g=first;g<last;g++) {
    unsigned bits=signs[(size_t(g)*4+lane/8)*n+row]>>((lane%8)*4);
    float4 a=reinterpret_cast<const float4*>(x)[(size_t(token)*groups+g)*32+lane];
    float dot=((bits&1)?a.x:-a.x)+((bits&2)?a.y:-a.y);
    dot+=((bits&4)?a.z:-a.z);dot+=((bits&8)?a.w:-a.w);
    sum=fmaf(dot,__half2float(scales[size_t(g)*n+row]),sum);
  }
  for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
  if(lane==0) {
    if(ns>1)partial[(size_t(token)*ns+split)*n+row]=sum;
    else {size_t i=size_t(token)*n+row;y[i]=beta==0?alpha*sum:fmaf(alpha,sum,beta*y[i]);}
  }
}
__global__ void continuation(const float* partial,float* y,int n,int ns,int r,float alpha,float beta) {
  int row=blockIdx.x*blockDim.x+threadIdx.x,token=blockIdx.y;
  if(row>=n) return;
  float sum=0;
  for(int s=0;s<ns;s++) sum+=partial[(size_t(token)*ns+s)*n+row];
  size_t i=size_t(token)*n+row;
  y[i]=beta==0 ? alpha*sum : fmaf(alpha,sum,beta*y[i]);
}
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* scales,const float* x,float* y,
                    float* tables,float* partial,float alpha,float beta,cudaStream_t st) {
  cudaError_t e=validate(p); if(e!=cudaSuccess) return e;
  if(!signs || !scales || !x || !y || (p.route==Route::Lut4&&!tables) || (splits(p)>1&&!partial)) return cudaErrorInvalidValue;
  int ns=splits(p), groups=p.k/128;
  if(p.route==Route::Lut4) {
    produce<<<(size_t(p.r)*(p.k/4)*16+255)/256,256,0,st>>>(x,tables,p.k,p.r);
    e=cudaGetLastError(); if(e!=cudaSuccess) return e;
    switch(p.r) {
      case 1: multi_launch<1>(p,signs,scales,tables,y,partial,alpha,beta,st); break;
      case 2: multi_launch<2>(p,signs,scales,tables,y,partial,alpha,beta,st); break;
      case 3: multi_launch<3>(p,signs,scales,tables,y,partial,alpha,beta,st); break;
      case 4: multi_launch<4>(p,signs,scales,tables,y,partial,alpha,beta,st); break;
      case 5: multi_launch<5>(p,signs,scales,tables,y,partial,alpha,beta,st); break;
      case 6: multi_launch<6>(p,signs,scales,tables,y,partial,alpha,beta,st); break;
      case 7: multi_launch<7>(p,signs,scales,tables,y,partial,alpha,beta,st); break;
      case 8: multi_launch<8>(p,signs,scales,tables,y,partial,alpha,beta,st); break;
    }
  } else if(p.route==Route::WarpSign) {
    warp_projection<<<dim3((p.n+p.threads/32-1)/(p.threads/32),ns,p.r),p.threads,0,st>>>(signs,scales,x,y,partial,p.n,groups,p.groups_per_split,ns,alpha,beta);
  } else {
    projection<<<dim3((p.n+p.threads-1)/p.threads,ns,p.r),p.threads,512,st>>>(signs,scales,x,tables,y,partial,p.n,groups,p.groups_per_split,ns,alpha,beta);
  }
  e=cudaGetLastError(); if(e!=cudaSuccess) return e;
  if(ns>1) { continuation<<<dim3((p.n+255)/256,p.r),256,0,st>>>(partial,y,p.n,ns,p.r,alpha,beta); e=cudaGetLastError(); }
  return e;
}
}
