#include "bitserial.cuh"
namespace bitserial {
static int splits(const Plan& p) {return (p.k/128+p.groups_per_split-1)/p.groups_per_split;}
cudaError_t validate(const Plan& p) {
  if(p.n<=0||p.k<=0||p.k%128||(p.r<1||p.r>16)||p.groups_per_split<=0||
     (p.threads!=64&&p.threads!=128&&p.threads!=256)||(p.route!=Route::Popcount&&p.route!=Route::Dp4a01))return cudaErrorInvalidValue;
  return cudaSuccess;
}
cudaError_t select(int n,int k,int r,Route route,Plan* out) {
  if(!out||n<=0||k<=0||k%128||(r<1||r>16)||(route!=Route::Popcount&&route!=Route::Dp4a01))return cudaErrorInvalidValue;
  int chunk,threads;
  if(n>=131072){chunk=route==Route::Dp4a01?(r==1||r>=12?k/128:16):16;threads=r==1&&route==Route::Popcount?256:128;}
  else if(n>=65536){chunk=route==Route::Dp4a01?(r==1||r>=12?16:4):16;threads=128;}
  else if(n<=256){chunk=1;threads=128;}
  else if(n<=5120&&k==5120){chunk=r>=12?4:1;threads=128;}
  else if(n==34816&&k==5120&&r>1&&r<=8&&route==Route::Dp4a01){chunk=16;threads=128;}
  else if(r==1&&k<=6144&&route==Route::Dp4a01){
    chunk=n==17408||n==34816?4:1;threads=n==34816||k==6144?256:128;
  }else {chunk=4;threads=r<=8&&(r>1||k>5120)?256:128;}
  *out=Plan{n,k,r,chunk,threads,route};return cudaSuccess;
}
cudaError_t select(int n,int k,int r,Plan* out){return select(n,k,r,Route::Dp4a01,out);}
size_t sign_bytes(int n,int k){return size_t(n)*(k/128)*16;}
size_t scale_bytes(int n,int k){return size_t(n)*(k/128)*2;}
size_t partial_bytes(const Plan& p){return splits(p)>1?size_t(p.r)*p.n*splits(p)*4:0;}
__global__ void transpose(const uint8_t* raw,uint32_t* signs,__half* scales,int n,int groups) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*groups)return;
  int row=i%n,g=i/n;const uint8_t* b=raw+(size_t(row)*groups+g)*18;
  scales[i]=*reinterpret_cast<const __half*>(b);
  #pragma unroll
  for(int w=0;w<4;w++){const uint8_t* q=b+2+w*4;signs[(size_t(g)*4+w)*n+row]=uint32_t(q[0])|(uint32_t(q[1])<<8)|(uint32_t(q[2])<<16)|(uint32_t(q[3])<<24);}
}
cudaError_t prepare(const uint8_t* raw,uint32_t* signs,__half* scales,int n,int k,cudaStream_t st) {
  if(!raw||!signs||!scales||n<=0||k<=0||k%128)return cudaErrorInvalidValue;
  transpose<<<(size_t(n)*(k/128)+255)/256,256,0,st>>>(raw,signs,scales,n,k/128);return cudaGetLastError();
}
template<bool PLANES> __global__ void producer(const float* x,int k,Carrier out) {
  int group=blockIdx.x,token=blockIdx.y,t=threadIdx.x,lane=t%32,warp=t/32;
  size_t base=(size_t(token)*(k/128)+group);float v=x[base*128+t];
  __shared__ float maxima[4],maximum,inverse;__shared__ int sums[4];__shared__ unsigned status;
  bool bad=!isfinite(v);float a=bad?0:fabsf(v);
  for(int d=16;d;d/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
  if(lane==0)maxima[warp]=a;
  int nonfinite=__syncthreads_or(bad);
  if(t==0) {
    maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
    status=nonfinite?1:maximum!=0&&(maximum<1e-30f||maximum>1e30f)?2:0;
    inverse=status||maximum==0?0:127.f/maximum;
    out.scales[base]=status?0:maximum/127.f;out.status[base]=static_cast<InputStatus>(status);
    if(status&&out.error)atomicMax(reinterpret_cast<unsigned*>(out.error),status);
  }
  __syncthreads();
  int q=status?0:__float2int_rn(v*inverse);
  if constexpr(PLANES) {
    #pragma unroll
    for(int p=0;p<8;p++){unsigned bits=__ballot_sync(0xffffffff,(unsigned(q)>>p)&1);if(lane==0)out.planes[base*32+p*4+warp]=bits;}
  }else out.q[base*128+t]=static_cast<int8_t>(q);
  for(int d=16;d;d/=2)q+=__shfl_xor_sync(0xffffffff,q,d);
  if(lane==0)sums[warp]=q;__syncthreads();
  if(t==0)out.dc[base]=sums[0]+sums[1]+sums[2]+sums[3];
}
cudaError_t produce(Route route,const float* x,int k,int r,Carrier out,cudaStream_t st) {
  if(!x||!out.scales||!out.dc||!out.status||k<=0||k%128||r<=0||r>65535||
     (route!=Route::Popcount&&route!=Route::Dp4a01)||(route==Route::Popcount&&!out.planes)||(route==Route::Dp4a01&&!out.q))return cudaErrorInvalidValue;
  if(route==Route::Popcount)producer<true><<<dim3(k/128,r),128,0,st>>>(x,k,out);
  else producer<false><<<dim3(k/128,r),128,0,st>>>(x,k,out);
  return cudaGetLastError();
}
template<int R,bool POP> __global__ void projection(const uint32_t* __restrict__ signs,const __half* __restrict__ scales,
                    Carrier in,float* y,float* partial,int n,int groups,int chunk,int ns,float alpha,float beta) {
  extern __shared__ uint32_t shared[];
  __shared__ float sx[R];__shared__ int dc[R];__shared__ unsigned error[R];
  int row=blockIdx.x*blockDim.x+threadIdx.x,split=blockIdx.y;
  int first=split*chunk,last=min(first+chunk,groups);float sum[R]={};
  for(int g=first;g<last;g++) {
    for(int t=threadIdx.x;t<R*32;t+=blockDim.x) {
      size_t base=(size_t(t/32)*groups+g)*32+t%32;
      shared[t]=POP?in.planes[base]:reinterpret_cast<const uint32_t*>(in.q)[base];
    }
    if(threadIdx.x<R) {size_t i=size_t(threadIdx.x)*groups+g;sx[threadIdx.x]=in.scales[i];dc[threadIdx.x]=in.dc[i];error[threadIdx.x]=unsigned(in.status[i]);}
    __syncthreads();
    if(row<n) {
      unsigned word[4];
      #pragma unroll
      for(int w=0;w<4;w++)word[w]=signs[(size_t(g)*4+w)*n+row];
      int selected[R]={};
      if constexpr(POP) {
        #pragma unroll
        for(int p=0;p<8;p++) {
          int count[R]={};
          #pragma unroll
          for(int w=0;w<4;w++) {
            #pragma unroll
            for(int t=0;t<R;t++)count[t]+=__popc(word[w]&shared[t*32+p*4+w]);
          }
          #pragma unroll
          for(int t=0;t<R;t++)selected[t]+=(p==7?-128:(1<<p))*count[t];
        }
      }else {
        int dots[R][4]={};
        #pragma unroll
        for(int w=0;w<4;w++) {
          #pragma unroll
          for(int j=0;j<8;j++) {
            unsigned expanded=(((word[w]>>(4*j))&15)*0x00204081u)&0x01010101u;
            #pragma unroll
            for(int t=0;t<R;t++)dots[t][w]=__dp4a(int(expanded),int(shared[t*32+w*8+j]),dots[t][w]);
          }
        }
        #pragma unroll
        for(int t=0;t<R;t++)selected[t]=(dots[t][0]+dots[t][1])+(dots[t][2]+dots[t][3]);
      }
      float ws=__half2float(scales[size_t(g)*n+row]);
      #pragma unroll
      for(int t=0;t<R;t++)sum[t]=error[t]?__int_as_float(0x7fc00000):fmaf(float(2*selected[t]-dc[t]),ws*sx[t],sum[t]);
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
  int row=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(row>=n)return;
  float sum=0;for(int s=0;s<ns;s++)sum+=partial[(size_t(t)*ns+s)*n+row];
  size_t i=size_t(t)*n+row;y[i]=beta==0?alpha*sum:fmaf(alpha,sum,beta*y[i]);
}
template<int R> static void launch(const Plan& p,const uint32_t* signs,const __half* scales,Carrier in,float* y,float* partial,float alpha,float beta,cudaStream_t st) {
  dim3 grid((p.n+p.threads-1)/p.threads,splits(p));
  if(p.route==Route::Popcount)projection<R,true><<<grid,p.threads,R*128,st>>>(signs,scales,in,y,partial,p.n,p.k/128,p.groups_per_split,splits(p),alpha,beta);
  else projection<R,false><<<grid,p.threads,R*128,st>>>(signs,scales,in,y,partial,p.n,p.k/128,p.groups_per_split,splits(p),alpha,beta);
}
cudaError_t project_carrier(const Plan& p,const uint32_t* signs,const __half* scales,Carrier in,float* y,float* partial,float alpha,float beta,cudaStream_t st) {
  cudaError_t e=validate(p);if(e!=cudaSuccess)return e;
  if(!signs||!scales||!in.scales||!in.dc||!in.status||!y||(splits(p)>1&&!partial)||
     (p.route==Route::Popcount&&!in.planes)||(p.route==Route::Dp4a01&&!in.q))return cudaErrorInvalidValue;
  #define ROWS(R) case R:launch<R>(p,signs,scales,in,y,partial,alpha,beta,st);break;
  switch(p.r){ROWS(1) ROWS(2) ROWS(3) ROWS(4) ROWS(5) ROWS(6) ROWS(7) ROWS(8)
    ROWS(9) ROWS(10) ROWS(11) ROWS(12) ROWS(13) ROWS(14) ROWS(15) ROWS(16)}
#undef ROWS
  e=cudaGetLastError();if(e!=cudaSuccess)return e;
  if(splits(p)>1){continuation<<<dim3((p.n+255)/256,p.r),256,0,st>>>(partial,y,p.n,splits(p),alpha,beta);e=cudaGetLastError();}
  return e;
}
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* scales,const float* x,Carrier scratch,float* y,float* partial,float alpha,float beta,cudaStream_t st) {
  cudaError_t e=validate(p);if(e!=cudaSuccess)return e;e=produce(p.route,x,p.k,p.r,scratch,st);if(e!=cudaSuccess)return e;
  return project_carrier(p,signs,scales,scratch,y,partial,alpha,beta,st);
}
}
