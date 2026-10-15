#include "batch_map.cuh"
namespace batch_map {
__global__ void count_kernel(const uint32_t* signs,uint8_t* counts,int n,int groups) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*groups)return;
  int g=i/n,row=i%n,c=0;
  for(int w=0;w<4;w++)c+=__popc(signs[(size_t(g)*4+w)*n+row]);counts[i]=c;
}
cudaError_t prepare_counts(const uint32_t* signs,uint8_t* counts,int n,int k,cudaStream_t st) {
  if(!signs||!counts||n<=0||k<=0||k%128)return cudaErrorInvalidValue;
  count_kernel<<<(size_t(n)*(k/128)+255)/256,256,0,st>>>(signs,counts,n,k/128);return cudaGetLastError();
}
__global__ void producer(const float* x,Carrier out) {
  int g=blockIdx.x,pair=blockIdx.y,j=threadIdx.x,lane=j%32,warp=j/32,groups=out.k/128;
  __shared__ float maxima[4],inverse;__shared__ int sums[4];__shared__ unsigned status;
  unsigned u[2];
  for(int h=0;h<2;h++) {
    int t=pair*2+h;bool present=t<out.rows;size_t base=size_t(t)*groups+g;
    float v=present?x[base*128+j]:0;bool bad=!isfinite(v);float a=bad?0:fabsf(v);
    for(int d=16;d;d/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
    if(lane==0)maxima[warp]=a;int nonfinite=__syncthreads_or(bad);
    if(j==0) {
      float maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
      status=nonfinite?1:maximum!=0&&(maximum<1e-30f||maximum>1e30f)?2:0;
      inverse=status||maximum==0?0:127.f/maximum;
      if(present){out.metadata.scales[base]=status?0:maximum/127.f;out.metadata.status[base]=static_cast<bitserial::InputStatus>(status);
        if(status&&out.metadata.error)atomicMax(reinterpret_cast<unsigned*>(out.metadata.error),status);}
    }
    __syncthreads();int q=status?0:__float2int_rn(v*inverse);u[h]=present?unsigned(q+128):0;
    for(int d=16;d;d/=2)q+=__shfl_xor_sync(0xffffffff,q,d);
    if(lane==0)sums[warp]=q;__syncthreads();
    if(j==0&&present)out.metadata.dc[base]=sums[0]+sums[1]+sums[2]+sums[3];__syncthreads();
  }
  out.packed[(size_t(pair)*groups+g)*128+j]=u[0]|(u[1]<<16);
}
cudaError_t produce(const float* x,Carrier out,cudaStream_t st) {
  if(!x||!out.packed||!out.metadata.scales||!out.metadata.dc||!out.metadata.status||out.k<=0||out.k%128||out.rows<1||out.rows>512)return cudaErrorInvalidValue;
  producer<<<dim3(out.k/128,(out.rows+1)/2),128,0,st>>>(x,out);return cudaGetLastError();
}
template<int R> __global__ void projection(const uint32_t* signs,const __half* scales,const uint8_t* counts,
    Carrier in,float* y,float* partial,int n,int chunk,int first_token,int ns,float alpha,float beta) {
  constexpr int P=(R+1)/2;extern __shared__ uint32_t shared[];
  __shared__ float sx[R];__shared__ int dc[R];__shared__ unsigned status[R];
  int row=blockIdx.x*blockDim.x+threadIdx.x,split=blockIdx.y,groups=in.k/128;
  int first=split*chunk,last=min(first+chunk,groups);float total[R]={};
  for(int g=first;g<last;g++) {
    for(int i=threadIdx.x;i<P*128;i+=blockDim.x)shared[i]=in.packed[(size_t(first_token/2+i/128)*groups+g)*128+i%128];
    if(threadIdx.x<R){int t=threadIdx.x;size_t base=size_t(first_token+t)*groups+g;
      sx[t]=in.metadata.scales[base];dc[t]=in.metadata.dc[base];status[t]=unsigned(in.metadata.status[base]);}
    __syncthreads();
    if(row<n) {
      uint32_t sums[P][4]={};
      #pragma unroll
      for(int w=0;w<4;w++) {
        uint32_t word=signs[(size_t(g)*4+w)*n+row];
        #pragma unroll
        for(int j=0;j<32;j++) {
          unsigned b=(word>>j)&1;
          #pragma unroll
          for(int p=0;p<P;p++) {
            unsigned a=shared[p*128+w*32+j];
            asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(sums[p][w]) : "r"(a),"r"(b),"r"(sums[p][w]));
          }
        }
      }
      int count=counts[size_t(g)*n+row];float ws=__half2float(scales[size_t(g)*n+row]);
      #pragma unroll
      for(int p=0;p<P;p++) {
        unsigned packed=(sums[p][0]+sums[p][1])+(sums[p][2]+sums[p][3]);
        #pragma unroll
        for(int h=0;h<2;h++)if(p*2+h<R) {
          int t=p*2+h,selected=(packed>>(h*16))&65535;
          int dot=2*selected-dc[t]-256*count;
          total[t]=status[t]?__int_as_float(0x7fc00000):fmaf(float(dot),ws*sx[t],total[t]);
        }
      }
    }
    __syncthreads();
  }
  if(row<n) {
    #pragma unroll
    for(int t=0;t<R;t++)if(ns>1)partial[(size_t(t)*ns+split)*n+row]=total[t];
    else {size_t i=size_t(t)*n+row;y[i]=beta==0?alpha*total[t]:fmaf(alpha,total[t],beta*y[i]);}
  }
}
__global__ void continuation(const float* partial,float* y,int n,int ns,float alpha,float beta) {
  int row=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(row>=n)return;float sum=0;
  for(int s=0;s<ns;s++)sum+=partial[(size_t(t)*ns+s)*n+row];size_t i=size_t(t)*n+row;
  y[i]=beta==0?alpha*sum:fmaf(alpha,sum,beta*y[i]);
}
cudaError_t project(const Plan& p,const uint32_t* signs,const __half* scales,const uint8_t* counts,
    Carrier in,float* y,float* partial,float alpha,float beta,cudaStream_t st) {
  if(!signs||!scales||!counts||!in.packed||!in.metadata.scales||!in.metadata.dc||!in.metadata.status||!y||p.n<=0||p.k<=0||p.k%128||p.k!=in.k||
     in.rows<1||in.rows>512||p.rows<1||p.rows>16||p.first_token<0||p.first_token>=in.rows||p.first_token%2||p.rows>in.rows-p.first_token||p.groups_per_split<=0||(p.threads!=128&&p.threads!=256))return cudaErrorInvalidValue;
  int ns=(p.k/128+p.groups_per_split-1)/p.groups_per_split;if(ns>1&&!partial)return cudaErrorInvalidValue;
  dim3 grid((p.n+p.threads-1)/p.threads,ns);
  #define ROW(R) case R:projection<R><<<grid,p.threads,((R+1)/2)*128*4,st>>>(signs,scales,counts,in,y,partial,p.n,p.groups_per_split,p.first_token,ns,alpha,beta);break;
  switch(p.rows){ROW(1) ROW(2) ROW(3) ROW(4) ROW(5) ROW(6) ROW(7) ROW(8) ROW(9) ROW(10) ROW(11) ROW(12) ROW(13) ROW(14) ROW(15) ROW(16) default:return cudaErrorInvalidValue;}
  #undef ROW
  auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
  if(ns>1){continuation<<<dim3((p.n+255)/256,p.rows),256,0,st>>>(partial,y,p.n,ns,alpha,beta);return cudaGetLastError();}return cudaSuccess;
}
}
