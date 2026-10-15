#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <algorithm>
#define CK(x) do{auto s=(x);if(s!=cudaSuccess){fprintf(stderr,"cuda %s\n",cudaGetErrorString(s));exit(2);}}while(0)
constexpr unsigned MOD=15015;
__device__ __forceinline__ unsigned compose(unsigned a,unsigned b){
 unsigned lane=threadIdx.x&31,g=lane>>2,t=lane&3,lo=0,hi=0;
 #pragma unroll
 for(int c=0;c<4;c++){
  unsigned v=__shfl_sync(0xffffffff,a,g*4+c);lo|=(v&255)<<(8*c);hi|=(v>>8)<<(8*c);
 }
 unsigned a0=(g<4&&t==g)?lo:0,a1=(g<4&&t==g)?hi:0,a2=(g>=4&&t==g-4)?lo:0,a3=(g>=4&&t==g-4)?hi:0;
 unsigned b0=__shfl_sync(0xffffffff,b,t*4+g/2),b1=__shfl_sync(0xffffffff,b,(t+4)*4+g/2);
 b0=((g&1)?b0>>8:b0&255)<<(8*(g/2));b1=((g&1)?b1>>8:b1&255)<<(8*(g/2));
 int c0=0,c1=0,c2=0,c3=0;
 asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
 unsigned product=unsigned(c0)+((unsigned(c1)+unsigned(c2))<<8)+(unsigned(c3)<<16),r;
 asm("rem.u32 %0,%1,15015;":"=r"(r):"r"(product));return r;
}
__global__ void pair(const uchar2* in,const uint16_t* enc,const uint8_t* obs,uint16_t* physical,uint8_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;uchar2 x=i<n?in[i]:make_uchar2(121,121);unsigned r=compose(enc[x.x],enc[x.y]);if(i<n){physical[i]=r;out[i]=obs[r];}
}
__global__ void chain(const uchar4* in,const uint16_t*enc,const uint8_t*obs,uint8_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;uchar4 x=i<n?in[i]:make_uchar4(121,121,121,121);
 unsigned r=compose(enc[x.x],enc[x.y]);r=compose(r,enc[x.z]);r=compose(r,enc[x.w]);if(i<n)out[i]=obs[r];
}
__global__ void direct(const uchar4* in,uint8_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uchar4 x=in[i];int a=x.x,b=x.y,c=x.z,d=x.w,z=0,m=1;
 #pragma unroll
 for(int k=0;k<5;k++){int v=(a%3-1)*(b%3-1)*(c%3-1)*(d%3-1);z+=(v+1)*m;m*=3;a/=3;b/=3;c/=3;d/=3;}out[i]=z;
}
int logical(int a,int b){int z=0,m=1;for(int k=0;k<5;k++){z+=((a%3-1)*(b%3-1)+1)*m;m*=3;a/=3;b/=3;}return z;}
__global__ void packed(const uchar4* in,const uint16_t*enc,const uint8_t*obs,uint8_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uchar4 x=in[i];unsigned a=enc[x.x],b=enc[x.y],c=enc[x.z],d=enc[x.w];unsigned z=(a&b&c&d)&31,s=((a^b^c^d)>>5)&z;out[i]=obs[z|(s<<5)];
}
int main(){
 int ps[5]={3,5,7,11,13};std::vector<uint16_t> e(243);std::vector<uint8_t>o(MOD,255);
 for(int a=0;a<243;a++){unsigned z=0;int q=a;for(int k=0;k<5;k++){int v=q%3-1;q/=3;int f=MOD/ps[k],inv=1;while((f*inv)%ps[k]!=1)inv++;z+=(v+ps[k])%ps[k]*f*inv;}e[a]=z%MOD;if(o[e[a]]!=255)return 3;o[e[a]]=a;}
 FILE*f=fopen("crt5-encoding.bin","wb");fwrite(e.data(),2,e.size(),f);fclose(f);f=fopen("crt5-observer.bin","wb");fwrite(o.data(),1,o.size(),f);fclose(f);
 uint16_t*de;uint8_t*doo;CK(cudaMalloc(&de,e.size()*2));CK(cudaMalloc(&doo,o.size()));CK(cudaMemcpy(de,e.data(),e.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(doo,o.data(),o.size(),cudaMemcpyHostToDevice));
 int np=243*243;std::vector<uchar2>p(np);for(int i=0;i<np;i++)p[i]=make_uchar2(i/243,i%243);uchar2*dp;uint16_t*ph;uint8_t*op;
 CK(cudaMalloc(&dp,np*sizeof(uchar2)));CK(cudaMalloc(&ph,np*2));CK(cudaMalloc(&op,np));CK(cudaMemcpy(dp,p.data(),np*sizeof(uchar2),cudaMemcpyHostToDevice));pair<<<(np+255)/256,256>>>(dp,de,doo,ph,op,np);CK(cudaGetLastError());CK(cudaDeviceSynchronize());std::vector<uint16_t>h(np);std::vector<uint8_t>ans(np);CK(cudaMemcpy(h.data(),ph,np*2,cudaMemcpyDeviceToHost));CK(cudaMemcpy(ans.data(),op,np,cudaMemcpyDeviceToHost));int wrong=0;for(int i=0;i<np;i++){unsigned want=unsigned(e[i/243])*e[i%243]%MOD;if(h[i]!=want||ans[i]!=logical(i/243,i%243))wrong++;}f=fopen("crt5-native-pairs.bin","wb");for(int i=0;i<np;i++){fwrite(&h[i],2,1,f);fwrite(&ans[i],1,1,f);}fclose(f);
 int n=262144;std::vector<uchar4>inputs(n);unsigned seed=43591;for(auto&x:inputs){unsigned char*v=(unsigned char*)&x;for(int j=0;j<4;j++){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;v[j]=seed%243;}}uchar4*di;uint8_t*nat,*ctl;CK(cudaMalloc(&di,n*sizeof(uchar4)));CK(cudaMalloc(&nat,n));CK(cudaMalloc(&ctl,n));CK(cudaMemcpy(di,inputs.data(),n*sizeof(uchar4),cudaMemcpyHostToDevice));chain<<<n/256,256>>>(di,de,doo,nat,n);direct<<<n/256,256>>>(di,ctl,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());std::vector<uint8_t>hn(n),hc(n);CK(cudaMemcpy(hn.data(),nat,n,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hc.data(),ctl,n,cudaMemcpyDeviceToHost));int cw=0;for(int i=0;i<n;i++){auto x=inputs[i];int want=logical(logical(logical(x.x,x.y),x.z),x.w);if(hn[i]!=want||hc[i]!=want)cw++;}
 std::vector<uint16_t> pe(243);std::vector<uint8_t> po(1024,255);for(int id=0;id<243;id++){int q=id;unsigned z=0,s=0;for(int k=0;k<5;k++){int v=q%3-1;q/=3;if(v)z|=1u<<k;if(v<0)s|=1u<<k;}pe[id]=z|(s<<5);po[pe[id]]=id;}uint16_t*dpe;uint8_t*dpo;CK(cudaMalloc(&dpe,486));CK(cudaMalloc(&dpo,1024));CK(cudaMemcpy(dpe,pe.data(),486,cudaMemcpyHostToDevice));CK(cudaMemcpy(dpo,po.data(),1024,cudaMemcpyHostToDevice));packed<<<n/256,256>>>(di,dpe,dpo,ctl,n);CK(cudaDeviceSynchronize());CK(cudaMemcpy(hc.data(),ctl,n,cudaMemcpyDeviceToHost));int pw=0;for(int i=0;i<n;i++)if(hc[i]!=hn[i])pw++;
 cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));std::vector<float>ms[3];for(int round=0;round<7;round++)for(int aa=0;aa<3;aa++){int arm=(aa+round)%3;CK(cudaEventRecord(start));for(int z=0;z<30;z++){if(arm==0)chain<<<n/256,256>>>(di,de,doo,nat,n);else if(arm==1)direct<<<n/256,256>>>(di,ctl,n);else packed<<<n/256,256>>>(di,dpe,dpo,ctl,n);}CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float v;CK(cudaEventElapsedTime(&v,start,end));ms[arm].push_back(v*1000/30);}for(auto&v:ms)std::sort(v.begin(),v.end());
 printf("{\"pairs\":%d,\"native_pair_errors\":%d,\"chain_cases\":%d,\"chain_errors\":%d,\"native_us\":%.6f,\"direct_us\":%.6f,\"encoding_bytes\":486,\"observer_bytes\":15015,\"mma_per_fourfactor_warp\":3,\"logical_outputs_per_warp\":160}\n",np,wrong,n,cw,ms[0][3],ms[1][3]);printf("{\"packed_errors\":%d,\"packed_us\":%.6f}\n",pw,ms[2][3]);return wrong||cw||pw?1:0;
}
