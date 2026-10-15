#include "runtime.hpp"
#include <cstdio>
#include <vector>
using namespace halo6000;
__global__ void check_words(const uint32_t*p,size_t n,uint32_t expected,unsigned long long*errors){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)if(p[i]!=expected)atomicAdd(errors,1ull);}
int main(){
 DeviceModel m;m.state.slots=2;m.state.context=8;
 std::vector<void*>owner;auto alloc=[&](size_t n,auto*&p){void*raw=nullptr;if(cudaMalloc(&raw,n)!=cudaSuccess)return false;owner.push_back(raw);p=reinterpret_cast<std::remove_reference_t<decltype(p)>>(raw);return cudaMemset(raw,0x3f,n)==cudaSuccess;};
 size_t kv=size_t(AttentionLayers)*2*8*KVHeads*HeadDim*2;
 if(!alloc(2*GdnFloatsPerSlot*4,m.state.gdn)||!alloc(2*ConvFloatsPerSlot*4,m.state.conv)||!alloc(kv,m.state.key)||!alloc(kv,m.state.value)||!alloc(8,m.state.positions))return 1;
 unsigned long long*errors=nullptr;if(!alloc(8,errors))return 1;
 cudaStream_t stream;if(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess)return 1;
 auto verify=[&](uint32_t e0,uint32_t e1){cudaMemsetAsync(errors,0,8,stream);for(int s=0;s<2;s++){uint32_t e=s?e1:e0;check_words<<<64,256,0,stream>>>(reinterpret_cast<uint32_t*>(m.state.gdn+size_t(s)*GdnFloatsPerSlot),GdnFloatsPerSlot,e,errors);check_words<<<64,256,0,stream>>>(reinterpret_cast<uint32_t*>(m.state.conv+size_t(s)*ConvFloatsPerSlot),ConvFloatsPerSlot,e,errors);check_words<<<1,32,0,stream>>>(reinterpret_cast<uint32_t*>(m.state.positions+s),1,e,errors);for(int l=0;l<AttentionLayers;l++){size_t off=(size_t(l)*2+s)*8*KVHeads*HeadDim;check_words<<<8,256,0,stream>>>(reinterpret_cast<uint32_t*>(m.state.key+off),8*KVHeads*HeadDim/2,e,errors);check_words<<<8,256,0,stream>>>(reinterpret_cast<uint32_t*>(m.state.value+off),8*KVHeads*HeadDim/2,e,errors);}}unsigned long long n=0;cudaMemcpyAsync(&n,errors,8,cudaMemcpyDeviceToHost,stream);return cudaStreamSynchronize(stream)==cudaSuccess&&n==0;};
 auto s=m.reset(0);if(!s.ok()||!verify(0,0x3f3f3f3f))return 2;
 s=m.reset(1,stream);if(!s.ok()||!verify(0,0))return 3;
 if(m.reset(2).ok()||m.reset(-2).ok())return 4;
 s=m.reset();if(!s.ok()||!verify(0,0))return 5;
 printf("{\"reset_default_to_nonblocking\":true,\"reset_nonblocking\":true,\"slot_isolation\":true,\"invalid_slots_rejected\":true,\"all_state_bits_zero\":true}\n");
 cudaStreamDestroy(stream);for(void*p:owner)cudaFree(p);return 0;
}
