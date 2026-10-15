#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cfenv>
static uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
int main(){
 if(std::fesetround(FE_TONEAREST))return 2;
 // Explicit sequential binary32 sums. Both sums of 4096 powers-of-two multiples
 // are exact; the last nonzero term is in the final K128 stage of K5120.
 volatile float reference=0,absorbed=0;
 for(int k=0;k<5120;k++){
  float x=k<4096?2048.f:k==5119?.5f:0.f;
  reference=reference+x;absorbed=absorbed+1.5f*x;
 }
 float original=reference*1.5f;
 uint32_t muBits=1011352371;float mu;std::memcpy(&mu,&muBits,4);
 float target=-.5f*mu,near=-.5f*.01220703125f;
 std::printf("{\"K\":5120,\"original_bits\":%u,\"absorbed_bits\":%u,\"actual_row_mu_bits\":%u,\"target_bits\":%u,\"nearest_bits\":%u}\n",bits(original),bits(absorbed),bits(mu),bits(target),bits(near));
 return bits(original)!=bits(absorbed) && bits(target)!=bits(near) ? 0:3;
}
