// g++ -O3 -std=c++17 -ffp-contract=off bench/q8_scale_stability.cpp -o /tmp/q8-scale-stability
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <algorithm>

using Row = std::array<float,128>;
struct Quantized { std::array<int,128> code; float scale; };
Quantized encode(const Row& x, bool ls) {
  float maximum=0;
  for(float v:x) maximum=std::fmax(maximum,std::fabs(v));
  Quantized out{};
  out.scale=maximum*(1.f/127.f);
  const float inv=out.scale>0 ? 1.f/out.scale : 0.f;
  float numerator=0, denominator=0;
  for(int i=0;i<128;i++) {
    const int q=std::max(-127,std::min(127,(int)std::nearbyint(x[i]*inv)));
    out.code[i]=q;
    numerator=std::fma(x[i],(float)q,numerator);
    denominator=std::fma((float)q,(float)q,denominator);
  }
  if(ls) out.scale=denominator>0 ? numerator/denominator : 0.f;
  return out;
}
Row decode(const Quantized& x) {
  Row y{};for(int i=0;i<128;i++) y[i]=(float)x.code[i]*x.scale;return y;
}
Row step(const Row& x,const Row& k,float g,float beta) {
  float sum[4]={};
  for(int i=0;i<128;i++) sum[i&3]=std::fma(x[i],k[i],sum[i&3]);
  const float dot=(sum[0]+sum[1])+(sum[2]+sum[3]);
  const float delta=beta*(-g*dot); // Zero innovation forcing; same delta-rule update.
  Row y{};for(int i=0;i<128;i++) y[i]=std::fma(g,x[i],delta*k[i]);return y;
}
double norm(const Row& x) { double s=0;for(float v:x)s+=(double)v*v;return std::sqrt(s); }
int main() {
  if(std::fesetround(FE_TONEAREST)) return 1;
  // Stay inside one rounding cell rather than relying on the 0.5 tie.
  const float r=0.5001f,beta=.9f,g=.9999f;
  const float inv=1.f/std::sqrt(1.f+r*r);
  Row k{};k[0]=-r*inv;k[1]=inv;
  const double qr=64./127.;
  const double prediction=(double)g*(1.+(double)beta*r*(qr-r)/(1.+(double)r*r));
  std::printf("{\"kind\":\"config\",\"r\":%.9g,\"g\":%.9g,\"beta\":%.9g,\"norm_k\":%.17g,\"predicted_max_scale_gain\":%.17g,\"exact_tie_example_gain\":%.17g}\n",r,g,beta,norm(k),prediction,636./635.);
  Row ref{},max_state{},ls_state{};ref[0]=max_state[0]=ls_state[0]=1;ref[1]=max_state[1]=ls_state[1]=r;
  const int checkpoints[]={0,1,100,1000,4000,8000,16000,24000,32768};
  int checkpoint=0;double max_ls_increase=0,max_max_increase=0;
  for(int t=0;t<=32768;t++) {
    if(t==checkpoints[checkpoint]) {
      auto q=encode(max_state,false);
      std::printf("{\"kind\":\"sample\",\"step\":%d,\"f32_l2\":%.17g,\"max_l2\":%.17g,\"ls_l2\":%.17g,\"max_codes\":[%d,%d],\"max_scale\":%.9g}\n",t,norm(ref),norm(max_state),norm(ls_state),q.code[0],q.code[1],q.scale);
      if(++checkpoint==9) break;
    }
    ref=step(ref,k,g,beta);
    const Row before_max=step(max_state,k,g,beta),before_ls=step(ls_state,k,g,beta);
    max_state=decode(encode(before_max,false));ls_state=decode(encode(before_ls,true));
    max_max_increase=std::max(max_max_increase,norm(max_state)/norm(before_max)-1.);
    max_ls_increase=std::max(max_ls_increase,norm(ls_state)/norm(before_ls)-1.);
  }
  std::printf("{\"kind\":\"float_rounding_observation\",\"max_l2_relative_quantizer_increase\":%.17g,\"ls_l2_relative_quantizer_increase\":%.17g}\n",max_max_increase,max_ls_increase);
  if(!std::isfinite(norm(max_state)) || !std::isfinite(norm(ls_state))) return 2;
  if(norm(max_state)<1e12 || norm(ls_state)>1.01 || norm(ref)>1.01) return 3;
  return 0;
}
