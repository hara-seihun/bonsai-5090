#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace h5::horizon {
// Per-request survival estimates; per-scheduler measured shape costs. No token
// contents enter the controller, and it never changes acceptance thresholds.
constexpr std::array<int,5> sizes{3,7,15,23,31};
struct Costs {
  std::array<double,33> seconds{};
  std::array<int,33> samples{};
  void observe(int n,double t) {
    if(n<1||n>32||!std::isfinite(t)||t<=0)return;
    // First-use graph construction is charged to the request, but should not
    // predict the cost of an already captured shape forever.
    seconds[n]=samples[n]++?std::min(seconds[n]*1.02,t):t;
  }
  double estimate(int n,int current) const {
    double base=samples[current]?seconds[current]:0.008;
    double predicted=base*(0.70+0.30*double(n+1)/(current+1));
    return samples[n]>1?std::min(seconds[n],predicted):predicted;
  }
};
struct State {
  std::array<double,32> yes{},seen{};
  std::array<int,33> counts{};
  int rounds=0,current=15;
  double draft_s=0;
  void observe(int n,int accepted,double verify_s,Costs& costs,bool eos) {
    if(n<1||n>32||accepted<0||accepted>n)return;
    costs.observe(n,verify_s+draft_s);counts[n]++;rounds++;
    for(int i=0;i<32;i++){yes[i]*=.9;seen[i]*=.9;}
    for(int i=0;i<std::min(n,accepted+int(!eos));i++){
      seen[i]+=1;yes[i]+=i<accepted;
    }
    current=n;
  }
  int choose(int capacity,int remaining,const Costs& costs) const {
    const int cap=std::max(1,std::min({32,capacity,remaining}));
    if(rounds<2){int initial=std::min(3,cap);for(int n:sizes)if(n<=std::min(15,cap))initial=n;return initial;}
    double total_yes=0,total_seen=0;for(int i=0;i<32;i++){total_yes+=yes[i];total_seen+=seen[i];}
    const double prior=(total_yes+1.8)/(total_seen+2.0);
    double survive=1,expected=1,best=-1,current_rate=0;int selected=std::min(3,cap);
    for(int n=1;n<=cap;n++){
      // Weak prior permits unseen later positions to become worthwhile after
      // several long accepted prefixes, while observed rejections dominate.
      double p=(yes[n-1]+2*prior)/(seen[n-1]+2.0);
      survive*=p;expected+=survive;
      bool candidate=(n==cap&&cap<3)||std::find(sizes.begin(),sizes.end(),n)!=sizes.end();
      if(!candidate)continue;
      double rate=expected/costs.estimate(n,current);
      if(n==current)current_rate=rate;
      if(rate>best){best=rate;selected=n;}
    }
    if(current<=cap&&current_rate>=best*.97)return current;
    return selected;
  }
};
}
