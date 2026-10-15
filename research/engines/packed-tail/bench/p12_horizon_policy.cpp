#include "../src/draft_horizon.h"
#include <cassert>
#include <cstdio>
int main(){
 h5::horizon::State s;h5::horizon::Costs c;
 assert(s.choose(32,96,c)==15);
 for(int i=0;i<40;i++){s.draft_s=.001;s.observe(16,0,.005,c,false);}
 assert(s.choose(32,96,c)==3);
 for(int cap=1;cap<=32;cap++)for(int remaining=1;remaining<100;remaining++){
  int n=s.choose(cap,remaining,c);assert(n>=1&&n<=cap&&n<=remaining);
 }
 h5::horizon::State high;
 for(int i=0;i<40;i++){high.draft_s=.001;high.observe(32,32,.007,c,false);}
 assert(high.choose(32,96,c)==31);
 auto seen=high.seen[31];high.observe(4,4,.005,c,false);assert(high.seen[31]==seen*.9);
 puts("horizon bounds, low/high acceptance and censored observations pass");
}
