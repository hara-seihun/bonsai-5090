#include "../src/f1_swar.cuh"
#include <cstdio>
#include <cstdlib>
static unsigned nibble(int a,int b){return (a!=0?2:0)|(a<0?8:0)|(b!=0?1:0)|(b<0?4:0);}
int main(){unsigned n=0;
// Distinct neighboring channel tuples catch inter-lane carry/borrow, not just broadcast inputs.
for(int left=0;left<81;left++)for(int right=0;right<81;right++){
  int v[8][4];unsigned x=0,y=0;
  for(int lane=0;lane<8;lane++){int z=(lane&1)?right:left;for(int j=0;j<4;j++){v[lane][j]=z%3-1;z/=3;}x|=nibble(v[lane][0],v[lane][1])<<(lane*4);y|=nibble(v[lane][2],v[lane][3])<<(lane*4);}
  unsigned r[4];f1::h4_swar(x,y,r);
  for(int p=0;p<4;p++)for(int lane=0;lane<8;lane++){
    int s=0;for(int j=0;j<4;j++)s+=(__builtin_parity((unsigned)(p&j))?-1:1)*v[lane][j];
    unsigned expected=abs(s)|(s<0?8:0),got=r[p]>>(4*lane)&15;
    if(got!=expected){printf("FAIL %d %d %d %d %u %u\n",left,right,lane,p,got,expected);return 1;}n++;
  }
}
printf("SWAR direct E2M1 map passed %u neighboring-channel checks\n",n);
}
