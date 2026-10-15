#include "codec.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
int main() {
  std::array<uint32_t,256> table{};
  for(int q=0;q<243;q++)table[q]=dense5::book(q);
  uint64_t checked=0;
  // All adjacent code labels and five nibble offsets: isolate the first two books.
  for(int q=0;q<243;q++)for(int r=0;r<243;r++)for(int start=0;start<5;start++) {
    uint64_t v=(uint64_t)table[q]|((uint64_t)table[r]<<20);
    unsigned a=q,b=r;
    int8_t t[10]; for(int j=0;j<5;j++,a/=3,b/=3){t[j]=a%3-1;t[j+5]=b%3-1;}
    for(int j=start;j<10;j++) {
      unsigned got=(v>>(j*4))&15;
      if(got!=dense5::fp4(t[j]))return 2;
      checked++;
    }
  }
  uint32_t rng=19;
  for(int trial=0;trial<128;trial++) {
    uint8_t chunk[832]{}; int8_t all[32][128];
    for(int lane=0;lane<32;lane++) {
      for(auto& t:all[lane]){rng=rng*1664525+1013904223;t=(rng%3)-1;}
      dense5::pack_lane(chunk,lane,all[lane]);
    }
    for(int lane=0;lane<32;lane++) {
      uint32_t labels[7]; for(int p=0;p<6;p++)memcpy(&labels[p],chunk+p*128+lane*4,4);
      uint16_t tail;memcpy(&tail,chunk+768+lane*2,2);labels[6]=tail;
      uint32_t expanded[26];for(int b=0;b<26;b++)expanded[b]=table[(labels[b/4]>>(8*(b%4)))&255];
      for(int w=0;w<16;w++) {
        int b=w*8/5,sh=(w*8%5)*4;
        unsigned got=(expanded[b]>>sh)|(expanded[b+1]<<(20-sh));
        if(sh>=12)got|=expanded[b+2]<<(40-sh);
        unsigned want=0;for(int j=0;j<8;j++)want|=dense5::fp4(all[lane][w*8+j])<<(j*4);
        if(got!=want)return 3;
        checked+=8;
      }
    }
  }
  printf("{\"stage\":\"host_format_certificate\",\"checked_nibbles\":%llu,\"adjacent_label_pairs\":59049,\"random_lane_streams\":4096,\"passed\":true}\n",(unsigned long long)checked);
}
