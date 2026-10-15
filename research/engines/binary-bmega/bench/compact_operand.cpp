#define __device__
#define __forceinline__ inline
#include "compact.cuh"
#include <cstdio>
#include <array>

int main() {
  alignas(128) std::array<uint8_t, 832> chunk{};
  uint32_t book[256];
  for (unsigned q=0; q<256; q++) book[q]=dense5::book(q);
  size_t compared=0;
  for (int byte=0; byte<26; byte++) for (int code=0; code<243; code++) {
    for (int lane=0; lane<32; lane++) {
      int8_t trits[128];
      for (int i=0; i<128; i++) trits[i]=int((i*17+lane*31+byte)%3)-1;
      dense5::pack_lane(chunk.data(),lane,trits);
      const int pos=byte<24?(byte/4)*128+lane*4+byte%4:768+lane*2+byte-24;
      chunk[pos]=uint8_t(code);
      uint32_t labels[7], reference[16];
      h5::dense5_labels(chunk.data(),lane,labels);
      h5::dense5_registers(chunk.data(),lane,book,reference);
      for (int w=0; w<16; w++) {
        auto got=h5::dense5_word(labels,book,w);
        if (got!=reference[w]) { fprintf(stderr,"byte=%d code=%d lane=%d word=%d\n",byte,code,lane,w); return 1; }
        compared++;
      }
    }
  }
  printf("dense5 operand words matched: %zu\n",compared);
}
