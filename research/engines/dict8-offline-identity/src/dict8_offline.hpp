#pragma once
#include "dict8_codec.hpp"
namespace h5::dict8::offline {
static inline void decode_weight_trits(const uint8_t* blk, int8_t* trit) {
  int e = 0;
  const int stages[2][2] = {{0, 16}, {16, 8}};
  for (int s = 0; s < 2; s++) {
    int j0 = stages[s][0], c = stages[s][1];
    for (int n = 0; n < 5; n++)
      for (int m = 0; m < c; m++) {
        unsigned q = blk[j0 + m];
        for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
        trit[e++] = (int8_t)((int)((q * 3u) >> 8) - 1);
      }
  }
  for (int n = 0; n < 4; n++)
    for (int h = 0; h < 2; h++) {
      unsigned q = blk[24 + h];
      for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
      trit[e++] = (int8_t)((int)((q * 3u) >> 8) - 1);
    }
}

inline void wf0(const int8_t (&trit)[32][128], uint8_t* chunk) {
 for(int l=0;l<32;l++){const int gid=l>>2,tq=l&3;
          for (int w = 0; w < 8; w++) {
            const int m = w >> 2, h = (w >> 1) & 1, r = w & 1, row = 16 * m + 8 * r + gid;
            uint32_t word = 0;
            for (int j = 0; j < 8; j++) {
              const int t0 = trit[row][64 * h + 8 * tq + j], t1 = trit[row][64 * h + 32 + 8 * tq + j];
              word |= (uint32_t)((t0 != 0) << 1 | (t0 < 0) << 3) << (4 * j);
              word |= (uint32_t)((t1 != 0) << 0 | (t1 < 0) << 2) << (4 * j);
            }
            memcpy(chunk + (w >> 2) * 512 + l * 16 + (w & 3) * 4, &word, 4);
          }
          }
}
}
