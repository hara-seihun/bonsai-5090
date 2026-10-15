#include "pf_chain.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>

struct Counts { unsigned bytes, pieces, ctas; };
Counts check(unsigned bytes, unsigned grid, unsigned threads) {
  unsigned cursor = 0, pieces = 0, ctas = 0;
  for (unsigned cta = 0; cta < grid; ++cta) {
    const auto s = h5::pf_chain_slice(bytes, grid, cta);
    assert(s.begin == cursor && !(s.begin & 15u) && !(s.bytes & 15u));
    assert(s.begin + s.bytes <= bytes);
    cursor += s.bytes;
    if (s.bytes) ++ctas;
    unsigned submitted = 0;
    for (unsigned lane = 0; lane < threads; ++lane) {
      for (unsigned offset = lane * 16384u; offset < s.bytes; offset += threads * 16384u) {
        const unsigned remaining = s.bytes - offset;
        const unsigned n = remaining < 16384u ? remaining : 16384u;
        assert(n && !(n & 15u) && !((s.begin + offset) & 15u));
        submitted += n; ++pieces;
      }
    }
    assert(submitted == s.bytes);
  }
  assert(cursor == bytes);
  return {cursor, pieces, ctas};
}
int main() {
  for (const unsigned cap : {0u, 16u, 512u, 16384u, 16777216u, 36208640u, 67108864u})
    for (const unsigned grid : {1u, 28u, 80u, 192u, 5120u})
      check(cap, grid, 128);
  const auto conv = check(16777216, 80, 128);
  const auto attn = check(16777216, 28, 256);
  assert(conv.pieces == 1040 && conv.ctas == 80);
  assert(attn.pieces == 1036 && attn.ctas == 28);
  assert(48 * conv.pieces + 16 * attn.pieces == 66496);
  assert(48 * conv.ctas + 16 * attn.ctas == 4288);
  printf("16MiB x 64 layers: intended_hint_bytes=1073741824 pieces_requests=66496 cta_requests=4288\n");
}
