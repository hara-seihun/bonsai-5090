#include "ternary_operands.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <random>

int main() {
  h5::TernaryOperands mode;
  assert(h5::parse_ternary_operands(nullptr, mode) && mode == h5::TernaryOperands::Packed);
  assert(h5::parse_ternary_operands("packed", mode) && mode == h5::TernaryOperands::Packed);
  assert(h5::parse_ternary_operands("expanded", mode) && mode == h5::TernaryOperands::Expanded);
  assert(h5::parse_ternary_operands("scale-words", mode) && mode == h5::TernaryOperands::ScaleWords);
  assert(!h5::parse_ternary_operands("", mode));
  assert(!h5::parse_ternary_operands("1", mode));
  std::mt19937 rng(17);
  for (int trial = 0; trial < 32; trial++) {
    uint8_t code[32][128];
    for (auto& row : code) for (auto& c : row) c = rng() % 4;
    std::array<uint8_t, 1024> packed{};
    for (int lane = 0; lane < 32; lane++) for (int w = 0; w < 8; w++) {
      const int row = 16 * (w >> 2) + 8 * (w & 1) + (lane >> 2);
      uint32_t v = 0;
      for (int j = 0; j < 8; j++) {
        const int k = 64 * ((w >> 1) & 1) + 8 * (lane & 3) + j;
        const unsigned c0 = code[row][k], c1 = code[row][k + 32];
        v |= (((c0 & 1) << 1) | ((c0 >> 1) << 3) | (c1 & 1) | ((c1 >> 1) << 2)) << (4 * j);
      }
      std::memcpy(packed.data() + (w >> 2) * 512 + lane * 16 + (w & 3) * 4, &v, 4);
    }
    std::array<uint8_t, 2048> expanded{};
    h5::expand_ternary_chunk(packed.data(), expanded.data());
    for (int lane = 0; lane < 32; lane++) for (int i = 0; i < 16; i++) {
      uint32_t word;
      std::memcpy(&word, expanded.data() + (i >> 2) * 512 + lane * 16 + (i & 3) * 4, 4);
      const int row = 16 * (i >> 3) + 8 * (i & 1) + (lane >> 2);
      for (int j = 0; j < 8; j++) {
        const int k = 64 * ((i >> 2) & 1) + 32 * ((i >> 1) & 1) + 8 * (lane & 3) + j;
        const unsigned c = code[row][k], nibble = ((c & 1) ? 2 : 0) | ((c & 2) ? 8 : 0);
        assert(((word >> (4 * j)) & 15) == nibble);
      }
    }
    uint8_t scales[64] = {}, expanded_scales[128];
    for (int i = 0; i < 32; i++) scales[i] = trial * 8 + i;
    h5::expand_ternary_scales(scales, expanded_scales);
    for (int i = 0; i < 128; i++) assert(expanded_scales[i] == scales[i / 4]);
  }
  std::puts("TERNARY_OPERANDS_CPU_OK");
}
