#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace h5 {
// WF6 carries WF2's exact MMA words, including the sign bit on zero, and
// pre-broadcasts its E4M3 group scales. Row scales and accumulation are unchanged.
constexpr int TERNARY_EXPANDED_WF = 6;
constexpr int TERNARY_EXPANDED_CW = 2048;
constexpr int TERNARY_EXPANDED_CS = 128;
constexpr int TERNARY_SCALE_WORDS_WF = 7;
enum class TernaryOperands { Packed, Expanded, ScaleWords };
inline bool parse_ternary_operands(const char* v, TernaryOperands& out) {
  if (!v || !std::strcmp(v, "packed")) { out = TernaryOperands::Packed; return true; }
  if (!std::strcmp(v, "expanded")) { out = TernaryOperands::Expanded; return true; }
  if (!std::strcmp(v, "scale-words")) { out = TernaryOperands::ScaleWords; return true; }
  return false;
}
inline void expand_ternary_chunk(const uint8_t* packed, uint8_t* expanded) {
  for (int lane = 0; lane < 32; lane++)
    for (int mm = 0; mm < 2; mm++)
      for (int h = 0; h < 2; h++) {
        const int w = mm * 4 + h * 2;
        uint32_t r0, r1;
        std::memcpy(&r0, packed + (w >> 2) * 512 + lane * 16 + (w & 3) * 4, 4);
        std::memcpy(&r1, packed + ((w + 1) >> 2) * 512 + lane * 16 + ((w + 1) & 3) * 4, 4);
        const uint32_t a[4] = {r0 & 0xAAAAAAAAu, r1 & 0xAAAAAAAAu,
                              (r0 << 1) & 0xAAAAAAAAu, (r1 << 1) & 0xAAAAAAAAu};
        std::memcpy(expanded + (mm * 2 + h) * 512 + lane * 16, a, sizeof(a));
      }
}
inline void expand_ternary_scales(const uint8_t* packed, uint8_t* expanded) {
  for (int i = 0; i < 32; i++) {
    const uint32_t word = uint32_t(packed[i]) * 0x01010101u;
    std::memcpy(expanded + i * 4, &word, 4);
  }
}
}
