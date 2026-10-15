#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>

constexpr std::uint32_t expand(std::uint32_t nibble) {
    return 0xb8b8b8b8u ^ ((((nibble * 0x00204081u) & 0x01010101u) << 7));
}

// Exact E4M3FN finite value, in units of 1/512. NaN is explicitly unset.
constexpr std::optional<int> decode_scaled512(std::uint8_t byte) {
    const unsigned exponent = (byte >> 3) & 15;
    const unsigned mantissa = byte & 7;
    if (exponent == 15 && mantissa == 7) return std::nullopt;
    const int magnitude = exponent == 0 ? static_cast<int>(mantissa)
        : static_cast<int>((8u + mantissa) << (exponent - 1));
    return (byte & 128) ? -magnitude : magnitude;
}

std::optional<float> decode_e4m3(std::uint8_t byte) {
    const unsigned exponent = (byte >> 3) & 15;
    const unsigned mantissa = byte & 7;
    if (exponent == 15 && mantissa == 7) return std::nullopt;
    const float magnitude = exponent == 0
        ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                     static_cast<int>(exponent) - 7);
    return (byte & 128) ? -magnitude : magnitude;
}

constexpr bool exhaustive_codes_and_exact_values() {
    for (std::uint32_t n = 0; n < 16; ++n) {
        for (unsigned lane = 0; lane < 4; ++lane) {
            const bool positive = ((n >> lane) & 1u) != 0;
            const auto byte = static_cast<std::uint8_t>(expand(n) >> (8 * lane));
            if (byte != (positive ? 0x38u : 0xb8u)) return false;
            if (((byte >> 3) & 15) != 7 || (byte & 7) != 0) return false;
            const auto decoded = decode_scaled512(byte);
            if (!decoded || *decoded != (positive ? 512 : -512)) return false;
        }
    }
    return true;
}

static_assert(exhaustive_codes_and_exact_values());
static_assert(!decode_scaled512(0x7f) && !decode_scaled512(0xff));
static_assert(decode_scaled512(0x7e) == 448 * 512);
static_assert(decode_scaled512(0x01) == 1);

int main() {
    unsigned cases = 0;
    for (std::uint32_t n = 0; n < 16; ++n) {
        for (unsigned lane = 0; lane < 4; ++lane) {
            const bool positive = ((n >> lane) & 1u) != 0;
            const auto byte = static_cast<std::uint8_t>(expand(n) >> (8 * lane));
            const auto decoded = decode_e4m3(byte);
            if (byte != (positive ? 0x38u : 0xb8u) || !decoded ||
                *decoded != (positive ? 1.0f : -1.0f)) {
                std::cerr << "FAIL nibble=" << n << " lane=" << lane << '\n';
                return 1;
            }
            ++cases;
        }
    }
    std::cout << "PASS: 16 nibbles, " << cases
              << " lane cases; bytes 0x38/0xb8, exponent 7, fraction 0, decoded +/-1\n";
}
