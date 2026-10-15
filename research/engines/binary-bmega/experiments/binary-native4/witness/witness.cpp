#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {
constexpr int K = 128;
using Signs = std::array<std::uint32_t, 4>;
using Nibbles = std::array<std::uint32_t, K / 8>;
using Activations = std::array<std::int8_t, K>;
struct WeightCarrier { Nibbles s4{}; std::int8_t half_dc = -64; };
struct ActivationCarrier { Nibbles low{}; Nibbles high{}; };

constexpr int signed_nibble(unsigned code) {
    return code < 8 ? int(code) : int(code) - 16;
}
constexpr unsigned encode_weight(bool positive) { return positive ? 1u : 15u; }
constexpr unsigned lane(const Nibbles& words, int i) {
    return (words[i / 8] >> (4 * (i % 8))) & 15u;
}
void set_lane(Nibbles& words, int i, unsigned code) {
    words[i / 8] |= std::uint32_t(code) << (4 * (i % 8));
}
bool positive(const Signs& signs, int i) {
    return ((signs[i / 32] >> (i % 32)) & 1u) != 0;
}
WeightCarrier pack_weights(const Signs& signs) {
    WeightCarrier w;
    for (int i = 0; i < K; ++i) {
        const bool p = positive(signs, i);
        set_lane(w.s4, i, encode_weight(p));
        if (p) ++w.half_dc;
    }
    return w;
}
ActivationCarrier pack_activations(const Activations& q) {
    ActivationCarrier a;
    for (int i = 0; i < K; ++i) {
        const unsigned u = unsigned(int(q[i]) + 128);
        set_lane(a.low, i, u % 16);
        set_lane(a.high, i, u / 16);
    }
    return a;
}
int projected(const WeightCarrier& w, const ActivationCarrier& a) {
    int lo = 0, hi = 0;
    for (int i = 0; i < K; ++i) {
        const int sign = signed_nibble(lane(w.s4, i));
        lo += sign * int(lane(a.low, i));
        hi += sign * int(lane(a.high, i));
    }
    return lo + 16 * hi - 256 * int(w.half_dc);
}
int dense_reference(const Signs& signs, const Activations& q) {
    int dot = 0;
    for (int i = 0; i < K; ++i) dot += (positive(signs, i) ? 1 : -1) * int(q[i]);
    return dot;
}
[[noreturn]] void fail(const char* what, int actual, int expected) {
    std::cerr << "FAIL " << what << ": " << actual << " != " << expected << '\n';
    std::exit(1);
}
void equal(const char* what, int actual, int expected) {
    if (actual != expected) fail(what, actual, expected);
}
std::uint64_t full_cases = 0;
void check_full(const Signs& signs, const Activations& q) {
    const int reference = dense_reference(signs, q);
    const auto packed = pack_weights(signs);
    int sum_w = 0;
    for (int i = 0; i < K; ++i) sum_w += positive(signs, i) ? 1 : -1;
    equal("half DC identity", 2 * int(packed.half_dc), sum_w);
    if (packed.half_dc < -64 || packed.half_dc > 64) fail("half DC bound", packed.half_dc, 64);
    equal("K128 signed-W4 unsigned-A4 correction", projected(packed, pack_activations(q)), reference);
    if (reference < -16384 || reference > 16384) fail("K128 bound", reference, 16384);
    ++full_cases;
}
std::int8_t signed_byte(int code) { return static_cast<std::int8_t>(code - 128); }
}

int main() {
    static_assert(std::numeric_limits<std::uint32_t>::digits == 32);
    static_assert(std::numeric_limits<std::int8_t>::min() == -128);
    static_assert(std::numeric_limits<std::int8_t>::max() == 127);
    static_assert(std::numeric_limits<int>::max() >= 2147483647);
    equal("decode s4 1", signed_nibble(1), 1);
    equal("decode s4 15", signed_nibble(15), -1);
    int scalar_cases = 0;
    for (int p = 0; p < 2; ++p) {
        const int w = p ? 1 : -1;
        for (int q = -128; q <= 127; ++q) {
            const int u = q + 128, low = u % 16, high = u / 16;
            if (low < 0 || low > 15 || high < 0 || high > 15) fail("digit bounds", low, high);
            equal("signed-byte decomposition", low + 16 * high - 128, q);
            const unsigned code = encode_weight(p != 0);
            equal("s4 weight", signed_nibble(code), w);
            equal("scalar projection", signed_nibble(code) * low + 16 * signed_nibble(code) * high - 128 * w, w * q);
            ++scalar_cases;
        }
    }
    int encoding_cases = 0;
    for (unsigned bits = 0; bits < 256; ++bits) {
        std::uint32_t packed = 0;
        for (int i = 0; i < 8; ++i) packed |= encode_weight(((bits >> i) & 1u) != 0) << (4 * i);
        for (int i = 0; i < 8; ++i) {
            const bool p = ((bits >> i) & 1u) != 0;
            const unsigned code = (packed >> (4 * i)) & 15u;
            const unsigned expected_bits = 1u | (p ? 0u : 14u);
            equal("s4 bit encoding", int(code), int(expected_bits));
            equal("s4 packed sign", signed_nibble(code), p ? 1 : -1);
            ++encoding_cases;
        }
    }
    const std::array<Signs, 8> patterns{{
        {0u, 0u, 0u, 0u}, {~0u, ~0u, ~0u, ~0u},
        {0xaaaaaaaau, 0x55555555u, 0xaaaaaaaau, 0x55555555u},
        {0x55555555u, 0xaaaaaaaau, 0x55555555u, 0xaaaaaaaau},
        {0u, ~0u, 0u, ~0u},
        {0x80000001u, 0x80000001u, 0x80000001u, 0x80000001u},
        {0x01234567u, 0x89abcdefu, 0xfedcba98u, 0x76543210u},
        {0xff00ff00u, 0x00ff00ffu, 0xf0f0f0f0u, 0x0f0f0f0fu}
    }};
    for (const auto& signs : patterns) {
        for (int code = 0; code < 256; ++code) {
            Activations constant{};
            constant.fill(signed_byte(code));
            check_full(signs, constant);
            Activations ramp{};
            for (int i = 0; i < K; ++i) ramp[i] = signed_byte((code + i) & 255);
            check_full(signs, ramp);
        }
    }
    for (int position = 0; position < K; ++position) {
        for (int code = 0; code < 256; ++code) {
            Activations q{};
            q[position] = signed_byte(code);
            Signs one{};
            one[position / 32] = 1u << (position % 32);
            check_full(one, q);
            for (auto& word : one) word = ~word;
            check_full(one, q);
        }
    }
    Activations extreme{};
    extreme.fill(-128);
    equal("positive sharp bound", projected(pack_weights(patterns[0]), pack_activations(extreme)), 16384);
    equal("negative sharp bound", projected(pack_weights(patterns[1]), pack_activations(extreme)), -16384);
    extreme.fill(127);
    equal("positive 127 endpoint", projected(pack_weights(patterns[1]), pack_activations(extreme)), 16256);
    equal("negative 127 endpoint", projected(pack_weights(patterns[0]), pack_activations(extreme)), -16256);
    std::cout << "PASS scalar_pairs=" << scalar_cases << " s4_packed_lane_encodings=" << encoding_cases
              << " K128_cases=" << full_cases << " sharp_bound=[-16384,16384]\n";
}
