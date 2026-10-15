#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {
constexpr int K = 128;
constexpr std::array<int, 8> coefficient{1, 2, 4, 8, 16, 32, 64, -128};
using SignWords = std::array<std::uint32_t, 4>;
using Activations = std::array<std::int8_t, K>;
using ActivationPlanes = std::array<SignWords, 8>;

struct PackedActivations {
    ActivationPlanes planes{};
    int sum = 0;
};

PackedActivations transpose(const Activations& q) {
    PackedActivations result;
    for (int i = 0; i < K; ++i) {
        const auto byte = static_cast<std::uint8_t>(q[i]);
        result.sum += q[i];
        for (int p = 0; p < 8; ++p)
            result.planes[p][i / 32] |= std::uint32_t((byte >> p) & 1u) << (i % 32);
    }
    return result;
}

int packed_dot(const SignWords& w, const PackedActivations& a) {
    int selected = 0;
    for (int p = 0; p < 8; ++p) {
        int count = 0;
        for (int j = 0; j < 4; ++j)
            count += std::popcount(w[j] & a.planes[p][j]);
        selected += coefficient[p] * count;
    }
    return 2 * selected - a.sum;
}

constexpr std::uint32_t expand_nibble(std::uint32_t nibble) {
    return (nibble * 0x00204081u) & 0x01010101u;
}

// Integer emulation of signed DP4A; only the reference reads byte lanes individually.
int dp4a_reference(std::uint32_t a, const std::array<std::int8_t, 4>& q) {
    int result = 0;
    for (int i = 0; i < 4; ++i)
        result += int((a >> (8 * i)) & 255u) * int(q[i]);
    return result;
}

int nibble_dot(const SignWords& w, const Activations& q) {
    int selected = 0;
    int sum = 0;
    for (int i = 0; i < K; i += 4) {
        const auto nibble = (w[i / 32] >> (i % 32)) & 15u;
        const std::array<std::int8_t, 4> lanes{q[i], q[i + 1], q[i + 2], q[i + 3]};
        selected += dp4a_reference(expand_nibble(nibble), lanes);
        for (auto lane : lanes) sum += lane;
    }
    return 2 * selected - sum;
}

int dense_reference(const SignWords& w, const Activations& q) {
    int result = 0;
    for (int i = 0; i < K; ++i)
        result += (((w[i / 32] >> (i % 32)) & 1u) ? 1 : -1) * int(q[i]);
    return result;
}

[[noreturn]] void fail(const char* check, int first, int second) {
    std::cerr << "FAIL " << check << ": " << first << " != " << second << '\n';
    std::exit(1);
}

void equal(const char* check, int first, int second) {
    if (first != second) fail(check, first, second);
}

std::uint64_t full_cases = 0;
void check_full(const SignWords& w, const Activations& q) {
    const auto a = transpose(q);
    const int exact = dense_reference(w, q);
    equal("K128 bitserial", packed_dot(w, a), exact);
    equal("K128 nibble DP4A affine", nibble_dot(w, q), exact);
    if (exact < -16384 || exact > 16384) fail("K128 bound", exact, 16384);
    ++full_cases;
}

std::int8_t signed_byte(int code) {
    return static_cast<std::int8_t>(code < 128 ? code : code - 256);
}
}

int main() {
    static_assert(std::numeric_limits<std::uint32_t>::digits == 32);
    static_assert(std::numeric_limits<std::int8_t>::min() == -128);
    static_assert(std::numeric_limits<std::int8_t>::max() == 127);
    int scalar_cases = 0;
    for (int w = 0; w < 2; ++w) {
        for (int code = 0; code < 256; ++code) {
            const int q = signed_byte(code);
            int decoded = 0, projected = 0;
            for (int p = 0; p < 8; ++p) {
                const int a = (code >> p) & 1;
                decoded += coefficient[p] * a;
                projected += coefficient[p] * (2 * (w & a) - a);
            }
            equal("signed-byte decode", decoded, q);
            equal("scalar bitserial", projected, (2 * w - 1) * q);
            equal("scalar affine", 2 * w * q - q, (2 * w - 1) * q);
            ++scalar_cases;
        }
    }
    int expansion_cases = 0, dp4a_lane_cases = 0;
    for (std::uint32_t nibble = 0; nibble < 16; ++nibble) {
        const auto expanded = expand_nibble(nibble);
        std::uint32_t expected = 0;
        for (int lane = 0; lane < 4; ++lane)
            expected |= ((nibble >> lane) & 1u) << (lane * 8);
        equal("nibble expansion", int(expanded), int(expected));
        ++expansion_cases;
        // Exhaust all signed bytes in every independent lane. DP4A is a lane sum.
        for (int lane = 0; lane < 4; ++lane) {
            for (int code = 0; code < 256; ++code) {
                std::array<std::int8_t, 4> q{};
                q[lane] = signed_byte(code);
                const int selected = int((nibble >> lane) & 1u) * int(q[lane]);
                equal("DP4A lane", dp4a_reference(expanded, q), selected);
                ++dp4a_lane_cases;
            }
        }
    }
    const std::array<SignWords, 8> patterns{{
        {0u, 0u, 0u, 0u},
        {~0u, ~0u, ~0u, ~0u},
        {0xaaaaaaaau, 0x55555555u, 0xaaaaaaaau, 0x55555555u},
        {0x55555555u, 0xaaaaaaaau, 0x55555555u, 0xaaaaaaaau},
        {0u, ~0u, 0u, ~0u},
        {0x80000001u, 0x80000001u, 0x80000001u, 0x80000001u},
        {0x01234567u, 0x89abcdefu, 0xfedcba98u, 0x76543210u},
        {0xff00ff00u, 0x00ff00ffu, 0xf0f0f0f0u, 0x0f0f0f0fu}
    }};
    for (const auto& w : patterns) {
        for (int code = 0; code < 256; ++code) {
            Activations constant{};
            constant.fill(signed_byte(code));
            check_full(w, constant);
            Activations ramp{};
            for (int i = 0; i < K; ++i) ramp[i] = signed_byte((code + i) & 255);
            check_full(w, ramp);
        }
    }
    for (int position = 0; position < K; ++position) {
        for (int code = 0; code < 256; ++code) {
            Activations q{};
            q[position] = signed_byte(code);
            SignWords one{};
            one[position / 32] = 1u << (position % 32);
            check_full(one, q);
            for (auto& word : one) word = ~word;
            check_full(one, q);
        }
    }
    Activations extreme{};
    extreme.fill(-128);
    equal("positive sharp bound", packed_dot(patterns[0], transpose(extreme)), 16384);
    equal("negative sharp bound", packed_dot(patterns[1], transpose(extreme)), -16384);
    extreme.fill(127);
    equal("positive 127 endpoint", packed_dot(patterns[1], transpose(extreme)), 16256);
    equal("negative 127 endpoint", packed_dot(patterns[0], transpose(extreme)), -16256);
    std::cout << "PASS scalar_cases=" << scalar_cases
              << " nibble_expansion_cases=" << expansion_cases
              << " dp4a_lane_cases=" << dp4a_lane_cases
              << " K128_cases=" << full_cases
              << " sharp_bound=[-16384,16384]\n";
}
