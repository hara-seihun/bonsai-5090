#include "../../src/dspark_quant.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using h5::dspark::QuantRowStatus;
constexpr int N = 64, K = 256, NG = N / 32;

static float bf(uint16_t bits) {
    uint32_t raw = uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &raw, 4);
    return value;
}
static float e4(uint8_t bits) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3)));
}
static unsigned code(float value) {
    float a = std::fabs(value);
    unsigned c = a < .25f ? 0 : a < .75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3 :
                 a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.f ? 6 : 7;
    return c | (value < 0 && c ? 8u : 0u);
}
struct Reference {
    std::vector<uint8_t> w, s, nib;
    std::vector<float> rs;
    std::vector<QuantRowStatus> status;
    Reference() : w(N*K/2), s(N*K/16), nib(N*K), rs(N), status(N) {}
};

// Forward layout copied from the existing engine's consumer contract. The GPU
// uses the inverse map; this oracle deliberately does not call that map.
static Reference reference(const std::vector<uint16_t>& input) {
    Reference out;
    for (int r = 0; r < N; ++r) {
        float maximum = 0;
        bool invalid = false;
        for (int k = 0; k < K; ++k) {
            uint16_t bits = input[r*K+k];
            invalid |= (bits & 0x7f80u) == 0x7f80u;
            if (std::isfinite(bf(bits))) maximum = std::max(maximum, std::fabs(bf(bits)));
        }
        out.rs[r] = invalid ? 0 : std::max(maximum / 2688.f, 1e-30f);
        out.status[r] = invalid ? QuantRowStatus::non_finite_input : QuantRowStatus::ready;
        if (invalid) continue;
        for (int b = 0; b < K / 16; ++b) {
            float bm = 0;
            for (int i = 0; i < 16; ++i) bm = std::max(bm, std::fabs(bf(input[r*K+b*16+i])));
            float target = bm / 6.f / out.rs[r];
            uint8_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
            if (e4(sb) < target && sb < 0x7e) ++sb;
            out.s[((b/8)*NG+r/32)*256+(r%32)*8+b%8] = sb;
            float scale = e4(sb) * out.rs[r];
            for (int i = 0; i < 16; ++i)
                out.nib[r*K+b*16+i] = scale > 0 ? code(bf(input[r*K+b*16+i]) / scale) : 0;
        }
    }
    for (int kb = 0; kb < K/128; ++kb) for (int g = 0; g < NG; ++g) {
        uint8_t* chunk = out.w.data() + (kb*NG+g)*2048;
        for (int l = 0; l < 32; ++l) for (int w = 0; w < 16; ++w) {
            int gid = l >> 2, tq = l & 3;
            int m = w >> 3, h = (w >> 2) & 1, j = w & 3;
            int r = g*32 + 16*m + 8*(j&1) + gid;
            int k0 = kb*128 + 64*h + 32*(j>>1) + 8*tq;
            uint32_t word = 0;
            for (int i = 0; i < 8; ++i) word |= uint32_t(out.nib[r*K+k0+i]) << (i*4);
            std::memcpy(chunk+(w>>2)*512+l*16+(w&3)*4, &word, 4);
        }
    }
    return out;
}

static bool inverse_layout(const Reference& ref) {
    std::vector<uint8_t> inverse(N*K/2);
    std::vector<unsigned> seen(N*K/8);
    for (int r = 0; r < N; ++r) for (int k = 0; k < K; k += 8) {
        int rr = r%32, kk = k%128;
        int gid = rr&7, tq = (kk>>3)&3;
        int j = ((kk>>5)&1)*2+((rr>>3)&1);
        int w = (rr>>4)*8+(kk>>6)*4+j;
        size_t offset = ((k/128)*NG+r/32)*512+(w>>2)*128+(gid*4+tq)*4+(w&3);
        ++seen[offset];
        uint32_t word = 0;
        for (int i = 0; i < 8; ++i) word |= uint32_t(ref.nib[r*K+k+i]) << (i*4);
        std::memcpy(inverse.data()+offset*4, &word, 4);
    }
    return inverse == ref.w && std::all_of(seen.begin(), seen.end(), [](unsigned n) { return n == 1; });
}

static bool check(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    return false;
}
#define CUDA(expr) do { if (!check((expr), #expr)) return 1; } while (0)

int main(int argc, char** argv) {
    bool cpu_only = argc == 2 && std::strcmp(argv[1], "--cpu-only") == 0;
    if (argc != 1 && !cpu_only) { std::fprintf(stderr, "usage: parity [--cpu-only]\n"); return 2; }
    std::vector<std::vector<uint16_t>> cases(4, std::vector<uint16_t>(N*K));
    uint32_t rng = 12;
    for (int i = 0; i < N*K; ++i) {
        rng = rng*1664525u+1013904223u;
        cases[0][i] = uint16_t((rng>>16)&0x807fu) | uint16_t((119+(rng%17))<<7);
        cases[1][i] = 0;
        cases[2][i] = uint16_t(rng>>16);
        if ((cases[2][i]&0x7f80u) == 0x7f80u) cases[2][i] ^= 0x0080u;
    }
    for (int k = 0; k < K; ++k) {
        cases[0][K+k] |= 0x8000u;
        cases[2][k] = (k&1) ? 0x8001 : 0x0001;
        cases[2][K+k] = (k&1) ? 0xff7f : 0x7f7f;
        cases[2][2*K+k] = (k&1) ? 0x8000 : 0;
    }
    cases[3] = cases[0];
    cases[3][0] = 0x7f80;
    cases[3][K+17] = 0xff80;
    cases[3][2*K+255] = 0x7fc0;
    cases[3][N*K-1] = 0x7fff;

    std::vector<Reference> refs;
    for (const auto& input : cases) {
        refs.push_back(reference(input));
        if (!inverse_layout(refs.back())) { std::fprintf(stderr, "CPU layout mismatch\n"); return 1; }
    }
    if (code(.25f) != 1 || code(.75f) != 2 || code(1.25f) != 3 ||
        code(1.75f) != 4 || code(2.5f) != 5 || code(3.5f) != 6 || code(5.f) != 7 ||
        code(-.25f) != 9 || code(-0.f) != 0) return 1;
    if (cpu_only) { std::puts("CPU forward/inverse WF1 parity PASS: N64 K256, four cases, threshold ties"); return 0; }

    uint16_t* input;
    uint8_t *w, *s;
    float* rs;
    QuantRowStatus* status;
    CUDA(cudaMalloc(&input, N*K*2));
    CUDA(cudaMalloc(&w, N*K/2));
    CUDA(cudaMalloc(&s, N*K/16));
    CUDA(cudaMalloc(&rs, N*sizeof(float)));
    CUDA(cudaMalloc(&status, N*sizeof(QuantRowStatus)));
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CUDA(h5::dspark::launch_prepare_nvfp4(input, N, K, w, s, rs, status, stream));
    cudaGraph_t graph;
    CUDA(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    CUDA(cudaGraphInstantiate(&exec, graph, 0));
    for (size_t c = 0; c < cases.size(); ++c) {
        CUDA(cudaMemcpyAsync(input, cases[c].data(), N*K*2, cudaMemcpyHostToDevice, stream));
        CUDA(cudaGraphLaunch(exec, stream));
        Reference got;
        CUDA(cudaMemcpyAsync(got.w.data(), w, N*K/2, cudaMemcpyDeviceToHost, stream));
        CUDA(cudaMemcpyAsync(got.s.data(), s, N*K/16, cudaMemcpyDeviceToHost, stream));
        CUDA(cudaMemcpyAsync(got.rs.data(), rs, N*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA(cudaMemcpyAsync(got.status.data(), status, N*sizeof(QuantRowStatus), cudaMemcpyDeviceToHost, stream));
        CUDA(cudaStreamSynchronize(stream));
        const auto& ref = refs[c];
        if (got.w != ref.w || got.s != ref.s || got.status != ref.status ||
            std::memcmp(got.rs.data(), ref.rs.data(), N*sizeof(float))) {
            for (size_t i = 0; i < got.w.size(); ++i) if (got.w[i] != ref.w[i]) { std::fprintf(stderr, "case %zu W[%zu] GPU=%u CPU=%u\n", c, i, got.w[i], ref.w[i]); break; }
            for (size_t i = 0; i < got.s.size(); ++i) if (got.s[i] != ref.s[i]) { std::fprintf(stderr, "case %zu S[%zu] GPU=%u CPU=%u\n", c, i, got.s[i], ref.s[i]); break; }
            for (int r = 0; r < N; ++r) if (got.status[r] != ref.status[r] || std::memcmp(&got.rs[r], &ref.rs[r], 4))
                std::fprintf(stderr, "case %zu row %d GPU=(%.9g,%u) CPU=(%.9g,%u)\n", c, r, got.rs[r], unsigned(got.status[r]), ref.rs[r], unsigned(ref.status[r]));
            return 1;
        }
    }
    CUDA(cudaGraphExecDestroy(exec));
    CUDA(cudaGraphDestroy(graph));
    CUDA(cudaFree(input)); CUDA(cudaFree(w)); CUDA(cudaFree(s)); CUDA(cudaFree(rs)); CUDA(cudaFree(status));
    CUDA(cudaStreamDestroy(stream));
    std::puts("GPU/CPU bitwise WF1 parity + graph replay PASS: generic, zero, all-finite BF16 ranges, nonfinite rows");
    return 0;
}
