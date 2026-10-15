#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc != 5) { std::fprintf(stderr, "refbench MODEL REF.bin STEPS REPEATS\n"); return 1; }
    int hdr[3];
    FILE* f = std::fopen(argv[2], "rb");
    if (!f || std::fread(hdr, sizeof(int), 3, f) != 3) { std::fprintf(stderr, "REFERENCE_HEADER_INVALID\n"); return 1; }
    const int steps = std::atoi(argv[3]), reps = std::atoi(argv[4]);
    const int nt = hdr[0], np = hdr[1], nv = hdr[2];
    if (np < 1 || nt <= np || nt > 65536 || nv != 248320 || steps < 1 || steps > nt - np || reps < 1 || reps > 100) {
        std::fprintf(stderr, "REFERENCE_GEOMETRY_INVALID\n"); return 1;
    }
    std::vector<llama_token> toks(nt);
    if (std::fread(toks.data(), sizeof(int), nt, f) != (size_t)nt) { std::fprintf(stderr, "REFERENCE_TOKENS_TRUNCATED\n"); return 1; }
    std::fclose(f);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "MODEL_LOAD_FAILED\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = nt + 64; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "CONTEXT_LOAD_FAILED\n"); return 1; }
    volatile int observed = 0;
    std::printf("{\"mode\":\"reference_decodebench\",\"prefix_tokens\":%d,\"decode_tokens\":%d,\"seconds\":[", np, steps);
    for (int r = -1; r < reps; r++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int c = 0; c < np; c += 512) {
            auto b = llama_batch_get_one(toks.data() + c, std::min(512, np - c));
            if (llama_decode(ctx, b)) { std::fprintf(stderr, "PREFILL_FAILED\n"); return 1; }
        }
        llama_synchronize(ctx);
        const double t0 = seconds();
        for (int i = 0; i < steps; i++) {
            auto one = llama_batch_get_one(toks.data() + np + i, 1);
            if (llama_decode(ctx, one)) { std::fprintf(stderr, "DECODE_FAILED\n"); return 1; }
            const float* logits = llama_get_logits_ith(ctx, -1);
            if (!logits) { std::fprintf(stderr, "LOGITS_UNSET\n"); return 1; }
            observed = int(std::max_element(logits, logits + nv) - logits);
        }
        const double elapsed = seconds() - t0;
        if (r >= 0) std::printf("%s%.12g", r ? "," : "", elapsed);
    }
    std::printf("]}\n");
    std::fprintf(stderr, "final argmax %d\n", observed);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
