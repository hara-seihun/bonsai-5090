#include "dspark_native.cuh"
#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace {
constexpr int D = 5120, C = 4, V = 248320, Q = 4, K = 5 * D;
constexpr int MASK = V - 1, RANK = 256, WARMUPS = 2, RUNS = 8;
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

enum class ErrorCode {
    usage, fixture, reference_load, reference_context, geometry, reference_decode,
    markov, environment, native_api, nonfinite, numerical_norm, cuda, bf16_fidelity
};
const char* error_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::usage: return "usage";
        case ErrorCode::fixture: return "fixture";
        case ErrorCode::reference_load: return "reference_load";
        case ErrorCode::reference_context: return "reference_context";
        case ErrorCode::geometry: return "geometry";
        case ErrorCode::reference_decode: return "reference_decode";
        case ErrorCode::markov: return "markov";
        case ErrorCode::environment: return "environment";
        case ErrorCode::native_api: return "native_api";
        case ErrorCode::nonfinite: return "nonfinite";
        case ErrorCode::numerical_norm: return "numerical_norm";
        case ErrorCode::cuda: return "cuda";
        case ErrorCode::bf16_fidelity: return "bf16_fidelity";
    }
    std::abort();
}
struct Failure { ErrorCode code; std::string stage, mode, detail; };
struct Done {};
using Result = std::variant<Done, Failure>;
Failure cuda_failure(const char* stage, const char* mode, cudaError_t error) {
    return {ErrorCode::cuda, stage, mode, cudaGetErrorString(error)};
}
std::string json_string(const std::string& text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) {
            char escaped[7]; std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            out += escaped;
        } else out += char(c);
    }
    return out + '"';
}
struct FileCloser { void operator()(FILE* file) const { std::fclose(file); } };
struct Reference {
    bool backend = false, has_batch = false;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_batch batch{};
    ~Reference() {
        if (has_batch) llama_batch_free(batch);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        if (backend) llama_backend_free();
    }
};
struct Device {
    cudaStream_t stream = nullptr;
    float* taps = nullptr;
    ~Device() {
        if (taps) cudaFree(taps);
        if (stream) cudaStreamDestroy(stream);
    }
};
struct Memory { size_t free = 0, total = 0; };
void print_memory(const char* name, const Memory& memory) {
    std::printf("\"%s\":{\"free_bytes\":%zu,\"total_bytes\":%zu}",
                name, memory.free, memory.total);
}
void print_ids(const char* name, const std::array<int, Q>& ids) {
    std::printf("\"%s\":[%d,%d,%d,%d]", name, ids[0], ids[1], ids[2], ids[3]);
}

Result benchmark(int argc, char** argv) {
    const char* stage = "fixture";
    const char* mode = "reference";
    // NativeDspark and the fork are foreign throwing APIs; classify at this boundary.
    try {
        if (argc != 2 && argc != 4)
            return Failure{ErrorCode::usage, stage, mode,
                "usage: dspark-fast-contract MODEL [TAPS_F32.bin ANCHOR_ID]; taps are exactly four rows of five concatenated 5120-wide post-block residuals"};
        int anchor = 198;
        std::vector<float> taps(size_t(C) * K);
        if (argc == 4) {
            char* end = nullptr;
            errno = 0;
            long value = std::strtol(argv[3], &end, 10);
            if (errno || end == argv[3] || *end || value < 0 || value >= V)
                return Failure{ErrorCode::fixture, stage, mode, "invalid anchor"};
            anchor = int(value);
            std::unique_ptr<FILE, FileCloser> file(std::fopen(argv[2], "rb"));
            if (!file) return Failure{ErrorCode::fixture, stage, mode, "cannot open tap fixture"};
            if (std::fread(taps.data(), sizeof(float), taps.size(), file.get()) != taps.size() ||
                std::fgetc(file.get()) != EOF || std::ferror(file.get()))
                return Failure{ErrorCode::fixture, stage, mode, "tap fixture must have exactly four rows"};
        } else {
            for (size_t i = 0; i < taps.size(); ++i)
                taps[i] = .1f * std::sin(float(i % 104729) * .017f);
        }
        for (float x : taps) if (!std::isfinite(x))
            return Failure{ErrorCode::nonfinite, stage, mode, "nonfinite tap fixture"};

        Reference reference;
        stage = "reference_load";
        auto began = Clock::now();
        llama_backend_init(); reference.backend = true;
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 999;
        reference.model = llama_model_load_from_file(argv[1], mp);
        if (!reference.model) return Failure{ErrorCode::reference_load, stage, mode, "fork model load failed"};
        llama_dspark_meta meta;
        if (!llama_model_dspark_get_meta(reference.model, &meta) || meta.n_embd != D ||
            meta.n_vocab != V || meta.n_capture != 5 || meta.n_embd_cap != K ||
            meta.block_size != Q || meta.mask_token_id != MASK || meta.markov_rank != RANK ||
            meta.graph_corrected || llama_model_n_layer(reference.model) != 6)
            return Failure{ErrorCode::geometry, stage, mode, "reference DSpark metadata must match raw base-logit/Markov contract"};
        const double reference_load_ms = elapsed_ms(began);
        stage = "reference_context";
        auto cp = llama_context_default_params();
        cp.n_ctx = 64; cp.n_batch = 32; cp.n_ubatch = 32; cp.n_seq_max = 1; cp.embeddings = false;
        reference.ctx = llama_init_from_model(reference.model, cp);
        if (!reference.ctx) return Failure{ErrorCode::reference_context, stage, mode, "fork context creation failed"};
        llama_set_causal_attn(reference.ctx, false);
        int32_t pos[C] = {0, 1, 2, 3};
        llama_set_dspark_ctx(reference.ctx, taps.data(), C, K, pos);
        reference.batch = llama_batch_init(C + Q, 0, 1); reference.has_batch = true;
        reference.batch.n_tokens = C + Q;
        for (int r = 0; r < C + Q; ++r) {
            reference.batch.token[r] = r < C ? 0 : r == C ? anchor : MASK;
            reference.batch.pos[r] = r;
            reference.batch.n_seq_id[r] = 1; reference.batch.seq_id[r][0] = 0;
            reference.batch.logits[r] = r >= C;
        }
        stage = "reference_decode"; began = Clock::now();
        if (llama_decode(reference.ctx, reference.batch))
            return Failure{ErrorCode::reference_decode, stage, mode, "fork decode failed"};
        std::vector<float> ref(size_t(Q) * V);
        for (int r = 0; r < Q; ++r) {
            const float* logits = llama_get_logits_ith(reference.ctx, C + r);
            if (!logits) return Failure{ErrorCode::reference_decode, stage, mode, "reference logits missing"};
            std::copy_n(logits, V, ref.data() + size_t(r) * V);
        }
        const double reference_decode_ms = elapsed_ms(began);
        stage = "reference_markov";
        std::vector<float> markov_a, markov_b;
        if (!llama_model_dspark_get_markov(reference.model, markov_a, markov_b))
            return Failure{ErrorCode::markov, stage, mode, "fork Markov weights missing"};
        if (markov_a.size() != size_t(V) * RANK || markov_b.size() != size_t(V) * RANK)
            return Failure{ErrorCode::geometry, stage, mode, "reference Markov shape"};
        std::printf("{\"stage\":\"reference\",\"fixture\":\"%s\",\"anchor_id\":%d,"
                    "\"load_ms\":%.9g,\"decode_and_read_ms\":%.9g,\"loads\":1}\n",
                    argc == 4 ? "file" : "synthetic", anchor, reference_load_ms, reference_decode_ms);
        std::fflush(stdout);

        Device device;
        stage = "tap_upload";
        auto ce = cudaStreamCreate(&device.stream);
        if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
        ce = cudaMalloc(&device.taps, taps.size() * sizeof(float));
        if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
        ce = cudaMemcpyAsync(device.taps, taps.data(), taps.size() * sizeof(float), cudaMemcpyHostToDevice, device.stream);
        if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
        ce = cudaStreamSynchronize(device.stream);
        if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);

        std::vector<int> domain;
        std::vector<float> selected_ref;
        std::array<int, Q> ref_ids{};
        bool bf16_fidelity = false;
        const std::array<h5::DsparkPrecision, 4> modes = {
            h5::DsparkPrecision::BF16, h5::DsparkPrecision::NVFP4A1,
            h5::DsparkPrecision::NVFP4A2, h5::DsparkPrecision::NVFP4HeadA2};
        for (auto precision : modes) {
            mode = h5::dspark_precision_name(precision);
            stage = "precision_selection";
            if (setenv("H5_DSPARK_PRECISION", mode, 1) != 0)
                return Failure{ErrorCode::environment, stage, mode, "setenv H5_DSPARK_PRECISION failed"};
            auto parsed = h5::read_dspark_precision(std::getenv("H5_DSPARK_PRECISION"));
            if (!std::holds_alternative<h5::DsparkPrecision>(parsed) || std::get<h5::DsparkPrecision>(parsed) != precision)
                return Failure{ErrorCode::geometry, stage, mode, "precision parser mismatch"};
            Memory before, loaded, executed, released;
            stage = "memory_before_load";
            ce = cudaMemGetInfo(&before.free, &before.total);
            if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
            double load_ms, rms, cosine, maxerr = 0, warmup_ms;
            int matched = 0, output_vocab = 0;
            std::array<int, Q> ids{};
            std::array<float, Q> confidence{};
            std::array<double, RUNS> times{};
            bool control_ok = false;
            {
                stage = "native_load"; began = Clock::now();
                h5::NativeDspark native(argv[1], 1, 64, 32, Q, device.stream);
                ce = cudaStreamSynchronize(device.stream);
                if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
                load_ms = elapsed_ms(began);
                stage = "memory_after_load";
                ce = cudaMemGetInfo(&loaded.free, &loaded.total);
                if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
                stage = "native_geometry";
                const auto& output_ids = native.output_ids();
                output_vocab = native.output_vocab();
                if (native.projection_precision() != precision || output_vocab < 1 || output_vocab > V ||
                    output_ids.size() != size_t(output_vocab) ||
                    native.capture_layers() != std::vector<int>({1, 16, 31, 46, 61}))
                    return Failure{ErrorCode::geometry, stage, mode, "native precision, taps or candidate shape mismatch"};
                for (size_t i = 0; i < output_ids.size(); ++i)
                    if (output_ids[i] < 0 || output_ids[i] >= V || (i && output_ids[i - 1] >= output_ids[i]))
                        return Failure{ErrorCode::geometry, stage, mode, "candidate IDs must be in-range, unique, sorted"};
                if (domain.empty()) {
                    domain = output_ids;
                    selected_ref.resize(size_t(Q) * output_vocab);
                    stage = "reference_domain_markov";
                    int previous = anchor;
                    for (int r = 0; r < Q; ++r) {
                        double best = -INFINITY; int winner = -1;
                        for (int i = 0; i < output_vocab; ++i) {
                            int token = domain[i];
                            float base = ref[size_t(r) * V + token];
                            if (!std::isfinite(base)) return Failure{ErrorCode::nonfinite, stage, mode, "nonfinite reference base logit"};
                            selected_ref[size_t(r) * output_vocab + i] = base;
                            if (token == MASK) continue;
                            double dot = 0;
                            for (int k = 0; k < RANK; ++k)
                                dot += double(markov_a[size_t(previous) * RANK + k]) * markov_b[size_t(token) * RANK + k];
                            double logit = double(base) + dot;
                            if (!std::isfinite(logit)) return Failure{ErrorCode::nonfinite, stage, mode, "nonfinite reference Markov logit"};
                            if (logit > best || (logit == best && token < winner)) { best = logit; winner = token; }
                        }
                        if (winner < 0) return Failure{ErrorCode::geometry, stage, mode, "reference domain has no non-mask candidate"};
                        previous = ref_ids[r] = winner;
                    }
                } else if (domain != output_ids)
                    return Failure{ErrorCode::geometry, stage, mode, "candidate domain changed between modes"};

                int slot = 0, count = C, row = 0, position = 0, start = C;
                auto run = [&]() {
                    native.run(device.taps, C, 1, &slot, &count, &row, &position, &anchor, &start, ids.data());
                };
                auto valid_output = [&]() {
                    if (native.query_rows() != Q || !native.confidence()) return false;
                    for (int r = 0; r < Q; ++r)
                        if (ids[r] == MASK || !std::binary_search(domain.begin(), domain.end(), ids[r])) return false;
                    return true;
                };
                stage = "native_warmups"; began = Clock::now();
                for (int i = 0; i < WARMUPS; ++i) {
                    native.reset(slot); run();
                    if (!valid_output()) return Failure{ErrorCode::geometry, stage, mode, "warmup output geometry"};
                }
                warmup_ms = elapsed_ms(began);
                for (int i = 0; i < RUNS; ++i) {
                    stage = "native_reset_outside_timing";
                    native.reset(slot);
                    ce = cudaStreamSynchronize(device.stream);
                    if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
                    stage = "native_timed_run"; began = Clock::now();
                    run(); times[i] = elapsed_ms(began);
                    if (!valid_output()) return Failure{ErrorCode::geometry, stage, mode, "timed output geometry"};
                    for (int r = 0; r < Q; ++r) {
                        float value = native.confidence()[r];
                        if (!std::isfinite(value)) return Failure{ErrorCode::nonfinite, stage, mode, "nonfinite learned confidence"};
                        if (value < 0 || value > 1) return Failure{ErrorCode::geometry, stage, mode, "learned confidence outside [0,1]"};
                        confidence[r] = value;
                    }
                }
                stage = "native_base_logits";
                std::vector<float> got(selected_ref.size());
                ce = native.read_base_logits(got.data(), got.size());
                if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
                double diff = 0, ref2 = 0, got2 = 0, dot = 0;
                for (size_t i = 0; i < got.size(); ++i) {
                    if (!std::isfinite(got[i])) return Failure{ErrorCode::nonfinite, stage, mode, "nonfinite native base logit"};
                    double e = double(got[i]) - selected_ref[i];
                    diff += e * e; ref2 += double(selected_ref[i]) * selected_ref[i];
                    got2 += double(got[i]) * got[i]; dot += double(selected_ref[i]) * got[i];
                    maxerr = std::max(maxerr, std::abs(e));
                }
                if (!(ref2 > 0 && got2 > 0)) return Failure{ErrorCode::numerical_norm, stage, mode, "zero logit norm"};
                rms = std::sqrt(diff / ref2); cosine = dot / std::sqrt(ref2 * got2);
                for (int r = 0; r < Q; ++r) matched += ids[r] == ref_ids[r];
                control_ok = matched == Q && rms < .01 && cosine > .9999;
                if (precision == h5::DsparkPrecision::BF16) bf16_fidelity = control_ok;
                stage = "memory_after_execution";
                ce = cudaMemGetInfo(&executed.free, &executed.total);
                if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
                stage = "native_release";
            }
            ce = cudaDeviceSynchronize();
            if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
            ce = cudaMemGetInfo(&released.free, &released.total);
            if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
            auto sorted = times; std::sort(sorted.begin(), sorted.end());
            double mean = 0; for (double t : times) mean += t / RUNS;
            std::printf("{\"stage\":\"mode_complete\",\"precision\":\"%s\",\"fixture\":\"%s\","
                        "\"query_rows\":%d,\"ingest_count\":%d,\"output_vocab\":%d,\"proposals_matched\":%d,"
                        "\"base_relative_rms\":%.17g,\"base_cosine\":%.17g,\"base_max_error\":%.17g,",
                        mode, argc == 4 ? "file" : "synthetic", Q, C, output_vocab, matched, rms, cosine, maxerr);
            print_ids("reference_ids", ref_ids); std::printf(","); print_ids("native_ids", ids);
            std::printf(",\"learned_confidence\":[%.9g,%.9g,%.9g,%.9g],\"quality_policy\":\"%s\",\"bf16_control_pass\":",
                        confidence[0], confidence[1], confidence[2], confidence[3],
                        precision == h5::DsparkPrecision::BF16 ? "strict_control" : "numerical_report_only");
            if (precision == h5::DsparkPrecision::BF16) std::printf("%s", control_ok ? "true" : "false");
            else std::printf("null");
            std::printf(",\"load_ms\":%.9g,\"warmup_runs\":%d,\"warmup_ms\":%.9g,"
                        "\"timing_scope\":\"NativeDspark::run including count4 ingest and host outputs; reset excluded\","
                        "\"run_count\":%d,\"run_wall_ms\":[", load_ms, WARMUPS, warmup_ms, RUNS);
            for (int i = 0; i < RUNS; ++i) std::printf("%s%.9g", i ? "," : "", times[i]);
            std::printf("],\"run_mean_ms\":%.9g,\"run_median_ms\":%.9g,\"run_min_ms\":%.9g,\"run_max_ms\":%.9g,",
                        mean, (sorted[RUNS / 2 - 1] + sorted[RUNS / 2]) / 2, sorted.front(), sorted.back());
            print_memory("device_before_load", before); std::printf(",");
            print_memory("device_after_load", loaded); std::printf(",");
            print_memory("device_after_execution", executed); std::printf(",");
            print_memory("device_after_release", released); std::printf("}\n");
            std::fflush(stdout);
        }
        stage = "cleanup"; mode = "all";
        ce = cudaFree(device.taps);
        if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
        device.taps = nullptr;
        ce = cudaStreamDestroy(device.stream);
        if (ce != cudaSuccess) return cuda_failure(stage, mode, ce);
        device.stream = nullptr;
        if (!bf16_fidelity) return Failure{ErrorCode::bf16_fidelity, "control_quality", "bf16",
            "BF16 requires 4/4 proposals, base_relative_rms < .01 and base_cosine > .9999; all quantized modes were reported"};
        std::printf("{\"stage\":\"comparison_complete\",\"modes_reported\":4,\"bf16_control_pass\":true,\"quantized_quality\":\"numerical_report_only\"}\n");
        return Done{};
    } catch (const std::exception& error) {
        return Failure{ErrorCode::native_api, stage, mode, error.what()};
    }
}
}

int main(int argc, char** argv) {
    Result result = benchmark(argc, argv);
    if (auto* failure = std::get_if<Failure>(&result)) {
        std::fprintf(stderr, "DsparkFastContractError[%s] stage=%s mode=%s: %s\n",
                     error_name(failure->code), failure->stage.c_str(), failure->mode.c_str(), failure->detail.c_str());
        std::printf("{\"stage\":\"error\",\"error_code\":%s,\"failed_stage\":%s,\"precision\":%s,\"detail\":%s}\n",
                    json_string(error_name(failure->code)).c_str(), json_string(failure->stage).c_str(),
                    json_string(failure->mode).c_str(), json_string(failure->detail).c_str());
        return failure->code == ErrorCode::usage ? 1 : 2;
    }
    return 0;
}
