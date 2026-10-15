#include "dspark_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef DSPARK_FIXTURE_IMPL
#define DSPARK_FIXTURE_IMPL "live-history"
#endif

using namespace h5::dspark;

static void cuda_ok(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA_ERROR operation=%s detail=%s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}
#define CUDA_OK(expr) cuda_ok((expr), #expr)

static void require(bool condition, const char* detail) {
    if (!condition) {
        std::fprintf(stderr, "FIXTURE_ERROR %s\n", detail);
        std::exit(1);
    }
}

template<class T> struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count;
    explicit DeviceBuffer(size_t n) : count(n) { CUDA_OK(cudaMalloc(&ptr, count * sizeof(T))); }
    ~DeviceBuffer() { CUDA_OK(cudaFree(ptr)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const std::vector<T>& data) {
        require(data.size() == count, "upload size mismatch");
        CUDA_OK(cudaMemcpy(ptr, data.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> data(count);
        CUDA_OK(cudaMemcpy(data.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return data;
    }
};

struct Seed {
    uint32_t state = 0x31415926;
    float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (float(state & 0xffff) / 32768.0f - 1.0f) * 0.7f;
    }
};

struct Spec {
    const char* name;
    int ring, window, anchor, context_begin, queries, layer;
    bool missing;
};

struct Fixture {
    KvGeometry g;
    Spec spec;
    std::vector<float> q;
    std::vector<__half> keys, values;
    std::vector<int32_t> tags, positions, slots, caches, block_ids;
    std::vector<AttentionBlock> blocks;
    std::vector<std::vector<int>> historical_caches;
    int missing_block = -1;
    size_t tag(int slot, int layer, int cache) const {
        return (size_t(slot) * g.layers + layer) * g.ring + cache;
    }
    size_t kv(int slot, int layer, int cache, int head, int d) const {
        return (tag(slot, layer, cache) * g.kv_heads + head) * g.head_dim + d;
    }
    int rows() const { return int(positions.size()); }
    explicit Fixture(Spec s) : g{2, 3, s.ring, 8, 2, 128}, spec(s) {
        const size_t tag_count = size_t(g.batch_slots) * g.layers * g.ring;
        tags.resize(tag_count);
        keys.assign(tag_count * g.kv_heads * g.head_dim, __float2half(2.0f));
        values.assign(keys.size(), __float2half(6.0f));
        Seed seed;
        for (int slot = 0; slot < g.batch_slots; ++slot) {
            const int anchor = s.anchor + slot * 3;
            const int begin = std::max(s.context_begin, std::max(0, anchor - s.window));
            blocks.push_back({slot, anchor, s.context_begin, rows(), s.queries});
            historical_caches.emplace_back();
            std::vector<bool> used(g.ring, false);
            for (int layer = 0; layer < g.layers; ++layer)
                for (int cache = 0; cache < g.ring; ++cache)
                    tags[tag(slot, layer, cache)] = anchor + 1000 + cache;
            auto populate = [&](int cache, int pos) {
                for (int layer = 0; layer < g.layers; ++layer) {
                    tags[tag(slot, layer, cache)] = pos;
                    for (int head = 0; head < g.kv_heads; ++head)
                        for (int d = 0; d < g.head_dim; ++d) {
                            const size_t i = kv(slot, layer, cache, head, d);
                            keys[i] = __float2half_rn(seed.next());
                            values[i] = __float2half_rn(seed.next() + float(layer) * 0.3f + float(slot) * 0.2f);
                        }
                }
                used[cache] = true;
            };
            for (int pos = begin; pos < anchor; ++pos) {
                const int cache = pos % g.ring;
                historical_caches.back().push_back(cache);
                populate(cache, pos);
            }
            for (int i = 0; i < s.queries; ++i) {
                int cache = -1;
                for (int candidate = 0; candidate < g.ring; ++candidate) {
                    bool natural_current = false;
                    for (int j = 0; j < s.queries; ++j)
                        natural_current |= candidate == (anchor + j) % g.ring;
                    if (!used[candidate] && !natural_current) { cache = candidate; break; }
                }
                require(cache >= 0, "no disjoint explicit current cache available");
                populate(cache, anchor + i);
                positions.push_back(anchor + i);
                slots.push_back(slot);
                caches.push_back(cache);
                block_ids.push_back(slot);
            }
            if (begin > 0 && !used[(begin - 1) % g.ring])
                populate((begin - 1) % g.ring, begin - 1);
        }
        q.resize(size_t(rows()) * g.q_heads * g.head_dim);
        for (float& v : q) v = seed.next();
        if (s.missing) {
            require(!historical_caches[0].empty(), "missing fixture requires history");
            const int cache = historical_caches[0][historical_caches[0].size() / 2];
            tags[tag(0, s.layer, cache)] += g.ring;
            missing_block = 0;
        }
    }

    std::vector<double> oracle() const {
        std::vector<double> result(q.size());
        for (int row = 0; row < rows(); ++row) {
            const int bid = block_ids[row];
            const AttentionBlock b = blocks[bid];
            if (bid == missing_block) {
                std::fill(result.begin() + size_t(row) * g.q_heads * g.head_dim,
                          result.begin() + size_t(row + 1) * g.q_heads * g.head_dim, NAN);
                continue;
            }
            std::vector<int> visible = historical_caches[bid];
            for (int i = 0; i < b.query_count; ++i) visible.push_back(caches[b.query_begin + i]);
            for (int qh = 0; qh < g.q_heads; ++qh) {
                const int kh = qh / (g.q_heads / g.kv_heads);
                const size_t qbase = (size_t(row) * g.q_heads + qh) * g.head_dim;
                std::vector<double> scores;
                double maximum = -INFINITY;
                for (int cache : visible) {
                    double score = 0;
                    for (int d = 0; d < g.head_dim; ++d)
                        score += double(q[qbase + d]) * double(__half2float(keys[kv(b.batch_slot, spec.layer, cache, kh, d)]));
                    score /= std::sqrt(double(g.head_dim));
                    scores.push_back(score);
                    maximum = std::max(maximum, score);
                }
                double denominator = 0;
                for (double& score : scores) { score = std::exp(score - maximum); denominator += score; }
                for (int d = 0; d < g.head_dim; ++d) {
                    double numerator = 0;
                    for (size_t i = 0; i < visible.size(); ++i)
                        numerator += scores[i] * double(__half2float(values[kv(b.batch_slot, spec.layer, visible[i], kh, d)]));
                    result[qbase + d] = numerator / denominator;
                }
            }
        }
        return result;
    }
};

struct DeviceFixture {
    DeviceBuffer<float> q, output;
    DeviceBuffer<__half> keys, values;
    DeviceBuffer<int32_t> tags, positions, slots, caches, block_ids;
    DeviceBuffer<AttentionBlock> blocks;
    DeviceBuffer<int> status;
    AttentionArgs args;
    explicit DeviceFixture(const Fixture& f)
        : q(f.q.size()), output(f.q.size()), keys(f.keys.size()), values(f.values.size()),
          tags(f.tags.size()), positions(f.positions.size()), slots(f.slots.size()),
          caches(f.caches.size()), block_ids(f.block_ids.size()), blocks(f.blocks.size()), status(1) {
        q.upload(f.q); keys.upload(f.keys); values.upload(f.values); tags.upload(f.tags);
        positions.upload(f.positions); slots.upload(f.slots); caches.upload(f.caches);
        block_ids.upload(f.block_ids); blocks.upload(f.blocks);
        CUDA_OK(cudaMemset(status.ptr, 0, sizeof(int)));
        CUDA_OK(cudaMemset(output.ptr, 0xff, output.count * sizeof(float)));
        args = {q.ptr, keys.ptr, values.ptr, tags.ptr, positions.ptr, slots.ptr, caches.ptr,
                block_ids.ptr, blocks.ptr, output.ptr, status.ptr, f.g, f.rows(), f.spec.layer,
                int(f.blocks.size()), f.spec.window, 1.0f / std::sqrt(float(f.g.head_dim))};
    }
};

static void check(const Fixture& f, const DeviceFixture& device, const char* phase) {
    const int expected_status = int(f.spec.missing ? DeviceError::missing_kv : DeviceError::success);
    const int actual_status = device.status.download()[0];
    if (actual_status != expected_status) {
        std::fprintf(stderr, "STATUS_MISMATCH case=%s phase=%s got=%d expected=%d\n",
                     f.spec.name, phase, actual_status, expected_status);
        std::exit(1);
    }
    const auto expected = f.oracle();
    const auto actual = device.output.download();
    double worst = 0;
    size_t nan_count = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(expected[i])) {
            require(std::isnan(actual[i]), "missing historical tag must produce NaN in every affected output");
            ++nan_count;
            continue;
        }
        const double error = std::abs(double(actual[i]) - expected[i]);
        worst = std::max(worst, error);
        if (!std::isfinite(actual[i]) || error > 2e-5 + 2e-5 * std::abs(expected[i])) {
            std::fprintf(stderr, "OUTPUT_MISMATCH case=%s phase=%s index=%zu got=%.9g expected=%.12g error=%.6g\n",
                         f.spec.name, phase, i, actual[i], expected[i], error);
            std::exit(1);
        }
    }
    std::printf("CORRECT impl=%s case=%s phase=%s layer=%d rows=%d ring=%d window=%d max_abs_error=%.8g nan_count=%zu status=%d\n",
                DSPARK_FIXTURE_IMPL, f.spec.name, phase, f.spec.layer, f.rows(), f.g.ring, f.spec.window,
                worst, nan_count, actual_status);
}

static void timing(const Fixture& f, DeviceFixture& device, cudaStream_t stream) {
    constexpr int warmup = 8, iterations = 64, samples = 5;
    for (int i = 0; i < warmup; ++i) CUDA_OK(launch_attention(device.args, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    cudaEvent_t start, end;
    CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&end));
    std::vector<float> microseconds;
    for (int sample = 0; sample < samples; ++sample) {
        CUDA_OK(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; ++i) CUDA_OK(launch_attention(device.args, stream));
        CUDA_OK(cudaEventRecord(end, stream));
        CUDA_OK(cudaEventSynchronize(end));
        float ms;
        CUDA_OK(cudaEventElapsedTime(&ms, start, end));
        microseconds.push_back(ms * 1000.0f / iterations);
    }
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(end));
    check(f, device, "post-timing");
    std::sort(microseconds.begin(), microseconds.end());
    const auto b = f.blocks[0];
    const int history = b.anchor_position - std::max(b.context_begin, std::max(0, b.anchor_position - f.spec.window));
    std::printf("TIMING impl=%s case=%s ring=%d history_slot0=%d window=%d queries_per_slot=%d slots=%d q_heads=%d kv_heads=%d head_dim=%d warmup=%d iterations=%d samples=%d min_us=%.6f median_us=%.6f max_us=%.6f\n",
                DSPARK_FIXTURE_IMPL, f.spec.name, f.g.ring, history, f.spec.window, f.spec.queries,
                f.g.batch_slots, f.g.q_heads, f.g.kv_heads, f.g.head_dim, warmup, iterations, samples,
                microseconds.front(), microseconds[samples / 2], microseconds.back());
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "--correctness") && std::strcmp(argv[1], "--timing"))) {
        std::fprintf(stderr, "usage: %s --correctness|--timing\n", argv[0]);
        return 2;
    }
    const bool measure = !std::strcmp(argv[1], "--timing");
    CUDA_OK(cudaSetDevice(0));
    cudaDeviceProp properties;
    CUDA_OK(cudaGetDeviceProperties(&properties, 0));
    std::printf("DEVICE impl=%s name=%s compute=%d.%d seed=0x31415926\n",
                DSPARK_FIXTURE_IMPL, properties.name, properties.major, properties.minor);
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const Spec cases[] = {
        {"ring8196-history48", 8196, 48, 8196 * 2 + 12, 0, 4, 0, false},
        {"ring8196-history176", 8196, 176, 8196 * 2 + 12, 0, 4, 2, false},
        {"wrap64-history48", 64, 48, 64 * 2 + 12, 0, 4, 2, false},
        {"wrap256-history176-block16", 256, 176, 256 * 2 + 12, 0, 16, 0, false},
        {"context-begin-clamp", 256, 176, 256 * 2 + 12, 256 * 2 + 12 - 9, 4, 1, false},
        {"anchor-zero", 64, 48, 0, 0, 4, 1, false},
        {"exact-tag-missing", 256, 176, 256 * 2 + 12, 0, 4, 2, true},
    };
    for (const Spec& s : cases) {
        Fixture fixture(s);
        DeviceFixture device(fixture);
        CUDA_OK(launch_attention(device.args, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        check(fixture, device, "initial");
        if (measure && (&s == &cases[0] || &s == &cases[1])) timing(fixture, device, stream);
    }
    CUDA_OK(cudaStreamDestroy(stream));
    std::printf("PASS impl=%s cases=%zu mode=%s\n", DSPARK_FIXTURE_IMPL, sizeof(cases) / sizeof(cases[0]), argv[1]);
    return 0;
}
