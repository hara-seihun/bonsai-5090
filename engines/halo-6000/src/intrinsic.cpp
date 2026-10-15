#include "intrinsic.hpp"
#include "nlohmann/json.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace halo6000 {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int Chunks = 16, Context = 512, First = 256, Last = 510, Rows = 255;
constexpr uint64_t OutputBytes = uint64_t(Chunks) * Rows * Vocab * sizeof(float);
constexpr const char * TokenHash = "aa910a0346ea2137c7c2f6bfdcb45b834c6a70e0639e72aabc625862adb5db99";
constexpr const char * ConfigId = "0062cb11becaec2eea537c4300f8c9619e7b057ca7bd41044d5432ef7cced9b7";
volatile sig_atomic_t interrupted = 0;
void on_signal(int signal) { interrupted = signal; }
Error fail(ErrorCode code, const std::string & message) { return {code, "intrinsic: " + message}; }
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }

struct Signals {
    struct sigaction term{}, interrupt{};
    bool installed = false;
    Status install() {
        interrupted = 0;
        struct sigaction action{}; action.sa_handler = on_signal; sigemptyset(&action.sa_mask);
        if (sigaction(SIGTERM, &action, &term)) return fail(ErrorCode::Io, "install SIGTERM handler");
        if (sigaction(SIGINT, &action, &interrupt)) { sigaction(SIGTERM, &term, nullptr); return fail(ErrorCode::Io, "install SIGINT handler"); }
        installed = true; return success();
    }
    ~Signals() { if (installed) { sigaction(SIGTERM, &term, nullptr); sigaction(SIGINT, &interrupt, nullptr); } }
};
struct Digest {
    EVP_MD_CTX * ctx = EVP_MD_CTX_new();
    Digest() = default;
    Digest(const Digest &) = delete;
    ~Digest() { EVP_MD_CTX_free(ctx); }
    bool begin() { return ctx && EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1; }
    bool update(const void * data, size_t size) { return EVP_DigestUpdate(ctx, data, size) == 1; }
    Result<std::string> finish() {
        unsigned char result[EVP_MAX_MD_SIZE]; unsigned int count = 0;
        if (EVP_DigestFinal_ex(ctx, result, &count) != 1 || count != 32) return fail(ErrorCode::Io, "SHA256 finalization");
        std::string text; const char * hex = "0123456789abcdef";
        for (unsigned i = 0; i < count; ++i) { text += hex[result[i] >> 4]; text += hex[result[i] & 15]; }
        return text;
    }
};
struct File {
    int fd = -1;
    File() = default; File(const File &) = delete;
    ~File() { if (fd >= 0) close(fd); }
    Status open_new(const fs::path & path) {
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) return fail(ErrorCode::Io, "create " + path.string() + ": " + strerror(errno));
        return success();
    }
    Status write_all(const void * data, size_t bytes) {
        const char * p = static_cast<const char *>(data);
        while (bytes) {
            if (interrupted) return fail(ErrorCode::Io, "interrupted by signal " + std::to_string(interrupted));
            ssize_t n = ::write(fd, p, bytes);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return fail(ErrorCode::Io, "write: " + std::string(strerror(errno)));
            p += n; bytes -= size_t(n);
        }
        return success();
    }
    Status finish() {
        if (fsync(fd)) return fail(ErrorCode::Io, "fsync: " + std::string(strerror(errno)));
        int current = fd; fd = -1;
        if (close(current)) return fail(ErrorCode::Io, "close: " + std::string(strerror(errno)));
        return success();
    }
};
struct Staging {
    fs::path path; bool committed = false;
    ~Staging() { if (!committed && !path.empty()) { std::error_code ec; fs::remove_all(path, ec); if (ec) fprintf(stderr, "intrinsic cleanup %s: %s\n", path.c_str(), ec.message().c_str()); } }
};
struct Events {
    cudaEvent_t start = nullptr, end = nullptr;
    ~Events() { if (start) cudaEventDestroy(start); if (end) cudaEventDestroy(end); }
};
Result<std::string> read_small(const fs::path & path, size_t limit) {
    std::error_code ec; auto size = fs::file_size(path, ec);
    if (ec) return fail(ErrorCode::Io, "stat " + path.string() + ": " + ec.message());
    if (size > limit) return fail(ErrorCode::Invalid, "file too large: " + path.string());
    std::ifstream stream(path, std::ios::binary); std::string result(size, '\0');
    stream.read(result.data(), result.size());
    if (!stream) return fail(ErrorCode::Io, "read " + path.string());
    return result;
}
Result<SourceIdentity> source_identity(const std::string & path) {
    struct stat st{};
    if (stat(path.c_str(), &st)) return fail(ErrorCode::Io, "stat source " + path + ": " + strerror(errno));
    SourceIdentity result; result.device = st.st_dev; result.inode = st.st_ino; result.size = st.st_size;
    result.mtime_ns = uint64_t(st.st_mtim.tv_sec) * 1000000000ULL + st.st_mtim.tv_nsec;
    result.ctime_ns = uint64_t(st.st_ctim.tv_sec) * 1000000000ULL + st.st_ctim.tv_nsec;
    return result;
}
Status sync_directory(const fs::path & path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return fail(ErrorCode::Io, "open directory " + path.string());
    int result = fsync(fd), error = errno; close(fd);
    if (result) return fail(ErrorCode::Io, "fsync directory " + path.string() + ": " + strerror(error));
    return success();
}
Status write_metadata(const fs::path & path, const std::string & text) {
    File file; auto s = file.open_new(path); if (!s.ok()) return s;
    s = file.write_all(text.data(), text.size()); if (!s.ok()) return s;
    return file.finish();
}
Status validate_spec(const json & spec) {
    if (spec.at("version") != 1 || spec.at("context") != Context || spec.at("chunks") != Chunks || spec.at("vocab") != Vocab ||
        spec.at("token_shape") != json::array({Chunks, Context}) || spec.at("output_shape") != json::array({Chunks, Rows, Vocab}) ||
        spec.at("token_dtype") != "little-endian-int32" || spec.at("output_dtype") != "little-endian-float32" ||
        spec.at("output_order") != "C: chunk, position, vocabulary-id; no header" ||
        spec.at("add_bos") != false || spec.at("add_eos") != false || spec.at("parse_special") != false ||
        spec.at("position_start") != 0 || spec.at("logit_positions") != json({{"first",First},{"last_inclusive",Last}}) ||
        spec.at("target_positions") != json({{"first",First+1},{"last_inclusive",Last+1}}) ||
        spec.at("config_id") != ConfigId || spec.at("tokens_sha256") != TokenHash)
        return fail(ErrorCode::Invalid, "not the frozen teacher-v1 protocol");
    const std::vector<std::string> expected = {"recurrent_state","conv_history","kv_cache","absolute_position"};
    auto resets = spec.at("reset_before_each_chunk").get<std::vector<std::string>>();
    auto sorted = expected; std::sort(sorted.begin(), sorted.end()); std::sort(resets.begin(), resets.end());
    if (resets != sorted) return fail(ErrorCode::Invalid, "reset families mismatch");
    if (!fs::path(spec.at("tokens_file").get<std::string>()).is_absolute()) return fail(ErrorCode::Invalid, "tokens_file must be absolute");
    return success();
}
}

Status dump_intrinsic(DeviceModel & model, DecodeWorkspace w, const Provenance & provenance,
                      const std::string & spec_path, const std::string & output_dir, int grid) {
    try {
        static_assert(std::endian::native == std::endian::little, "raw teacher format requires little endian");
        if (model.state.slots < 1 || model.state.context < Context || w.context < model.state.context || !w.logits || !model.device_weights ||
            !model.state.gdn || !model.state.conv || !model.state.key || !model.state.value || !model.state.positions || grid < 1 || grid > 256)
            return fail(ErrorCode::Capacity, "initialized full model/state/workspace, context>=512 and grid1..256 required");
        if (model.weights.head.n != Vocab || model.weights.head.k != Hidden || model.weights.embedding.n != Vocab || model.weights.embedding.k != Hidden ||
            provenance.version != CacheVersion || provenance.geometry != GeometryTag || provenance.cache_path.empty())
            return fail(ErrorCode::Geometry, "model/cache geometry or provenance mismatch");
        auto before = source_identity(provenance.source_path); if (!before.ok()) return before.error();
        if (!(before.value() == provenance.source)) return fail(ErrorCode::CacheMismatch, "source changed since cache load");
        auto spec_text = read_small(spec_path, 1024 * 1024); if (!spec_text.ok()) return spec_text.error();
        json spec = json::parse(spec_text.value()); auto s = validate_spec(spec); if (!s.ok()) return s;
        auto token_text = read_small(spec.at("tokens_file").get<std::string>(), Chunks * Context * 4);
        if (!token_text.ok()) return token_text.error();
        if (token_text.value().size() != Chunks * Context * 4) return fail(ErrorCode::Invalid, "token byte count mismatch");
        Digest token_digest; if (!token_digest.begin() || !token_digest.update(token_text.value().data(), token_text.value().size())) return fail(ErrorCode::Io, "token SHA256");
        auto token_hash = token_digest.finish(); if (!token_hash.ok()) return token_hash.error();
        if (token_hash.value() != TokenHash) return fail(ErrorCode::Invalid, "frozen token SHA256 mismatch");
        std::vector<int32_t> tokens(Chunks * Context); memcpy(tokens.data(), token_text.value().data(), token_text.value().size());
        for (int32_t token : tokens) if (token < 0 || token >= Vocab) return fail(ErrorCode::Invalid, "teacher token outside vocabulary");
        fs::path output = fs::absolute(output_dir), parent = output.parent_path(); std::error_code ec;
        if (output_dir.empty() || fs::exists(output)) return fail(ErrorCode::Io, "output directory empty or already exists");
        fs::create_directories(parent, ec); if (ec) return fail(ErrorCode::Io, "create output parent: " + ec.message());
        auto space = fs::space(parent, ec); if (ec || space.available < OutputBytes + 16 * 1024 * 1024ULL) return fail(ErrorCode::Capacity, "insufficient free space for 4.05GB raw logits");
        Staging stage; stage.path = output.string() + ".partial-" + std::to_string(getpid());
        if (!fs::create_directory(stage.path, ec)) { stage.path.clear(); return fail(ErrorCode::Io, "staging directory exists or cannot be created: " + ec.message()); }
        Signals signals; s = signals.install(); if (!s.ok()) return s;
        File raw; s = raw.open_new(stage.path / "logits.f32"); if (!s.ok()) return s;
        Digest raw_digest; if (!raw_digest.begin()) return fail(ErrorCode::Io, "raw SHA256 initialization");
        Events events; s = cuda_status(cudaEventCreate(&events.start), "intrinsic start event"); if (!s.ok()) return s;
        s = cuda_status(cudaEventCreate(&events.end), "intrinsic end event"); if (!s.ok()) return s;
        std::vector<float> row(Vocab); json chunk_timings = json::array();
        double gpu_ms = 0, copy_ms = 0, write_ms = 0, finite_hash_ms = 0, reset_ms = 0;
        auto wall_start = Clock::now();
        for (int chunk = 0; chunk < Chunks; ++chunk) {
            auto reset_start = Clock::now(); s = model.reset(); if (!s.ok()) return s;
            s = cuda_status(cudaDeviceSynchronize(), "intrinsic sequence reset completion"); if (!s.ok()) return s;
            reset_ms += ms(reset_start); double chunk_gpu_ms = 0; auto chunk_start = Clock::now();
            for (int position = 0; position < Context; ++position) {
                if (interrupted) return fail(ErrorCode::Io, "interrupted by signal " + std::to_string(interrupted));
                s = cuda_status(cudaEventRecord(events.start), "intrinsic token start"); if (!s.ok()) return s;
                s = run_token(model, w, tokens[chunk * Context + position], position, 0, grid); if (!s.ok()) return s;
                s = cuda_status(cudaEventRecord(events.end), "intrinsic token end"); if (!s.ok()) return s;
                s = cuda_status(cudaEventSynchronize(events.end), "intrinsic token completion"); if (!s.ok()) return s;
                float elapsed = 0; s = cuda_status(cudaEventElapsedTime(&elapsed, events.start, events.end), "intrinsic token GPU time"); if (!s.ok()) return s;
                gpu_ms += elapsed; chunk_gpu_ms += elapsed;
                if (position < First || position > Last) continue;
                auto start = Clock::now(); s = cuda_status(cudaMemcpy(row.data(), w.logits, Vocab * sizeof(float), cudaMemcpyDeviceToHost), "intrinsic full-vocab logits copy"); if (!s.ok()) return s;
                copy_ms += ms(start); start = Clock::now();
                for (int id = 0; id < Vocab; ++id) if (!std::isfinite(row[id])) return fail(ErrorCode::Invalid, "nonfinite logit chunk=" + std::to_string(chunk) + " position=" + std::to_string(position) + " id=" + std::to_string(id));
                if (!raw_digest.update(row.data(), row.size() * sizeof(float))) return fail(ErrorCode::Io, "raw SHA256 update");
                finite_hash_ms += ms(start); start = Clock::now(); s = raw.write_all(row.data(), row.size() * sizeof(float)); if (!s.ok()) return s;
                write_ms += ms(start);
            }
            chunk_timings.push_back({{"chunk",chunk},{"teacher_tokens",Context},{"scored_rows",Rows},{"gpu_ms",chunk_gpu_ms},{"wall_ms",ms(chunk_start)}});
            fprintf(stdout, "intrinsic chunk %d/%d complete GPU %.3f ms\n", chunk+1, Chunks, chunk_gpu_ms); fflush(stdout);
        }
        auto flush_start = Clock::now(); s = raw.finish(); if (!s.ok()) return s; double flush_ms = ms(flush_start);
        if (fs::file_size(stage.path / "logits.f32") != OutputBytes) return fail(ErrorCode::Io, "completed raw file byte count mismatch");
        auto raw_hash = raw_digest.finish(); if (!raw_hash.ok()) return raw_hash.error();
        auto after = source_identity(provenance.source_path); if (!after.ok()) return after.error();
        if (!(after.value() == before.value())) return fail(ErrorCode::CacheMismatch, "source changed during intrinsic execution");
        Digest spec_digest; if (!spec_digest.begin() || !spec_digest.update(spec_text.value().data(),spec_text.value().size())) return fail(ErrorCode::Io, "spec SHA256");
        auto spec_hash = spec_digest.finish(); if (!spec_hash.ok()) return spec_hash.error();
        json source = {{"device",provenance.source.device},{"inode",provenance.source.inode},{"size",provenance.source.size},{"mtime_ns",provenance.source.mtime_ns},{"ctime_ns",provenance.source.ctime_ns}};
        json manifest = {{"version",1},{"state","succeeded"},{"backend","halo6000_native"},{"model",provenance.source_path},{"model_bytes",provenance.source.size},
                         {"cache",{{"path",provenance.cache_path},{"version",provenance.version},{"geometry",provenance.geometry},{"source_identity",source}}},
                         {"spec",spec},{"spec_sha256",spec_hash.value()},{"tokens_sha256",token_hash.value()},{"tokens_first16",std::vector<int32_t>(tokens.begin(),tokens.begin()+16)},
                         {"logits_file","logits.f32"},{"logits_sha256",raw_hash.value()},{"logits_bytes",OutputBytes},{"dtype","little-endian-float32"},
                         {"shape",{Chunks,Rows,Vocab}},{"grid",grid},{"execution_route","all-serial-native-token-id-teacher-forcing"},
                         {"slots_used",1},{"state_reset_before_each_chunk",true},{"fork_inference",false},{"finite_logits",true},
                         {"timings",{{"gpu_forward_ms",gpu_ms},{"copy_ms",copy_ms},{"finite_check_sha256_ms",finite_hash_ms},{"write_ms",write_ms},
                                    {"reset_ms",reset_ms},{"flush_ms",flush_ms},{"wall_ms",ms(wall_start)},{"teacher_tokens",Chunks*Context},{"scored_rows",Chunks*Rows},
                                    {"gpu_teacher_tokens_per_second",Chunks*Context*1000./gpu_ms},{"includes_prompt_decode_all_positions",true},{"model_load_included",false}}},
                         {"chunk_timings",chunk_timings}};
        s = write_metadata(stage.path / "spec.json",spec_text.value()); if (!s.ok()) return s;
        s = write_metadata(stage.path / "manifest.json",manifest.dump(2)+"\n"); if (!s.ok()) return s;
        s = sync_directory(stage.path); if (!s.ok()) return s;
        if (interrupted) return fail(ErrorCode::Io, "interrupted before publication");
        if (syscall(SYS_renameat2, AT_FDCWD, stage.path.c_str(), AT_FDCWD, output.c_str(), 1 /* RENAME_NOREPLACE */)) return fail(ErrorCode::Io, "atomic output publication: " + std::string(strerror(errno)));
        stage.committed = true;
        s = sync_directory(parent); if (!s.ok()) return s;
        fprintf(stdout,"intrinsic published %s (%llu bytes)\n",output.c_str(),static_cast<unsigned long long>(OutputBytes));fflush(stdout);
        return success();
    } catch (const json::exception & error) { return fail(ErrorCode::Invalid, "spec/metadata JSON: " + std::string(error.what())); }
      catch (const fs::filesystem_error & error) { return fail(ErrorCode::Io,error.what()); }
      catch (const std::exception & error) { return fail(ErrorCode::Invalid,error.what()); }
}
}
