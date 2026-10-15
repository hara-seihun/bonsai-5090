#include "grammar.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

using Json = nlohmann::ordered_json;
static Json load(const char* path) {
    std::ifstream input(path);
    if (!input) { std::cerr << "cannot open " << path << '\n'; std::exit(1); }
    return Json::parse(input);
}
int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) { std::cerr << "usage: grammar-mask-bench VOCAB_JSON GRAMMAR_CASES_JSON\n"; return 1; }
    std::vector<std::string> pieces;
    std::vector<int> eos;
    const std::string path = argv[1];
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".bin") {
        std::ifstream input(path, std::ios::binary);
        auto word = [&]() {
            uint8_t b[4]; input.read(reinterpret_cast<char*>(b), 4);
            if (!input) { std::cerr << "truncated vocabulary\n"; std::exit(1); }
            return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
        };
        const uint32_t count = word();
        for (uint32_t i = 0; i < count; ++i) {
            std::string piece(word(), '\0'); input.read(piece.data(), piece.size());
            if (!input) { std::cerr << "truncated piece\n"; return 1; }
            pieces.push_back(std::move(piece));
        }
    } else {
      const auto data = load(argv[1]);
      if (data.is_array()) pieces = data.get<std::vector<std::string>>();
      else {
        eos = data.at("eos").get<std::vector<int>>();
        for (const auto& hex : data.at("pieces_hex")) {
            const auto text = hex.get<std::string>();
            if (text.size() % 2) { std::cerr << "invalid token hex\n"; return 1; }
            std::string piece;
            for (size_t i = 0; i < text.size(); i += 2) {
                const std::string byte = text.substr(i, 2);
                size_t used = 0; const auto value = std::stoul(byte, &used, 16);
                if (used != 2) { std::cerr << "invalid token byte\n"; return 1; }
                piece.push_back(char(value));
            }
            pieces.push_back(std::move(piece));
        }
      }
    }
    if (pieces.size() == 248320 && eos.empty()) eos = {248044, 248046};
    // Add byte tokens for exact prefix replay independent of BPE segmentation.
    const int bytes = pieces.size();
    for (int c = 0; c < 256; ++c) pieces.emplace_back(1, char(c));
    const auto begin = std::chrono::steady_clock::now();
    const auto replay_pieces = pieces;
    std::unordered_map<std::string, int> piece_ids;
    for (int t = 0; t < (int)pieces.size(); ++t) if (!pieces[t].empty()) piece_ids.emplace(pieces[t], t);
    h5::grammar::CompileOptions options;
    if (argc == 4) {
        if (std::string(argv[3]) != "canonical_fixed") return 1;
        options.mode = h5::grammar::Mode::CanonicalFixed;
        options.encode = [&](const std::string& text) {
            std::vector<int> encoded;
            for (size_t at = 0; at < text.size();) {
                size_t n = text.size() - at;
                for (; n; --n) if (piece_ids.count(text.substr(at,n))) break;
                if (!n) return std::vector<int>{};
                encoded.push_back(piece_ids.at(text.substr(at,n))); at += n;
            }
            return encoded;
        };
    }
    auto vocab = std::make_shared<h5::grammar::Vocabulary>(std::move(pieces), eos);
    const auto ready = std::chrono::steady_clock::now();
    std::cerr << "vocabulary_prepare_ms=" << std::chrono::duration<double,std::milli>(ready - begin).count() << '\n';
    std::vector<uint32_t> mask((vocab->size() + 31) / 32);
    const auto cases = load(argv[2]);
    for (const auto& item : cases.at("cases")) {
        auto compiled = h5::grammar::compile(item.at("schema"), vocab, options);
        if (!compiled) { std::cerr << compiled.error << '\n'; return 1; }
        const std::string text = item.at("expected").dump();
        double cold_ms = 0, cached_ms = 0;
        uint64_t digest = 14695981039346656037ull;
        size_t prefixes = 0, forced = 0;
        for (size_t i = 0; i <= text.size();) {
            ++prefixes;
            const auto start = std::chrono::steady_clock::now();
            const int allowed = compiled.matcher->mask(mask.data());
            const auto first = std::chrono::steady_clock::now();
            if (allowed <= 0) { std::cerr << "empty prefix mask\n"; return 1; }
            for (uint32_t word : mask) { digest ^= word; digest *= 1099511628211ull; }
            for (int repeat = 0; repeat < 10; ++repeat) {
                if (compiled.matcher->mask(mask.data()) != allowed) return 1;
            }
            const auto last = std::chrono::steady_clock::now();
            cold_ms += std::chrono::duration<double,std::milli>(first - start).count();
            cached_ms += std::chrono::duration<double,std::milli>(last - first).count() / 10;
            if (i == text.size()) break;
            int selected = bytes + uint8_t(text[i]); size_t consumed = 1;
            if (argc == 4) {
                selected = -1; consumed = 0;
                for (int t = 0; t < (int)replay_pieces.size(); ++t) {
                    const auto& piece = replay_pieces[t];
                    if (piece.size() <= consumed || !(mask[t/32] & (1u << (t%32)))) continue;
                    if (!text.compare(i,piece.size(),piece)) { selected=t; consumed=piece.size(); }
                }
                if (selected < 0) { std::cerr << "no canonical continuation at byte " << i << ' ' << item.at("id") << '\n'; return 1; }
            }
            if (allowed == 1) ++forced;
            if (!compiled.matcher->accept(selected)) {
                std::cerr << "fixture prefix rejection at byte " << i << '\n'; return 1;
            }
            i += consumed;
        }
        if (!compiled.matcher->complete()) return 1;
        std::cout << Json{{"case",item.at("id")},{"prefixes",prefixes},{"forced_tokens",forced},{"mode",argc==4?"canonical_fixed":"ordinary"},{"mask_digest",digest},
                         {"cold_total_ms",cold_ms},{"cached_total_ms",cached_ms}}.dump() << '\n';
    }
}
