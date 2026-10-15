#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace h5::grammar {

class Vocabulary {
public:
    // Empty non-EOS pieces are disabled. EOS IDs never consume their piece.
    Vocabulary(std::vector<std::string> pieces, std::vector<int> eos);
    size_t size() const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    friend class Matcher;
    friend struct CompilerAccess;
};

class Matcher {
public:
    Matcher(const Matcher&);
    Matcher& operator=(const Matcher&);
    Matcher(Matcher&&) noexcept;
    Matcher& operator=(Matcher&&) noexcept;
    ~Matcher();
    // Failed acceptance leaves the matcher unchanged. Accepted EOS ends it;
    // an ended matcher permits EOS-only, idempotently, for speculative padding.
    bool accept(int token);
    bool complete() const;
    // Writes ceil(vocabulary.size()/32) words; bit 1 means allowed.
    int mask(uint32_t* words) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit Matcher(std::unique_ptr<Impl>);
    friend struct CompilerAccess;
};

struct Result {
    std::unique_ptr<Matcher> matcher;
    std::string error;
    explicit operator bool() const { return bool(matcher); }
};

// Output subset: object properties follow schema insertion order, finite
// enum/const values use canonical JSON serialization, integer has no fraction
// or exponent. Whitespace is unrestricted. Unsupported constraints fail
// compilation explicitly; they are never ignored. anyOf keeps live prefix
// alternatives; oneOf requires provably disjoint semantic alternatives.
// Local JSON Pointer $ref/$defs are expanded; recursive references and
// reference siblings imposing constraints fail explicitly.
// CanonicalFixed retains the ordinary full-vocabulary byte grammar at every
// ambiguous byte state. Only already byte-deterministic spans are tokenizer-
// encoded into singleton next-token choices, recanonicalizing after each token.
// Payload tokens may cross structural boundaries exactly as in ordinary mode.
inline constexpr const char* CANONICAL_FIXED_LAW = "ordinary-byte-grammar-forced-spans-only-v3";
enum class Mode { Ordinary, CanonicalFixed };
struct CompileOptions {
    Mode mode = Mode::Ordinary;
    // Required for CanonicalFixed. Token pieces must exactly reproduce the text;
    // controls, EOS, empty pieces and out-of-range tokens produce compile errors.
    std::function<std::vector<int>(const std::string&)> encode;
};
Result compile(const nlohmann::ordered_json& schema,
               std::shared_ptr<const Vocabulary> vocabulary);
Result compile(const nlohmann::ordered_json& schema,
               std::shared_ptr<const Vocabulary> vocabulary, const CompileOptions& options);

} // namespace h5::grammar
