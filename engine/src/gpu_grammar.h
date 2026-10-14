#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json_fwd.hpp>

namespace h5::gpu_grammar {
constexpr uint32_t INVALID = 0xffffffffu;
struct State { uint32_t id = 0; uint32_t ended = 0; };
struct Plan {
  std::vector<uint32_t> transitions; // state-major 256 byte transitions; INVALID rejects
  std::vector<uint8_t> accepting;
  uint32_t initial = 0;
  uint32_t states() const { return static_cast<uint32_t>(accepting.size()); }
  size_t bytes() const { return transitions.size()*sizeof(uint32_t)+accepting.size(); }
};
enum class CompileError { None, InvalidSchema, UnsupportedKeyword, InvalidReference, OverlappingOneOf, StateBudget, InvalidUtf8, Internal };
struct CompileResult {
  Plan plan;
  CompileError error = CompileError::None;
  std::string detail;
  bool ok() const { return error == CompileError::None; }
};
CompileResult compile(const nlohmann::ordered_json& schema, uint32_t state_budget = 32768);
struct CacheStats {
  uint64_t resident_bytes = 0, budget_bytes = 0, hits = 0, misses = 0, evictions = 0;
  double compile_s = 0;
  uint64_t token_plan_bytes = 0, wide_states = 0, sparse_edges = 0;
};
struct TokenTable { std::vector<uint32_t> offsets; std::vector<uint8_t> bytes; std::vector<uint8_t> eos; };
struct Row { void* plan = nullptr; State state; uint32_t prefix_offset = 0, prefix_length = 0; };
struct RowResult {
  State before, after;
  uint32_t allowed = 0, complete = 0, error = 0;
};
}

extern "C" {
int h5_gpu_grammar_vocabulary(void* engine, const h5::gpu_grammar::TokenTable*);
int h5_gpu_grammar_upload(void* engine, const h5::gpu_grammar::Plan*, void** handle);
int h5_gpu_grammar_release(void* engine, void* handle);
int h5_gpu_grammar_cache_stats(void* engine, void* handle, h5::gpu_grammar::CacheStats*);
int h5_gpu_grammar_target(void* engine, int rows, const h5::gpu_grammar::Row*, int prefix_count, const int* prefix_tokens);
int h5_gpu_grammar_results(void* engine, int rows, h5::gpu_grammar::RowResult*);
int h5_draft_gpu_grammar(void* engine, int ns, const int* slot, const int* n_ing, const int* ing_row0,
  const int* ing_pos, const int* anchor, const int* start, int* output,
  const h5::gpu_grammar::Row* blocks);
}
