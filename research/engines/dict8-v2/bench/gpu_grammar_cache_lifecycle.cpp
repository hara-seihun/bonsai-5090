#include "gpu_grammar.h"
#include "model.h"
#include <cstdio>
extern "C" void* h5_create(const char*, int, int, int);
extern "C" int h5_step(void*, int, const int*, const int*, const int*, int*);
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int gpu_grammar_cache_lifecycle_probe(void* engine) {
  using namespace h5::gpu_grammar;
  TokenTable table;
  table.offsets.resize(h5::VOCAB + 1);
  table.eos.resize(h5::VOCAB);
  table.bytes = {'a', 'b'};
  for (int token = 0; token <= h5::VOCAB; ++token) table.offsets[token] = token < 2 ? token : 2;
  table.eos[2] = 1;
  CHECK(h5_gpu_grammar_vocabulary(engine, &table) == 0);
  Plan source;
  source.accepting = {0, 0, 1};
  source.transitions.assign(3 * 256, INVALID);
  source.transitions['a'] = 1;
  source.transitions[256 + 'b'] = 2;
  void *first = nullptr, *second = nullptr, *third = nullptr;
  CacheStats cold, shared, retained, empty;
  CHECK(h5_gpu_grammar_upload(engine, &source, &first) == 0);
  CHECK(h5_gpu_grammar_cache_stats(engine, first, &cold) == 0);
  CHECK(cold.token_plan_bytes > 0 && cold.sparse_edges == 3 && cold.wide_states == 0);
  CHECK(h5_gpu_grammar_upload(engine, &source, &second) == 0 && first != second);
  CHECK(h5_gpu_grammar_cache_stats(engine, second, &shared) == 0);
  CHECK(shared.hits == cold.hits + 1 && shared.misses == cold.misses);
  CHECK(shared.resident_bytes == cold.resident_bytes && shared.compile_s == cold.compile_s);
  CHECK(h5_gpu_grammar_vocabulary(engine, &table) == -1);
  Row row{second, State{0, 0}, 0, 1};
  const int prefix = 0;
  CHECK(h5_gpu_grammar_target(engine, 1, &row, 1, &prefix) == 0);
  CHECK(h5_gpu_grammar_release(engine, first) == -3);
  CHECK(h5_gpu_grammar_target(engine, 0, nullptr, 0, nullptr) == 0);
  CHECK(h5_gpu_grammar_release(engine, first) == 0);
  CHECK(h5_gpu_grammar_target(engine, 1, &row, 1, &prefix) == 0);
  const int slot = 0, token = 0, position = 0;
  int selected = -1;
  CHECK(h5_step(engine, 1, &slot, &token, &position, &selected) == 0);
  RowResult result;
  CHECK(h5_gpu_grammar_results(engine, 1, &result) == 0);
  CHECK(selected == 1 && result.before.id == 1 && result.after.id == 2 && result.complete && !result.error);
  CHECK(h5_gpu_grammar_release(engine, second) == 0);
  CHECK(h5_gpu_grammar_upload(engine, &source, &third) == 0);
  CHECK(h5_gpu_grammar_cache_stats(engine, third, &retained) == 0);
  CHECK(retained.hits == shared.hits + 1 && retained.compile_s == cold.compile_s);
  CHECK(retained.resident_bytes == cold.resident_bytes);
  CHECK(h5_gpu_grammar_release(engine, third) == 0);
  CHECK(h5_gpu_grammar_vocabulary(engine, &table) == 0);
  CHECK(h5_gpu_grammar_cache_stats(engine, nullptr, &empty) == 0 && !empty.resident_bytes);
  std::puts("GPU grammar cached handle/prefix/decode/lifetime contract passed");
  return 0;
}
#ifdef GPU_GRAMMAR_CACHE_PROBE_MAIN
int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]); return 2; }
  void* engine = h5_create(argv[1], 1, 64, 32);
  if (!engine) return 1;
  return gpu_grammar_cache_lifecycle_probe(engine);
}
#endif
