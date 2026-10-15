#include "gpu_grammar_cache.h"
#include <cstdio>
#include <memory>
using namespace h5::gpu_grammar;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
struct Entry { uint64_t refs, used; };
int main() {
  TokenLayout layout;
  const std::vector<uint32_t> counts{0, 1, 2, 512, 513, 777};
  CHECK(token_layout(counts, 777, 100, &layout) == TokenCacheError::None);
  CHECK(layout.offsets == std::vector<uint32_t>({0, 0, 1, 3, 0, 1}));
  CHECK(layout.sparse_edges == 515 && layout.wide_states == 2);
  CHECK(layout.bytes == 100 + 6 * 9 + 515 * 8 + 2 * (777 * 4 + 25 * 4 + 512 * 4));
  CHECK(token_layout({778}, 777, 0, &layout) == TokenCacheError::Overflow);
  CHECK(token_layout({}, 777, 0, &layout) == TokenCacheError::Overflow);
  CHECK(token_layout({1}, 0, 0, &layout) == TokenCacheError::Overflow);
  CHECK(token_layout({1000000000}, 1000000000, 0, &layout) == TokenCacheError::None);
  CHECK(layout.bytes > TOKEN_CACHE_BUDGET);
  std::vector<std::unique_ptr<Entry>> entries;
  entries.emplace_back(new Entry{2, 0});
  entries.emplace_back(new Entry{0, 10});
  entries.emplace_back(new Entry{0, 20});
  auto* active = entries[0].get();
  auto* second = entries[2].get();
  CHECK(token_cache_victim(entries)->get() == entries[1].get());
  entries.erase(token_cache_victim(entries));
  CHECK(entries[0].get() == active && entries[1].get() == second);
  --active->refs;
  CHECK(token_cache_victim(entries)->get() == second);
  --active->refs;
  active->used = 30;
  CHECK(token_cache_victim(entries)->get() == second);
  second->refs = 1;
  CHECK(token_cache_victim(entries)->get() == active);
  active->refs = 1;
  CHECK(token_cache_victim(entries) == entries.end());
  std::puts("GPU token cache layout and inactive-LRU contract passed");
}
