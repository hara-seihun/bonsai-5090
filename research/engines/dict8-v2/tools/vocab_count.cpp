// Token frequency over text files with the model's own tokenizer (vocab-only load).
//   vocab-count MODEL.gguf OUT.tsv FILE|@LIST...   (OUT.tsv: token_id<TAB>count, every id that occurred;
//   @LIST names a file holding one path per line)
// Files are tokenized in 64 KiB pieces split at newlines; no special-token parsing.
#include "llama.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: vocab-count MODEL OUT.tsv FILE...\n"); return 2; }
  llama_backend_init();
  auto mp = llama_model_default_params(); mp.vocab_only = true;
  llama_model* m = llama_model_load_from_file(argv[1], mp);
  if (!m) { fprintf(stderr, "model load failed\n"); return 1; }
  const llama_vocab* v = llama_model_get_vocab(m);
  const int n_vocab = llama_vocab_n_tokens(v);
  std::vector<unsigned long long> cnt(n_vocab, 0);
  std::vector<llama_token> toks;
  unsigned long long total = 0;
  std::vector<std::string> paths;
  for (int f = 3; f < argc; f++) {
    if (argv[f][0] != '@') { paths.push_back(argv[f]); continue; }
    std::ifstream l(argv[f] + 1); if (!l) { fprintf(stderr, "missing list %s\n", argv[f] + 1); return 1; }
    for (std::string line; std::getline(l, line);) if (!line.empty()) paths.push_back(line);
  }
  for (const auto& path : paths) {
    std::ifstream in(path, std::ios::binary); if (!in) { fprintf(stderr, "skip %s\n", path.c_str()); continue; }
    std::stringstream ss; ss << in.rdbuf(); const std::string s = ss.str();
    size_t p = 0;
    while (p < s.size()) {
      size_t e = std::min(s.size(), p + 65536);
      if (e < s.size()) { size_t nl = s.rfind('\n', e); if (nl != std::string::npos && nl > p) e = nl + 1; }
      const int len = (int)(e - p);
      toks.resize(len + 16);
      int n = llama_tokenize(v, s.data() + p, len, toks.data(), (int)toks.size(), false, false);
      if (n < 0) { toks.resize(-n); n = llama_tokenize(v, s.data() + p, len, toks.data(), (int)toks.size(), false, false); }
      for (int i = 0; i < n; i++) if (toks[i] >= 0 && toks[i] < n_vocab) { cnt[toks[i]]++; total++; }
      p = e;
    }
  }
  FILE* o = fopen(argv[2], "w"); if (!o) return 1;
  for (int i = 0; i < n_vocab; i++) if (cnt[i]) fprintf(o, "%d\t%llu\n", i, cnt[i]);
  fclose(o);
  fprintf(stderr, "n_vocab=%d tokens=%llu\n", n_vocab, total);
  llama_model_free(m);
  return 0;
}
