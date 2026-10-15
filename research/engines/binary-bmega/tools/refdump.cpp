// Reference logits from the PrismML llama.cpp fork.
// refdump MODEL PROMPT_FILE N_GEN OUT [raw|chat]
// Evaluates the prompt, then generates N_GEN greedy tokens one at a time. Writes
//   int32 n_tok, n_prompt, vocab; int32 tokens[n_tok]; float logits[n_tok - n_prompt + 1][vocab]
// where logits row i predicts token n_prompt + i (row 0 = last prompt position).
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: refdump MODEL PROMPT_FILE N_GEN OUT [raw|chat]\n"); return 1; }
  std::ifstream pf(argv[2]); std::stringstream ss; ss << pf.rdbuf(); std::string prompt = ss.str();
  const int n_gen = atoi(argv[3]);
  const bool chat = argc > 5 && !strcmp(argv[5], "chat");
  const bool teacher = argc > 5 && !strcmp(argv[5], "teacher");
  if (n_gen <= 0 || (argc > 5 && strcmp(argv[5], "raw") && !chat && !teacher)) {
    fprintf(stderr, "invalid generation count or mode\n"); return 1;
  }
  llama_backend_init();
  auto mp = llama_model_default_params(); mp.n_gpu_layers = 999;
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int nv = llama_vocab_n_tokens(vocab);
  if (chat) {
    llama_chat_message m{"user", prompt.c_str()};
    const char* tmpl = llama_model_chat_template(model, nullptr);
    std::vector<char> buf(prompt.size() * 2 + 4096);
    int n = llama_chat_apply_template(tmpl, &m, 1, true, buf.data(), buf.size());
    prompt.assign(buf.data(), n);
    prompt += "<think>\n\n</think>\n\n";
  }
  std::vector<llama_token> toks(prompt.size() + 16);
  int np = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
  if (np <= 0) { fprintf(stderr, "tokenization failed\n"); return 1; }
  toks.resize(np);
  std::vector<llama_token> held;
  if (teacher) {
    if (np < 128 + n_gen) { fprintf(stderr, "teacher input requires at least %d tokens\n", 128 + n_gen); return 1; }
    held.assign(toks.begin() + 128, toks.begin() + 128 + n_gen);
    np = 128; toks.resize(np);
  }
  auto cp = llama_context_default_params(); cp.n_ctx = np + n_gen + 16; cp.n_batch = cp.n_ctx; cp.n_ubatch = 512;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) { fprintf(stderr, "context load failed\n"); return 1; }
  std::vector<float> out;
  llama_batch b = llama_batch_init(np, 0, 1);
  for (int i = 0; i < np; i++) { b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i == np - 1; }
  b.n_tokens = np;
  if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
  for (int g = 0;; g++) {
    const float* lg = llama_get_logits_ith(ctx, -1);
    out.insert(out.end(), lg, lg + nv);
    if (g == n_gen) break;
    int best = 0; for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
    toks.push_back(teacher ? held[g] : best);
    llama_batch one = llama_batch_get_one(&toks.back(), 1);
    if (llama_decode(ctx, one)) { fprintf(stderr, "decode failed\n"); return 1; }
  }
  FILE* f = fopen(argv[4], "wb");
  if (!f) { perror("reference output"); return 1; }
  int hdr[3] = {(int)toks.size(), np, nv};
  fwrite(hdr, 4, 3, f); fwrite(toks.data(), 4, toks.size(), f); fwrite(out.data(), 4, out.size(), f); fclose(f);
  std::string text;
  for (size_t i = np; i < toks.size(); i++) { char p[256]; int n = llama_token_to_piece(vocab, toks[i], p, sizeof p, 0, true); text.append(p, n > 0 ? n : 0); }
  fprintf(stderr, "prompt %d tokens, %s: %s\n", np, teacher ? "teacher-forced" : "generated", text.c_str());
  llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free();
  return 0;
}
