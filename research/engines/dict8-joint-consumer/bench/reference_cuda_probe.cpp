#include "llama.h"
#include "ggml-backend.h"
#include "sampling.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using Json = nlohmann::ordered_json;
static bool read_file(const char* path, std::string& out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  out.assign(std::istreambuf_iterator<char>(file), {});
  return !file.bad();
}
static int fail(int code, const char* reason) {
  std::fprintf(stderr, "reference_error[%d]: %s\n", code, reason);
  return code;
}
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::puts("reference_cuda_probe MODEL PROMPT_UTF8 TEACHER_IDS_BIN OUT_PREFIX GENERATE_STEPS SAMPLING_JSON");
    std::puts("Teacher IDs: uint32 LE count then int32 LE IDs. Sampling: explicit temperature,top_p,top_k,min_p,presence_penalty,repetition_penalty,seed.");
    std::puts("Outputs: .logits int32 LE header[rows,vocab,prompt_tokens,teacher_tokens], then records int32 continuation_position + float32[vocab]; saves positions0,32,64,128,256 plus finalteacher. .rows.jsonl mapping; .tokens.json IDs; .text generated continuation.");
    return 0;
  }
  if (argc != 7) return fail(2, "wrong argument count; use --help");
  std::string prompt, teacher_text, sampling_text;
  if (!read_file(argv[2], prompt) || !read_file(argv[3], teacher_text) || !read_file(argv[6], sampling_text)) return fail(3, "input file read failed");
  Json sampling_json = Json::parse(sampling_text, nullptr, false);
  if (!sampling_json.is_object() || teacher_text.size() < 4) return fail(4, "malformed teacher/sampling input");
  uint32_t teacher_count;
  std::memcpy(&teacher_count, teacher_text.data(), 4);
  if (teacher_count > 8192 || teacher_text.size() != 4 + size_t(teacher_count) * 4) return fail(4, "teacher binary size mismatch");
  std::vector<llama_token> teacher(teacher_count);
  std::memcpy(teacher.data(), teacher_text.data()+4, size_t(teacher_count)*4);
  for (auto id : teacher) if (id < 0) return fail(4, "negative teacher token");
  for (const char* key : {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "seed"}) {
    if (!sampling_json.contains(key)) return fail(4, "sampling configuration missing required field");
  }
  char* end = nullptr;
  long steps = std::strtol(argv[5], &end, 10);
  if (!end || *end || steps < 0 || steps > 4096) return fail(4, "generate steps must be integer in [0,4096]");
  h5::sampling::Config sampling;
  std::string error;
  if (!h5::sampling::parse(sampling_json, sampling, error)) return fail(4, error.c_str());
  llama_backend_init();
  ggml_backend_load_all();
  if (!llama_supports_gpu_offload()) return fail(5, "CUDA offload unavailable");
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = -1;
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
  if (!model) return fail(6, "model load failed");
  const auto* vocab = llama_model_get_vocab(model.get());
  const int nv = llama_vocab_n_tokens(vocab);
  for (auto id : teacher) if (id >= nv) return fail(4, "teacher token outside vocabulary");
  int count = llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, false, true);
  if (count >= 0) return fail(7, "prompt must tokenize to nonempty sequence");
  std::vector<llama_token> tokens(-count);
  count = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
  if (count <= 0) return fail(7, "prompt tokenization failed");
  tokens.resize(count);
  if (tokens.size() + teacher.size() + steps > 8192) return fail(7, "trajectory exceeds 8192 positions");
  auto cp = llama_context_default_params();
  cp.n_ctx = tokens.size() + teacher.size() + steps + 32;
  cp.n_batch = 256;
  cp.n_ubatch = 128;
  cp.n_threads = 8;
  cp.n_threads_batch = 8;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
  if (!context) return fail(8, "context initialization failed");
  auto batch = llama_batch_init(256, 0, 1);
  int pos = 0;
  while (pos < count) {
    batch.n_tokens = std::min(256, count - pos);
    for (int i = 0; i < batch.n_tokens; ++i) {
      batch.token[i] = tokens[pos + i];
      batch.pos[i] = pos + i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = pos + i == count - 1;
    }
    if (llama_decode(context.get(), batch)) return fail(9, "prompt decode failed");
    pos += batch.n_tokens;
  }
  std::string prefix = argv[4];
  std::ofstream logits(prefix + ".logits", std::ios::binary);
  std::ofstream rows(prefix + ".rows.jsonl");
  std::ofstream text(prefix + ".text", std::ios::binary);
  if (!logits || !rows || !text) return fail(10, "output file open failed");
  std::vector<unsigned char> seen(nv, 0);
  std::mt19937_64 rng(sampling.seed);
  std::vector<llama_token> generated;
  size_t row = 0;
  int32_t header[4] = {0, nv, count, int32_t(teacher.size())};
  logits.write(reinterpret_cast<const char*>(header), sizeof(header));
  auto save_row = [&](const char* consumed, int token) {
    int32_t continuation = pos - count;
    if (continuation != 0 && continuation != 32 && continuation != 64 && continuation != 128 && continuation != 256 && continuation != int32_t(teacher.size())) return true;
    const float* values = llama_get_logits_ith(context.get(), -1);
    if (!values) return false;
    for (int i = 0; i < nv; ++i) if (!std::isfinite(values[i])) return false;
    auto offset = logits.tellp();
    logits.write(reinterpret_cast<const char*>(&continuation), sizeof(continuation));
    logits.write(reinterpret_cast<const char*>(values), size_t(nv) * sizeof(float));
    rows << Json{{"row", row}, {"vocab", nv}, {"byte_offset", int64_t(offset) + 4}, {"next_position", pos}, {"continuation_consumed", pos-count}, {"consumed_kind", consumed}, {"consumed_token", token}}.dump() << '\n';
    ++row;
    logits.flush(); rows.flush();
    return bool(logits) && bool(rows);
  };
  if (!save_row("prompt", tokens.back())) return fail(10, "logit output failed or nonfinite logits");
  auto consume = [&](llama_token id) {
    batch.n_tokens = 1;
    batch.token[0] = id; batch.pos[0] = pos; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
    if (llama_decode(context.get(), batch)) return false;
    seen[id] = 1; ++pos;
    return true;
  };
  for (auto id : teacher) {
    if (!consume(id)) return fail(9, "teacher decode failed");
    if (!save_row("teacher", id)) return fail(10, "teacher logit output failed");
  }
  for (long step = 0; step < steps; ++step) {
    auto chosen = h5::sampling::select(llama_get_logits_ith(context.get(), -1), nv, sampling, seen, rng);
    if (!chosen.error.empty()) return fail(11, chosen.error.c_str());
    generated.push_back(chosen.token);
    if (llama_vocab_is_eog(vocab, chosen.token)) break;
    int length = llama_token_to_piece(vocab, chosen.token, nullptr, 0, 0, true);
    if (length > 0) return fail(11, "unexpected token piece sizing result");
    std::vector<char> piece(-length);
    length = llama_token_to_piece(vocab, chosen.token, piece.data(), piece.size(), 0, true);
    if (length < 0) return fail(11, "token piece conversion failed");
    text.write(piece.data(), length); text.flush();
    if (!text) return fail(10, "continuation output failed");
    if (!consume(chosen.token)) return fail(9, "generated decode failed");
    if (!save_row("generated", chosen.token)) return fail(10, "generated logit output failed");
  }
  header[0] = int32_t(row);
  logits.seekp(0);
  logits.write(reinterpret_cast<const char*>(header), sizeof(header));
  logits.flush();
  if (!logits) return fail(10, "logit header finalization failed");
  std::ofstream manifest(prefix + ".tokens.json");
  manifest << Json{{"prompt", tokens}, {"teacher", teacher}, {"generated", generated}, {"sampling", sampling_json}, {"vocab", nv}, {"rows", row}, {"dtype", "float32-le"}, {"flash_attention", false}, {"gpu_layers", "all"}}.dump(2) << '\n';
  if (!manifest) return fail(10, "token manifest write failed");
  llama_batch_free(batch);
  std::fprintf(stderr, "reference_complete: prompt=%d teacher=%zu generated=%zu rows=%zu vocab=%d\n", count, teacher.size(), generated.size(), row, nv);
  return 0;
}
