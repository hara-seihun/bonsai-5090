#include "vision.h"
#include "model.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>

extern "C" {
void* h5_create(const char*, int, int, int);
void h5_reset(void*, int);
int h5_step_visual(void*, int, const int*, const int*, const int*, const int*, const float*, int*);
int h5_step_visual_logits(void*, int, const int*, const int*, const int*, const int*, const float*, int*, float*);
}
using json = nlohmann::ordered_json;
static std::string base64(const std::string& s) {
  const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string r; unsigned v = 0; int bits = 0;
  for (unsigned char c : s) { v = (v << 8) | c; bits += 8; while (bits >= 6) { bits -= 6; r += a[(v >> bits) & 63]; } }
  if (bits) r += a[(v << (6 - bits)) & 63];
  while (r.size() % 4) r += '=';
  return r;
}
static json compare(const float* a, const float* b) {
  double ma = *std::max_element(a, a + h5::VOCAB), mb = *std::max_element(b, b + h5::VOCAB);
  double sa = 0, sb = 0, aa = 0, bb = 0, ab = 0;
  for (int i = 0; i < h5::VOCAB; i++) { sa += exp(a[i] - ma); sb += exp(b[i] - mb); aa += a[i]*a[i]; bb += b[i]*b[i]; ab += a[i]*b[i]; }
  double kld = 0;
  for (int i = 0; i < h5::VOCAB; i++) kld += exp(b[i] - mb) / sb * ((b[i] - mb - log(sb)) - (a[i] - ma - log(sa)));
  return {{"native_top1", std::max_element(a, a + h5::VOCAB) - a}, {"reference_top1", std::max_element(b, b + h5::VOCAB) - b},
          {"reference_to_native_kld", kld}, {"logit_cosine", ab / sqrt(aa*bb)}};
}
int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: h5-vision-check MODEL MMPROJ IMAGE [--encode-only|--encode-gpu|--reference|--reference-only|--native-compare] [OUTPUT.json]\n"); return 1; }
  bool encoder_gpu = argc > 4 && std::string(argv[4]) == "--encode-gpu";
  bool encode_only = encoder_gpu || (argc > 4 && std::string(argv[4]) == "--encode-only");
  bool reference_only = argc > 4 && std::string(argv[4]) == "--reference-only";
  bool native_compare = argc > 4 && std::string(argv[4]) == "--native-compare";
  bool reference = reference_only || (argc > 4 && std::string(argv[4]) == "--reference");
  std::vector<float> native_logits(h5::VOCAB);
  llama_backend_init(); ggml_backend_load_all();
  auto mp = llama_model_default_params(); mp.vocab_only = false; mp.no_alloc = false; mp.n_gpu_layers = 0;
  llama_model* model = llama_model_load_from_file(argv[1], mp); if (!model) return 2;
  h5::Vision encoder; std::string error;
  if (!encoder.init(argv[2], model, encoder_gpu, 256, error)) { fprintf(stderr, "%s\n", error.c_str()); return 3; }
  std::ifstream f(argv[3], std::ios::binary); std::string bytes((std::istreambuf_iterator<char>(f)), {});
  h5::VisualPrompt prompt;
  const std::string text = std::string("<|im_start|>user\n") + mtmd_default_marker() + "\nWhat color is the large shape? Answer with one color word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  if (!encoder.prepare(text, {"data:image/png;base64," + base64(bytes)}, 2048, prompt, error)) { fprintf(stderr, "%s\n", error.c_str()); return 4; }
  json report = {{"prompt_rows", prompt.tokens.size()}, {"image_tokens", prompt.image_tokens}, {"next_rope_pos", prompt.next_pos}, {"encode_s", prompt.encode_s}};
  for (size_t i = 0; i < prompt.tokens.size(); i++) if (prompt.tokens[i] == -1) { report["first_visual_rope"] = prompt.rope[i]; break; }
  if (!encode_only && !reference_only) {
    void* engine = h5_create(argv[1], 1, 4096, 64); if (!engine) return 5;
    std::vector<float> logits((size_t)64 * h5::VOCAB), input((size_t)64 * h5::D);
    std::vector<int> slot(64, 0), pos(64), out(64);
    int last_token = -1;
    for (size_t start = 0; start < prompt.tokens.size(); start += 64) {
      int n = std::min((size_t)64, prompt.tokens.size() - start);
      std::fill(input.begin(), input.end(), 0.f);
      for (int j = 0; j < n; j++) { pos[j] = start + j; int er = prompt.embedding_row[start+j]; if (er >= 0) std::copy_n(prompt.embeddings.data() + (size_t)er*h5::D, h5::D, input.data() + (size_t)j*h5::D); }
      bool last = start + n == prompt.tokens.size();
      int status = last ? h5_step_visual_logits(engine, n, slot.data(), prompt.tokens.data()+start, pos.data(), prompt.rope[start].data(), input.data(), out.data(), logits.data())
                        : h5_step_visual(engine, n, slot.data(), prompt.tokens.data()+start, pos.data(), prompt.rope[start].data(), input.data(), out.data());
      if (status) { fprintf(stderr, "native step failed %d\n", status); return 6; }
      if (last) { std::copy_n(logits.data() + (size_t)(n-1)*h5::VOCAB, h5::VOCAB, logits.data()); last_token = out[n-1]; }
    }
    report["native_first_token"] = last_token;
    native_logits.assign(logits.begin(), logits.begin() + h5::VOCAB);
    if (native_compare) {
      if (argc < 6) return 9;
      std::vector<float> reference_logits(h5::VOCAB);
      std::ifstream file(std::string(argv[5]) + ".logits", std::ios::binary);
      if (!file.read((char*)reference_logits.data(), reference_logits.size()*sizeof(float))) return 10;
      report["comparison"] = compare(native_logits.data(), reference_logits.data());
    }
  }
    if (reference) {
      auto cp = llama_context_default_params(); cp.n_ctx = 4096; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_threads = 4; cp.n_threads_batch = 4;
      auto context = llama_init_from_model(model, cp); if (!context) return 7;
      for (size_t start = 0; start < prompt.tokens.size();) {
        bool visual = prompt.tokens[start] == -1; size_t end = start + 1;
        while (end < prompt.tokens.size() && end-start < 64 && (prompt.tokens[end] == -1) == visual) end++;
        int n = end-start;
        std::vector<int> positions(4*n), nseq(n, 1); std::vector<int8_t> outputs(n, 0);
        llama_seq_id zero = 0; std::vector<llama_seq_id*> seq(n, &zero);
        std::vector<float> emb((size_t)n*h5::D);
        for (int j = 0; j < n; j++) { for (int k = 0; k < 4; k++) positions[k*n+j] = prompt.rope[start+j][k];
          if (visual) std::copy_n(prompt.embeddings.data() + (size_t)prompt.embedding_row[start+j]*h5::D, h5::D, emb.data() + (size_t)j*h5::D); }
        outputs[n-1] = end == prompt.tokens.size();
        llama_batch batch{n, visual ? nullptr : prompt.tokens.data()+start, visual ? emb.data() : nullptr, positions.data(), nseq.data(), seq.data(), outputs.data()};
        if (llama_decode(context, batch)) return 8;
        start = end;
      }
      const float* ref_logits = llama_get_logits_ith(context, -1);
      if (reference_only) {
        report["reference_first_token"] = std::max_element(ref_logits, ref_logits+h5::VOCAB)-ref_logits;
        if (argc > 5) { std::ofstream file(std::string(argv[5])+".logits", std::ios::binary); file.write((const char*)ref_logits, h5::VOCAB*sizeof(float)); }
      } else report["comparison"] = compare(native_logits.data(), ref_logits);
      llama_free(context);
    }
  puts(report.dump(2).c_str());
  if (argc > 5) { std::ofstream file(native_compare ? std::string(argv[5])+".native.json" : argv[5]); file << report.dump(2) << '\n'; }
  llama_model_free(model);
}
