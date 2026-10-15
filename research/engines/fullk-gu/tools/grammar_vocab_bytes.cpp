#include <llama.h>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr,"usage: grammar-vocab-bytes MODEL_GGUF\n"); return 2; }
  llama_log_set([](ggml_log_level,const char*,void*){},nullptr);
  auto params = llama_model_default_params();
  params.vocab_only = true; params.n_gpu_layers = 0; params.no_alloc = false;
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],params),llama_model_free);
  if (!model) { std::fprintf(stderr,"vocabulary metadata load failed\n"); return 1; }
  const auto* vocab = llama_model_get_vocab(model.get());
  constexpr int count = 248320, eos1 = 248046, eos2 = 248044;
  if (!vocab || llama_vocab_n_tokens(vocab) < count) { std::fprintf(stderr,"vocabulary geometry mismatch\n"); return 1; }
  uint64_t bytes = 0, disabled = 0;
  for (int id = 0; id < count; ++id) {
    if (llama_vocab_is_control(vocab,id) || id == eos1 || id == eos2) { ++disabled; continue; }
    std::string piece(256,'\0');
    int size = llama_token_to_piece(vocab,id,piece.data(),piece.size(),0,false);
    if (size < 0) { piece.resize(-size); size = llama_token_to_piece(vocab,id,piece.data(),piece.size(),0,false); }
    if (size > 0) bytes += size; else ++disabled;
  }
  const uint64_t offsets = uint64_t(count+1)*sizeof(uint32_t), eos = count;
  std::printf("{\"vocabulary\":%d,\"token_bytes\":%llu,\"offset_bytes\":%llu,\"eos_bytes\":%llu,\"shared_device_vocabulary_bytes\":%llu,\"disabled_piece_ids\":%llu,\"eos_ids\":[%d,%d],\"scope\":\"same control/EOS filtering and raw token pieces as server; CPU vocab-only load\"}\n", count,(unsigned long long)bytes,(unsigned long long)offsets,(unsigned long long)eos,(unsigned long long)(bytes+offsets+eos),(unsigned long long)disabled,eos1,eos2);
  return 0;
}
