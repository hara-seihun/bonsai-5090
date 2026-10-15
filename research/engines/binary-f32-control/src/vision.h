#pragma once
#include <array>
#include <mutex>
#include <string>
#include <vector>
#include "llama.h"
#include "mtmd.h"

namespace h5 {
struct VisualPrompt {
  std::vector<int> tokens;
  std::vector<std::array<int, 4>> rope;
  std::vector<int> embedding_row; // -1 for text, otherwise row in embeddings
  std::vector<float> embeddings; // primal (unrotated) language embeddings
  int next_pos = 0, image_tokens = 0;
  double encode_s = 0;
};
class Vision {
  mtmd_context* ctx = nullptr;
  std::mutex mu;
public:
  ~Vision();
  bool init(const char* mmproj, const llama_model* model, bool gpu, int max_image_tokens, std::string& error);
  bool prepare(const std::string& prompt, const std::vector<std::string>& urls, int max_rows, VisualPrompt& out, std::string& error);
  bool enabled() const { return ctx != nullptr; }
};
}
