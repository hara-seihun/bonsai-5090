#include "vision.h"
#include "model.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "mtmd-helper.h"
#include "httplib.h"
#include <algorithm>
#include <chrono>
#include <memory>

namespace h5 {
namespace {
constexpr size_t MAX_IMAGE_BYTES = 20 * 1024 * 1024;
bool image_bytes(const std::string& url, std::vector<unsigned char>& bytes, std::string& error) {
  if (url.compare(0, 11, "data:image/") == 0) {
    auto at = url.find(";base64,");
    if (at == std::string::npos || url.size() - at - 8 > (MAX_IMAGE_BYTES * 4 / 3 + 4)) {
      error = "image data URL must be base64 and at most 20 MiB"; return false;
    }
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    unsigned acc = 0; int bits = 0; bool pad = false;
    for (size_t i = at + 8; i < url.size(); i++) {
      const char c = url[i];
      if (c == '=') { pad = true; continue; }
      auto v = alphabet.find(c);
      if (pad || v == std::string::npos) { error = "invalid image base64"; return false; }
      acc = (acc << 6) | (unsigned)v; bits += 6;
      if (bits >= 8) { bits -= 8; bytes.push_back((acc >> bits) & 255); }
    }
  } else if (url.compare(0, 7, "http://") == 0 || url.compare(0, 8, "https://") == 0) {
    const size_t scheme = url.find("://") + 3, slash = url.find('/', scheme);
    const std::string origin = url.substr(0, slash);
    httplib::Client client(origin);
    client.set_connection_timeout(10); client.set_read_timeout(20);
    client.set_follow_location(false);
    auto result = client.Get(slash == std::string::npos ? "/" : url.substr(slash),
      [&](const char* data, size_t n) {
        if (bytes.size() + n > MAX_IMAGE_BYTES) return false;
        bytes.insert(bytes.end(), data, data + n); return true;
      });
    if (!result || result->status != 200) { error = "image URL download failed (redirects are not followed)"; return false; }
  } else { error = "image_url requires an http(s) URL or base64 image data URL"; return false; }
  if (bytes.empty() || bytes.size() > MAX_IMAGE_BYTES) { error = "empty or oversized image"; return false; }
  return true;
}
}
Vision::~Vision() { if (ctx) mtmd_free(ctx); }
bool Vision::init(const char* mmproj, const llama_model* model, bool gpu, int max_image_tokens, std::string& error) {
  if (llama_model_n_embd_inp(model) != D) { error = "vision input dimension does not match native engine"; return false; }
  if (gpu && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
    error = "GPU vision requested but libmtmd has no GPU backend; build GGML_CUDA=ON or use --vision-gpu 0"; return false;
  }
  if (gpu) {
    try {
      Gguf projection; projection.open(mmproj);
      for (const auto& tensor : projection.tensors) {
        if (tensor.type != GGML_TYPE_F32 && tensor.type != GGML_TYPE_F16 &&
            tensor.type != GGML_TYPE_BF16 && tensor.type != GGML_TYPE_Q8_0) {
          error = "GPU vision supports only Q8_0/F16/F32/BF16 projector weights: " + tensor.name; return false;
        }
      }
    } catch (const std::exception& ex) { error = ex.what(); return false; }
  }
  auto params = mtmd_context_params_default();
  if (gpu) params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  params.use_gpu = gpu; params.n_threads = 4; params.warmup = false;
  params.image_max_tokens = max_image_tokens;
  ctx = mtmd_init_from_file(mmproj, model, params);
  if (!ctx) { error = "cannot load mmproj"; return false; }
  if (!mtmd_support_vision(ctx) || !mtmd_decode_use_mrope(ctx)) {
    error = "native vision requires the Qwen M-RoPE projector"; return false;
  }
  return true;
}
bool Vision::prepare(const std::string& prompt, const std::vector<std::string>& urls, int max_rows,
                     VisualPrompt& out, std::string& error) {
  if (!ctx) { error = "vision is disabled; start with --mmproj PATH"; return false; }
  using Bitmap = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
  std::vector<Bitmap> owned; std::vector<const mtmd_bitmap*> bitmaps;
  for (const auto& url : urls) {
    std::vector<unsigned char> bytes;
    if (!image_bytes(url, bytes, error)) return false;
    auto decoded = mtmd_helper_bitmap_init_from_buf(ctx, bytes.data(), bytes.size(), false);
    Bitmap bitmap(decoded.bitmap, mtmd_bitmap_free);
    std::unique_ptr<mtmd_helper_video, decltype(&mtmd_helper_video_free)> video(decoded.video_ctx, mtmd_helper_video_free);
    if (video || !bitmap || mtmd_bitmap_is_audio(bitmap.get())) { error = "cannot decode image (PNG/JPEG recommended)"; return false; }
    const uint64_t pixels = (uint64_t)mtmd_bitmap_get_nx(bitmap.get()) * mtmd_bitmap_get_ny(bitmap.get());
    if (pixels > 64000000) { error = "image exceeds 64 megapixels"; return false; }
    bitmaps.push_back(bitmap.get()); owned.push_back(std::move(bitmap));
  }
  std::lock_guard<std::mutex> lock(mu);
  std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
  mtmd_input_text text{prompt.data(), prompt.size(), false, true};
  if (mtmd_tokenize(ctx, chunks.get(), &text, bitmaps.data(), bitmaps.size())) { error = "vision preprocessing failed"; return false; }
  size_t rows = 0;
  for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); i++) rows += mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks.get(), i));
  if (rows == 0 || rows > (size_t)max_rows) { error = "image and text prompt exceeds native context"; return false; }
  out = VisualPrompt(); out.tokens.reserve(rows); out.rope.reserve(rows); out.embedding_row.reserve(rows);
  int pos = 0;
  for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); i++) {
    const auto chunk = mtmd_input_chunks_get(chunks.get(), i);
    const auto type = mtmd_input_chunk_get_type(chunk);
    const size_t n = mtmd_input_chunk_get_n_tokens(chunk);
    if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t count; const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
      for (size_t j = 0; j < count; j++, pos++) {
        out.tokens.push_back(tokens[j]); out.rope.push_back({pos, pos, pos, pos}); out.embedding_row.push_back(-1);
      }
    } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      if (mtmd_decode_use_non_causal(ctx, chunk)) { error = "noncausal projector is not supported by native Qwen attention"; return false; }
      auto begin = std::chrono::steady_clock::now();
      if (mtmd_encode_chunk(ctx, chunk)) { error = "vision encoder failed"; return false; }
      out.encode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      const auto image = mtmd_input_chunk_get_tokens_image(chunk);
      const float* embeddings = mtmd_get_output_embd(ctx);
      int first = out.embeddings.size() / D;
      out.embeddings.insert(out.embeddings.end(), embeddings, embeddings + n * D);
      for (size_t j = 0; j < n; j++) {
        const auto p = mtmd_image_tokens_get_decoder_pos(image, pos, j);
        out.tokens.push_back(-1); out.rope.push_back({(int)p.t, (int)p.y, (int)p.x, (int)p.z});
        out.embedding_row.push_back(first + j);
      }
      out.image_tokens += n;
      pos += mtmd_input_chunk_get_n_pos(chunk);
    } else { error = "only image chunks are supported"; return false; }
  }
  out.next_pos = pos;
  return true;
}
}
