#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <stdexcept>
namespace h5 {
enum class DsparkVocabErrorCode { invalid_source_vocab, empty_path, open_failed, read_failed, empty_selection, malformed_id, out_of_range, duplicate_id, only_mask };
struct DsparkVocabError { DsparkVocabErrorCode code; std::string detail; };
using DsparkVocabSelection = std::variant<std::vector<int>, DsparkVocabError>;
// Unset path selects the full source domain; set path is whitespace-separated
// unsigned decimal IDs, nonempty, unique and sorted into token-ID tie order.
DsparkVocabSelection read_dspark_vocab(const char* path, int source_vocab);
class DsparkVocabException : public std::runtime_error {
public:
  DsparkVocabErrorCode code;
  explicit DsparkVocabException(const DsparkVocabError& error)
    : std::runtime_error("DSpark vocabulary: " + error.detail), code(error.code) {}
};
// BF16 reference checkpoint route. Own embedding/head, native six-layer trunk,
// BF16 cuBLAS projections and sequential Markov resampling. Greedy only;
// query row0 predicts the token after the anchor, unlike DFlash2's discarded row0.
class NativeDspark {
  struct State;
  std::unique_ptr<State> state;
public:
  NativeDspark(const std::string& path, int slots, int context, int rows, int proposals, cudaStream_t stream);
  ~NativeDspark();
  const std::vector<int>& capture_layers() const;
  const float* confidence() const;
  int query_rows() const;
  int output_vocab() const;
  const std::vector<int>& output_ids() const;
  // Tight [query_rows(),output_vocab()] logits in output_ids() column order.
  cudaError_t read_base_logits(float* output, size_t elements) const;
  void reset(int slot);
  void copy_slot(int source, int destination);
  void run(const float* taps, int captured_rows, int streams, const int* slots, const int* ingest_count,
           const int* ingest_row, const int* ingest_position, const int* anchors, const int* starts, int* drafts);
};
}
