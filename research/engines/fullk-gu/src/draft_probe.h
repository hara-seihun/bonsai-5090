#pragma once
#include <cstddef>

namespace h5 {
enum class DraftProbeKind { TrunkResidual, HiddenProjection, TopValues, TopIndices };
enum class DraftProbeError { None, BadArgument, NoPrediction, BufferTooSmall, Cuda };
struct DraftProbeResult {
  DraftProbeError error;
  int cuda_code;
  int rows;
  int columns;
  std::size_t bytes;
};
}

// Copies a completed drafter's retained output, never inserts an in-flight pipeline fence.
// Float buffers for the first three kinds; int buffers for TopIndices. Only None has data.
extern "C" h5::DraftProbeResult h5_draft_probe(
    void* engine, h5::DraftProbeKind kind, void* host_output, std::size_t capacity_bytes);
