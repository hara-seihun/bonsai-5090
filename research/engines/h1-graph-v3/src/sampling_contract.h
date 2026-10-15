#pragma once
#include <cstdint>

struct H5SamplingConfig {
  double temperature;
  double top_p;
  double min_p;
  double presence_penalty;
  double repetition_penalty;
  int top_k;
};

extern "C" {
// Uses retained target-head rows. Prefix contains rows-1 proposed tokens;
// row i applies penalties to seen plus prefix[0..i). Noise is keyed only by
// request seed, absolute generated-token index and vocabulary token ID.
int h5_sample_target(void* engine, const H5SamplingConfig* config,
                     uint64_t seed, uint64_t first_index,
                     const unsigned char* seen, const int* prefix,
                     int first_row, int rows, int* out);

// Same causal ingest/anchor contract as h5_draft; sampled single-slot proposals
// share target noise keys. Approximate draft scores never replace target draws.
int h5_draft_sampled(void* engine, int ns, const int* slot,
                     const int* n_ing, const int* ing_row0, const int* ing_pos0,
                     const int* anchor, const int* start, int* drafts,
                     const H5SamplingConfig* config, uint64_t seed,
                     uint64_t first_index, const unsigned char* seen);
}
