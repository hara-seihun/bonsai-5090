#include "draft_contract.h"
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace h5;
int main() {
  float margin;
  assert(parse_draft_margin("128", margin) && margin == 128.f);
  assert(!parse_draft_margin("inf", margin) && !parse_draft_margin("nan", margin));
  assert(!parse_draft_margin("-1", margin) && !parse_draft_margin("8oops", margin));
  assert(parse_proposals("64") == 64 && parse_proposals("64oops") == -1);
  assert(parse_proposals("999999999999999999999999") == -1);
  assert(draft_config(15, 1, false, 4096, 128).error == DraftError::UnsupportedLength);
  assert(draft_config(16, 2, false, 4096, 128).error == DraftError::SingleStreamRequired);
  assert(draft_config(64, 1, true, 4096, 128).error == DraftError::TreeLengthUnsupported);
  assert(draft_config(64, 1, false, 65, 128).error == DraftError::Capacity);
  assert(draft_config(7, INT_MAX, false, 4096, 128).error == DraftError::Capacity);
  assert(draft_config(7, 32, true, 4096, 512).engine_rows == 776);

  for (int proposals : {7, 16, 32, 64}) {
    const auto c = draft_config(proposals, 1, false, 4096, 128);
    assert(c.error == DraftError::None && c.verifier_rows == proposals + 1);
    assert(c.query_rows == c.verifier_rows && c.pending_rows == c.verifier_rows);
    // All accepted-prefix/rollback choices, with both full-block and single-row next visits.
    for (int committed = 0; committed <= c.pending_rows; committed++) {
      for (int rows : {1, c.pending_rows}) {
        std::vector<double> prior(c.pending_rows), current(rows), pending(rows);
        for (int i = 0; i < c.pending_rows; i++) prior[i] = i + 1;
        for (int i = 0; i < rows; i++) current[i] = 100 + i;
        double state = 3., saved = state;
        int replayed = 0, done = 0, visits = 0;
        do {
          const auto chunk = replay_chunk(committed, rows, replayed, done, 16);
          assert(chunk.replay_rows + chunk.new_rows <= 16);
          assert(chunk.replay_rows + chunk.new_rows > 0);
          if (!committed && visits == 0) saved = state;
          for (int t = 0; t < chunk.replay_rows; t++) state += prior[chunk.replay_offset + t];
          if (chunk.replay_rows && replayed + chunk.replay_rows == committed) saved = state;
          for (int t = 0; t < chunk.new_rows; t++) {
            const int offset = chunk.new_offset + t;
            pending[offset] = current[offset]; state += current[offset];
          }
          replayed += chunk.replay_rows; done += chunk.new_rows; visits++;
          assert(visits <= 10);
        } while (replayed < committed || done < rows);
        assert(saved == 3. + committed * (committed + 1.) / 2.);
        assert(pending == current);
        assert(state == saved + rows * 100. + rows * (rows - 1.) / 2.);
      }
    }
    // Query tiles must see the ENTIRE noncausal block, not just their own 8 rows.
    for (int start : {0, 1020, 1118, 2241}) {
      for (int block = 0; block < 2; block++) {
        const int b0 = block * c.query_rows;
        for (int tile = 0; tile < c.query_rows; tile += 8) {
          assert(draft_key_end(start + tile, b0 + tile, c.query_rows) == start + c.query_rows);
        }
      }
      // Rejected speculative rows must not overwrite still-visible accepted context.
      constexpr int ring = 1120, window = 1024;
      std::vector<int> cache(ring, -1);
      for (int pos = std::max(0, start - window + 1); pos < start; pos++) cache[pos % ring] = pos;
      for (int pos = start; pos < start + c.query_rows; pos++) cache[pos % ring] = pos;
      for (int accepted = 0; accepted <= proposals; accepted++) {
        const int next = start + accepted + 1;
        for (int pos = std::max(0, next - window + 1); pos < start; pos++) assert(cache[pos % ring] == pos);
      }
    }
    std::vector<int> draft(proposals, 4), target(proposals + 1, 4);
    assert(draft_accepted_prefix(target.data(), draft.data(), proposals) == proposals);
    for (int mismatch = 0; mismatch < proposals; mismatch++) {
      target[mismatch] = 5;
      assert(draft_accepted_prefix(target.data(), draft.data(), proposals) == mismatch);
      target[mismatch] = 4;
    }
    draft[0] = target[0] = 248044;
    assert(draft_accepted_prefix(target.data(), draft.data(), proposals) == 1);
    printf("proposals=%d query_rows=%d verifier_rows=%d pending_rows=%d: capacity/replay/rollback/attention/ring/EOS pass\n", proposals, c.query_rows, c.verifier_rows, c.pending_rows);
  }
  assert(relaxed_draft_allowed(4, 5, 10.f, -20.f, 30.f, 248320));
  assert(!relaxed_draft_allowed(4, 5, 10.f, -20.f, 29.f, 248320));
  assert(!relaxed_draft_allowed(248044, 5, 10.f, 10.f, 1e10f, 248320));
  assert(!relaxed_draft_allowed(4, 248046, 10.f, 10.f, 1e10f, 248320));
  assert(!relaxed_draft_allowed(4, 5, INFINITY, 10.f, 1e10f, 248320));
  assert(!relaxed_draft_allowed(4, 5, 10.f, NAN, 1e10f, 248320));
  puts("relaxed severe-loss margin and strict EOS/nonfinite contracts pass; bonus remains outside relaxed rows");
}
