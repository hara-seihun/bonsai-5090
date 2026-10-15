# Coupled sampling trace analysis

`analyze_coupled_trace.py` is the CPU-only owner of offline trace analysis. It uses
Python 3's standard library; no model, GPU, NumPy, or build is needed.

```sh
python3 bench/analyze_coupled_trace.py --self-test
python3 bench/analyze_coupled_trace.py trace.jsonl --seeds 256 > analysis.json
```

## Input contract

Each nonempty JSONL line is a block object with these required fields (additional
metadata is allowed):

```json
{"seed":42,"first_index":0,"temperature":1,"top_p":1,"min_p":0,"top_k":1,"draft_temperature":1,"draft_top_p":1,"draft_min_p":0,"rows":[{"target_ids":[7],"target_scores":[0],"draft_ids":[7],"draft_scores":[0],"draft_token":7,"target_token":7}],"accepted_prefix":1}
```

- `seed` and `first_index` are uint64 integers. Row `i` uses the absolute sampling
  index `(first_index+i) mod 2^64`, as in the native sampler.
- Scores are finite **post-presence/repetition-penalty, pre-temperature** values.
  Do not pass probabilities or logits before penalties.
- Target arrays contain exactly the `top_k` highest target candidates, including
  score-tie selection by ascending token ID. `top_k` must be 1–128; this trace
  format cannot reconstruct an unbounded (`top_k=0`) target distribution.
- Draft arrays contain the transition scores of the raw 16-candidate proposal
  lattice. Smaller arrays (1–16) are accepted for synthetic fixtures. Candidate
  IDs within either array must be distinct nonnegative int32 integers. Arrays
  need not be sorted.
- Both temperatures must be in `(0,2]`, both `top_p` values in `(0,1]`, and both
  `min_p` values in `[0,1]`. Greedy/temperature-zero traces are unsupported.
- `accepted_prefix` is the number of consecutive matching rows at the start.
  It must match the recorded tokens: when less than `len(rows)`, the next row
  must disagree. Only those matching rows **and the first mismatching row** are
  analyzed. Later rows are schema-checked but do not contribute to any metric
  or token replay check. No unrecorded bonus row is invented.

The sampler semantics come from [`src/sampling.h`](../src/sampling.h), the draft
selector in [`src/draft.cu`](../src/draft.cu), and the uint64 keyed noise in
[`src/sampling_noise.h`](../src/sampling_noise.h). The analyzer sorts by score
descending/token ID ascending, applies top-k, applies relative `min_p`, then
keeps the inclusive `top_p` crossing token and renormalizes. Optional
`draft_top_k` independently selects proposal top-k (0 means all raw candidates);
when omitted it uses target `top_k`. Positive proposal top-k is bounded by the
number of raw draft candidates. Noise races use the original
relative scores divided by temperature, avoiding unnecessary normalization
roundoff. The recorded seed is replayed for both distributions on every realized
row; a mismatch is an explicit error, not an extra measurement.

## Output

The single JSON result includes block/realized/ignored-row counts, observed
agreement, mean target entropy in nats, and mean **retained-target probability
mass** covered by the raw draft candidate IDs. Means give equal weight to each
realized row. `baseline` holds the recorded proposal configuration.

`ranking` compares ten proposal configurations: draft-temperature multipliers
`0.7, 1, 1.3, 1.6, 2`, each with either recorded proposal filters (`configured`)
or `top_p=1, min_p=0` (`untruncated`). Untruncated means the full proposal lattice
subject to the unchanged top-k, not the model's full vocabulary. Temperatures
are capped at the runtime maximum 2; effective ranges and clamped-row counts
are reported, and coincident configurations remain separately labeled.

For each configuration, the result reports empirical shared-Gumbel agreement,
mean proposal entropy, mean target mass on retained proposal support, and mean
retained proposal support size. Agreement uses `--seeds` races on **each** realized
row, with uint64 seed keys `(recorded_seed+1)..(recorded_seed+seeds)` modulo 2^64.
Each trial shares token-keyed noise across target and every proposal configuration;
the recorded seed itself is not counted. Ranking is descending agreement, then
retained target mass, then configuration name. `agreement_count` and
`draws_per_config` expose the denominator.

These comparisons hold the trace's realized prefixes fixed; they do not fabricate
new prefixes or extrapolate a whole generated trajectory. The target distribution
is unchanged across proposal comparisons.

Malformed JSON, missing/invalid fields, unsupported sampling configurations,
contradictory acceptance counts, and recorded-token replay mismatches exit 2 with
a file/line/row error on stderr and no partial JSON report. An empty input is an
error. `--self-test` checks identical distributions agree on all 256 races,
inclusive top-p crossing, tied-score ID ordering, the uint64 mixer, and canonical
C++ keyed-noise vectors.
