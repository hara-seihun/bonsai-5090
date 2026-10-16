# Bonsai2 27B: paired custom needle retrieval

RTX 5090, October 16, 2026. Fully rendered prompt lengths; output headroom is additional.

| Prompt tokens | Halo | Same-weight native reference |
|---:|---:|---:|
| 32,768 | 20/20 | 20/20 |
| 65,536 | 20/20 | 20/20 |
| 131,072 | 19/20 | 20/20 |
| Total | **59/60 (98.33%)** | **60/60 (100%)** |

Custom NIAH panel: one- and four-needle exact-JSON retrieval, two content seeds
(20261015/20261016), five depth phases (0, .25, .5, .75, 1), three lengths.
For four needles the positions are `(depth_phase+i)/4`, `i=0..3`.
Both engines receive the identical rendered prompt, with thinking enabled/xhigh,
temperature 1, top_p .95, top_k 20, min_p 0, 8,192 output-token cap.
Same GGUF SHA256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
Native reference retains F32 recurrence/F16 KV; Halo's complete environment is in
[runtime-settings.json](runtime-settings.json). Seeds match, sampling implementations differ.
This panel is distinct from the published PrismML suite and full RULER.

The only miss is Halo `131072-multi-d000-s20261015`: 8,192 output tokens,
unclosed reasoning and an empty final answer. The other 119 responses exactly match
the gold JSON. No correct response was capped; treating every capped response as wrong
leaves the table unchanged. All 60 pairs/120 records are present; all six arms completed
and teardown was recorded. Original controller completed at 04:39:05Z.

## Immutable evidence

- [Validated summary](validated-summary.json), exact original SHA256
  `2b4befc646baa4be01f29c3436326e3eafa579b5d15357d2f26dcfccc98ee169`.
- [All 120 scored records](results.json), including the failed response.
- [Prompts](prompts.tar.gz): all 60 complete prepared case JSONs, rendered text,
  exact token IDs, gold answers, seed/sampling parameters and measured needle offsets;
  the original preparation manifest is included. No prompt truncation.
- [Raw evidence](responses.tar.gz): 660 byte-identical files: all 120 raw responses,
  HTTP requests, score records and response acknowledgements, plus all 60 native
  tokenizer requests/responses/acknowledgements. Raw responses retain reasoning,
  final text, token usage and finish reasons.
- [Original 711-file manifest](raw-manifest.json), SHA256
  `f907e7906799849e3d3ff6a229946e2a41055b3be92801497343c3ae670644cc`.
  The 51 operational controller/host files are hash-bound there rather than redistributed.
- [Source provenance and archive member hashes](source-provenance.json).
  [protocol.py](protocol.py) extracts the original generator, renderer and scoring functions
  byte-for-byte; private host/process orchestration is omitted. The original full source
  and frozen packet are SHA256-bound. The existing [engine](../../../engine/)
  and [build instructions](../../../README.md) supply the inference source;
  measured binary `quality_custom.dll` SHA256 `a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c`
  matches the existing [throughput receipt](../../../receipts/serving-b64x128.json).
  Native reference is the [pinned PrismML fork](../../../NOTICE).
- Exact [chat template](assets/chat_template.jinja) and immutable upstream
  [tokenizer URLs/hashes](assets/tokenizer/source.json). These assets originate from
  PrismML's Apache-2.0 model. No weights are redistributed.

## Offline reproduction

From the repository root (Python standard library, no GPU/network):

```sh
python3 -B benchmarks/niah/paired-20261016/replay.py
```

This checks compressed archives and every member hash, prompt/request identities,
all raw scores, and matched native tokenization. It reports 59/60 versus 60/60.
Original lifecycle assessment remains in the validated summary; this portable replay
recomputes quality from the retained evidence without launching processes.

## Fresh inference commands

These are equivalent portable commands, not a claim of a new measurement. Build the
repository and acquire the hash-pinned GGUF using the root instructions. For each
length run with context `length+8192`, one slot, prefill 256. For example, 128K:

```sh
python3 engine/serve.py --model "$MODEL" --profile profiles/serving.json \
  --slots 1 --ctx 139264 --prefill 256 --port 18571 --kv fp4-direct --layout planes
```

Use the exact environment in `runtime-settings.json`, including `H5_NO_SPEC=1`.
For the reference, use its recorded `native_command` (replace binary/model paths),
F16 K/V, `-np 1`, `-c 139264`, `-b 256 -ub 256`, no context shift/cache reuse.
Repeat for contexts 40960 and 73728. Send the archived raw request for that engine
without rerendering or changing sampling, e.g. in a separate fresh output directory:

```sh
mkdir retained-niah
cd retained-niah
tar xzf ../benchmarks/niah/paired-20261016/responses.tar.gz
curl --fail-with-body --max-time 600 --header 'Content-Type: application/json' \
  --data-binary @halo-c139264/131072-multi-d000-s20261015/wire-request.json \
  http://127.0.0.1:18571/v1/completions > fresh-response.json
```

The matching reference request omits `prompt_token_ids`; its tokenizer must first
match every supplied ID in the prepared case. Each HTTP request is one independent
response. Stop each owned server before switching engines. Keep fresh observations
separate from this retained generation.
