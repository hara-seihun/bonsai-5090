# Native image chat

Build against PrismML-Eng/llama.cpp `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`.
Dependencies: CUDA 12.8+ with SM120a, C++17, OpenSSL development headers/libs,
Prism `libllama`, `libggml`, `libggml-base`, and `libmtmd` (`cmake --build LLAMA/build --target mtmd -j4`).

```sh
make -j4 h5-server h5-vision-check LLAMA=/root/llama LLAMA_BUILD=/root/llama/build-vision
H5_THREADS=4 H5_WSCALE=e4m3 ./h5-server MODEL.ptq1.gguf DRAFT.safetensors \
  --slots 2 --ctx 4096 --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --vision-gpu 0 --image-max-tokens 1024
```

Official assets: `prism-ml/Ternary-Bonsai-2-27B-gguf` language PTQ1_0 and separate
`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (about 0.63 GB).
`--vision-gpu 0` is the explicit CPU encoder (default). `--vision-gpu 1` requires
a GPU backend in the linked ggml build; a missing backend is an error, not a silent CPU fallback.
Only the encoder changes device; language inference is always native CUDA.

### Focused CUDA build

The general Prism CUDA build instantiates every MMQ/FA quantization and MoE kernel;
`GGML_CUDA_FORCE_CUBLAS` does not remove these compilation units. Our encoder profile
owns a separate immutable-source copy and removes all those template families from
its build graph. Supported Q8_0/F16/F32/BF16 matmuls execute cuBLAS. MoE/flash and
excluded dispatch families explicitly fail; the projector loader rejects unsupported
weight types before encoding, rather than silently moving an unknown quantization
to CPU. GPU attention is ordinary dense attention, not flash attention.

```sh
python3 tools/prepare_vision_cuda.py --upstream /root/llama \
  --source /root/llama-vision-focused --build /root/llama/build-vision-focused \
  --cpu-build /root/llama/build-vision --threads 4 --compile
make -j4 h5-server h5-vision-check LLAMA=/root/llama \
  LLAMA_BUILD=/root/llama/build-vision-focused
# Serve with --mmproj PATH --vision-gpu 1.
```

The recipe pins Prism `1a07bfa5`, verifies the generated build graph has no
`template-instances`, and never modifies the parent source or CPU libraries.
The optional same-revision CPU library memo reuses `libllama` and `libmtmd` only:
mtmd has no CUDA-specific compilation and dispatches via ggml's backend registry.
The new ggml CUDA libraries remain in the focused build. Reused binary hashes are
recorded in `halo-reused-libraries.json`. Without `--cpu-build`, the same recipe
also builds those two libraries from source. A manifest prevents unnecessary
re-extraction/rebuild when the owning recipe's outputs have not changed.
Use `h5-vision-check ... --encode-gpu` for a real GPU encoder-only check.
The Prism vocab-only loader omits geometry; its `no_alloc` loader asserts on
Hadamard metadata. Thus a vision-enabled server also holds a CPU mmap model view
for mtmd's vocabulary/geometry. It creates no llama language context and never calls
`llama_decode` for served requests. Text-only mode remains vocab-only.

Image content parts use the OpenAI `image_url` form (object with `url`, or a string).
PNG/JPEG data URLs and HTTP(S) URLs work; downloads are limited to 20 MiB and
redirects are rejected. Up to eight images are accepted, in content order, including
multi-turn history. `detail` is accepted but resolution is controlled by
`--image-max-tokens`. Invalid/unsupported inputs are explicit HTTP 400 errors;
a server without `--mmproj` rejects images instead of silently dropping them.
Audio/video and projectors requiring noncausal attention are not supported.

The encoder emits primal 5120-dimensional vectors. They replace the native residual
input *after* the ternary embedding table's Hadamard inverse, so they are not rotated
twice. Image tokens have a sentinel ID, never a token-table lookup. Native Qwen35
IMRoPE uses GGUF sections `[11,11,10,0]` and mtmd's `(t,y,x,z)` coordinates.
Causal/cache positions and GDN sequence order remain contiguous row ordinals,
independent of the compressed visual temporal positions. Subsequent text and
output tokens continue from mtmd's `next_pos`, not image token count.

Image requests use ordinary native greedy decode by default. Set request field
`vision_speculation: true` to feed captured native image-prefill features into
DFlash and strictly verify proposals against the native M-RoPE target. Drafter
positions remain ordinal (proposal quality); target positions remain compressed
visual coordinates, and pending GDN/KV commitment remains ordinal. No raw image
embedding is sent to a token-table lookup in the drafter. Text-only requests keep
their existing speculative path, including mixed scheduling batches. Native image
steps bypass CUDA graph capture and the persistent megakernel. Image speculation
is opt-in. Paired measurements and exact-control parity are reported below. HTTP timings include image encoding in TTFT,
`image_encode_s`, `image_tokens`, native prompt prefill wall time
(`native_prefill_s`, shared scheduling steps are charged to each participating request),
`proposed_drafts`, `accepted_drafts`, `verify_s`, and `draft_s`.

## Bounded tests

The synthetic OCR/chart/spatial harness `tools/vision_tasks.py` additionally needs
Python Pillow and DejaVu Sans (`python3-pil` and `fonts-dejavu-core` on Ubuntu).
The combined compact+GPU-vision build and text-serving repeat are recorded in
[the integration report](../../results/integrated-vision/README.md).

```sh
make check-chat LLAMA=/root/llama
H5_THREADS=4 H5_WSCALE=e4m3 python3 tools/vision_smoke.py \
  --model MODEL --draft DRAFT --mmproj MMPROJ --output results/vision-smoke
./h5-vision-check MODEL MMPROJ bench/vision/red.png --encode-only encode.json
./h5-vision-check MODEL MMPROJ bench/vision/green-large.png --encode-gpu gpu-encode.json
# One server lifetime: GPU encoding, full smoke, and paired ordinary/speculation measurements.
H5_THREADS=4 H5_WSCALE=e4m3 python3 tools/vision_smoke.py \
  --model MODEL --draft DRAFT --mmproj MMPROJ --vision-gpu 1 \
  --speculate --bench-spec --output results/vision-gpu-spec
# Exact-control parity: omit H5_WSCALE and use H5_ACT=int8 H5_WEIGHTS=int8
# H5_STATE=f32 H5_KV=f16 with --strict-match. Longer than a shell-tool turn on startup.
# CPU reference may take longer than a shell-tool turn; use an attended/durable job.
./h5-vision-check MODEL MMPROJ bench/vision/red.png --reference-only reference.json
H5_THREADS=4 H5_WSCALE=e4m3 ./h5-vision-check MODEL MMPROJ bench/vision/red.png \
  --native-compare reference.json
```

The reference uses the same real encoder output and decoder coordinates, but Prism
`llama_decode` for language inference. It writes `reference.json.logits`; the native
comparison writes `reference.json.native.json` with top-1, reference-to-native KLD,
and logit cosine. This is not a substitute for task accuracy or a broad vision eval.

The two 112×112 fixtures have an unlabeled red or blue square on white; the
512×512 fixture has a green circle. Image colors do not appear in request text.
October 6 dedicated RTX5090 smoke passed the two small color cases, two-image
streamed ordering, text generation afterward, and malformed-image rejection,
with the Q8_0 CPU encoder. Each small image has 16 visual tokens; encoder time was
about 0.33 seconds. Real same-embedding/position reference comparison on the red
fixture gave reference-to-native KLD **0.0245973**, cosine **0.9937296**, and different
first token IDs (native 5844, reference 1114). This is not bit identity.
[Raw smoke/comparison evidence](../../results/vision-smoke/) is preserved.
The parent's INT8-activation/F32-state/F16-KV control matched reference top-1
(1114), with KLD **0.00004032** and cosine **0.9999658**. This separates embedding/
IMRoPE mapping from production FP4 quantization noise.
### Focused GPU encoder and image speculation results

The focused CUDA backend passed real encoder and full HTTP tests on the dedicated
RTX5090. The 256-token green image crosses 128-row native prefill chunks; concurrent
text/image, two-image SSE ordering, text-after-image, and malformed-input tests pass
with opt-in image speculation too. Language inference remains native throughout.
Cold standalone encoder calls took 216 ms (16 image tokens) and 199 ms (256);
steady-state server GPU encoding took 4.33 ms and 27.73 ms respectively.
[Encoder evidence](../../results/vision-gpu-encode/).

Three paired 128-output-token measurements with `H5_WSCALE=e4m3`:

| Fixture/prompt | Ordinary decode tok/s | Image spec tok/s | Accepted/proposed |
| --- | ---: | ---: | ---: |
| Red square, detailed caption | 157.5 | 266.4 | 72/399 |
| Green circle, Pillow explanation | 156.7 | 469.9 | 96/224 |
| Blue square, 1582-row long prefix | 155.8 | 314.4 | 81/336 |

These are measured strict target acceptance, not forced acceptance. Production
FP4 greedy strings differ between ordinary and speculative batch shapes. Both
paths identified the fixture colors/shapes correctly, but captions and the Pillow
explanation were token-capped; this is not a runnable-code accuracy result.
[Paired outputs/timings](../../results/vision-gpu-spec/).

A shared-head precision defect initially crashed the alternate-precision DFlash
control: NVFP4 drafter scope also selected a nonexistent NVFP4 representation of
the INT8 native LM head. The fix scopes head activation preparation and projection
to target precision, restores the previous scope value, and rejects missing GEMM
weight representations explicitly. Drafter projections and default FP4 behavior
are unchanged. After this repair, `H5_ACT=int8 H5_WEIGHTS=int8 H5_STATE=f32
H5_KV=f16` (without `H5_WSCALE`) passed all three **identical 128-token outputs and
usage**, plus full multi-image/concurrency smoke. Control ordinary/spec decode
rates were 37.7/79.3, 37.7/96.5, and 37.6/94.0 tok/s.
[Exact-control evidence](../../results/vision-gpu-spec-control-fixed/).

The parent's deterministic OCR/inventory/table, bar-chart reading/arithmetic, and
spatial layout suite scored **10/10 ordinary and 10/10 speculative** using default
FP16 weight scales and a 1024-image-token cap. Fixture images, questions/expected
answers, actual responses, and scores are [preserved](../../results/vision-tasks-fp16scale/);
its generating/running harness is `tools/vision_tasks.py` in the parent's integrated
branch. These synthetic tasks and color fixtures are bounded functional checks,
not a broad VQA benchmark or a claim about arbitrary photos.
