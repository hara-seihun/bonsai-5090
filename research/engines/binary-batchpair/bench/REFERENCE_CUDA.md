# Focused PrismML CUDA reference

`build_cuda_reference.sh`, `prepare_cuda_reference.py`, and
`reference_cuda_probe.cpp` own an independent later-position reference for the
Bonsai Chat native-engine repair. This is generated diagnostic tooling, not the
public engine or its tokenizer ABI. Parent evidence lives in
[`results/chat-repetition/`](../results/chat-repetition/README.md).

Existing gpu-host source `/root/llama` is pinned at
`1a07bfa5f4144274c8f1c9963821dd9d9a51854b`. The generated snapshot, isolated build,
probe and `artifacts.sha256` live under `/root/llama-reference-cuda/`. Existing
`/root/llama/build/bin` and native serving are never replaced.

The preparation script reads the model GGUF tensor table, retains MMQ template
families actually used by that model, and removes other MMQ dispatch cases.
Unsupported MMQ types abort explicitly. Flash attention is disabled; its entry
points are replaced with rejecting stubs and its template families are not
compiled. The current model tensor types are PTQ1_0 (402), F32 (353), and BF16
(96); only PTQ1_0 needs an MMQ template. Other CUDA operations and model code
come from the pinned source.
CUDA 12.8, sm120a and g++13 are selected explicitly. Only target `llama` and the
reference probe are built; examples, common utilities, app, server and tests are
disabled. `/root/llama-reference-cuda/source.sha` is the completed snapshot marker.
Remove the generated `source/`, `build/` and marker together to regenerate after
changing the preparation script. The entire generated remote directory is
removable without affecting serving or models.

## Build receipt, October8

Local kjob `bonsai-cuda-reference` completed with exit0. Probe `--help` and dynamic
link resolution succeeded; all llama/ggml libraries resolve to the isolated
build directory. Original `/root/llama` remains clean. No model load or GPU
inference was performed by this build task.

- Probe SHA256: `5a21c14e18050f0a8729cc124c64281d9ac781bc87075c3a7eb3997904da764f`.
- CUDA library SHA256: `4392717d7d47da1360b9731d8ff1aca35b32052d48b749c4dea684af79c14850`.
- Full hashes: remote `/root/llama-reference-cuda/artifacts.sha256`.
- Durable log: `~/.local/state/kjob/bonsai-cuda-reference/log`.
- GPU after build:12422MiB used of32607MiB.

The later coordinated teacher-prefix run completed with30prompt and256teacher
positions, no generated continuation, and five saved vocabulary rows. All65/65
layers were offloaded. Reported CUDA buffers were5395.33MiB model,32MiB KV,
149.62MiB recurrent state, and132.50MiB compute at512allocated context positions.
It exited and unloaded before the full-context public engine deployment. The
paired raw logits and measured distribution differences are retained in the
[parent evidence](../results/chat-repetition/README.md).

Copy these three source files to that remote directory, then submit a durable
local kjob whose foreground command is:

```sh
ssh gpu-host bash /root/llama-reference-cuda/build_cuda_reference.sh
```

The build does not perform inference. Coordinate GPU use with the serving owner
before executing the probe. The probe requests all model layers on CUDA and
prints llama load/buffer diagnostics to stderr, so retain stderr for actual GPU
footprint and backend confirmation.

## Invocation and output

```sh
/root/llama-reference-cuda/reference_cuda_probe \
  /root/models/bonsai.gguf /path/prompt.txt /path/teacher.bin \
  /path/result-prefix 0 /path/sampling.json
```

`prompt.txt` is exact UTF-8 prompt bytes; tokenization uses `add_special=false`,
`parse_special=true`. `teacher.bin` is little-endian uint32 token count followed
by that many little-endian int32 token IDs. A zero count requests no teacher
prefix. `GENERATE_STEPS` is an explicit integer 0 through4096. Total trajectory
must not exceed8192 positions. Context size is the trajectory length plus32.

`sampling.json` requires all fields, for example:

```json
{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0,"presence_penalty":1.5,"repetition_penalty":1,"seed":42}
```

Sampling uses the same shared native `sampling.h` selector. Teacher tokens mark
presence but do not consume RNG draws. After replay, generation starts from the
explicit seed; raw-logit comparison is independent of sampling configuration.

`result-prefix.logits` matches the native teacher probe format:

- Four little-endian int32 header fields:
  `[saved_rows,vocab,prompt_tokens,teacher_tokens]`.
- Each record: little-endian int32 continuation position, then `vocab`
  little-endian float32 logits.
- Position0 predicts the first teacher token. Position k predicts after
  consuming k continuation tokens. Saved positions are0,32,64,128,256 and the
  final teacher position, without duplicates. Only reached positions are saved.
- The saved-row count is finalized on successful completion. A nonzero process
  exit means outputs are incomplete and must not be treated as a finished run.

`.rows.jsonl` provides the same positions, next absolute position, consumed token
and float-array byte offset. `.tokens.json` records prompt, teacher and generated
IDs and configuration. `.text` contains only generated continuation text. A
successful run prints `reference_complete` with token/row counts to stderr.
Generated token pieces preserve special markers such as `</think>`; EOG still
stops generation before text output.
