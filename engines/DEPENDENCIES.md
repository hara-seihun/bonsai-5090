# Building the exported engines

No dependencies, compiler outputs or weights are vendored beyond the small explicitly licensed HIP HTTP/JSON source dependencies. Use a separate dependency source directory outside this repository.

## PrismML llama.cpp

The exported engines use this API revision (different from the root B64 engine's pin):

- repository: <https://github.com/PrismML-Eng/llama.cpp>
- commit: `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`
- license: MIT, included in `licenses/llama.cpp-MIT.txt`

A source-only download avoids importing Git history:

```sh
mkdir prism-llama
curl -fL https://github.com/PrismML-Eng/llama.cpp/archive/1a07bfa5f4144274c8f1c9963821dd9d9a51854b.tar.gz \
  | tar xz --strip-components=1 -C prism-llama
cmake -S prism-llama -B prism-llama/build -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DLLAMA_CURL=OFF
cmake --build prism-llama/build --target llama llama-common mtmd -j4
```

Install compiler/CMake/OpenMP/OpenSSL development prerequisites appropriate to your OS first. CUDA engines require CUDA12.8+ (`sm_120a`); HIP requires a ROCm compiler supporting `gfx1151`. The CPU ggml build suffices for vocabulary and CPU image encoding. `mtmd` is only required for the5090 server/image targets. Halo6000 links `llama-common`; its include paths expect upstream `common/` and vendor headers.

The HIP Makefile historically links from `LLAMA/build-hip/bin`. Either configure/build the same upstream dependency in `build-hip` or explicitly override `LDFLAGS` to its actual library directory. `make LLAMA=/absolute/path/to/prism-llama` overrides the dependency root. Always use absolute library paths for embedded rpaths.

For the optional5090 CUDA image encoder, the included `halo-5090/tools/prepare_vision_cuda.py` prepares a focused same-revision build; inspect its `--help`. The native language engine remains CUDA in CPU-encoder mode. A missing requested encoder backend is an error, not an alternate language execution route.

## Python harnesses

Core clients use Python3 standard-library HTTP/JSON plus `requests` where imported. Evaluation tools additionally use `numpy`, `datasets`, `transformers`, `math-verify` and task-specific scorers as named by their imports/CLI. Vision fixture generation uses Pillow and a DejaVu font. Code-evaluation harnesses execute generated code: run them in your own isolated evaluation environment. Dataset/model downloads are separate explicit operations.

## Model inputs

- Both27B engines: `prism-ml/Ternary-Bonsai-2-27B-gguf`, language PTQ1_0 (HIP also supports PQ2_0). DFlash2: `incoai/Qwen3.8-27B-DFlash2`. HIP MTP uses `Qwen/Qwen3.8-27B` shard18.5090 image input additionally needs official `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`.
- Halo6000: custom ternary Qwen3.6-35B-A3B v1 and/or its measured practical hybrid, not an arbitrary stock Qwen checkpoint. Source/cache identifiers are in retained receipts. Its custom model preparation and frozen intrinsic fixture acquisition are not fully bundled.

Historical benchmark input hashes remain in receipts. This export does not silently substitute current model revisions for them. Root `scripts/fetch_model.py` serves the root engine's separate pinned model contract; inspect identities before sharing a downloaded input across variants.
