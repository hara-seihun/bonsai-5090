# Building the research sources

The export contains source, not prebuilt artifacts. Nothing in this guide was executed as a GPU run during publication. Rebuilt artifacts have new identities; retained measurements refer to the source/binary/compiler identities in their own receipts.

## Toolchains and inputs

- Blackwell RTX5090 kernels require CUDA with `sm_120a` support (recorded builds CUDA13.0/13.1), a compatible host C++17 compiler, and the CUDA SDK. Use explicit `-gencode arch=compute_120a,code=sm_120a`; generic compute_120 PTX cannot express every instruction here.
- CPU witnesses use GCC/Clang C++17 or C++20 as named below. Dict8 CPU cache tooling links OpenSSL Crypto on Linux or BCrypt on Windows. Python3 is used for source generation/test harnesses. Lean proofs use the Lean4 version recorded by their receipt; packing Round23 pins 4.33.0.
- No dependency source tree is vendored here. Binary engine Makefiles expect a separately built PrismML llama.cpp source/build via `LLAMA`/`LLAMA_BUILD`. The recorded binary oracle pin is `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`; target `Bonsai-27B-Q1_0.gguf` is 3,803,452,480 bytes, SHA256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`. DSpark BF16 is separately pinned in `engines/binary-spec/tools/binary/README.md`.
- Five-wide/Dict8 are the **ternary Bonsai2** line, not that binary checkpoint. Exact model SHA256 is `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`, 5,946,648,928 bytes. Original model/cache/weight-row/activation/logit files are deliberately absent. Supply your own licensed matching model; payload identity/ordered-layer pins remain enforced.

## Binary engine variants

Choose one directory under `engines/`; never overlay variant source files together. Linux builds use its own Makefile, for example:

```sh
cd research/engines/binary-spec
make -j2 NVCC=/usr/local/cuda/bin/nvcc \
  LLAMA=/absolute/prism-llama LLAMA_BUILD=/absolute/prism-llama/build h5 h5-server
```

The source's Makefile enumerates all engine translation units, binary experiment dependencies and local headers. `binary-windows-current` also retains the original Windows CMakeLists; configure from an x64 MSVC developer shell with explicit `LLAMA_ROOT` and `LLAMA_BUILD`. Its source-only `experiments/b32-imma` dependency is included. `binary-bmega` retains its distinct Makefile changes and `src/bmega.*`; `H5_BMEGA=1` is opt-in. Binary mode, state/KV formats and experimental environment selectors are in each `tools/binary/README.md` and experiment README. Do not import a ternary serving preset into binary measurements.

Some authoring scripts mention the original remote host, private object cache, profiler transport or omitted benchmark data. They are historical experiment drivers, not mandatory build dependencies. Use the local Makefile/source and explicitly supplied model/dependency paths. No automatic publication/remote-host replay is part of these instructions.

## Ternary whole-engine integrations

`fullk-gu`, `packed-tail`, `h1-graph-v3`, `dict8-v2`, `dict8-offline-identity` and `dict8-joint-consumer` retain their complete engine source and generated headers. Their original top-level CMakeLists predate the experiment; native experiment build scripts originally linked **hash-bound inherited objects**. Those objects are excluded. Use the source-only research CMake recipe instead:

```sh
cmake -S research -B /absolute/research-build \
  -DRESEARCH_VARIANT=dict8-offline-identity \
  -DLLAMA_ROOT=/absolute/prism-llama
cmake --build /absolute/research-build --parallel 2
```

This compiles the retained original API/Q6 source plus the exported frozen-width dispatcher and variant translation units, rather than borrowing opaque `.obj` files. `external-serving/README.md` describes their identities and the dispatcher boundary. This source rebuild is not a claim of byte-identical reproduction of any recorded Windows DLL, nor renewed numerical/performance acceptance. The build creates a library only and runs nothing. The independently generated joint consumers are compiler/CPU candidates; they are not automatically inserted into the Dict8 Engine.

## Dict8 offline cache and shared consumers

Start at `engines/dict8-offline-identity/experiments/gu-dict8-cache/README.md`. The readable source pack already includes generated `dict8_model.hpp`, `dict8_cache.hpp`, `dict8_binding.hpp`, codec and parser. `generate.py`/`integrate.py` describe how they were derived. Don't rerun a generator merely to build the supplied source: the semantic binding intentionally covers the exact generator and source; changing a binding changes the cache generation.

Compile CPU cache producer (no CUDA initialization):

```sh
cd research/engines/dict8-offline-identity
g++ -std=c++17 -O3 -Isrc experiments/gu-dict8-cache/cache_tool.cpp \
  src/gguf.cpp -lcrypto -pthread -o /absolute/cache_tool
```

CLI: `cache_tool prepare|verify|compare MODEL CACHE THREADS MAX_BYTES HEADROOM_BYTES`. All paths/budgets are explicit. It refuses overwrites and mismatched model/cache/header/payload identities; no runtime encode fallback exists. Windows uses MSVC C++17 and BCrypt. The single-scan identity, ABI, fault-receipt and query tests can be run individually with Python3 from that snapshot. The historical `test_cpu.py` real-row test requires excluded original row fixtures; it is not a portable synthetic substitute. CPU receipts are retained under `evidence/dict8-identity/`.

Joint search source is `engines/dict8-joint-consumer/experiments/joint-consumer/`. `generate.py` requires the exact baseline `src/dict8.cu`. `reproduce.py` documents all eight compile flags and source-suffix checks, but its actual-row stage requires excluded gate/up fixtures. Build a consumer without running CUDA or requiring those fixtures:

```sh
cd research/engines/dict8-joint-consumer
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_120a -lineinfo \
  -DH5_PROFILE_GEMM -DH5_PDL_EARLY_GEMM_RELEASE -DH5_PDL_GDN_STATE_OVERLAP \
  -DH5_NO_L2PF -DH5_PROFILE_SCALE_CARRIER -DH5_WEIGHT_TMA_CACHE_POLICY=2 \
  -Isrc -Iexperiments/joint-consumer -cubin \
  experiments/joint-consumer/shared8_compact.cu -o /absolute/shared8_compact.cubin
```

The self-contained `circuit/witness.cpp` exercises the finite six-gate encoding without a model fixture; `circuit/Carrier.lean` and `JointCarrier.lean` state the formal contract. Source-only compiler closure is distinct from engine integration. Split53/literal3 require a new payload/cache semantic identity; shared8 still needs deliberate integrated-consumer binding.

## Standalone five-wide and FP4/TF32 kernels

These `.cu` files are complete executable harnesses. Compile each as its own program; they intentionally define separate `main`s. Preserve the receipt's flags. Representative full-K build:

```sh
nvcc -O3 --fmad=false -std=c++17 \
  -gencode arch=compute_120a,code=sm_120a -Xptxas=-v \
  research/five-wide-native/fullk_rolling.cu -o /absolute/fullk_rolling
```

Synthetic full-K executable arguments are `[M] [scale_context]`, e.g. 17408/0 for the retained full-geometry scope. It allocates large validation images and writes generated synthetic binaries; do not execute it on a shared GPU without independent run ownership. `crt5*.cu`, `affine5*.cu`, `fullk_*.cu` and `scaled_word*.cu` retain wins, losses and compile/layout failures separately.

FP4/TF32 CPU searches:

```sh
g++ -std=c++20 -O3 research/fp4-chain/five_search.cpp -o /absolute/five_search
g++ -std=c++20 -O3 research/fp4-chain/tf32_four_search.cpp -o /absolute/tf32_four_search
lean research/fp4-chain/Observer.lean
```

These produce/check mathematical finite tables, not model activations. Native `native_prime.cu` generates `native-prime-observer.bin`; the later `native_readout`/`native_tile_*` programs consume that immutable synthetic witness. Its binary is excluded; source generation is retained. Build with the standalone CUDA flags above, plus any source-specific flags documented in its receipt. K16/K64 retain their individual sources and numeric receipts. No generated witness is a learned model kernel.

## Packing Round23

`packing-round23/method/packing-swarm/tools/round23/README.md` owns the original recipe; `packing-round22` supplies inherited source/proofs/invoice. Use an installed Lean4.33.0 executable in the proof replay's `LEAN` setting (the historical absolute toolchain path is a sanitized locator). `proof/replay.py` receives explicit `--root` and repeated `--contract`; the four Lean modules sit together in `proof/`. It audits finite declarations/axioms, not CUDA.

All saved model/tape payloads and compiler expansions are omitted. CPU tape replay and full cost expansion require supplying the original pinned tapes where their APIs request them; the export does not silently regenerate trained inputs. The native packed readers, build generators, inherited headers, proof source and compact compiler/CPU/invoice receipts are included. Source/invoice closure does not establish hardware timing, occupancy, numerical model acceptance or a winning candidate.
