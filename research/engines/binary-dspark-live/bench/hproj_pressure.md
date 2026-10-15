# Native hproj under CTA pressure

Owner: batch2; `bench/hproj_pressure.{cu,mk,md}` and `bench/hproj_pressure_native.py`.
No model/engine objects or HTTP service are needed. This is the small kernel-level successor
to the unresolved hproj diagnostics in `bench/multi_engine_batch.md`.

## What it tests

- Hproj: native `gemm_fp4_body<4,2,32,0,1>`, WF1 cfg3, R448/N256/K5120, ks1,
  MODE0, 14 CTAs. The wrapper has the production kernel's parameter order and launch bound.
- Pressure: native `gemm_fp4_body<8,1,64,0,2>`, WF2 cfg1, R128/N65536/K5120,
  ks1, MODE0, 512 CTAs per call: the actual pruned-head GEMM geometry, not a spin kernel.
- Every call has independent activations, activation scales, weights, weight scales, row
  scales, output and optional trace allocation. Inputs are immutable after setup. No alias,
  split reduction, unordered output atomic or inter-call read-after-write is introduced.
- Packed E2M1 nibbles cover all finite codes, positive E4M3 scale bytes vary by physical
  position, and finite activation/output row scales vary by row. Hash-derived operands are
  deterministic; each pressure call has its own seed. All operand bytes, including reserved
  scale bytes and row padding, are initialized. No model files or real weights are loaded.
- One isolated eager output for each work object is the baseline. Every later comparison
  checks **every FP32 bit**, finite values and both 64-word output guards. Work objects and
  operands stay identical across arm comparisons; MODE0 overwrites the output.
- Eight arms run each repeat: isolated hproj, isolated pressure, serial HP→pressure,
  serial pressure→HP, concurrent launches HP-first and pressure-first on two nonblocking
  streams, and both same-stream orders. `--pdl 0|1` is explicit and applies to the latter
  chains too. No CUDA event is inserted between kernels in a PDL chain.
- `--backend eager|graphs|both` uses identical kernel arguments. Graphs cover each independent
  branch and both same-stream chains. Concurrent graph arms launch the two independent graphs
  on separate streams. Baselines are shared across backends, so graph geometry is not compared
  only with itself.

It isolates the native pipeline under competing real kernel work. It does **not** reproduce
the model's activation preparation, writable g4 recycling, drafter ring, host metadata or
full CUDA graph topology. Passing here would narrow the defect to a path absent from this
unit; it would not establish model equivalence. With `RING_RELEASE=native`, helpers and body
are taken byte-for-byte from the selected `SOURCE_ROOT`; the generator stops at the unique ordinary-kernel wrapper
boundary. Changed/ambiguous boundaries fail the build. The all-shape/carrier launch registries
are excluded, so only the two tested kernel instantiations compile, with all selected native
defines retained. Generated code belongs in `BUILD`, not production source.

## Build without using a GPU

From the checkout containing these bench files, point `SOURCE_ROOT` at the **immutable
candidate source**, not necessarily the checkout containing this harness. For the known-failing
CUDA12.8 baseline, select the same native source as the model diagnostic (2809846), not the
later coherent-scale source. The harness also builds against the corrected source to compare
it, but those are separate build identities.

```sh
NATIVE_DEFS='-DH5_PROFILE_GEMM -DH5_PDL_EARLY_GEMM_RELEASE -DH5_PDL_GDN_STATE_OVERLAP -DH5_NO_L2PF -DH5_PROFILE_SCALE_CARRIER -DH5_WEIGHT_TMA_CACHE_POLICY=2'
# Set SOURCE_ROOT to the actual immutable candidate checkout.
make -f bench/hproj_pressure.mk SOURCE_ROOT="$SOURCE_ROOT" \
  BUILD=build-hproj-pressure NVCC=/usr/local/cuda/bin/nvcc \
  NATIVE_DEFS="$NATIVE_DEFS" CTA_TRACE=0 RING_RELEASE=native hproj-pressure
# Optional residency witness: separate binary/build identity, because instrumentation can affect scheduling.
make -f bench/hproj_pressure.mk SOURCE_ROOT="$SOURCE_ROOT" \
  BUILD=build-hproj-pressure-trace NVCC=/usr/local/cuda/bin/nvcc \
  NATIVE_DEFS="$NATIVE_DEFS" CTA_TRACE=1 RING_RELEASE=native hproj-pressure
build-hproj-pressure/hproj-pressure --help
```

`SOURCE_ROOT`, `BUILD`, `NVCC`, `NATIVE_DEFS`, `CTA_TRACE`, and `RING_RELEASE` are required. The selected headers,
full native source, harness/generator/makefile, compiler version and defines are hashed in
`BUILD/compile-config`; the JSON configuration emits its SHA256 build identity. Keep that
receipt with stdout/stderr and the binary. The build regenerates only when inputs change.
Local CUDA13.1 compiled both variants successfully, with exactly two kernels confirmed by
`cuobjdump --dump-resource-usage`; help/configuration checks do not launch kernels. GPU work
is reserved for the campaign owner's existing card after the numerical gates.

## Bounded GPU run

```sh
mkdir -p out/hproj-pressure
# No model startup or prefix replay. Use the matched CUDA12.8 build on the existing GPU.
timeout 45s ./build-hproj-pressure/hproj-pressure \
  --repeats 8 --pressure-calls 1 --backend both --pdl 1 --device 0 \
  --dump first-difference --output-prefix out/hproj-pressure/plain \
  >out/hproj-pressure/plain.stdout 2>out/hproj-pressure/plain.stderr
printf '%s\n' "$?" >out/hproj-pressure/plain.exit
# Repeat PDL0 under a new prefix, not by replacing the first receipt.
timeout 45s ./build-hproj-pressure-trace/hproj-pressure \
  --repeats 2 --pressure-calls 1 --backend both --pdl 1 --device 0 \
  --dump first-difference --output-prefix out/hproj-pressure/trace \
  >out/hproj-pressure/trace.stdout 2>out/hproj-pressure/trace.stderr
printf '%s\n' "$?" >out/hproj-pressure/trace.exit
```

All seven CLI options are required. Repeats1–64, pressure calls1–8. At one pressure call,
operands/guarded outputs occupy about126MB GPU memory; each extra pressure call adds about
123MB and its full baseline adds32MiB host memory. Allocation failures retain native CUDA
code/name/message. Baseline NaNs or damaged guards are runtime errors, never a valid reference.
The program records resources and per-arm wall time excluding initialization, output reset,
D2H comparison and dumps. These timings are not model throughput measurements.

With `CTA_TRACE=0`, concurrent means queued on independent streams; actual CTA overlap is
unknown. `CTA_TRACE=1` adds one thread0 timestamp at native-body entry and exit per CTA,
without a new barrier or dependency. It reports overlapping CTA-residency intervals and their
maximum intersection. This witnesses kernel residency, not just overlapping enqueue windows;
it does not count arithmetic cycles or establish same-SM co-residency. Use the uninstrumented
binary for the primary arithmetic/sanitizer test: timestamps can perturb the defect. All-exact
traced runs with no observed pressure exit4 rather than claiming pressure coverage.

A `comparison` event records full-output bit count/first index, guard differences, nonfinite
pairs, max/RMS delta and baseline/current hashes. On the **first failure in the whole run**,
`--dump first-difference` saves the entire baseline/current output of that work object as
native-endian uint32 words (`.baseline.u32`/`.current.u32`). The event identifies the prefix,
work index (0=hproj; 1+=pressure call), and element count; reshape with the emitted work
rows/n. Later failures still compare all elements, but do not produce unbounded duplicate
dumps. `--dump none` performs the same checks without artifacts.

Exit0 = all requested output checks exact (trace-off does not assert residency);
exit1 = native/runtime/IO/nonfinite-baseline/cleanup error;
exit2 = invalid configuration; exit3 = an output/guard difference;
exit4 = traced arithmetic exact but no pressure witnessed;
exit124 = timeout/unfinished, not an arithmetic or sanitizer result.

## Empty-barrier diagnostic control

`RING_RELEASE=lanes` generates a deliberate alternative to the native empty-barrier
handoff: expected arrivals become FW×32 and every consumer lane performs a release
arrival after its reads. Existing warp synchronization, full-barrier transactions,
stage counts, phases, input bytes and arithmetic are unchanged. This removes reliance
on the transitive `lane reads → syncwarp → leader arrival` aggregation. The generator
requires exactly one initialization and three release sites; unexpected source fails.
It preserves source line counts, and JSON/build identity names the selected protocol.
This is **not a production repair** or the byte-for-byte native control.

Build against the same immutable source/compiler/defines into a separate directory:

```sh
make -f bench/hproj_pressure.mk SOURCE_ROOT="$SOURCE_ROOT" \
  BUILD=build-hproj-pressure-lanes NVCC=/usr/local/cuda/bin/nvcc \
  NATIVE_DEFS="$NATIVE_DEFS" CTA_TRACE=0 RING_RELEASE=lanes hproj-pressure
```

Use the same full-output run and isolated one-launch sanitizer filter below. A difference
between protocols localizes the tool/source handoff question; it does not alone identify
whether the original protocol or the instrumenter's indirect-dependency tracking is wrong.

## Compute Sanitizer

The VM's actual `compute-sanitizer --help` was read from
`~/work/tune-5090/batch2/compute-sanitizer.help`. It supports `kne` (full mangled
name), `kns` (substring), and `regex`; launch counts include only matching kernels. These
wrappers have C linkage, so `kne=hproj_probe` matches exactly. Do not enable blocking launches
or force per-launch synchronization: that would remove the pressure being tested.

```sh
compute-sanitizer --help >out/hproj-pressure/compute-sanitizer.help
# Inspect installed help before applying these flags on another installation.
timeout 45s compute-sanitizer --tool racecheck \
  --kernel-name kne=hproj_probe --launch-count 16 --error-exitcode 5 \
  --racecheck-memcpy-async yes --racecheck-report all \
  ./build-hproj-pressure/hproj-pressure \
  --repeats 2 --pressure-calls 1 --backend eager --pdl 1 --device 0 \
  --dump first-difference --output-prefix out/hproj-pressure/race \
  >out/hproj-pressure/race.stdout 2>out/hproj-pressure/race.stderr
printf '%s\n' "$?" >out/hproj-pressure/race.exit
# One isolated baseline plus seven hproj launches/repeat =15 filtered launches.
```

Pressure kernels stay native/uninstrumented with that filter. Run a separate pressure-filter
check if its output differs. A bounded `--tool memcheck --kernel-name kne=hproj_probe
--error-exitcode 5` run checks hproj accesses; `--tool synccheck` checks synchronization.
Sanitizer exit5 is distinct from the harness's output mismatch3. A clean filtered racecheck
only covers the checked shared-memory/TMA path; the full output checks still decide arithmetic
repeatability. All model-specific state/lifetime questions remain with the two-engine harness.

## First GPU and racecheck receipts

The campaign owner's original-source PDL1, original-source PDL0 and coherent-scale PDL1
runs each passed all224 full-output checks. The separately instrumented residency witness
passed56 checks and observed16 CTA-pressure events. The immutable-input native unit therefore
has not reproduced the model's arithmetic drift.

Original CUDA12.8 code under Compute Sanitizer **2025.1.0.0/build35583870** reports internal
shared-memory WAR hazards even in the first isolated hproj baseline. The short analysis
run (`--launch-count 1 --racecheck-report analysis --print-limit 10`) completed in6.77s,
exit5: **16 errors, zero warnings**,10 displayed hazards; all14 harness output checks
remained exact. Reads at `fp4.cu:511/522` and `ldsm4` are paired with global→shared
bulk writes. Full receipt: `batch2/hproj-race-one/{stdout,stderr,receipt.json}` under
`~/work/tune-5090`. The earlier45s byte-report timeout was unfinished and
is superseded for tool completion, not deleted.

The source handoff is `consumer reads → syncwarp → leader empty.release → producer
empty.acquire → bulk overwrite`. PTX12.8
[successful acquire mbarrier wait](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)
guarantees pre-release-arrive regular accesses have been performed;
[bulk operations](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-operations)
are ordered after prior instructions in their issuing thread. NVIDIA's
[TMA staging example](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies-tma-one-dim-staging)
uses shared reads→CTA synchronization→overwrite without a per-reuse proxy fence.
No missing proxy fence is established for this read-only reuse path.

[Sanitizer release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
added global→shared bulk-copy support in2024.4, and subsequently fixed syncwarp false
positives (2025.1), transactional mbarrier arrive (2025.2), mbarrier false positives
(2025.4), and complete-tx false positives (2026.3). Those are concrete reasons to compare
the installed tool with a current tool on the **same binary**, not a diagnosis that this
report is false.

The same original binary also reports16 errors/exit5 under **2026.3.0** (6.24s), while
its all-lane release control exits0 (2.93s), with every output check exact. Receipts:
`batch2/hproj-race-latest/` and `batch2/hproj-race-lanes/`. This distinguishes the empty
handoff protocols under the current tool; it does not yet establish the model drift's cause.

The all-lane release only satisfied the tool: its SASS still issues the arrives with shared loads
pending. The model drift was the missing `fence.proxy.async` before the consumer release, which
ptxas exploited by issuing the arrive ahead of the slot's last loads. Fixed in bae0822
(`ring_release`); see [root cause](../results/integrated/DRAFT-RACE.md).

The full-model candidate **226c37f**, compiled CUDA12.8 with the flag and run eager,
allKS1,PDL1,`H5_DRAFT_HPROJ_FENCE=none`, **does not repair model repeatability**:
375 prefix checkpoints and verifier tokens remain exact, but215 draft tokens and801 confidence
values differ. Sixteen of20 HP checkpoints differ (204,927 FP32 bit-pattern differences);
trunk residual and top16 values/indices stay exact. Raw `batch2/ring-release/`. The runner
stopped after this negative; graphs/pair were not run for this candidate. Diagnostic wall time
includes full checkpoint readback and is not a throughput measurement. The sanitizer-clean
native protocol and model HP drift are therefore distinct measured facts. AFTER-hproj remains
the repeatable single-engine control; the underlying model cause is still unresolved.
