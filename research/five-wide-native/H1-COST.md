# H1: whole-route cost evidence

## Caller-binding correction

The caller-bound CPU successor now selects actual `CARRIER=true` cfg13/WF0 in frozen DLLa836fd68:2288 static instructions,0SHFL.IDX/0SHFL.BFLY,32OMMA sites;126registers,stack/local0,static shared1024B. Exact symbol, source, tool and image bindings live in [CPU successor](gu-scale-absorb-cpu-v1/bindings.json), with [selected SASS](gu-scale-absorb-cpu-v1/serving-carrier-true.sass). The original invoice's CARRIER=false2427/no-SHFL/136register comparator remains withdrawn; its sealed package is unchanged. Corrected static comparison supplies no measured latency attribution. Every numerical/TPS receipt stands.

Target: exceed actual frozen-serving B64×128 throughput with its accepted arithmetic/layout, including the whole producer/consumer/model route. No GPU run was made for this diagnosis.

## Measured facts

- Ordinary H1 plain4022.795777 vs frozen serving6777.352952tok/s; generation2.0363947 vs1.2087315s, leaving0.8276632s per8192-token batch to recover. Deterministic H1 plain3947.455855 vs6583.070035. Full strict numerical seam closes on the tested teacher-forced corpus; no speedup/adoption.
- H1 halves terminal carrier traffic against parent tail; ordinary plain throughput changes4012.525034→4022.795777, only0.256%. These are successive retained runs, not simultaneous timing pairs. Plain already loses40.644%, so mixed relabeling cannot explain the shared deficit. H1 mixed is only0.536% below its plain arm.
- Exact image-bound H1 disassembly: plain DLL398a9d2b has5604 instructions,256 SHFL.IDX plus64 SHFL.BFLY; mixed5949 instructions with the same shuffle totals. Both have32 static OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X sites. Static totals include cold/optional paths and are neither executed counts nor cycles. Actual serving CARRIER=true has2288 static instructions,0SHFL.IDX/0SHFL.BFLY and32OMMA sites; the corrected source/caller/image binding is above.
- In H1 plain's main K loop0x5b50→0xad80, the backedge returns to0x5b50;128 SHFL.IDX and32 OMMA sites lie inside it. `src/fp4.cu:247–249,483–489` puts cooperative byte-book expansion and cross-lane gathering before MMA at every128-K stage; `kb1−kb0=5120/128=40`. Mixed additionally conjugates accumulators before/after each stage (`586–595,672–681`). Plain retains cooperative operand reconstruction.
- Linked resource readings: H1 producers131plain/142mixed registers; stack/local0. These do not support an H1 register-spill diagnosis. Actual serving CARRIER=true uses126registers,stack/local0. Consumers40plain/47mixed registers. Ptxas reports4096B consumer shared storage; linked cuobjdump reports5120B—both retained, no occupancy claim made from their differing accounting.

## Profile seam

Exact frozen engine source SHA256 `e2ae33c409fc8a4fe668f8c5cbf51009f65ea76745741fd5d3135c329a5b18c4`, lines2584–2586:

```cpp
const char* dclear_env = getenv("H5_DOWN_CLEAR");
if (dclear_env && strcmp(dclear_env, "0") && strcmp(dclear_env, "1")) throw std::runtime_error("H5_DOWN_CLEAR must be 0 or 1");
const bool down_clear = dclear_env && dclear_env[0] == '1' && R <= TAIL_MAX_ROWS && prep_clear;
```

Unset and0 both selectfalse. The old ordinary/deterministic controls and new H1 timings are behaviorally matched on this setting; different DLLs/GU routes remain intentional. Strict reference/candidate both explicitly pin0. The source is retained exact bytes from the RTX 5090 host's custom-wide256 source, not recreated from the absent source-staging pointer.

## Whole-target continuation

H1 repeatedly reconstructs ordinary FP4 operands from dense5 codes. That work is source/SASS-bound; its latency share and dominance are unmeasured. It is evidence for search, not a prescribed method. The complete producer, boundary conversions, dynamic traffic and sign/Hadamard1024/FP4 consumer must improve together while preserving rounded K5120 arithmetic and row means. Fixed weights may be prepared once, with the prepared resident footprint priced. Moving an epilogue scale by itself earns no speed credit.

The decider remains complete model TPS above6777.352952 ordinary/6583.070035 deterministic with strict graph/logit/endpoint conformance. Current work is CPU-only and separate from the operator's paired-WF0/K128/radix512 frontier. [Fixed-row-mean absorption](gu-scale-absorb-result-v1/README.md) supplies a route-specific falsifier: all64 sampled real fixed rows fail one-plane FP4/UE4M3 exact coefficient capacity. Its separate1.5 K5120 witness uses RN binary32 sums; [rounding-scope correction](GU-ABSORPTION-ROUNDING.md) records the missing native K-loop rounding/boundary binding, so it does not establish serving-loop noncommutation. CPU source/Lean/host-compiler witnesses are retained; no new native model/TPS acceptance or window nominee follows.

[Original opcode/resource/source invoice](h1-cost-invoice.json) and its sealed package retain all seven selected-function dumps and earlier failures. The frozen image binding is correct; the serving specialization selection was wrong. No GPU window, serving change or experiment replay occurred during this correction.
