> This branch now implements the [offline cache successor](../gu-dict8-cache/README.md). The material below records the parent v2 construction. Original v2 runtime settlement is `~/work/five-wide-native/DICT8-RESULT.md`; that completed matrix is never replayed.

# Dict8 private whole-engine startup successor — CPU-staged, not run

`H5_GU_DICT8=1` selects the eight-trit dictionary/ring4 GU only at actual B64/config13/split1. Other rows use the retained original WF0 payload. The source starts from exact frozen engine `e2ae33c409fc8a4fe668f8c5cbf51009f65ea76745741fd5d3135c329a5b18c4`, not the H1 derivative. Original activation formats, separate gate/up native words and MMA dependency order, activation/row means, SiLU, sign/Hadamard1024/FP4 observer and non-GU projections are retained. There is no RN distribution assumption.

## Built and staged

- Private DLL `4098ba03c9d8b53e0672e9f6800d1461c453bce46e80923c46e389edbc415dfa`, CPU build11.018208504s, exit0. Timing and endpoint-instrumented kernels132registers, stack/local0. Linked static shared1024B, dynamic ring66624B; no occupancy or latency inference.
- Twelve hash-bound unchanged frozen objects, including wide256 dispatch, are retained. Benchmark `3fd47ff37205bdd50738194ef92c13c21d24fc77a9d1be1197b69128a2173b12` is unchanged.
- Frozen serving DLL `a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c` and production plan `713c060b627ef9955d68edfef17e12d75f07093c68b74c3fd8ecf4d19b3507bd` remain unchanged.
- Native CPU build custody: `~/work/five-wide-native/dict8-engine-build-v2.zip`,81148274B, SHA256 `e72a27de31d11ec9edcb000fd42ec8a91a341c375168f7e7dbfa01b0e3322710`. Contains source, all inherited/private objects, private DLL/benchmark, commands/logs, resources/symbols and receipt. CUDA/MSVC/llama/model inputs remain separately owned host dependencies.
- On the RTX 5090 host, private destination is `models/experiments/bonsai-quality-20261013/gu-dict8-engine-build-v2`. `cpu-stage-receipt.json` records actual isolated Python contract exit0, PE exports checked without loading, exact model/binary hashes, no handback file and no runtime receipt. Harness plan SHA256 `29ad08e111d352a09f6384ce1740f6ed6c2b43a3bc164adacb83aafa132fdb5f`.

## Original failed window and CPU repair

Original745cc824d/DLL501b5a1c consumed its sole run: driverab21bfcf exited7 after24.740974s. Two frozen control children passed; candidate failed at `qc_create` with an access violation before creation returned. All three children ceased, mutex release/close returned1, production remained exact, and8614 independently confirmed physical handback. All24 retained files are preserved in112149572-byte `dict8-engine-window-v1-custody.zip`, SHA531626a25dd76c2df807c51f23b0b04aff85af9cf26792443e97b40405256026. No candidate numerical rejection or TPS was observed.

[STARTUP.md](STARTUP.md) owns the source-order cause and repair: dictionary upload consumed an uninitialized engine stream. Identical preparation now follows successful stream creation. The extracted-source CPU lifecycle regression rejects the old order and accepts the successor;69 shared headers match frozen/candidate copies, and API handles pass through directly. Runtime source changes only upload placement.

## Complete finite harness

`driver.py` is the sole future GPU entry. It requires a fresh explicit PrismML handback clearing every current/queued reservation and separate source-bound run authorization, then acquires `Local\BonsaiGPU` atomically once for the complete window. No handback is fabricated. MATH500 has priority; this successor has no reserved slot or run authorization.

One≤240s window; each direct native child≤50s shortened by remaining budget. Prefix failures/timeouts remain. The driver never resets/replays the window, never mutates serving, and releases/closes in `finally` after each invoked child returned or was killed and waited. A timed-out process is not numerical rejection.

1. Unchanged frozen strict reference, independent repeat, candidate: three warmups followed by two cached graph observations; warmup counters/inputs/tokens retained separately; observed hits2/captures0 required. Exact initial/prefill tokens, teacher-forced inputs/tokens and all64×248320 finite F32 logits×2 (127139840B) must match.
2. Candidate strict mode additionally checks rounded separate gate/up after activation scales/row means, SiLU and complete q4/s4/g4. Checks reset after prefill and between warmup/observation; exact per-step counts across all64layers are required. The per-layer comparator is a **new source-bound WF0 diagnostic kernel**, not a claim that the frozen DLL exposes those endpoints. The frozen DLL supplies independent logits/control.
3. Fresh unchanged-serving/candidate whole-model A/B, separately ordinary and deterministic: B64×128, three passes, eight warmup tokens/stream,8192tokens/pass. Match benchmark/plan hashes and workload; compare profile-specific medians. Endpoint checks are absent from timing. Historical baselines6777.352952ordinary/6583.070035deterministic remain reference facts, not substituted measurements.

**No GPU/model/DLL execution, GPU query or mutex acquisition has occurred for this candidate. Strict numerical acceptance, fit/cache/latency and whole-model TPS remain unset. No adoption.**

## Full cost invoice

`invoice.py` derives `cost-invoice.json` from actual dimensions/source geometry:

- Prepared GU36208640B/layer vs original44564480B:18.75% fewer recurring weight bytes. Original remains resident for other rows/reference. Added prepared weights2317352960B across64layers plus26244B global dictionary =2317379204B added requested device allocation.
- Original GU weights2852126720B, FP16 scales178257920B and means8912896B remain. Total GU-related original+prepared+dictionary5356676740B, excluding unchanged other model/state. Strict-only check buffers add22909232B; timing allocates none of them. Physical allocator granularity/complete VRAM fit are unset.
- Weight TMA interface72417280B/layer/decode vs89128960B native; B64 has two token CTAs per weight group. Exact model geometry requires zero weight-transfer padding. Reserved1024B scale hole/stage is not transferred under CARRIER=true. Activations/scales and full downstream observer remain priced separately.
- Dictionary adds32load sites/warp/K128 stage:178257920logical lane bytes/layer/decode, **not DRAM bytes**. Caching/latency are unmeasured.
- Startup retains original preparation and adds1426063360native-word keys /11408506880nibble decodes across64layers, a36208640B host working vector/layer and2317379204B extra requested upload. Actual encode/upload nanoseconds and complete engine-create wall time are instrumented for the authorized run; no setup latency is invented.

## CPU/proof scope and failures

`check.cpp` tests all6561dictionary codes/invalid nibbles and matches40960actual original operand words plus66560prepared bytes against the independent real-row fixture,0 mismatches. `Dict8.lean` proves nibble/key identities and conditional opaque replay; neither establishes compiler/CUDA correspondence, native rounding or model equivalence. `test_harness.py` exercises pure warmup/graph/endpoint acceptance under actual isolated Python without executing top-level native loads.

The first local stage generator failed with `KeyError: 'src/dict8.cu'`: native receipt paths use Windows separators. `stage-failed-v1/` retains the generator/log; the successor normalizes source paths once, rejects collisions/missing required bindings, and staged the unchanged DLL successfully. The failed prefix reached no native staging or GPU work. Original03382927f checkpoint, compiler failures, absorption/RN correction and original failed release receipt remain intact.

Private candidate source/harness lives on `agent/gu-dict8-startup`; source bundle and independent readable result in `~/work/five-wide-native/` survive writer release. Native private stage persists, but provides no authorization to run it.
