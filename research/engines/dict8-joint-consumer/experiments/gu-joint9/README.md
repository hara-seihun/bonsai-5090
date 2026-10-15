# Joint native-operand carriers (CPU-only, ongoing)

Target: whole-model TPS above6777.352952 ordinary /6583.070035 deterministic, preserving the actual rounded K5120 contraction, activation q4/s4/g4, row means and complete SiLU/sign/Hadamard1024/FP4 observer. No GPU/window/serving change. the PrismML quality campaign holds the GPU; its IFEval006→worker0142faa9 benchmark reservation is unaffected.

## Arithmetic contract

Keep gate and up as **two independent original native E2M1 operand streams**. Joint coordinates are a weight representation, not permission to replace their contractions by one g+3u result. Native rounding is opaque: the same words, activation scales, constant weight scale0x38383838, per-output K128/half64 MMA dependency order, AG/row-mean multiplication and SiLU code must remain. The frozen source is `0754bd730848620791da50c9aaadd6ba0a8f431434a75428b916969464164f52`; selected carrier call is cfg13/FW8/WG2/TOK32/split1/CARRIER=true. The earlier1.5 absorption witness is strictly RN algebra, not native RTZ evidence; [scope correction](~/work/five-wide-native/GU-ABSORPTION-ROUNDING.md) leaves its sealed packet intact.

`Joint9.lean` proves the pair coordinate inverse and observer composition conditional on identical opaque native instruction streams. It supplies no CUDA floating-rounding model. `check.cpp` exhausts59049 five-pair blocks and proves all4096 lane coordinates visited once. `actual_rows.py` serializes actual block0 first32 gate/up rows across all40 K128 stages:40960 separate native operand words,0 mismatches. Native acceptance and full-engine integration remain unset.

## Joint five-pair labels

Five GU pairs use ten ternary digits in two five-digit bytes;52B/lane,1664B per32 logical output features and K128. Pair-specific256-entry tables decode both streams without cross-lane gathers. Full source-derived compiler probes preserve the original MMA/epilogue body and compare:

| Probe | Static instructions | Registers | Stack/local | Static MMA | Static IDX | Static LDS |
|---|---:|---:|---:|---:|---:|---:|
| Matched nativeWF0 |2288|126|0|32|0|16|
| Independent dense5 |2976|140|0|32|0|78|
| Joint five-pair |2896|143|0|32|0|77|

The smallest backward region containing all32 MMA sites is160 instruction lines for nativeWF0,381 for independent dense5 and491 for joint five-pair. These are static compiler regions, not executed counts or cycles. Total-static improvement against dense5 hides a worse selected hot region. No bottleneck attribution or speed claim follows.

Joint weight payload is36,208,640B/layer versus44,564,480B nativeWF0,18.75% less;2,317,352,960B across64 layers, before retaining original weights or other model state. The template's unused scale-layout reservation and its2048B shared lookup-table setup also enter the bill. Original GU output, activation conversion and complete consumer are unchanged; no H1 carrier is introduced. `cost-invoice-v1.json` records schedule, payload, additional table setup and explicit unmeasured costs.

## Eight-trit native-word dictionary successor

Eight ternary weights identify one original32-bit E2M1 operand word:6561 possibilities,13-bit keys. A gate/up sibling word pair is26bits;16 pairs occupy exactly52B/lane, the same18.75% payload reduction. A26244B immutable prepared dictionary replaces joint trit reconstruction and 20-bit stitching. It still yields separate gate/up words and preserves the original contraction/epilogue.

`actual_dict8.py` serializes the same real rows, all40 stages:40960 original words,0 mismatches. Packed66560B SHA256 `1e9716ff2c3ca95bbec54ccb276be55ca1159a21c89b89aaeee3e00ffcdd8b60`; dictionary26244B SHA256 `c852b5dcceec90e78b2a19dca75dd896062ed3ff606d6b51adb475e610cb7b6a`.

Compiler v1 failed before producing a cubin because frozen serving predates the global-book argument; source/error preserved in `dict8-failed-v1`. V2 explicitly passes the dictionary (native0 passes none):

| Probe | Static instructions | Registers | Stack/local | MMA | IDX | LDG | LDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native0 |2288|126|0|32|0|76|16|
| Dict8 global readonly lookup |2584|134|0|32|0|108|25|

Smallest enclosing backward regions: native0 `0x3d40–0x4730`,160 lines/0LDG/16LDS; dict8 `0x4790–0x5990`,289 lines/32LDG/25LDS. The dictionary removes much of joint5's hot reconstruction (491→289 static lines), while adding32 recurring global lookup sites. Cache behavior, actual latency and complete model TPS remain unmeasured. No window nominee yet.

Each layer/step has136 CTAs,8 consumer warps/CTA,40 K128 stages and32 warp MMA sites/stage:1392640 warp MMA instructions. Dict8 introduces32 lookup sites/warp/stage, which means44564480 logical lane-word reads (178257920B)/layer/step and1460288880640B across64layers×128steps **at the instruction interface**, not DRAM traffic. The26244B dictionary may cache; no measured cache/hit assumption is made. Replacing native weights reduces their logical TMA payload18.75%; retaining both payloads adds2.317GB prepared weight state rather than saving resident memory. Non-GU projections, GU output, scalar means and entire consumer have zero claimed delta. Their existing absolute costs are not newly measured.

`bank_price.py` models a prospective shared dictionary against actual sampled indices under4B/32-bank addressing with equal-address broadcast:1280 warp loads, average3.5359375 distinct-address rounds, maximum7. It is address geometry, not a measured GPU conflict or cycle result. This constrains simply moving the LUT into shared memory; capacity/ring-depth and setup traffic must be priced too.

Explicit four-stage ring successor is also compiled, using the same dictionary/payload and native per-output arithmetic order:2448 static instructions,132registers,stack/local0; smallest32-MMA backward region297 lines/32LDG/25LDS. The matched native0 stays2288/126/160. This lowers total-static/register cost against automatic ring depth, while its selected hot region rises289→297; no executed-time claim follows. CPU compiler archive `168137a76dad687c9fe8932978fa5098b8c8b18aa61b7c2b6f0c1eaedf7f1ba4` preserves it.

## Current next move

Price and compile a small shared/dictionary or instruction-composition successor that removes expensive recurring lookup/address work without losing the byte saving or identical native operand stream. Choose by the whole route, not a decoder-only win. Engine/parser/admission integration, warmed strict full logits/observer endpoints and ordinary/deterministic whole-model timing are all still unset. the GPU remains with the PrismML quality campaign; no model/DLL execution or GPU query occurred.

## Custody

Compiler-only jobs use existing CUDA/MSVC tooling through gpu_host.py and never load a native library or call CUDA. Guarded generations refuse an existing directory. Archive identities:

- Joint5 compile v1: `91764037aa490321d8eed81720665f27bff81fe45efaee5dbcb3ab95e4adb204`.
- Dict8 failed compile v1: `fcbf63606a685b6ebfe89498b8240e0f68d85f4f8eeb0076c13751a7c3331c19`.
- Dict8 compile v2: `131461438d894bc9907fd03bc6ba0aeac11fe14ad3facd43947a41417b9afd89`.

Readable compiler receipts/SASS/cubins live under `~/work/five-wide-native/gu-joint9-{compile-v1,dict8-compile-v1,dict8-compile-v2}/`; source generators, proof, fixtures and invoices belong to this private branch. Current writer: gu-joint9 registry00000000-0000-0000-0000-000000000000, budgeted512MiB/8GiB headroom. Push/bundle every coherent checkpoint before release; do not mutate serving or older sealed custody.
