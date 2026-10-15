# bmega measured negatives (October 10)

Complete alternative sources of `src/bmega.cu`, retained because each one changes a measured trade-off
and loses on the full single-stream step. Same card/protocol as the `RESULTS.md` binary persistent-kernel
section; decodebench Wiki128+128 seconds, lower is better. The production source is `src/bmega.cu`
(FP4 three-plane exact A8, one TMA per chunk, per-warp piece fetch): full 0.388-0.391 s.

| Variant | Full step s/128 | No-dependency streaming floor (NOWAIT=2) | Note |
|---|---:|---:|---|
| production (FP4 3 planes, GROUP 1, per-warp fetch) | 0.388-0.391 | 0.33-0.36 | best full step |
| INT8 m16n8k32 (same exact integer dot, bit-identical logits), GROUP 1 | 0.413 | 0.36 | heavier ALU sign expansion lengthens ring-resident chunk latency |
| INT8 + dedicated input-fetch warp, GROUP 4 (`bmega-int8-fetcher-group4.cu`) | 0.456-0.460 | 0.29 | fetcher batches readiness: later first piece |
| INT8 + per-warp fetch, GROUP 4 (`bmega-int8-group4.cu`) | 0.426 | **0.294 (435 tok/s)** | best streaming floor; coarser ring flow control costs the chain |
| FP4 + GROUP 4 (`bmega-fp4-group4.cu`) | 0.432 | 0.320 | |

GROUP = chunks per TMA copy and ring barrier. A single producer thread issuing one 2.3 KB copy per
chunk costs about 200 ns per chunk handshake (NOWAIT=3 consumer-only floor 0.251 s at GROUP 1 versus
0.185 s at GROUP 4), so grouping raises the pure streaming ceiling from ~390 to ~435 tok/s; on the full
step it loses because the dependency chain, not streaming, is the binding constraint.
