# Code views for each ladder step

Each `.patch` is a unified diff between **excerpts**: the relevant functions from PrismML's llama.cpp fork (or the previous step) and from the measured source. They are for reading alongside [../ladder.md](../ladder.md), not for applying. File names inside the diffs are synthetic. Line counts describe the excerpts, not the size or effort of the real change.

| Unit | What it changes | Excerpt lines +/− |
|---|---|---:|
| [01a-sampling](01a-sampling.patch) | Batch selection and token-only readback | +211 / −71 |
| [01b-embedding-placement](01b-embedding-placement.patch) | Place the embedding table on CUDA | +8 / −0 |
| [01c-embedding-reader](01c-embedding-reader.patch) | Read packed ternary embedding rows | +31 / −79 |
| [02a-direct-state](02a-direct-state.patch) | Direct recurrent state access | +46 / −8 |
| [02b-cg8-update](02b-cg8-update.patch) | Eight cooperating lanes per state row | +56 / −203 |
| [03a-state-allocation](03a-state-allocation.patch) | Allocate 16-bit recurrent state | +2 / −2 |
| [03b-state-io](03b-state-io.patch) | Load/store half values, retain float arithmetic | +21 / −10 |
| [03c-state-init](03c-state-init.patch) | Initialize the new state format | +5 / −0 |
| [04a-weight-layout](04a-weight-layout.patch) | Pack static weights for their consumer | +53 / −7 |
| [04b-operand-map](04b-operand-map.patch) | Turn packed bits into FP4 operand registers | +58 / −88 |
| [04c-activation-prep](04c-activation-prep.patch) | Fuse normalization, signs, rotation and FP4 packing | +587 / −382 |
| [04d-tensor-core](04d-tensor-core.patch) | Use FP4 block-scaled Tensor Core instructions | +72 / −185 |
| [04e-scale-carrier](04e-scale-carrier.patch) | Move a row scale outside the reduction | +53 / −0 |
| [04f-projection-layout](04f-projection-layout.patch) | Group projections sharing an input | +111 / −118 |
| [04g-output-fusion](04g-output-fusion.patch) | Fuse SiLU, gate multiplication and residual writes | +51 / −85 |
| [04h-copy-pipeline](04h-copy-pipeline.patch) | Overlap tile copies and matrix work | +148 / −0 |
| [04i-recurrence-integration](04i-recurrence-integration.patch) | Integrate convolution and recurrence with the new engine | +308 / −103 |
| [04j-attention](04j-attention.patch) | Fuse attention preparation and use FP8 KV | +265 / −133 |
| [04k-execution](04k-execution.patch) | Own graph replay, scheduling and matrix plans | +287 / −92 |
| [05a-state-codec](05a-state-codec.patch) | Pack six-bit state into three word planes | +44 / −20 |
| [05b-single-token-update](05b-single-token-update.patch) | Specialize the recurrent update for one new token per stream | +152 / −56 |
| [06a-kv-writer](06a-kv-writer.patch) | Write packed FP4 KV and block scales | +24 / −29 |
| [06b-kv-reader](06b-kv-reader.patch) | Read FP4 directly into attention fragments | +36 / −42 |
| [06c-kv-wiring](06c-kv-wiring.patch) | Connect the matching KV writer and reader | +28 / −0 |

Steps 04a–04k were measured only as one bundle, 02a+02b, 03a–c and 05a+05b only together; see the ladder for which numbers belong to which units.
