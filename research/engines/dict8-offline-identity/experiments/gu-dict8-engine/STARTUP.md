# Dict8 startup-order repair

Original source745cc824d/DLL501b5a1c failed its sole authorized run in `qc_create`: access violation reading0xFFFFFFFFFFFFFFFF. Stable independent frozen controls passed; candidate engine creation did not return. No candidate numerical rejection, TPS or adoption follows.

## Cause

`Engine::st` is uninitialized until the constructor's successful `cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking)`. The new Dict8 dictionary preparation called `upload()` earlier. That helper consumes `st` in `cudaMemcpyAsync` and `cudaStreamSynchronize`. The candidate therefore passed an indeterminate stream handle into CUDA at startup.

The successor moves the identical dictionary construction/upload immediately after successful nonblocking-stream creation. It retains the explicit stream dependency; neither null initialization nor a default-stream path masks it. Dictionary encoding, native kernels, arithmetic, scales, means, endpoints and all original controls remain unchanged.

## CPU evidence

- `test_startup.py` compiles and runs the actual extracted source upload helper and constructor create/upload ordering against host CUDA stubs. The fixture injects an invalid-stream sentinel: original ordering is rejected before stream creation; successor creates one stream and performs one26,244-byte upload. This does not reproduce or identify the hardware faulting instruction/address.
- `startup-source-delta.json` establishes that relocation is the entire runtime-source delta.
- `abi-boundary.json`: TMat is TU-local.69 common headers are byte-equal across installed frozen custom/wide and original candidate; the successor changes none. Frozen api.cpp has direct `qc_create→h5_create` opaque-Engine passthrough. These observations give no shared-TMat incompatibility evidence; they do not reconstruct the historical compiler dependency closure.
- `startup-cpu-first-failure.json` retains the host-fixture namespace compile error and its correction.

## Custody and next gate

Original source, DLL, driver, plan, controls and failed window are immutable. `startup-v1-engine.cu` is the source-bound negative regression fixture. The source generator owns the corrected ordering.

Successor build/staging are CPU-only in fresh `gu-dict8-engine-build-v2`; no candidate DLL/model is loaded. A new source-bound authorization and fresh8614 all-reservation handback are required for execution. MATH500 has priority; no GPU successor is reserved by this repair. One fresh≤240s slot request follows complete source publication and CPU staging; each native child remains≤50s. Original window is never reset or replayed.
