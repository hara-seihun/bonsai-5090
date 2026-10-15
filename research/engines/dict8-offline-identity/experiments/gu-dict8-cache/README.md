# Private Dict8 single-scan model identity

CPU-first successor to `5bdb08fafd6958f1ae46427f1058158a6f961d0b`. Cache-enabled actual Engine construction now hashes its owned model backing **once**, before the extracted GGUF parser. Reader validates an identity capability for that same live backing without another model read/hash. Dictionary encoding remains absent from Engine creation. Unset Dict8 keeps the original `Gguf::open`/repack/dispatch route and ignores the cache variable.

**Local CPU source/tests only. Native build, full-model CPU preparation/compare, cached Engine startup, device allocation/upload, numerical acceptance and TPS for this successor remain unmeasured here.** Parent owns native delivery into distinct `gu-dict8-identity-build-v2`; v4 and its custody are untouched. No GPU query/load/run, reservation, serving mutation, profiling, rental, download or v2 replay belongs to this work.

## Owned identity and platform contract

`dict8_model.hpp` is generated from `model.hpp.in` plus the existing parser. `ExactModel` is non-copyable/non-movable, owns storage and private parsed GGUF, and mints `Identity` only after successful admission, SHA and parse. Its parsed GGUF has no independently owned raw mapping. Actual Engine tensor/metadata consumers use `model()`, selecting the owned ExactModel when enabled. A caller cannot supply a digest, pointer/length or boolean to mint a capability or open Reader.

Capabilities are weak ownership references; Reader requires both the exact owner and its identity. Expired identity returns `StaleIdentity`, another owner's identity returns `SubstitutedIdentity`, including identical bytes reopened at the same path. Reader pins the backing through its own lifetime, so Engine/producer destruction cannot invalidate it. Mapping cleanup is RAII on acquisition, hash, parse and cache failures. The typed error enum is shared in `dict8_common.hpp`; C creation still preserves its existing null/error ABI.

**Windows: retained read-only mapping on checked local NTFS.** One UTF-8→UTF-16 pathname open uses `GENERIC_READ`, `FILE_SHARE_READ` only. All admission queries, size, section and view come from that handle. Admission requires:

- `SetLastError(0)` before the file-type query; disk file; `FileRemoteProtocolInfo` fails with exactly `ERROR_INVALID_PARAMETER` (87), with success rejected as remote and other failures as `AdmissionQuery`;
- filesystem exactly NTFS, a volume-GUID final path and `DRIVE_FIXED` root;
- no reparse tag, REPARSE_POINT, OFFLINE, RECALL_ON_OPEN or RECALL_ON_DATA_ACCESS attributes;
- neither directory nor delete-pending; exact `FileStandardInfo.EndOfFile` size.

Sharing violation returns `ModelInUse`; unadmitted storage/query/mapping failures are typed, with no snapshot or encode fallback. The read-only view is hashed once. Backing unmaps it, closes its section, **then** closes the deny-write file handle. Successful share-deny admission excludes already-existing write handles/sections/views; later ordinary writes, rename and deletion remain denied for the backing lifetime. Other read-only openers remain permitted. Opus's independent Win32 synthetic-file evidence: `~/work/dict8-identity-mutation-probe-v1/README.md` and `results.json` (plus part-2 admission probes). The Windows fixture test additionally exercises this implementation's positive route, prior writable section/view and release lifecycle when parent runs it natively.

**Linux: private anonymous snapshot.** Ordinary `MAP_PRIVATE` does not freeze a mutable source file. ExactModel opens once, checks size, reads directly into owned anonymous pages, checks EOF, applies `PROT_READ`, hashes that snapshot once, then parses and consumes it. Mutating/truncating/replacing the source afterward cannot change tensor bytes. Concurrent mutation during copying either produces the pinned SHA or `ModelIdentity`; size/mtime/path are never identity evidence.

Boundary: normal process API discipline and ordinary user-mode file actors. Arbitrary same-process VM writes/reprotection, PROCESS_VM_WRITE, administrator/kernel/raw-volume/filter-driver or impersonating third-party-filesystem actors are outside it. On Windows, later page-in storage faults are fatal hardware exceptions under the existing engine process policy; this work does not translate `EXCEPTION_IN_PAGE_ERROR` into success or a cache fallback. Storage returning identical bytes remains assumed.

## Immutable payload and source binding

Supported production model: **5,946,648,928 bytes**, SHA256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.

`D8CACHE1` retains its 2,176-byte fixed header and **2,317,352,960-byte** payload (64 × 36,208,640), total **2,317,355,136 bytes**. The independently compiled digest-table pin remains `f331dfecd01847d69872633f843c7639c86e9301092662b28e9d9c20ef671169`. Every ordered layer must match its pinned digest before Reader succeeds; caller-rewritten tables, permutations and corrupt labels cannot substitute a payload. Gate/up rounding, MMA order, scales/means, SiLU, sign/Hadamard/FP4 observer and original WF0 routes are unchanged.

`generate.py` binds actual generated Engine/parser/headers, generator/template, integrator, codec/encoder, frozen engine, producer and build definitions in `semantic-binding.json`. The new semantic hash **rejects v4's old header**, although the deterministic payload/table are unchanged. Prepare a separate file in the new generation; never rewrite old custody. `payload-identity.json` retains the original full-model deterministic payload evidence, not a claim that the successor was run.

## Recurring invoice

The 13-field `h5_dict8_cache_stats` ABI is preserved. Slot 1 (Reader model-hash ns) is now **zero by construction**; slot 9 still reports the sole pre-parse model hash. Slot 0 identifies the admitted model bytes; it is no longer an extra Reader read. Other payload read/hash/allocation/free and retained GU repack/upload/mean components keep their meanings. Original 10-field setup slot 8 (dictionary encode ns) remains zero.

New `h5_dict8_identity_stats(engine,out,14)` fields, in order:

1. source open/size ns
2. backing allocation/map ns
3. source read/copy ns
4. protection ns
5. sole model SHA ns
6. parse ns
7. explicit source-read bytes
8. copied bytes
9. requested anonymous allocation bytes (page-rounded)
10. identity-hashed bytes
11. identity scans
12. file-mapped bytes
13. admission-query ns
14. storage kind: 1 Windows retained local NTFS, 2 Linux private snapshot

Windows: one **5,946,648,928-byte SHA**, file mapping of that size, **zero model-copy bytes and zero anonymous model allocation**. Explicit source-read bytes are zero because demand paging occurs inside SHA/later accesses; this does not mean zero storage I/O. Linux: source-read/copy **5,946,648,928 bytes** plus one SHA of those bytes, page-rounded anonymous allocation and protection, all recurring and exposed. The snapshot remains through model/Reader lifetime. With a 4 KiB page it requests **5,946,650,624 bytes**. Add the unchanged payload host allocation **2,317,352,960 bytes**, ordinary model metadata/working allocations and OS paging. RSS/residency and end-to-end speed are unmeasured. Times are component observations, not additive startup predictions.

## CPU reproduction and native handoff

Run from the repository:

```text
python3 experiments/gu-dict8-engine/integrate.py
python3 experiments/gu-dict8-cache/generate.py
python3 experiments/gu-dict8-cache/test_identity.py
python3 experiments/gu-dict8-cache/test_identity_abi.py
python3 experiments/gu-dict8-cache/test_windows_queries.py
python3 experiments/gu-dict8-cache/test_fault_receipts.py
python3 experiments/gu-dict8-cache/test_cpu.py
python3 experiments/gu-dict8-cache/test_boundary.py
```

`test_identity.py` compiles the **actual generated Engine bootstrap and model selector**, production ExactModel/Reader algorithm, and explicit isolated tiny fixture SHA/size/cache geometry. Production contains no fixture minting API. An independent crypto-call counter proves one model hash through bootstrap, zero additional hashes through Reader. It checks stale/substituted capabilities, retained backing lifetime, wrong-size/hash-before-unsafe-parse, payload/table/header corruption, order/terminal failures, failures before the device boundary and the original disabled route. 28 injected admission-fact failures cover remote/reparse/unsupported/query/size states; these are predicate tests, not observed remote-volume negatives. Linux additionally changes the actual weight byte (fixture offset 64) through a pre-existing shared writable alias and source rewrite, re-hashes the full snapshot after each test-only mutation, checks read-only protection, and six injected allocation/protection/read/crypto failures with cleanup. `identity-test-receipt.json` identifies the compiled fixture and production inputs.

`test_windows_queries.py` compiles the actual generated Windows query caller against injected Win32 APIs: stale error clearing, GUID buffer growth, missing root and query failures (14 cases), without claiming live-volume negatives. `test_identity_abi.py` compiles the actual exported stats function against a CPU sentinel owner, checks all 14 ordered fields and null/count/unset errors, without loading a DLL. All C++ assertion translation units reject `NDEBUG` at compile time.

`test_cpu.py` keeps the actual upload branch CPU stub and independent 66,560-byte real-row operand fixture. `test_boundary.py` keeps allocation/generic C-constructor error/reset coverage. Full-model `fault_test.cpp` now uses the owned ExactModel API and asserts its scan count remains one across all Reader opens. The exact full fault sequence is also run with the isolated tiny fixture. Its payload-corruption case restores a valid header/exact size after the truncation cases; expected/actual error diagnostics name the source line. `fault_host.execute` retains stdout/stderr, per-process timings and terminal receipt in `finally` even on nonzero/timeout/launch/parse failures. Five injected process scenarios exercise this receipt contract and owned-probe cleanup. The original v1 native fault failure belongs to its original custody.

`ship_build.py` targets **gu-dict8-identity-build-v2**, includes `src/`, producer/fault sources and the portable tiny fixture runner, and runs both portable identity/API and extracted-export ABI fixtures with the absolute `owner/tools/python/python.exe` in the configured MSVC environment. `export_host.py` requires the new identity-stats export and checks producer/fault executable imports contain no cudart. It compiles producer/fault executables with `src/gguf.cpp`; parent runs it. Existing frozen object/header checks remain. Prepare/compare/fault/export host scripts also target only the new generation.

Producer command is unchanged:

```text
cache_tool prepare|verify|compare MODEL CACHE THREADS MAX_BYTES HEADROOM_BYTES
```

Its JSON now exposes all model identity/backing components above. Preparation preserves staged validation, no-overwrite publication, explicit storage/headroom, failure cleanup and no hidden replay. Parent may run full CPU prepare+compare in the separate generation. Previous measurements and immutable source/native custody remain under `~/work/five-wide-native/dict8-cache-result/`; none are successor startup/TPS evidence.
