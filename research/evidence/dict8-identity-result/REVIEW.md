# Dict8 model-identity source review: commit 662ed50e0

This is an independent read-only CPU review of `~/work/clones/dict8-model-identity` at commit `662ed50e0` ("Bind Dict8 cache to one owned model identity scan"). The review was done on 2026-10-13. Repairs belong to the writer, conversation b77ad563. The parent is conversation 12f2cc5b.

## Scope, bound to `git show 662ed50e0:<path>` SHA-256

| Path | SHA-256 |
|---|---|
| src/dict8_model.hpp | 4509361a01d780bf1a92fbb48a64c5d2f9e56f68aab892b741af1f9d12482dcc |
| src/dict8_cache.hpp | 80cc62ffbea0d81e7a3390af371d245e064f024c7b0ca98943e596f19c398b5c |
| src/dict8_common.hpp | 06158e15585fc1db7c37df24d14907a0968e122fd4492bc2dd2d3436d115daa6 |
| src/dict8_binding.hpp | 8f908acca91e7ad1febe009fc1f23dce23909bd12fb8f83566420d19bdc803af |
| src/engine.cu | 6445a05870929ae76b314ae96f39c9ce3f426bb8a0ddf71616ffe53b7abe183b |
| experiments/gu-dict8-cache/identity_test.cpp | cb7b6e0b0249416e07ec410a36e5dc1faed4db71f7e1b413ab1293f679087e29 |
| experiments/gu-dict8-cache/semantic-binding.json | aa3e8864db0b9b9de94558e0ca11ae4cd4511effabc0ad569bfcaf64f9daeb01 |

The bindings in `dict8_binding.hpp` at 662ed50e0 are:

| Binding | SHA-256 |
|---|---|
| model | 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3 |
| semantics | c7a8b1434ebf1b4cc0bcdb515fd593a7d681bf0ced478cf0f92bdba561bf2a1c |
| payload table | f331dfecd01847d69872633f843c7639c86e9301092662b28e9d9c20ef671169 (unchanged from 5bdb08faf) |

**Correction.** The first review messages gave the semantics hash as `0d021b97…`. That value came from an in-progress generator state in the uncommitted checkout. The committed value is `c7a8b143…`. Every source hash above matches the files that were read.

The Linux identity and CPU tests were rerun from a /tmp copy. Both passed, and the fixture hashes match the writer's receipt. The Windows native fixture evidence comes from the parent's identity-build-v1 run; this review did not run it. Not done: GPU work, DLL load or run, mutex or reservation use, or a v2 replay.

## Verified, no finding

- **One model SHA scan.** `ExactModel::open` hashes exactly once. `Reader::open` adds zero scans, and `model_hash_ns` is 0.
- **Hash before parse.** A wrong-magic model fails with ModelIdentity.
- **Parser lifetime.** `parsed_.base` is null. Only Backing unmaps. Tensor pointers live as long as the `ExactModel` and any Reader's `model_lifetime_`.
- **Identity token.** It can't be forged: it has a private constructor and a weak_ptr. A stale token fails with StaleIdentity and a substituted one with SubstitutedIdentity; both are tested.
- **Windows admission.**
  - Every check runs on one retained GENERIC_READ handle opened with share=READ. A sharing violation maps to ModelInUse.
  - The admission checks:
    - the file type is DISK;
    - the remote query fails with error 87;
    - the file system is exactly NTFS;
    - the GUID volume root is DRIVE_FIXED;
    - no reparse, offline, recall or directory attributes, and no pending delete;
    - EOF equals model_size.
  - The section and view come from that same handle. Teardown order is unmap, then section, then file, on every path.
  - The pure predicate is tested with 28 injected rejections.
- **Linux snapshot.**
  - Private anonymous mapping, filled by bounded EINTR-safe reads.
  - A short read or an extra byte fails with ModelIdentity.
  - PROT_READ is set before the hash.
  - Six injected faults leave no live mappings.
- **Engine.** All model access goes through `model()`. The disabled branch is unchanged. The cache is destroyed before the model. A Reader failure releases the model before the CUDA stream exists.
- **Cache binding.** Header validation is unchanged and the payload table hash is retained. The semantics hash changed because its bound source changed, so caches from before this change are rejected.
- **v2 shared headers.** All are byte-identical on the RTX 5090 host.

## Findings, repaired by the writer toward v2

1. **The Linux mutation-independence asserts are vacuous** (`identity_test.cpp` lines 104–106).
   - The writes go to bytes 255 and 0. The asserts read tensor offsets 64 and 67.
   - Fix: mutate offset 64, or re-hash the snapshot after each mutation.
2. **Some Windows `storage_facts` branches are untested.**
   - The untested branches are the stale `GetLastError` check after `GetFileType`, the `GetFinalPathNameByHandleW` grow loop, and a root parse failure.
   - Add `SetLastError(0)` before `GetFileType`.
3. **The new `h5_dict8_identity_stats` export is unchecked.**
   - `export_host.py` doesn't assert it exists, and no harness records its 14 fields.
   - Cache stats now report `model_hash_ns=0`. The hash cost and storage mode are visible only there.
4. **The tests depend on `assert`.** Keep `NDEBUG` out of their compiles, or use always-on checks. The ship build's 50 s timeout now also covers the identity test; v1 took 16.38 s in total.

`fault_test.cpp`, which resizes to file_bytes+1 and then expects Corrupt, is a separate defect that the writer identified.

## Stated boundaries, not defects

- Windows zero-copy pages bytes in again after the hash. A media error then surfaces as EXCEPTION_IN_PAGE_ERROR, which is untyped.
- These are outside the boundary:
  - same-process code, `WriteProcessMemory` and a PAGE_WRITECOPY reprotect;
  - admin, raw-volume and filter-driver writers;
  - third-party file systems that report NTFS on a fixed drive.
- While an engine with dict8 enabled is alive, the model file can't be written, renamed or deleted. Readers are still allowed.
- The Linux snapshot holds about 5.95 GB of anonymous memory.
- `ExactModel::Identity` is redundant with holding the model, but harmless.

## Evidence

The mutation and admission probe that backs the Windows invariant is in this directory: `README.md`, `probe.py`, `probe2_admission.py`, `run1.log`, `run2.log` and `results.json`.

## Delta check of the repaired final: 9387d08aed0127afb523db5517a58ba5f43f308d

This was a read-only delta check of the diff from 662ed50e0 to 9387d08ae, plus a CPU rerun of `git archive 9387d08ae` in /tmp. It was not a full re-review.

Source SHA-256 from `git show 9387d08ae`:
- **Changed:**
  - `dict8_model.hpp` is 15880a28ee2f4704…. The only change is `SetLastError(NO_ERROR)` before `GetFileType`.
  - `dict8_binding.hpp` is 0ddaef13548df290…. The semantics hash changed to 18cc9ecfb30d4d4009d8bc610ca2c2a57f1996207633b9fcc7eaf61d400cb799.
- **Unchanged:**
  - `engine.cu` 6445a058
  - `dict8_cache.hpp` 80cc62ff
  - `dict8_common.hpp` 06158e15
  - the model hash 53107f53 and the payload table f331dfec

Repairs of the four findings:
1. **Mutation tests.** Both the writable alias and the file rewrite now mutate offset 64, which is the tensor's first byte. After each mutation the test asserts `tensor[0]==42`. It also hashes the full 256-byte snapshot after each mutation and checks it against the pinned model SHA. **Repaired.**
2. **`SetLastError` and untested query branches.** `SetLastError` is present. `test_windows_queries.py` extracts the generated `storage_facts` body and runs 14 mocked cases, compiled on Linux. The cases cover:
   - a stale error is cleared;
   - an unknown file type, both with and without an error;
   - a remote file system succeeds, plus 4 non-87 errors;
   - volume, path, tag and standard query failures;
   - the GUID path buffer growing;
   - no GUID root, which is rejected without a drive query.

   This is mock evidence, not a live Windows run. **Repaired.**
3. **Export coverage.** `export_host.py` asserts `h5_dict8_identity_stats` is exported (via dumpbin) and also dumps `fault_test.exe` imports. `test_identity_abi.py` compiles the actual exported function body against a sentinel owner. It checks 14 ordered fields and the error codes -1/-2, and that a failing call does not write to the output. **Repaired.**
4. **Assert dependence and the fault fixture.** The identity, fault, ABI and query tests carry `#error` guards against NDEBUG. `fault_test` now reports the line and the expected/actual error names, and calls `write_header(original_header)` before the Corrupt expectation, which fixes the +1-size defect. `fault_host.py` writes a started receipt first. Its finally block keeps the state, the streams and the failure type. It refuses to overwrite an existing receipt, and cleans up the staged and malformed fixtures. `test_fault_receipts.py` covers 5 injected scenarios: failure, timeout, launch error, bad JSON and pass. `test_identity.py` now runs the real `fault_test.cpp` against the fixture; the only edit it makes is a smaller headroom value. **Repaired.**

All 6 CPU tests pass from the archive: test_identity, test_identity_abi, test_windows_queries, test_fault_receipts, test_cpu and test_boundary. There are no further actionable defects. The native v2 build and the Windows receipts belong to the parent. The stated boundaries above are unchanged.
