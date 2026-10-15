# Dict8 model-identity mutation probe v1

Synthetic 1 MiB zero file, NTFS, the RTX 5090 host Windows 11 26200, non-admin user, Python 3.13 ctypes direct Win32 calls.
Each actor is a separate child process (`agent.py`), so these are cross-process kernel results.
No model, GPU, CUDA DLL, mutex, reservation or serving state was touched. The `.bin` files were deleted after the run.

- Source: `probe.py` (sha256 ae1717bf…)
- Raw run: `run1.log` (315b1426…), structured: `results.json` (e3a73278…, 58 steps)
- the RTX 5090 host copy: `C:/work/models/experiments/bonsai-quality-20261013/dict8-identity-mutation-probe-v1/{agent.py,results.json}`

## Findings

| Scenario | Result |
|---|---|
| S0: current `h5_map_file` (share READ\|DELETE, file handle closed after mapping) | A second process opens RW, `WriteFile` bytes appear in the hashed view. Shrink is blocked (1224). **Current code is mutable after hash.** |
| S1: writable view exists, all writer handles closed | Deny-write open (share READ or READ\|DELETE) **fails with ERROR_SHARING_VIOLATION (32)**. The store goes to disk. After unmap, the deny-write open succeeds. |
| S1: writable section handle retained, no views | Deny-write open still fails (32). |
| S2: retained GENERIC_READ share=READ handle | Blocked: GENERIC_WRITE, FILE_WRITE_DATA, FILE_APPEND_DATA, DELETE, CREATE_ALWAYS (32). Rename/delete (32), replace-onto (5), RW section from a read handle (5). A hardlink can be created, but writing through it fails (32). WRITECOPY section and COPY view stores stay private, and the reader view and disk are unchanged. Readers are still allowed. |
| S3: same process | Store into the RO view faults. VirtualProtect RW fails (87), but PAGE_WRITECOPY succeeds. `WriteProcessMemory` into the RO view **succeeds**: the view changes privately and the disk is unchanged. |
| S4: deny handle closed before the view | Writer opens and `WriteFile` mutates the view. |
| S5: writer handle exists before the deny open | Deny open fails (32). |

## Invariant (local NTFS, user-mode file actors)

Sharing alone is sufficient against both prior and later file writers **if** the following hold, in this order:

1. Open GENERIC_READ with share=FILE_SHARE_READ only. Success proves that no write/delete handle and no writable section or view exists at that instant.
2. Create the section and view **from that handle**, and take the size from that handle.
3. Hash the view once.
4. Retain the handle until after `UnmapViewOfFile`.

A failed open is a typed error (sharing violation), not a fallback path.

## Part 2: admission queries (`probe2_admission.py`, `run2.log`)

Observed on a local NTFS file on C: (fixed):

| Query | Value |
|---|---|
| `GetFileType` | 1 (`FILE_TYPE_DISK`) |
| `GetVolumeInformationByHandleW` | file system "NTFS" |
| `FileStandardInfo` | EOF equals size, 1 link, not delete-pending, not a directory |
| `FileAttributeTagInfo` | attributes 0x20, reparse tag 0 |
| `FileRemoteProtocolInfo` | fails with error 87 |
| `GetFinalPathNameByHandleW(VOLUME_NAME_GUID)` | `\\?\Volume{438333cc-…}\…` |
| `GetDriveTypeW` on the volume root | 3 (`DRIVE_FIXED`) |

These negatives could not be exercised live:

- Remote: `\\localhost\C$` was access denied (5) as a non-admin user, and the RTX 5090 host has no other share.
- Symlink: creation needs a privilege (error 1314).

Remote and reparse rejection therefore rest on documented semantics plus injected-fact unit tests.

## Outside this invariant (untested)

- Same-process code, and other processes with PROCESS_VM_WRITE. These apply equally to a heap snapshot.
- Admin/kernel/raw-volume writers.
- Filter drivers.
- Network or non-NTFS filesystems.
- The Linux `MAP_PRIVATE` branch, which does not freeze the file.

## Source review

[REVIEW-662ed50e0.md](REVIEW-662ed50e0.md) is the review of commit 662ed50e0, bound to `git show` hashes. It includes the corrected semantics hash c7a8b143. The repaired-final delta check is pending.
