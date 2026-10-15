#!/usr/bin/env python3
"""Read-only replay of sealed CPU witnesses; no receipt or log overwrite."""
import hashlib, json, pathlib, subprocess
import replay
ROOT = replay.ROOT
receipt = json.loads((replay.RESULT/'receipt.json').read_text())
assert receipt['state'] == 'all_full_paired_saved_tape_witnesses_pass'
for relative, identity in receipt['artifact_sha256'].items():
    assert replay.sha(ROOT/relative) == identity, relative
for relative, identity in receipt['build']['source_sha256'].items():
    assert replay.sha(ROOT/relative) == identity, relative
assert replay.sha(ROOT/receipt['native_paired_source']) == receipt['native_paired_source_sha256']
seal = json.loads(replay.SEAL.read_text())
assert replay.sha(replay.SEAL) == receipt['build']['round20_receipt_sha256']
for name, variants in receipt['cases'].items():
    family, batch, suffix = name.split('-')
    B = int(batch.removeprefix('B'))
    assert suffix == 'final'
    data, manifest, manifest_path, packed = replay.evidence(name, seal)
    payload = b''.join(data[passname+'.'+suffix] for passname in ('primary','innovation') for suffix in ('q','s','g.f32'))
    payload += b''.join(data[arm+'.u32'] for arm in ('candidate','mergedFW8','current19'))
    payload += b''.join(data[arm+'.trace.u64'] for arm in ('candidate','mergedFW8','current19'))
    for kind in ('ordinary','ubsan'):
        expected = variants[kind]
        assert replay.sha(packed) == expected['inherited_archive_sha256']
        assert replay.sha(manifest_path) == expected['inherited_manifest_sha256']
        assert manifest['members'] == expected['inherited_members']
        binary = replay.ARCHIVE/kind
        assert replay.sha(binary) == receipt['build']['binaries'][kind]
        observed = subprocess.run([str(binary),family,str(B)],input=payload,capture_output=True,timeout=45)
        assert observed.returncode == 0 and not observed.stderr, (name,kind,observed.stderr)
        assert json.loads(observed.stdout) == expected['observed'], (name,kind)
print('All eighteen full paired CPU/UBSan saved-word witnesses replay exactly; source/native identities and inherited member custody match')
