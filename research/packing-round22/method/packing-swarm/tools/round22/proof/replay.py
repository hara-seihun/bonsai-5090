#!/usr/bin/env python3
"""Definitions-only paired dictionary, pinned Lean compiler and exhaustive axiom audit."""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = Path(__file__).resolve().parent
LEAN = Path('~/.elan/toolchains/leanprover--lean4---v4.33.0/bin/lean')
ARCHIVE = ROOT / 'archive/packing-swarm/round22/proof' / ('run-' + str(time.time_ns()))
RESULT = ROOT / 'results/packing-swarm/round22/proof/receipt.json'
MODULES = [
    ('Packing20', ROOT / 'method/packing-swarm/tools/round20/proof/NativeJoin.lean'),
    ('Packing21', ROOT / 'method/packing-swarm/tools/round21/proof/StridedPrep.lean'),
    ('Packing22', SOURCE / 'PairedTile.lean'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    names = {}
    for namespace, path in MODULES:
        text = path.read_text()
        if re.search(r'\b(sorry|admit|axiom|unsafe|native_decide)\b', text):
            raise SystemExit('unchecked proof declaration: ' + str(path))
        entries = re.findall(r'^theorem (\w+)', text, re.M)
        if not entries or len(set(entries)) != len(entries):
            raise SystemExit('missing/duplicate theorem: ' + str(path))
        names[namespace] = entries
        (ARCHIVE / path.name).write_text(text)
    audit = ARCHIVE / 'Audit.lean'
    audit.write_text('import PairedTile\n' + ''.join(
        '#print axioms ' + ns + '.' + name + '\n'
        for ns, entries in names.items() for name in entries))
    env = dict(os.environ, LEAN_PATH=str(ARCHIVE))
    stages = {}

    def run(name, args, seconds):
        start = time.monotonic()
        try:
            process = subprocess.run(args, cwd=ROOT, env=env, timeout=seconds,
                                     capture_output=True, text=True)
            log, code = process.stdout + process.stderr, process.returncode
            state = 'succeeded' if code == 0 else 'failed'
        except (subprocess.TimeoutExpired, OSError) as error:
            log, code, state = str(error), None, 'boundary_error'
        (ARCHIVE / (name + '.log')).write_text(log)
        stages[name] = dict(command=args, elapsed_seconds=time.monotonic() - start,
                            timeout_seconds=seconds, exit_code=code, status=state)
        save(ARCHIVE / (name + '.json'), stages[name])
        if state != 'succeeded' or re.search(r'\b(error|warning):', log):
            raise SystemExit('proof error: ' + str(ARCHIVE / (name + '.log')))
        return log

    version = run('version', [str(LEAN), '--version'], 2)
    if 'version 4.33.0,' not in version:
        raise SystemExit('wrong Lean compiler')
    for _, path in MODULES:
        run(path.stem, [str(LEAN), '--root=' + str(ARCHIVE), '-o',
                       str(ARCHIVE / (path.stem + '.olean')), str(ARCHIVE / path.name)], 12)
    log = run('audit', [str(LEAN), '--root=' + str(ARCHIVE), str(audit)], 12)
    actual = {}
    for line in log.splitlines():
        empty = re.fullmatch(r"'(Packing(?:20|21|22)\.\w+)' does not depend on any axioms", line)
        used = re.fullmatch(r"'(Packing(?:20|21|22)\.\w+)' depends on axioms: \[(.*)\]", line)
        if empty or used:
            name = (empty or used)[1]
            if name in actual:
                raise SystemExit('duplicate audit theorem')
            actual[name] = [] if empty else used[2].split(', ')
    expected = {ns + '.' + name for ns, entries in names.items() for name in entries}
    if set(actual) != expected or any(set(a) - {'propext', 'Quot.sound', 'Classical.choice'}
                                      for a in actual.values()):
        raise SystemExit('incomplete/unapproved audit')
    contracts = [ROOT / 'method/packing-swarm/tools/round22/native' / n
                 for n in ['paired.cuh', 'generate.py', 'binding.hpp', 'generation.hpp',
                           'views.cuh', 'stage.cuh', 'session.cuh']]
    external = [Path('~/work/packing-mma-round12/method/packing-swarm/tools/round11/residual/literal.cuh')]
    receipt = dict(
        status='checked_structural_lean', theorem_count=len(names['Packing22']),
        inherited_theorem_count=len(names['Packing20']) + len(names['Packing21']),
        total_audited_theorems=len(expected), printed_axioms=actual,
        compiler=dict(path=str(LEAN), version=version.strip(), sha256=sha(LEAN),
                      Std_sha256=sha(LEAN.parent.parent / 'lib/lean/Std.olean')),
        stages=stages,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in SOURCE.iterdir() if p.is_file()},
        inherited_source_sha256={str(p.relative_to(ROOT)): sha(p) for _, p in MODULES[:2]},
        source_contract_sha256={str(p.relative_to(ROOT)): sha(p) for p in contracts},
        external_source_contract_sha256={str(p): sha(p) for p in external},
        artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in ARCHIVE.iterdir() if p.is_file()},
        scope='definitions-only equal 8group256feature/32token paired shared dictionary; candidateFW4WG2/controlFW8WG1 group reconstruction and uniqueness; full output addresses at native stride with complete merged16480/14336/34816 and B1/8/32 including3group tail; shared stage9216 and ring73856 plus256 globals=74112; staged-once event dictionary; swizzle reconstruction; ascending40 K128 folds with two independent opaque accumulators; exact primary-first literal MODE1 additions; complete per-pass output/status/owner and exact manifest drain requirement before inherited whole release',
        primitive_semantics='arbitrary opaque MMA/FMA/mul/add word functions; dictionaries and load events are mathematical contracts, not CUDA execution traces',
        Cpp_CUDA_refinement='unset', native_word_refinement='unset', native_fault_refinement='unset',
        native_execution=False, native_time='unset', physical_capacity='unset',
        cache_occupancy_granularity_residency='unset', controller_refinement='unset',
        packing_bound='unchanged', symmetry_narrowing=False, model_task_HTTP='unset')
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    save(RESULT, receipt)
    print(json.dumps(dict(status=receipt['status'], theorems=receipt['theorem_count'],
                          inherited=receipt['inherited_theorem_count'], audited=len(expected),
                          receipt=str(RESULT.relative_to(ROOT)))))


if __name__ == '__main__':
    main()
