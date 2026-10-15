#!/usr/bin/env python3
"""Bounded definitions-only replay; exact predecessor custody and complete axiom audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

SOURCE = Path(__file__).resolve().parent
LEAN = Path('~/.elan/toolchains/leanprover--lean4---v4.33.0/bin/lean')
MODULES = [
    ('Packing20', 'NativeJoin', '690225f0ecc3227b02402037e9509905a12c94b80ded54aec03266bdb919cda6'),
    ('Packing21', 'StridedPrep', '2d108cf1021ea91978bdb3638bdf28a054cd9c20508882f2ccc6e87f5fdd0cfb'),
    ('Packing22', 'PairedTile', '78633b6e9f116f39b49a1b565b45634f7417b7912265a30e9b16cc322aec2539'),
    ('Packing23', 'SerializedGroup', None),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Evidence root: writes archive/packing-swarm/round23/proof and results/packing-swarm/round23/proof')
    parser.add_argument('--contract', type=Path, action='append', required=True,
                        help='Exact native contract input; repeat for every contract source')
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.monotonic()
    deadline = started + 53
    archive = root / 'archive/packing-swarm/round23/proof' / ('run-' + str(time.time_ns()))
    result = root / 'results/packing-swarm/round23/proof/receipt.json'
    archive.mkdir(parents=True, exist_ok=False)
    result.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='running', archive=str(archive), stages={},
                   elapsed_seconds=None, timeout_seconds=53)
    save(result, receipt)

    def run(name, command, seconds):
        available = deadline - time.monotonic()
        if available <= 0:
            raise RuntimeError('whole replay deadline exhausted')
        bound = min(seconds, available)
        then = time.monotonic()
        try:
            process = subprocess.run(command, cwd=archive, env=dict(os.environ, LEAN_PATH=str(archive)),
                                     capture_output=True, text=True, timeout=bound)
            log = process.stdout + process.stderr
            stage = dict(status='succeeded' if process.returncode == 0 else 'failed',
                         exit_code=process.returncode)
        except (subprocess.TimeoutExpired, OSError) as error:
            log = str(error)
            stage = dict(status='boundary_error', error_type=type(error).__name__, exit_code=None)
        stage.update(command=command, elapsed_seconds=time.monotonic() - then, timeout_seconds=bound)
        (archive / (name + '.log')).write_text(log)
        save(archive / (name + '.json'), stage)
        receipt['stages'][name] = stage
        save(result, receipt)
        if stage['status'] != 'succeeded' or re.search(r'\b(error|warning):', log):
            raise RuntimeError('proof stage failed: ' + str(archive / (name + '.log')))
        return log

    try:
        names = {}
        for namespace, module, pinned in MODULES:
            path = SOURCE / (module + '.lean')
            text = path.read_text()
            if pinned is not None and sha(path) != pinned:
                raise RuntimeError('predecessor hash mismatch: ' + str(path))
            if re.search(r'\b(sorry|admit|axiom|unsafe|native_decide)\b', text):
                raise RuntimeError('unchecked declaration: ' + str(path))
            entries = re.findall(r'^theorem (\w+)', text, re.M)
            if not entries or len(set(entries)) != len(entries):
                raise RuntimeError('missing or duplicate theorem: ' + str(path))
            names[namespace] = entries
            (archive / path.name).write_text(text)
        audit = archive / 'Audit.lean'
        audit.write_text('import SerializedGroup\n' + ''.join(
            '#print axioms ' + ns + '.' + name + '\n'
            for ns, entries in names.items() for name in entries))
        contracts = [p.resolve(strict=True) for p in args.contract]
        contract_hashes = {str(p): sha(p) for p in contracts}
        version = run('version', [str(LEAN), '--version'], 2)
        if 'version 4.33.0,' not in version:
            raise RuntimeError('wrong compiler version')
        for _, module, _ in MODULES:
            run(module, [str(LEAN), '--root=' + str(archive), '-o',
                         str(archive / (module + '.olean')), str(archive / (module + '.lean'))], 12)
        log = run('audit', [str(LEAN), '--root=' + str(archive), str(audit)], 8)
        actual = {}
        pattern = re.compile(r"'(Packing(?:20|21|22|23)\.\w+)' (?:does not depend on any axioms|depends on axioms: \[([^\]]*)\])")
        for match in pattern.finditer(log):
            name = match[1]
            if name in actual:
                raise RuntimeError('duplicate audit theorem: ' + name)
            actual[name] = [] if match[2] is None else re.split(r'\s*,\s*', match[2].strip())
        if pattern.sub('', log).strip():
            raise RuntimeError('unexpected audit output')
        expected = {ns + '.' + name for ns, entries in names.items() for name in entries}
        if set(actual) != expected:
            raise RuntimeError('audit omitted or added a theorem')
        if any(set(used) - {'propext', 'Quot.sound', 'Classical.choice'} for used in actual.values()):
            raise RuntimeError('unapproved axiom dependency')
        if {str(p): sha(p) for p in contracts} != contract_hashes:
            raise RuntimeError('native contract changed during replay')
        receipt.update(
            status='checked_structural_lean', theorem_count=len(names['Packing23']),
            inherited_theorem_count=sum(len(names[ns]) for ns in ['Packing20', 'Packing21', 'Packing22']),
            total_audited_theorems=len(expected), printed_axioms=actual,
            compiler=dict(path=str(LEAN), version=version.strip(), sha256=sha(LEAN),
                          Std_sha256=sha(LEAN.parent.parent / 'lib/lean/Std.olean')),
            source_sha256={p.name: sha(p) for p in SOURCE.iterdir() if p.is_file() and p.suffix in ['.lean', '.py', '.md']},
            predecessor_source_sha256={module + '.lean': pinned for _, module, pinned in MODULES if pinned is not None},
            native_contract_sha256=contract_hashes,
            artifact_sha256={p.name: sha(p) for p in archive.iterdir() if p.is_file()},
            scope='definitions-only original pass/n/m/i and serialized q/pass/n/mm/i schedules for both FW4WG2 and FW8WG1; exact schedule permutation, unique complete coordinates, arbitrary independent-coordinate updates and all-K loop interchange; forty ascending K128 per-coordinate folds; q-local saved-word/scale/feature reconstruction; opaque floating payload dictionary and literal primary-first output',
            primitive_semantics='arbitrary opaque MMA/FMA/mul/add, word expansion and saved-word functions; equality of mathematical functions, not CUDA execution traces',
            Cpp_CUDA_refinement='unset', floating_primitive_refinement='unset',
            native_primitive_binding='both arms ordered_fold: volatile inline PTX fma.rn.f32; mathematical FMA remains opaque',
            native_word_refinement='unset', native_fault_refinement='unset',
            native_execution=False, native_time='unset', physical_capacity='unset',
            cache_occupancy_granularity_residency='unset', controller_refinement='unset',
            packing_bound='unchanged', symmetry_narrowing=False, model_task_HTTP='unset')
    except (RuntimeError, OSError, ValueError) as error:
        receipt.update(status='failed', error=str(error))
    receipt['elapsed_seconds'] = time.monotonic() - started
    save(result, receipt)
    save(archive / 'receipt.json', receipt)
    print(json.dumps(dict(status=receipt['status'], theorems=receipt.get('theorem_count'),
                          audited=receipt.get('total_audited_theorems'), receipt=str(result),
                          elapsed_seconds=receipt['elapsed_seconds'])))
    return 0 if receipt['status'] == 'checked_structural_lean' else 1


if __name__ == '__main__':
    raise SystemExit(main())
