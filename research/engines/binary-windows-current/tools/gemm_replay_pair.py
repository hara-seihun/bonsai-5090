#!/usr/bin/env python3
"""ABBA fixed-work whole-target replay; requires an already authorized idle GPU."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-replay', type=Path, required=True)
    p.add_argument('--candidate-replay', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--control-plan', type=Path, required=True)
    p.add_argument('--candidate-plan', type=Path, required=True)
    p.add_argument('--mode', choices=['ordinary', 'verify'], required=True)
    p.add_argument('--pdl', choices=['0', '1'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for key in ('control_replay', 'candidate_replay', 'model', 'events', 'control_plan', 'candidate_plan'):
        path = getattr(args, key).resolve()
        if not path.is_file():
            p.error(f'{key}: not a file: {path}')
        setattr(args, key, path)
    if args.output.exists():
        p.error('output directory exists; choose a fresh path')
    args.output.mkdir(parents=True)
    env = dict(os.environ)
    for name in ('H5_PROF', 'H5_NOGRAPH', 'H5_TREE2', 'H5_FORCE_CFG', 'H5_PLAN_WRITE', 'H5_REPLAY_WRITE', 'H5_PDLCLOCK'):
        if name in env:
            p.error(f'unset interfering environment: {name}')
    env.update(H5_COMPACT='dense5', H5_PDL=args.pdl)
    records = []
    for index, variant in enumerate(('control', 'candidate', 'candidate', 'control')):
        plan = args.control_plan if variant == 'control' else args.candidate_plan
        env['H5_PLAN_READ'] = str(plan)
        prefix = args.output.resolve() / f'{index}-{variant}'
        replay = args.control_replay if variant == 'control' else args.candidate_replay
        command = [str(replay), str(args.model), str(args.events), str(prefix), '--ordinary' if args.mode == 'ordinary' else '--decode-only']
        with prefix.with_suffix('.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            p.exit(1, f'replay failed ({result.returncode}); inspect {prefix}.log\n')
        with prefix.with_suffix('.csv').open() as f:
            rows = list(csv.reader(f))
        decode = [r for r in rows if int(r[0]) > 0]
        if not decode or args.mode == 'ordinary' and any(int(r[1]) != 1 for r in decode):
            p.exit(1, 'fixture is not expected fixed work\n')
        records.append({'variant': variant, 'mean_ms': statistics.mean(float(r[2])*1000 for r in decode),
                        'rounds': len(decode), 'row_counts': [int(r[1]) for r in decode],
                        'predictions': [list(map(int, r[3:])) for r in decode]})
    if any((r['rounds'], r['row_counts']) != (records[0]['rounds'], records[0]['row_counts']) for r in records):
        p.exit(1, 'replays executed different amounts of work\n')
    controls = [r['mean_ms'] for r in records if r['variant'] == 'control']
    candidates = [r['mean_ms'] for r in records if r['variant'] == 'candidate']
    changed = [sum(a != b for ar, br in zip(records[0]['predictions'], r['predictions']) for a, b in zip(ar, br)) for r in records]
    summary = {'mode': args.mode, 'pdl': int(args.pdl), 'control_ms': statistics.mean(controls),
               'candidate_ms': statistics.mean(candidates),
               'latency_reduction_percent': 100*(1-statistics.mean(candidates)/statistics.mean(controls)),
               'prediction_changes_vs_first_control': changed, 'runs': records,
               'inputs': {key: {'path': str(getattr(args, key)), 'sha256': sha(getattr(args, key))}
                          for key in ('control_replay', 'candidate_replay', 'events', 'control_plan', 'candidate_plan')},
               'plans_by_variant': {'control': str(args.control_plan), 'candidate': str(args.candidate_plan)},
               'model': str(args.model), 'environment': {k: v for k, v in env.items() if k.startswith('H5_')},
               'scope': 'teacher-forced fixed work, prefill excluded; not task quality or fresh generated-request speed'}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('runs', 'inputs', 'environment')}, indent=2))


if __name__ == '__main__':
    main()
