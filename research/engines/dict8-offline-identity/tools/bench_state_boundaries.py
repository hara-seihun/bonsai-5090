#!/usr/bin/env python3
"""Fixed-work target comparisons. The caller owns GPU admission; this runner rents nothing."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--model', required=True, type=Path)
p.add_argument('--evidence', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--replayer', type=Path, default=Path('./profile-replay'))
p.add_argument('--repeats', required=True, type=int)
p.add_argument('--variants', nargs='+', choices=['control', 'state', 'attention', 'both'], required=True)
a = p.parse_args()
if a.repeats < 1:
    p.error('--repeats must be positive')
for f in [a.model, a.replayer, a.evidence / 'frozen-plan.txt', a.evidence / 'ordinary-events.txt', a.evidence / 'prompt-events.txt']:
    if not f.is_file():
        p.error(f'missing input: {f}')
a.output.mkdir(parents=True, exist_ok=False)
base = dict(os.environ)
for key in ['H5_PROF', 'H5_NOGRAPH', 'H5_TREE2', 'H5_MEGA', 'H5_PDLCLOCK', 'H5_PLAN_WRITE', 'H5_REPLAY_WRITE', 'H5_WSCALE', 'H5_GDN_CG', 'H5_GDN_DELTA', 'H5_ATTN_TPS', 'H5_ATTN8']:
    base.pop(key, None)
base.update(H5_COMPACT='dense5', H5_PLAN_READ=str((a.evidence / 'frozen-plan.txt').resolve()), H5_PDL='1', H5_STATE='f16', H5_KV='fp8')
variants = {'control': ('0', '0'), 'state': ('1', '0'), 'attention': ('0', '1'), 'both': ('1', '1')}
receipt = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'runs': []}
(a.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
for repeat in range(a.repeats):
    for workload, events, option in [('ordinary', 'ordinary-events.txt', '--ordinary'), ('verify', 'prompt-events.txt', '--decode-only')]:
        panel = []
        order = a.variants if repeat % 2 == 0 else list(reversed(a.variants))
        for variant in order:
            name = f'{workload}-{repeat}-{variant}'
            prefix = a.output / name
            env = dict(base, H5_STATE_FUSION=variants[variant][0], H5_ATTN_DIRECT=variants[variant][1])
            cmd = [str(a.replayer.resolve()), str(a.model.resolve()), str((a.evidence / events).resolve()), str(prefix.resolve()), option]
            started = time.time()
            with prefix.with_suffix('.log').open('w') as log:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            run = {'name': name, 'command': cmd, 'state_fusion': env['H5_STATE_FUSION'], 'attention_direct': env['H5_ATTN_DIRECT'], 'returncode': result.returncode, 'wall_s': time.time() - started}
            receipt['runs'].append(run)
            if result.returncode == 0:
                rows = list(csv.reader(prefix.with_suffix('.csv').open()))[1:]
                run['rounds'] = len(rows)
                run['mean_ms'] = sum(float(r[2]) for r in rows) * 1000 / len(rows)
                predictions = [r[3:] for r in rows]
                panel.append((variant, run, predictions))
            (a.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
            print(json.dumps(run), flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
        controls = [pred for variant, _, pred in panel if variant == 'control']
        if len(controls) == 1:
            for _, run, pred in panel:
                if list(map(len, pred)) != list(map(len, controls[0])):
                    raise RuntimeError('fixed-work prediction shape mismatch')
                run['prediction_mismatches'] = sum(x != y for r, s in zip(pred, controls[0]) for x, y in zip(r, s))
                run['prediction_count'] = sum(map(len, pred))
        (a.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
