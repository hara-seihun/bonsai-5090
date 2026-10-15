#!/usr/bin/env python3
import json
from pathlib import Path
root = Path(__file__).resolve().parent
summary = []
for file in sorted((root / 'results').glob('*-r*.jsonl')):
    records = [json.loads(line) for line in file.read_text().splitlines()]
    if len(records) != 21:
        raise SystemExit(f'incomplete sweep {file}: {len(records)}')
    baseline = next(x for x in records if x['route'] == 'dp4a')
    best = min((x for x in records if x['route'] == 'fp4_digits'), key=lambda x: x['inclusive_cold_us'])
    row = {'kind': baseline['kind'], 'r': baseline['r'], 'n': baseline['n'], 'k': baseline['k'],
           'baseline_cold_us': baseline['inclusive_cold_us'], 'joint_cold_us': best['inclusive_cold_us'],
           'speedup': baseline['inclusive_cold_us'] / best['inclusive_cold_us'],
           'cfg': best['cfg'], 'splits': best['splits'],
           'max_error': max(x['max_error'] for x in records),
           'relative_rms': max(x['relative_rms'] for x in records)}
    summary.append(row)
    print(f"{row['kind']} R{row['r']} {row['baseline_cold_us']:.3f} → {row['joint_cold_us']:.3f} us "
          f"{row['speedup']:.3f}x cfg{row['cfg']} split{row['splits']} max{row['max_error']:.4g} rms{row['relative_rms']:.4g}")
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
