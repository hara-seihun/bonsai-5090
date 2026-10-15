#!/usr/bin/env python3
import json
from pathlib import Path
root = Path(__file__).resolve().parent / 'results' / 'packed1'
summary = []
for file in sorted(root.glob('packed1-*-r*.jsonl')):
    records = [json.loads(line) for line in file.read_text().splitlines()]
    if len(records) != 21:
        raise SystemExit(f'incomplete Packed1 sweep {file}: {len(records)}')
    controls = [json.loads(line) for line in (root / file.name.replace('packed1-', 'wf8-', 1)).read_text().splitlines()]
    if len(controls) != 21:
        raise SystemExit(f'incomplete WF8 control {file}')
    base = next(x for x in records if x['route'] == 'dp4a')
    candidates = [x for x in records if x['route'] == 'fp4_digits']
    if not all(x['wf8_packed1_bitexact'] for x in candidates):
        raise SystemExit(f'parity failed {file}')
    best = min(candidates, key=lambda x: x['inclusive_cold_us'])
    first_cfg, first_split = {'gu': (0, 4), 'down': (2, 8), 'head': (0, 1)}[base['kind']]
    first = next(x for x in candidates if (x['cfg'], x['splits']) == (first_cfg, first_split))
    wf8 = next(x for x in controls if (x['cfg'], x['splits']) == (first_cfg, first_split))
    row = dict(kind=base['kind'], r=base['r'], n=base['n'], k=base['k'], baseline_cold_us=base['inclusive_cold_us'],
               best_cold_us=best['inclusive_cold_us'], speedup=base['inclusive_cold_us']/best['inclusive_cold_us'],
               cfg=best['cfg'], splits=best['splits'], first_engine_cfg=first_cfg, first_engine_splits=first_split,
               first_engine_cold_us=first['inclusive_cold_us'], matched_wf8_cold_us=wf8['inclusive_cold_us'],
               wf8_packed1_bitexact=True, max_error=max(x['max_error'] for x in candidates),
               max_relative_rms=max(x['relative_rms'] for x in candidates))
    summary.append(row)
    print(f"{row['kind']} R{row['r']} DP4A {row['baseline_cold_us']:.3f} → Packed1 {row['best_cold_us']:.3f} "
          f"us {row['speedup']:.3f}x best{row['cfg']}/{row['splits']}; "
          f"engine {row['first_engine_cold_us']:.3f} vs WF8sameplan {row['matched_wf8_cold_us']:.3f} us")
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
