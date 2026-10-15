#!/usr/bin/env python3
import json
from pathlib import Path
root = Path(__file__).resolve().parent / 'results' / 'ragged'
rows = []
for file in sorted(root.glob('*-r*.jsonl')):
    records = [json.loads(line) for line in file.read_text().splitlines()]
    if len(records) != 21:
        raise SystemExit(f'incomplete sweep: {file}: {len(records)}')
    base = next(x for x in records if x['route'] == 'dp4a')
    candidates = [x for x in records if x['route'] == 'fp4_digits']
    if not all(x['wf8_packed1_bitexact'] for x in candidates):
        raise SystemExit(f'parity failed: {file}')
    row = dict(kind=base['kind'], r=base['r'], n=base['n'], k=base['k'],
               baseline_cold_us=base['inclusive_cold_us'], baseline_warm_us=base['inclusive_warm_us'],
               max_error=max(x['max_error'] for x in candidates),
               max_relative_rms=max(x['relative_rms'] for x in candidates), wf8_packed1_bitexact=True)
    for tag, subset in [('all_schedules', candidates), ('selected_F4G2T16', [x for x in candidates if x['cfg'] == 1])]:
        best = min(subset, key=lambda x: x['inclusive_cold_us'])
        row[tag] = dict(cold_us=best['inclusive_cold_us'], warm_us=best['inclusive_warm_us'],
                        cfg=best['cfg'], splits=best['splits'], speedup=base['inclusive_cold_us']/best['inclusive_cold_us'])
    rows.append(row)
    b, s = row['all_schedules'], row['selected_F4G2T16']
    print(f"{row['kind']} R{row['r']} DP4A{row['baseline_cold_us']:.3f} → best{b['cold_us']:.3f} us "
          f"{b['speedup']:.3f}x cfg{b['cfg']}/{b['splits']}; F4G2T16 {s['cold_us']:.3f} us "
          f"{s['speedup']:.3f}x split{s['splits']}")
(root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
