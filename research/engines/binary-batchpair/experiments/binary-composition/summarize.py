#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for r in (1, 8, 16):
    path = root / 'results' / f'r{r}.jsonl'
    if not path.exists():
        raise SystemExit(f'missing GPU receipt: {path}')
    rows.extend(json.loads(line) for line in path.read_text().splitlines())
summary = []
for r in (1, 8, 16):
    for include_down in (False, True):
        group = [x for x in rows if x['r'] == r and x['down_included'] == include_down]
        baseline = next(x for x in group if x['route'] == 'unfused_selected')
        best = min((x for x in group if x['route'] == 'fused'), key=lambda x: x['cold_us'])
        same = next(x for x in group if x['route'] == 'unfused_matching' and x['chunk'] == best['chunk'])
        summary.append({'r': r, 'down_included': include_down, 'unfused_selected': baseline,
                        'fused_best': best, 'unfused_same_chunk': same,
                        'selected_over_best': baseline['cold_us'] / best['cold_us'],
                        'matching_over_best': same['cold_us'] / best['cold_us']})
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('| R | down | selected us | fused us | tile/chunk | selected/fused | same-chunk/fused |')
print('|---:|:---:|---:|---:|:---:|---:|---:|')
for s in summary:
    b = s['fused_best']
    print(f"|{s['r']}|{'yes' if s['down_included'] else 'no'}|"
          f"{s['unfused_selected']['cold_us']:.3f}|{b['cold_us']:.3f}|"
          f"{b['tile']}/{b['chunk']}|{s['selected_over_best']:.3f}×|{s['matching_over_best']:.3f}×|")
