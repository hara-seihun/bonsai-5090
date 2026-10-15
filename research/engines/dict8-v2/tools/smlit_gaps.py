#!/usr/bin/env python3
"""smlit_gaps.py LAUNCHES_CSV ROUND: transition gaps (next first wait release - previous last exit) by kernel pair."""
import csv, collections, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['round'] == sys.argv[2]]
rows.sort(key=lambda r: float(r['entry_us']))
lab = lambda r: r['name'].replace('k_', '')[:10] + '/' + r['grid']
gap = collections.defaultdict(lambda: [0, 0.0]); tot = 0
for a, b in zip(rows, rows[1:]):
    g = float(b['wait_release_first_us']) - float(a['exit_us']); k = lab(a) + ' -> ' + lab(b)
    gap[k][0] += 1; gap[k][1] += g; tot += g
print('total transition gap us', round(tot, 1), 'span', rows[-1]['exit_us'])
for k, v in sorted(gap.items(), key=lambda kv: -kv[1][1])[:16]:
    print(f'{v[1]:8.1f} us  n={v[0]:3d} avg {v[1] / v[0]:5.2f}  {k}')
