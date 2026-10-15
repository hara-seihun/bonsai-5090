#!/usr/bin/env python3
"""smlit_placement.py DUMP SRC: for each big GEMM after a latency kernel, SM placement vs predecessor (first round)."""
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import smlit_analyze as S
names = S.tag_names(sys.argv[2])
first, b, r = S.load(sys.argv[1]); r = r[:int(b[0])]
L = S.launches(r); L.sort(key=lambda x: min(int(g['t0'].min()) for g in x['ctas']))
T0 = int(r['t0'].min()); seen = collections.Counter()
for i, x in enumerate(L):
    if names[x['tag']] == 'k_gemm_fp4' and x['grid'] >= 100 and seen[x['grid']] < 2:
        seen[x['grid']] += 1; prev = L[i - 1]
        psms = set(int(g['sm'][0]) for g in prev['ctas'])
        cnt = collections.Counter(int(g['sm'][0]) for g in x['ctas'])
        missing = set(range(170)) - set(cnt); doubled = [s for s, c in cnt.items() if c > 1]
        ex = {}
        for g in x['ctas']: ex.setdefault(cnt[int(g['sm'][0])], []).append((int(g['t1'].max()) - T0) / 1e3)
        print(f"{names[prev['tag']]}/{prev['grid']} -> gemm/{x['grid']}: missing {len(missing)} (prev SMs {len(missing & psms)}), doubled {len(doubled)}; "
              f"exit single max {max(ex.get(1, [0])):.2f} med {np.median(ex.get(1, [0])):.2f} doubled max {max(ex.get(2, [0])):.2f}")
