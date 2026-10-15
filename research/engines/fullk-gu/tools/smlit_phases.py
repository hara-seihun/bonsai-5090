#!/usr/bin/env python3
"""smlit_phases.py DUMP SRC [NAME_SUBSTR]: median per-warp phase cycles from H5_LIT_MARK stamps (first round)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import smlit_analyze as S
names = S.tag_names(sys.argv[2]); want = sys.argv[3] if len(sys.argv) > 3 else 'prep'
first, b, r = S.load(sys.argv[1]); r = r[:int(b[0])]
for tag in sorted(set(r['tag'])):
    nm = names[int(tag)]
    if want not in nm: continue
    for g in sorted(set(r['grid'][r['tag'] == tag])):
        x = r[(r['tag'] == tag) & (r['grid'] == g)]
        pts = np.column_stack([x['cw'], x['mk'], x['cx']]).astype(float)
        d = np.median(np.diff(pts, axis=1), axis=0)
        print(f"{nm} grid {g} threads {x['threads'][0]} n {len(x)} pre-wait {np.median(x['cw']):.0f} | loads {d[0]:.0f} norm {d[1]:.0f} silu/sign/had {d[2]:.0f} after/store {d[3]:.0f} quant {d[4]:.0f} | post-wait {np.median(x['cx']-x['cw']):.0f} cyc")
