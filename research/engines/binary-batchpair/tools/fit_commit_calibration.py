#!/usr/bin/env python3
"""Fit the expected-commit calibration p(accept position i | earlier accepted) = sigmoid(a + b*gap_i + c*i)
from an H5_COMMIT_TRACE file (one JSON line per round: selected, accepted, matched_eos, gaps).

  fit_commit_calibration.py TRACE.jsonl   -> prints "a,b,c" on stdout, fit report on stderr

Position i (0-based) is observed when every earlier proposal was accepted and i < selected; its outcome
is accepted (i < accepted) or rejected (i == accepted). Rounds ending in a matched EOS stop at the EOS.
"""
import json, sys
import numpy as np

xs, ys = [], []
for line in open(sys.argv[1]):
    r = json.loads(line)
    sel, acc, g = r['selected'], r['accepted'], r['gaps']
    last = acc if r['matched_eos'] else min(acc + 1, sel)
    for i in range(last):
        xs.append((1.0, float(g[i]), float(i))); ys.append(1.0 if i < acc else 0.0)
X, y = np.array(xs), np.array(ys)
w = np.zeros(3)
for _ in range(50):
    p = 1 / (1 + np.exp(-X @ w))
    H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(3)
    step = np.linalg.solve(H, X.T @ (y - p))
    w += step
    if np.abs(step).max() < 1e-9: break
p = 1 / (1 + np.exp(-X @ w))
ll = float(np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
base = float(y.mean())
ll0 = base * np.log(base) + (1 - base) * np.log(1 - base)
print(f'observations {len(y)}, accept rate {base:.3f}, mean log-lik {ll:.4f} (constant {ll0:.4f})', file=sys.stderr)
for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 1e9)):
    m = (X[:, 1] >= lo) & (X[:, 1] < hi)
    if m.any(): print(f'gap [{lo},{hi}): n {m.sum()} observed {y[m].mean():.3f} fitted {p[m].mean():.3f}', file=sys.stderr)
print(f'{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}')
