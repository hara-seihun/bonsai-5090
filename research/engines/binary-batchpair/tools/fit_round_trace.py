#!/usr/bin/env python3
"""Fit p(proposal j accepted | proposals < j accepted, selector gap_j, j) from H5_ROUND_TRACE files.

  fit_round_trace.py TRACE [TRACE...]

Trace lines: "nact nd accepted gap0 ... gap(P-1)" (one per verify round, fixed commitment). Only rounds that
verified every proposal (nd == P) are uncensored beyond the cut; positions j < min(nd, accepted+1) are used.
Prints the logistic fit sigmoid(a + b*gap + c*j) as H5_SPEC_PCAL and a calibration table.
"""
import sys
import numpy as np

rows = []
for path in sys.argv[1:]:
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        nact, nd, acc = int(f[0]), int(f[1]), int(f[2]); gaps = [float(x) for x in f[3:]]
        if len(gaps) < nd:
            continue   # torn last line of a live trace
        for j in range(min(nd, acc + 1)):
            rows.append((gaps[j], j, 1.0 if acc > j else 0.0))
X = np.array([[1.0, g, j] for g, j, _ in rows]); y = np.array([r[2] for r in rows])
print(f"{len(rows)} position observations from {len(sys.argv) - 1} trace(s); base acceptance {y.mean():.3f}")
w = np.zeros(3)
for _ in range(50):   # Newton / IRLS
    p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p)
    H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(3); g = X.T @ (y - p)
    step = np.linalg.solve(H, g); w += step
    if np.abs(step).max() < 1e-8:
        break
print(f"H5_SPEC_PCAL={w[0]:.4f},{w[1]:.4f},{w[2]:.4f}")
p = 1 / (1 + np.exp(-X @ w))
edges = [-1e9, 0.5, 1, 2, 4, 1e9]
gaps = X[:, 1]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (gaps >= lo) & (gaps < hi)
    if m.any():
        print(f"gap [{lo:g},{hi:g}): n={m.sum():6d} observed {y[m].mean():.3f} model {p[m].mean():.3f}")
for j in range(int(X[:, 2].max()) + 1):
    m = X[:, 2] == j
    print(f"position {j}: n={m.sum():6d} observed {y[m].mean():.3f} model {p[m].mean():.3f}")
