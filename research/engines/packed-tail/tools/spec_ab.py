#!/usr/bin/env python3
"""spec_ab.py PREFIX: summarize a spec-replay run (PREFIX.csv, PREFIX.log with ab_arms, PREFIX-tokens.txt).

Per arm: tokens/s over the measured rounds (accepted+1 per round over summed round wall time),
mean round, verifier, host and drafter wall time, and mean tokens per round. B/A is reported on
tokens/s and on round time; if the arms produced different token streams, that is printed, since
then the two arms ran different work.
"""
import csv, re, statistics, sys
from pathlib import Path

p = sys.argv[1]
rows = list(csv.DictReader(open(p + ".csv")))
log = Path(p + ".log").read_text() if Path(p + ".log").exists() else ""
m = re.search(r"ab_arms (\w+)", log)
arms = m.group(1) if m else None
toks = Path(p + "-tokens.txt").read_text().splitlines()
by = {}
for r in rows:
    by.setdefault(int(r["pass"]), []).append(r)
res = {}
for ps, rs in sorted(by.items()):
    arm = arms[ps] if arms else "A"
    t = sum(float(r["round_s"]) for r in rs); n = sum(int(r["accepted"]) + 1 for r in rs)
    res.setdefault(arm, []).append({"tps": n / t, "round": 1e3 * t / len(rs), "verify": 1e3 * statistics.mean(float(r["verify_s"]) for r in rs),
                                    "host": 1e3 * statistics.mean(float(r["host_s"]) for r in rs), "draft": 1e3 * statistics.mean(float(r["draft_s"]) for r in rs),
                                    "tpr": n / len(rs), "rounds": len(rs)})
tag = Path(p).name
for a in sorted(res):
    v = res[a]; f = lambda k: statistics.mean(x[k] for x in v)
    print(f"{tag} arm {a}: passes {len(v)} tok/s {f('tps'):.1f}  round {f('round'):.4f} ms = verify {f('verify'):.4f} + host {f('host'):.4f} + draft {f('draft'):.4f}  tok/round {f('tpr'):.3f}  rounds {f('rounds'):.0f}  per-pass tok/s {[round(x['tps'], 1) for x in v]}")
if arms and len(res) == 2:
    A = statistics.mean(x["tps"] for x in res["A"]); B = statistics.mean(x["tps"] for x in res["B"])
    RA = statistics.mean(x["round"] for x in res["A"]); RB = statistics.mean(x["round"] for x in res["B"])
    print(f"{tag} tok/s B/A = {B / A:.4f} ({100 * (B / A - 1):+.2f}%)   round B/A = {RB / RA:.4f} ({100 * (RB / RA - 1):+.2f}%)")
    sa = {toks[i] for i in range(len(toks)) if arms[i] == "A"}; sb = {toks[i] for i in range(len(toks)) if arms[i] == "B"}
    print(f"{tag} token streams: A {len(sa)} distinct, B {len(sb)} distinct, {'same' if sa == sb else 'DIFFERENT between arms'}")
else:
    print(f"{tag} token streams: {len(set(toks))} distinct over {len(toks)} passes")
