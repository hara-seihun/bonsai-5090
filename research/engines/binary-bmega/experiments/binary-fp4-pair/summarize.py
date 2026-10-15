#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for path in sorted((root / "results").glob("*-r*.jsonl")):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != 56:
        raise SystemExit(f"incomplete sweep: {path}: {len(records)}")
    kinds = {(r["kind"], r["n"], r["k"], r["r"]) for r in records}
    if len(kinds) != 1:
        raise SystemExit(f"mixed shape receipt: {path}")
    kind, n, k, r = kinds.pop()
    result = dict(kind=kind, n=n, k=k, r=r)
    for cache in ("warm", "cold"):
        field = f"inclusive_{cache}_us"
        best = {route: min((x for x in records if x["route"] == route), key=lambda x: x[field])
                for route in ("ordered_two_pass", "joint")}
        base, joint = best["ordered_two_pass"], best["joint"]
        result[cache] = {"baseline_us": base[field], "joint_us": joint[field],
                         "speedup": base[field] / joint[field],
                         "baseline_cfg": base["cfg"], "baseline_splits": base["splits"],
                         "joint_cfg": joint["cfg"], "joint_splits": joint["splits"]}
    result["max_error"] = max(x["max_error_ordered"] for x in records if x["route"] == "joint")
    result["max_relative_rms"] = max(x["relative_rms_ordered"] for x in records if x["route"] == "joint")
    rows.append(result)
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("| Projection | R | Warm ordered → joint µs | Cold ordered → joint µs | Cold speedup | Joint cfg/split |")
print("|---|---:|---:|---:|---:|---:|")
for row in rows:
    w, c = row["warm"], row["cold"]
    print(f'| {row["kind"]} | {row["r"]} | {w["baseline_us"]:.3f} → {w["joint_us"]:.3f} | '
          f'{c["baseline_us"]:.3f} → {c["joint_us"]:.3f} | {c["speedup"]:.3f}× | {c["joint_cfg"]}/{c["joint_splits"]} |')
