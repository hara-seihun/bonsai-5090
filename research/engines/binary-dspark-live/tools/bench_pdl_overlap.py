#!/usr/bin/env python3
"""Native, fixed-work ABBA comparisons. Never provisions or acquires a GPU."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def file_id(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_run(prefix, expected_work):
    with prefix.with_suffix(".csv").open() as f:
        rows = list(csv.reader(f))
    if not rows or rows[0][0] != "0":
        raise ValueError("missing prefill round")
    work = [(int(r[0]), int(r[1])) for r in rows[1:]]
    elapsed = [float(r[2]) for r in rows[1:]]
    predictions = [list(map(int, r[3:])) for r in rows[1:]]
    if work != expected_work:
        raise ValueError("replay omitted or changed recorded work")
    if not work or any(not math.isfinite(t) or t <= 0 for t in elapsed):
        raise ValueError("invalid decode timings")
    if any(len(p) != n for (_, n), p in zip(work, predictions)):
        raise ValueError("incomplete predictions")
    return work, statistics.mean(elapsed) * 1000, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", action="append", required=True, help="LABEL=/absolute/replayer")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True, help="retained halo-profile directory")
    p.add_argument("--output", type=Path, required=True, help="new directory; creation refuses existing")
    p.add_argument("--pairs", type=int, required=True)
    p.add_argument("--pdl", choices=("0", "1"), required=True)
    p.add_argument("--timeout", type=int, required=True, help="per process seconds")
    a = p.parse_args()
    if a.pairs < 1 or not 0 < a.timeout <= 1800:
        p.error("positive pairs and timeout <=1800 required")
    binaries = {"control": a.baseline.resolve(strict=True)}
    for item in a.candidate:
        label, sep, path = item.partition("=")
        if not sep or not label.isidentifier() or label in binaries:
            p.error("candidate must be unique LABEL=PATH")
        binaries[label] = Path(path).resolve(strict=True)
    model = a.model.resolve(strict=True)
    plan = (a.evidence / "frozen-plan.txt").resolve(strict=True)
    fixtures = {w: (a.evidence / f"{w}-events.txt").resolve(strict=True)
                for w in ("ordinary", "prompt")}
    expected = {}
    for workload, fixture in fixtures.items():
        steps = [int(line.split()[1]) for line in fixture.read_text().splitlines()
                 if line.startswith("STEP ")]
        expected[workload] = list(enumerate(steps[1:], 1))
        if not expected[workload]:
            p.error("fixture contains no decode work")
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = {"binaries": {k: file_id(v) for k, v in binaries.items()},
                "plan": file_id(plan), "events": {k: file_id(v) for k, v in fixtures.items()},
                "model": str(model), "pdl": a.pdl, "runs": [], "pairs": []}
    # Eliminate inherited tuning/diagnostic interventions; these are the only H5 inputs.
    env = {k: v for k, v in os.environ.items() if not k.startswith("H5_")}
    env.update(H5_COMPACT="dense5", H5_PLAN_READ=str(plan), H5_PDL=a.pdl)
    for candidate in list(binaries)[1:]:
        for workload, events in fixtures.items():
            for pair in range(a.pairs):
                quartet = []
                for slot, label in enumerate(("control", candidate, candidate, "control")):
                    name = f"{candidate}-{workload}-{pair}-{slot}-{label}"
                    prefix = a.output / name
                    cmd = [str(binaries[label]), str(model), str(events), str(prefix),
                           "--ordinary" if workload == "ordinary" else "--decode-only"]
                    with prefix.with_suffix(".log").open("w") as log:
                        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=a.timeout, check=True)
                    lines = prefix.with_suffix(".log").read_text().splitlines()
                    metrics = json.loads(lines[-1])
                    if metrics["graph_replays"] <= 0:
                        raise ValueError("no native graph replay")
                    work, ms, pred = read_run(prefix, expected[workload])
                    run = {"name": name, "label": label, "workload": workload,
                           "mean_ms": ms, "graph_metrics": metrics}
                    manifest["runs"].append(run)
                    quartet.append((work, ms, pred))
                    if quartet[0][0] != work:
                        raise ValueError("work changed across pair")
                    (a.output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
                control_ms = statistics.mean([quartet[0][1], quartet[3][1]])
                candidate_ms = statistics.mean([quartet[1][1], quartet[2][1]])
                differences = lambda x, y: sum(i != j for r, s in zip(x, y) for i, j in zip(r, s))
                result = {"candidate": candidate, "workload": workload, "pair": pair,
                          "control_ms": control_ms, "candidate_ms": candidate_ms,
                          "latency_reduction_pct": 100 * (1 - candidate_ms / control_ms),
                          "control_prediction_changes": differences(quartet[0][2], quartet[3][2]),
                          "candidate_prediction_changes": differences(quartet[1][2], quartet[2][2]),
                          "cross_prediction_changes": differences(quartet[0][2], quartet[1][2])}
                manifest["pairs"].append(result)
                (a.output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        print(f"benchmark failed: {e}", file=sys.stderr)
        sys.exit(1)
