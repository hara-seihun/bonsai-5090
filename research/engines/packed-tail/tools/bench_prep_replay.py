#!/usr/bin/env python3
"""Paired full-target replay for the independent preparation candidate."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def read_csv(path):
    with path.open() as handle:
        rounds = [[float(x) if i == 2 else int(x) for i, x in enumerate(row)]
                  for row in csv.reader(handle)]
    decode = rounds[1:]
    if not decode or any(not math.isfinite(row[2]) or row[2] <= 0 for row in decode):
        raise ValueError(f"invalid replay timing: {path}")
    return decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "plan", "ordinary-events", "verify-events", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--g4", required=True, type=float)
    parser.add_argument("--pdl", required=True, choices=("0", "1"))
    parser.add_argument("--mode", required=True, choices=("1", "2", "3", "4"), help="1=all chunks; 2=nonnorm chunks; 3=RMS prepass; 4=coalesced row norm")
    parser.add_argument("--repeats", required=True, type=int)
    args = parser.parse_args()
    if not math.isfinite(args.g4) or args.g4 <= 0 or args.repeats <= 0:
        parser.error("g4 must be finite positive and repeats positive")
    for path in (args.binary, args.model, args.plan, args.ordinary_events, args.verify_events):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    # Ambient engine switches are not part of the comparison contract.
    base = {k: v for k, v in os.environ.items() if not k.startswith("H5_")}
    base.update(H5_COMPACT="dense5", H5_PLAN_READ=str(args.plan.resolve()), H5_PDL=args.pdl)
    summary = []
    for workload, fixture, option in (("ordinary", args.ordinary_events, "--ordinary"),
                                       ("verify", args.verify_events, "--decode-only")):
        records = {"row": [], "chunk": []}
        mismatches = compared = 0
        for rep in range(args.repeats):
            pair = {}
            for mode in (("row", "chunk") if rep % 2 == 0 else ("chunk", "row")):
                prefix = args.output / f"{workload}-{rep}-{mode}"
                env = dict(base, H5_PREP_CHUNK="0" if mode == "row" else args.mode)
                if mode == "chunk":
                    env["H5_PREP_G4"] = repr(args.g4)
                command = [str(args.binary.resolve()), str(args.model.resolve()),
                           str(fixture.resolve()), str(prefix.resolve()), option]
                with prefix.with_suffix(".log").open("w") as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=900, check=True)
                pair[mode] = read_csv(prefix.with_suffix(".csv"))
                records[mode].append(statistics.mean(row[2] for row in pair[mode]) * 1000)
            if [(r[0], r[1]) for r in pair["row"]] != [(r[0], r[1]) for r in pair["chunk"]]:
                raise ValueError("replay work differs")
            for left, right in zip(pair["row"], pair["chunk"]):
                if len(left[3:]) != len(right[3:]):
                    raise ValueError("prediction count differs")
                mismatches += sum(a != b for a, b in zip(left[3:], right[3:]))
                compared += len(left[3:])
        row_ms, chunk_ms = (statistics.mean(records[m]) for m in ("row", "chunk"))
        summary.append(dict(workload=workload, pdl=int(args.pdl), mode=int(args.mode), fixed_g4=args.g4,
                            row_ms=row_ms, chunk_ms=chunk_ms,
                            latency_reduction_percent=(1-chunk_ms/row_ms)*100,
                            repeats_ms=records, prediction_mismatches=mismatches,
                            compared_predictions=compared,
                            note="Teacher-forced fixed work; predictions never select later work. Not task quality."))
    encoded = json.dumps(summary, indent=2) + "\n"
    (args.output / "summary.json").write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
