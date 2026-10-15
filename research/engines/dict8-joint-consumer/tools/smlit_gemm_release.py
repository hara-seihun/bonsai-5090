#!/usr/bin/env python3
"""Read matched-release GEMM phase marks, not the generic post-PDL-wait activity proxy.

Usage: smlit_gemm_release.py DUMP SOURCE_DIR OUTPUT_JSON
The dump and source must belong to the same exact instrumented build.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from smlit_analyze import launches, load, tag_names


def distribution(values):
    return {"n": len(values), "p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def analyze(dump, source):
    first, bounds, recs = load(dump)
    names = tag_names(source)
    long = recs["cx"] > 20000
    if not np.any(long):
        raise ValueError("no long records for clock calibration")
    ns_per_cycle = float(np.median((recs["t1"][long] - recs["t0"][long]) / recs["cx"][long]))
    report = {"ns_per_cycle": ns_per_cycle, "rounds": []}
    lo = 0
    for ri, hi in enumerate(bounds):
        records = recs[lo:int(hi)]
        lo = int(hi)
        origin = int(records["t0"].min())
        rows = []
        for index, launch in enumerate(launches(records)):
            name = names.get(launch["tag"], str(launch["tag"]))
            if name not in ("k_gemm_fp4", "k_gemm_fp4_tail"):
                continue
            if len(launch["seen"]) != launch["grid"]:
                raise ValueError(f"incomplete GEMM launch {index}")
            warps = np.concatenate(launch["ctas"])
            if np.any(warps["mk"] == 0):
                raise ValueError(f"GEMM {index} lacks matched-release marks; wrong build or unsupported route")
            if np.any(np.diff(warps["mk"].astype(np.int64), axis=1) < 0):
                raise ValueError(f"nonmonotonic GEMM marks {index}")
            producer = warps["warp"] == warps["threads"] // 32 - 1
            consumers = warps[~producer]
            producers = warps[producer]
            if not len(producers) or not len(consumers):
                raise ValueError(f"GEMM {index} is missing a warp role")
            # Ring issue and data-ready marks use clock64; convert using this dump's observed clock.
            p = producers["mk"].astype(np.float64) * ns_per_cycle / 1000
            c = consumers["mk"].astype(np.float64) * ns_per_cycle / 1000
            entry_ns = (warps["t0"] - np.uint64(origin)).astype(np.float64)
            consumer_entry_ns = (consumers["t0"] - np.uint64(origin)).astype(np.float64)
            releases = (entry_ns + warps["mk"][:, 3] * ns_per_cycle) / 1000
            ready = (consumer_entry_ns + consumers["mk"][:, 0] * ns_per_cycle) / 1000
            entry = entry_ns / 1000
            rows.append({
                "launch": index, "name": name, "grid": launch["grid"],
                "threads": int(warps["threads"][0]), "aux": int(warps["aux"][0]),
                "entry_first_us": float(entry.min()), "entry_last_us": float(entry.max()),
                "exit_last_us": (int(warps["t1"].max()) - origin) / 1000,
                "release_first_us": float(releases.min()), "release_last_us": float(releases.max()),
                "sms_used": len(set(map(int, warps["sm"]))),
                "consumer_first_stage_ready_us": distribution(ready),
                "producer_entry_to_ring_issue_us": distribution(p[:, 0]),
                "producer_readiness_wait_us": distribution(p[:, 1] - p[:, 0]),
                "consumer_first_stage_to_loop_end_us": distribution(c[:, 1] - c[:, 0]),
                "consumer_late_grid_acquire_us": distribution(c[:, 2] - c[:, 1]),
                "consumer_acquire_to_release_us": distribution(c[:, 3] - c[:, 2]),
            })
        if not rows:
            raise ValueError("no marked GEMM launches")
        report["rounds"].append({"round": first + ri, "gemms": rows})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    try:
        report = analyze(args.dump, args.source)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"matched-release trace: {error}\n")


if __name__ == "__main__":
    main()
