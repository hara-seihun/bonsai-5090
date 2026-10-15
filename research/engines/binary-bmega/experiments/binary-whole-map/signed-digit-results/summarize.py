#!/usr/bin/env python3
import json
from pathlib import Path
root = Path(__file__).resolve().parent
summary = {"hidden7": [], "exact8": []}
for family, glob in (("hidden7", "down-r*.jsonl"), ("exact8", "exact8-*-r*.jsonl")):
    for path in sorted(root.glob(glob)):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            raise RuntimeError(f"empty receipt: {path}")
        for boundary in sorted({x["boundary"] for x in rows}):
            records = [x for x in rows if x["boundary"] == boundary]
            suffix = "given7" if family == "hidden7" else "given8"
            names = ["dp4a_" + suffix, "four_digits_" + suffix,
                     ("signed3_" if family == "hidden7" else "exact3_") + suffix]
            if not all(any(x["route"] == route for x in records) for route in names):
                continue
            selected = {route: min((x for x in records if x["route"] == route), key=lambda x: x["cold_us"])
                        for route in names}
            candidate = selected[names[2]]
            matched = next(x for x in records if x["route"] == names[1]
                           and x["cfg"] == candidate["cfg"] and x["splits"] == candidate["splits"])
            item = {"source": path.name, "kind": rows[0]["kind"], "r": rows[0]["r"],
                    "boundary": boundary,
                    "continuation": ("none" if "next_a8" not in boundary else
                                     {"down": "residual_add_then_original_a8", "gu": "silu_gate_times_up_then_original_a8", "head": "argmax"}[rows[0]["kind"]]),
                    "best": selected, "matched_four": matched,
                    "candidate_over_best_four_speedup": selected[names[1]]["cold_us"] / candidate["cold_us"],
                    "candidate_over_dp4a_speedup": selected[names[0]]["cold_us"] / candidate["cold_us"],
                    "same_plan_four_digit_bitexact": all(x["same_plan_four_digit_bitexact"] for x in records if x["route"] == names[2]),
                    "max_error": max(x["max_error"] for x in records),
                    "max_relative_rms": max(x["relative_rms"] for x in records)}
            summary[family].append(item)
            if "next_a8" in boundary:
                print(f'{family} {item["kind"]} R{item["r"]}: dp4a={selected[names[0]]["cold_us"]:.3f}, '
                      f'four_best={selected[names[1]]["cold_us"]:.3f}, three={candidate["cold_us"]:.3f}, '
                      f'cfg/split={candidate["cfg"]}/{candidate["splits"]}, matched_four={matched["cold_us"]:.3f}')
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
