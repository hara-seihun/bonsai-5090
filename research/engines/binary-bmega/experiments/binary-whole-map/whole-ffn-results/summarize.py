import json
from pathlib import Path

root = Path(__file__).parent
summary = {"scope": "real layer0 Q1 gate/up/down; deterministic inputs; original K128 A8; complete down", "units": "us", "row_results": {}}
for rows in (16, 32):
    records = [json.loads(line) for line in (root / f"r{rows}.jsonl").read_text().splitlines()]
    row = {"preparation": records[0], "witnesses": [r for r in records if r["kind"] in ("witness", "complete_wrapper_witness", "contracts")], "boundaries": {}}
    for boundary in ("given_a8_encode", "f32_producer_encode"):
        complete = [r for r in records if r.get("boundary") == boundary and r["complete_down"]]
        grouped = {route: [r for r in complete if r["route"] == route] for route in ("materialized_exact3", "compound_exact3")}
        best = {route: {cache: min(items, key=lambda r: r[f"{cache}_us"]) for cache in ("warm", "cold")} for route, items in grouped.items()}
        current = next(r for r in grouped["materialized_exact3"] if (r["gu_cfg"], r["gu_splits"], r["down_cfg"], r["down_splits"]) == (1, 4, 1, 8))
        best["current_split4_control"] = current
        best["improvement_percent"] = {cache: 100 * (1 - best["compound_exact3"][cache][f"{cache}_us"] / best["materialized_exact3"][cache][f"{cache}_us"]) for cache in ("warm", "cold")}
        row["boundaries"][boundary] = best
    row["removed_dynamic_global_bytes"] = rows * 17408 * (8 + 4 + 1)
    summary["row_results"][str(rows)] = row
summary["tail_witnesses"] = {}
for rows in (5, 17, 31, 65):
    records = [json.loads(line) for line in (root / f"r{rows}.jsonl").read_text().splitlines()]
    summary["tail_witnesses"][str(rows)] = [r for r in records if r["kind"] in ("witness", "complete_wrapper_witness", "contracts")]
summary["retained_fallback_extra_static_bytes_64_layers"] = 25067520 * 64
summary["complete_control_windows"] = {}
for rows in (5, 16, 32):
    records = [json.loads(line) for line in (root / f"controls-r{rows}.jsonl").read_text().splitlines()]
    windows = {}
    for boundary in ("given_a8_encode", "f32_producer_encode"):
        routes = {}
        for route in sorted({r["route"] for r in records}):
            items = [r for r in records if r["route"] == route and r["boundary"] == boundary]
            routes[route] = {cache: min(items, key=lambda r: r[f"{cache}_us"]) for cache in ("warm", "cold")}
        windows[boundary] = routes
    summary["complete_control_windows"][str(rows)] = windows
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for rows, row in summary["row_results"].items():
    for boundary, b in row["boundaries"].items():
        print(rows, boundary, "control", *(b["materialized_exact3"][cache][f"{cache}_us"] for cache in ("warm", "cold")), "compound", *(b["compound_exact3"][cache][f"{cache}_us"] for cache in ("warm", "cold")), "improvement", b["improvement_percent"])
