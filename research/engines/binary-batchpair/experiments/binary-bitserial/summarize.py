#!/usr/bin/env python3
import json
from pathlib import Path
root = Path(__file__).resolve().parent
rows = []
for path in sorted((root / "results").glob("*.jsonl")):
    trials = [json.loads(line) for line in path.read_text().splitlines()]
    for route in dict.fromkeys(v["route"] for v in trials):
        best = min((v for v in trials if v["route"] == route), key=lambda v: v["cold_us"])
        rows.append(dict(receipt=path.name, **best))
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
for path in dict.fromkeys(v["receipt"] for v in rows):
    group = [v for v in rows if v["receipt"] == path]
    print(path, " ".join(f'{v["route"]}={v["cold_us"]:.3f}us' for v in group))
