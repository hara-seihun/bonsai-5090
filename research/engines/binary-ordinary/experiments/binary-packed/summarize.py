#!/usr/bin/env python3
import json
from pathlib import Path
root = Path(__file__).resolve().parent
rows = []
for shape in ["square", "up", "down", "head", "small"]:
    for r in range(1, 9):
        name = f"{shape}-r{r}" + ("-v2" if r == 1 else "") + ".jsonl"
        choices = [json.loads(line) for line in (root / "results" / name).read_text().splitlines()]
        best = min(choices, key=lambda v: v["cold_us"])
        rows.append(dict(shape=shape, receipt=name, **best))
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("| N × K | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for shape in ["square", "up", "down", "head", "small"]:
    values = [v for v in rows if v["shape"] == shape]
    print(f'| {values[0]["n"]} × {values[0]["k"]} | ' + " | ".join(f'{v["cold_us"]:.2f}' for v in values) + " |")
for field in ["max_error_f32", "relative_rms_f32", "max_error_f64_sample"]:
    print(field, max(v[field] for v in rows))
