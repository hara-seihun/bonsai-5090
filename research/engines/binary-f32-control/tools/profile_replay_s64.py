#!/usr/bin/env python3
"""Replicate the recorded ordinary workload and select its S64 decode timings."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ORIGIN_SHA256 = "e40c1ba21c3f3d93548e69d165b2cf7508f1579fe28adf27b1f96daf6663e088"
SLOTS = 64
PREFILL_ROWS = 48
DECODE_ROUNDS = 127
ROWCAP = 520


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def recorded_steps(data):
    if sha256(data) != ORIGIN_SHA256:
        raise ValueError("source SHA256 differs from the recorded ordinary workload")
    events = [line.split() for line in data.decode("ascii").splitlines()]
    if len(events) != 1 + 2 * (1 + DECODE_ROUNDS) or events[0] != ["RESET", "0"]:
        raise ValueError("recording must be RESET 0 followed by 128 KEEP/STEP pairs")
    steps = []
    for i in range(1 + DECODE_ROUNDS):
        if events[1 + 2 * i] != ["KEEP", "0", "0"]:
            raise ValueError("recording contains non-ordinary state transitions")
        step = events[2 + 2 * i]
        rows = PREFILL_ROWS if i == 0 else 1
        if not step or step[0] != "STEP" or len(step) != 2 + 3 * rows:
            raise ValueError("recording has an unexpected STEP shape")
        values = [int(v) for v in step[1:]]
        if values[0] != rows:
            raise ValueError("recording has an unexpected row count")
        triples = [values[j:j + 3] for j in range(1, len(values), 3)]
        positions = list(range(PREFILL_ROWS)) if i == 0 else [PREFILL_ROWS + i - 1]
        if [r[2] for r in triples] != positions or any(r[0] != 0 or r[1] < 0 for r in triples):
            raise ValueError("recording has unexpected slots, tokens or positions")
        steps.append(triples)
    return steps


def step_line(rows):
    return "STEP " + str(len(rows)) + " " + " ".join(str(v) for row in rows for v in row)


def write_new(path, data):
    with path.open("xb") as f:
        f.write(data)


def generate(source, output, manifest_path):
    paths = [source.resolve(), output.resolve(), manifest_path.resolve()]
    if len(set(paths)) != 3 or output.exists() or manifest_path.exists():
        raise ValueError("source, output and manifest must be distinct; outputs must not exist")
    data = source.read_bytes()
    steps = recorded_steps(data)
    events = [f"RESET {slot}" for slot in range(SLOTS)]
    for slot in range(SLOTS):
        events.append(f"KEEP {slot} 0")
        events.append(step_line([[slot, token, pos] for _, token, pos in steps[0]]))
    for original in steps[1:]:
        _, token, pos = original[0]
        events.extend(f"KEEP {slot} 0" for slot in range(SLOTS))
        events.append(step_line([[slot, token, pos] for slot in range(SLOTS)]))
    result = ("\n".join(events) + "\n").encode("ascii")
    manifest = {
        "schema": "h5-profile-replay-s64-v1",
        "origin": {"path": str(source.resolve()), "sha256": sha256(data), "bytes": len(data),
                   "events": 257, "prefill_rows": PREFILL_ROWS, "decode_rounds": DECODE_ROUNDS},
        "fixture": {"path": str(output.resolve()), "sha256": sha256(result), "bytes": len(result),
                    "events": len(events), "slots": SLOTS, "rowcap": ROWCAP, "min_ctx": 175,
                    "prefill_steps": SLOTS, "step_count": SLOTS + DECODE_ROUNDS,
                    "prefill_tokens": SLOTS * PREFILL_ROWS, "decode_tokens": SLOTS * DECODE_ROUNDS},
        "measurements": {"rows_filter": SLOTS, "prefill_rows": PREFILL_ROWS,
                         "decode_rounds_per_pass": DECODE_ROUNDS,
                         "first_decode_round": SLOTS, "last_decode_round": SLOTS + DECODE_ROUNDS - 1,
                         "csv_columns": ["round", "rows", "seconds", "output_tokens..."]},
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    write_new(output, result)
    try:
        write_new(manifest_path, encoded_manifest)
    except OSError:
        output.unlink()
        raise
    return manifest


def summarize(csv_path, manifest_path, arms, decode_csv, output):
    if not arms or any(arm not in "AB" for arm in arms):
        raise ValueError("arms must explicitly identify each pass, e.g. ABBA")
    paths = [csv_path.resolve(), manifest_path.resolve(), decode_csv.resolve(), output.resolve()]
    if len(set(paths)) != 4 or decode_csv.exists() or output.exists():
        raise ValueError("measurement inputs/outputs must be distinct; outputs must not exist")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "h5-profile-replay-s64-v1" or manifest["origin"]["sha256"] != ORIGIN_SHA256:
        raise ValueError("manifest does not name the recorded ordinary S64 workload")
    fixture = manifest["fixture"]
    if fixture["slots"] != SLOTS or fixture["step_count"] != SLOTS + DECODE_ROUNDS:
        raise ValueError("manifest has unexpected fixture dimensions")
    if sha256(Path(fixture["path"]).read_bytes()) != fixture["sha256"]:
        raise ValueError("fixture SHA256 differs from the manifest")
    data = csv_path.read_bytes()
    rows = list(csv.reader(io.StringIO(data.decode("ascii"))))
    step_count = SLOTS + DECODE_ROUNDS
    if len(rows) != len(arms) * step_count:
        raise ValueError("CSV must contain every prefill and all 127 decode rounds for each pass")
    passes = []
    selected = []
    for index, arm in enumerate(arms):
        seconds = 0.0
        for round_index, row in enumerate(rows[index * step_count:(index + 1) * step_count]):
            expected_rows = PREFILL_ROWS if round_index < SLOTS else SLOTS
            if len(row) != 3 + expected_rows or int(row[0]) != round_index or int(row[1]) != expected_rows:
                raise ValueError(f"invalid CSV shape/order at pass {index} round {round_index}")
            elapsed = float(row[2])
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("CSV seconds must be finite and positive")
            if expected_rows == SLOTS:
                selected.append(row)
                seconds += elapsed
        passes.append({"pass": index, "arm": arm, "decode_rounds": DECODE_ROUNDS,
                       "decode_tokens": SLOTS * DECODE_ROUNDS, "decode_seconds": seconds,
                       "aggregate_tokens_per_second": SLOTS * DECODE_ROUNDS / seconds,
                       "per_slot_tokens_per_second": DECODE_ROUNDS / seconds})
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(selected)
    report = {"schema": "h5-profile-replay-s64-measurements-v1", "csv_sha256": sha256(data),
              "fixture_sha256": fixture["sha256"], "origin_sha256": ORIGIN_SHA256,
              "rows_filter": SLOTS, "arms": arms, "passes": passes}
    write_new(decode_csv, stream.getvalue().encode("ascii"))
    try:
        write_new(output, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    except OSError:
        decode_csv.unlink()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("source", type=Path)
    gen.add_argument("output", type=Path)
    gen.add_argument("manifest", type=Path)
    summary = commands.add_parser("summarize")
    summary.add_argument("csv", type=Path)
    summary.add_argument("manifest", type=Path)
    summary.add_argument("--arms", required=True)
    summary.add_argument("--decode-csv", required=True, type=Path)
    summary.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            result = generate(args.source, args.output, args.manifest)
        else:
            result = summarize(args.csv, args.manifest, args.arms, args.decode_csv, args.output)
    except (OSError, ValueError, KeyError, UnicodeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
