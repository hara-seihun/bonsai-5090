#!/usr/bin/env python3
"""CPU-only contract probe for the native fixed-eight-row flag comparison."""
import argparse
import json
from pathlib import Path


# Config indices used by launch_scale_carrier_tail's TAIL_SIDE admission.
SIDE = {19: (4, 2, 0), 21: (8, 1, 0), 23: (4, 2, 4), 26: (8, 1, 4)}
FAMILIES = {
    "attn_in": (14336, 5120, 8, 0, 3),
    "gdn_in": (16384, 5120, 8, 0, 3),
    "out": (5120, 6144, 8, 1, 3),
    "gu": (34816, 5120, 8, 0, 100),
    "down": (5120, 17408, 8, 1, 3),
}


def probe(plan_path, event_path, overrides):
    plan = {}
    for line in Path(plan_path).read_text().splitlines():
        if not line.strip():
            continue
        values = tuple(map(int, line.split()))
        if len(values) != 7 or values[:5] in plan:
            raise ValueError("plan requires unique seven-integer rows")
        plan[values[:5]] = values[5:]
    for entry in overrides:
        shape, selection = entry.split("=")
        n, k, add, wf = map(int, shape.split(":"))
        ci, split = map(int, selection.split(":"))
        plan[(n, k, 8, add, wf)] = (ci, split)
    steps = []
    first_event = True
    for line in Path(event_path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        op, values = fields[0], tuple(map(int, fields[1:]))
        if first_event and (op != "RESET" or values != (0,)):
            raise ValueError("native replay must begin with RESET 0")
        first_event = False
        if op == "RESET":
            if values != (0,):
                raise ValueError("fixed-eight workload requires RESET 0")
        elif op == "KEEP" or op == "ACCEPT":
            if len(values) != 2 or values[0] != 0 or not 0 <= values[1] <= 8:
                raise ValueError("invalid native recurrent-state event")
        elif op == "STEP":
            if not values or not 1 <= values[0] <= 520 or len(values) != 1 + 3 * values[0]:
                raise ValueError("malformed STEP")
            rows = [values[1 + 3 * i:4 + 3 * i] for i in range(values[0])]
            if any(slot != 0 or not 0 <= pos < 4096 or not 0 <= token < 248320 for slot, token, pos in rows):
                raise ValueError("fixed-eight workload requires one valid sequence slot")
            if any(rows[i][2] != rows[0][2] + i for i in range(len(rows))):
                raise ValueError("native rows must be consecutive within each step")
            steps.append(values[0])
        else:
            raise ValueError("native verifier recipe accepts STEP, not ordinary or HEADSTEP work")
    if len(steps) < 2 or steps[0] <= 8 or any(rows != 8 for rows in steps[1:]):
        raise ValueError("expected one prefill followed only by native STEP 8 events")
    report = {"prefill_rows": steps[0], "fixed8_steps": len(steps) - 1, "families": {}}
    for family, key in FAMILIES.items():
        ci, split = plan[key]
        if ci not in SIDE or not 1 <= split <= key[1] // 128:
            raise ValueError(f"{family}: not a flag-supported geometry/split")
        fw, wg, pipe = SIDE[ci]
        if (family == "gu") != (wg == 2) or (family == "gu" and split != 1):
            raise ValueError(f"{family}: not a flag-supported mode")
        # Projections use TAIL_SIDE mode 1/2. Non-add, nonsplit stores would need mode 0.
        if family.endswith("_in") and split == 1:
            raise ValueError(f"{family}: flag projection requires split-K or add mode")
        stage = fw * wg * (832 + 64) + 8 * 72
        stages = pipe or min(8, 96 * 1024 // stage)
        nkb = key[1] // 128
        min_split_kb = nkb // split
        max_split_kb = (nkb + split - 1) // split
        initial = [min(5, stages, kb) * fw * wg * 832 for kb in (min_split_kb, max_split_kb)]
        report["families"][family] = {
            "cfg": ci, "splits": split, "ctas": (key[0] // 32 + fw * wg - 1) // (fw * wg) * split,
            "stages": stages, "prewait0_weight_bytes_per_cta": 0,
            "prewait5_weight_bytes_per_cta_range": initial,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("events")
    parser.add_argument("--override", action="append", required=True, help="N:K:ADD:WF=CFG:SPLIT; use none for an explicitly empty set")
    args = parser.parse_args()
    overrides = [] if args.override == ["none"] else args.override
    try:
        print(json.dumps(probe(args.plan, args.events, overrides), indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"flag contract: {error}\n")


if __name__ == "__main__":
    main()
