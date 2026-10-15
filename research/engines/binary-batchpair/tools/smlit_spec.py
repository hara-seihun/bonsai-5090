#!/usr/bin/env python3
"""smlit_spec.py DUMP SRC_DIR --draft-weights CSV: verifier/drafter split of a spec-replay SM-lit dump.

spec-replay writes two bounds per round (after the verifier step, after the drafter), so segments
alternate verify, draft. Per round: verifier span, host gap verifier -> drafter (last kernel exit to
first kernel entry), drafter span, host gap drafter -> next verifier. Then the drafter's kernels
aggregated over all recorded rounds. Drafter traffic requires captured matrix metadata;
SM-lit aux stores only N*K/128, not N, K or format. CSV columns: N,K,bytes_per_128
(dense5=26, tern=34, b2=36, b3=52, nvfp4=72). Includes FP32 output-row scales except dense5.
"""
import argparse, collections, csv, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from smlit_analyze import load, launches, tag_names, validate_tags

DRAFT_FORMATS = {26: "dense5", 34: "tern", 36: "b2", 52: "b3", 72: "nvfp4"}


@dataclass(frozen=True)
class DraftWeight:
    N: int
    K: int
    bytes_per_128: int

    @property
    def format(self):
        return DRAFT_FORMATS[self.bytes_per_128]

    @property
    def bytes(self):
        return self.N * self.K // 128 * self.bytes_per_128 + (0 if self.format == "dense5" else 4 * self.N)


def load_draft_weights(path):
    weights = {}
    try:
        with Path(path).open(newline="") as file:
            reader = csv.DictReader(file)
            if not {"N", "K", "bytes_per_128"}.issubset(reader.fieldnames or []):
                return "draft weight CSV requires N,K,bytes_per_128 columns"
            for line, row in enumerate(reader, 2):
                weight = DraftWeight(*(int(row[key]) for key in ("N", "K", "bytes_per_128")))
                if weight.N <= 0 or weight.K <= 0 or weight.K % 128 or weight.N % 32:
                    return f"invalid drafter packed matrix shape on CSV line {line}"
                if weight.bytes_per_128 not in DRAFT_FORMATS:
                    return f"unknown drafter bytes_per_128 on CSV line {line}: {weight.bytes_per_128}"
                blocks = weight.N * weight.K // 128
                if blocks in weights and weights[blocks] != weight:
                    return f"ambiguous drafter metadata for aux={blocks}: SM-lit cannot distinguish these shapes/formats"
                weights[blocks] = weight
    except (OSError, ValueError, TypeError, csv.Error) as error:
        return f"invalid draft weight CSV: {error}"
    if not weights:
        return "draft weight CSV contains no matrices"
    return weights


def seg_stats(r, names, ns_per_cyc, origin):
    L = launches(r)
    rows = []
    for x in L:
        e = min(int(g["t0"].min()) - origin for g in x["ctas"]); f = max(int(g["t1"].max()) - origin for g in x["ctas"])
        wends = [int(w["t0"]) - origin + int(w["cw"]) * ns_per_cyc for g in x["ctas"] for w in g]
        aux = int(x["ctas"][0]["aux"][0])
        rows.append({"name": names.get(x["tag"], str(x["tag"])), "grid": x["grid"], "entry": e, "exit": f,
                     "release": min(wends), "aux": aux, "complete": len(x["seen"]) == x["grid"]})
    rows.sort(key=lambda z: z["entry"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("source")
    parser.add_argument("--draft-weights", required=True, metavar="CSV")
    parser.add_argument("--tag-name", action="append", default=[], metavar="TAG=KERNEL")
    args = parser.parse_args()
    weights = load_draft_weights(args.draft_weights)
    if isinstance(weights, str):
        parser.error(weights)
    names = tag_names(args.source, args.tag_name)
    first, bounds, recs = load(args.dump)
    validate_tags(recs, names)
    m = recs["cx"] > 20000
    ns_per_cyc = float(np.median((recs["t1"][m] - recs["t0"][m]) / recs["cx"][m]))
    if len(bounds) % 2:
        raise SystemExit("odd segment count: not a spec-replay dump")
    segs, lo = [], 0
    origin = int(recs["t0"].min())
    for hi in bounds:
        segs.append(seg_stats(recs[lo:int(hi)], names, ns_per_cyc, origin)); lo = int(hi)
    used_blocks = {z["aux"] for d in segs[1::2] for z in d if z["aux"]}
    missing = sorted(used_blocks - weights.keys())
    if missing:
        parser.error(f"missing drafter metadata for aux blocks {missing}; no format can be inferred from the dump")
    bad = sum(not z["complete"] for s in segs for z in s)
    if bad:
        print(f"WARNING {bad} incomplete launches")
    print(f"clock {1 / ns_per_cyc:.3f} GHz, rounds {first}..{first + len(segs) // 2 - 1}")
    print(f"{'round':>5s} {'verify':>8s} {'v->d gap':>8s} {'draft':>8s} {'d->v gap':>8s} {'v launch':>8s} {'d launch':>8s}  (us)")
    tot = collections.Counter()
    nr = len(segs) // 2
    for i in range(nr):
        v, d = segs[2 * i], segs[2 * i + 1]
        vs = (v[-1]["exit"] - v[0]["entry"]) / 1e3; ds = (d[-1]["exit"] - d[0]["entry"]) / 1e3
        g1 = (d[0]["entry"] - v[-1]["exit"]) / 1e3
        g2 = (segs[2 * i + 2][0]["entry"] - d[-1]["exit"]) / 1e3 if 2 * i + 2 < len(segs) else float("nan")
        print(f"{first + i:5d} {vs:8.1f} {g1:8.1f} {ds:8.1f} {g2:8.1f} {len(v):8d} {len(d):8d}")
        tot["verify"] += vs; tot["vd"] += g1; tot["draft"] += ds
        if g2 == g2: tot["dv"] += g2; tot["ndv"] += 1
    print(f"mean  {tot['verify'] / nr:8.1f} {tot['vd'] / nr:8.1f} {tot['draft'] / nr:8.1f} {tot['dv'] / max(tot['ndv'], 1):8.1f}")
    # drafter kernel aggregate (internal gaps = release of k minus exit of k-1)
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    for i in range(nr):
        d = segs[2 * i + 1]
        for j, z in enumerate(d):
            weight = weights[z["aux"]] if z["aux"] else None
            key = f"{z['name']}/g{z['grid']}" + (f"/{weight.format}/{weight.N}x{weight.K}" if weight else "")
            A = agg[key]
            A[0] += 1; A[1] += (z["exit"] - z["entry"]) / 1e3; A[2] += (z["exit"] - z["release"]) / 1e3
            A[3] += weight.bytes if weight else 0
            if j: A[4] += max(0.0, (z["release"] - d[j - 1]["exit"]) / 1e3)
    # verifier kernel aggregate, keyed by kernel, grid and weight blocks (aux); dense5 26 B per block,
    # the 96-row alpha/beta NVFP4 projection 72 B
    vagg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    for i in range(nr):
        v = segs[2 * i]
        for j, z in enumerate(v):
            key = f"{z['name']}/g{z['grid']}/b{z['aux']}"
            A = vagg[key]
            A[0] += 1; A[1] += (z["exit"] - z["entry"]) / 1e3; A[2] += (z["exit"] - z["release"]) / 1e3
            A[3] += z["aux"] * (72 if z["aux"] == 96 * 5120 // 128 else 26)
            if j: A[4] += max(0.0, (z["release"] - v[j - 1]["exit"]) / 1e3)
    print(f"\nverifier kernels, per round: {'n':>5s} {'span':>8s} {'postrel':>8s} {'gap-in':>8s} {'MB':>8s} {'GB/s':>6s}")
    for k, A in sorted(vagg.items(), key=lambda kv: -kv[1][2]):
        bw = A[3] / (A[2] * 1e3) if A[3] else 0
        print(f"{k:40s} {A[0] / nr:5.1f} {A[1] / nr:8.1f} {A[2] / nr:8.1f} {A[4] / nr:8.1f} {A[3] / nr / 1e6:8.1f} {bw:6.0f}")
    print(f"\ndrafter weight bytes per round: {sum(A[3] for A in agg.values()) / nr:,.0f} (including row scales)")
    print(f"drafter kernels, per round: {'n':>5s} {'span':>8s} {'postrel':>8s} {'gap-in':>8s} {'MB':>8s} {'GB/s':>6s}")
    for k, A in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        bw = A[3] / (A[2] * 1e3) if A[3] else 0
        print(f"{k:34s} {A[0] / nr:5.1f} {A[1] / nr:8.1f} {A[2] / nr:8.1f} {A[4] / nr:8.1f} {A[3] / nr / 1e6:8.1f} {bw:6.0f}")


if __name__ == "__main__":
    main()
