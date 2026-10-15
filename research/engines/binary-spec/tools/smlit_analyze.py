#!/usr/bin/env python3
"""Analyze an H5_SMLIT replay dump (PREFIX-smlit.bin).

Usage: smlit_analyze.py DUMP SRC_DIR OUT_DIR [--sms 170]

Reconstructs launches from per-warp stamps, then reports per round:
  * step span, per-SM resident and post-wait-active fractions
  * time with fewer than all SMs doing post-wait work (the "dark silicon" budget)
  * per-launch spans, PDL wait, tail, and GEMM weight bandwidth
Launch reconstruction relies on single-stream PDL ordering: every CTA of launch k starts
before any CTA of launch k+2 of the same kernel. A CTA seen twice in one open launch starts a
new launch; incomplete launches are reported as errors, not silently merged.
"""
import argparse, json, re, struct, sys
from pathlib import Path
import numpy as np

REC = np.dtype([("tag", "<u4"), ("cta", "<u4"), ("grid", "<u4"), ("aux", "<u4"), ("warp", "<u2"), ("sm", "<u2"),
                ("threads", "<u4"), ("t0", "<u8"), ("t1", "<u8"), ("cw", "<u4"), ("cx", "<u4"), ("mk", "<u4", (4,))])
assert REC.itemsize == 64
MODULE = {"kernels.cu": 1, "attn.cu": 1, "prep_chunk.cuh": 1, "mega.cu": 1, "fp4.cu": 2, "draft.cu": 3, "graph_memory.cu": 4, "sk.cu": 7}


def tag_names(src, overrides=()):
    src = Path(src)
    candidates, seen = {}, set()
    site_file = src / "smlit.cuh"
    sites = {}
    if site_file.exists():
        text = site_file.read_text()
        enum = re.search(r"enum class SiteTag\s*:\s*uint32_t\s*\{([^}]+)\}", text)
        if enum:
            sites = {name: int(value, 0) for name, value in
                     re.findall(r"(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+)u?", enum[1])}
            if len(set(sites.values())) != len(sites):
                raise SystemExit("duplicate stable SM-lit site tags")

    def visit(path, mid):
        key = (path, mid)
        if key in seen:
            return
        seen.add(key)
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            include = re.match(r'\s*#include\s+"([^"]+)"', line)
            if include:
                child = path.parent / include[1]
                if child.exists():
                    visit(child, mid)
            if "define" in line or not re.search(r"\bH5_LIT(?:_SITE)?\(", line):
                continue
            block = "\n".join(lines[max(0, i - 4):i])
            kernel = re.findall(r"(k_\w+|graph_\w+|finish_\w+)\s*\(", block)
            site = re.search(r"H5_LIT_SITE\(\s*(\w+)\s*,", line)
            if site and site[1] not in sites:
                raise SystemExit(f"unknown stable SM-lit site {site[1]} in {path}:{i}")
            tag = sites[site[1]] if site else mid * 100000 + i
            name = kernel[-1] if kernel else f"{path.name}:{i}"
            candidates.setdefault(tag, set()).add(name)

    for filename, mid in MODULE.items():
        path = src / filename
        if path.exists():
            visit(path, mid)
    names = {tag: "|".join(sorted(values)) for tag, values in candidates.items()}
    for item in overrides:
        match = re.fullmatch(r"(0x[0-9a-fA-F]+|\d+)=(\w+)", item)
        if not match:
            raise SystemExit(f"invalid --tag-name {item!r}: expected TAG=KERNEL")
        tag, name = int(match[1], 0), match[2]
        if tag not in candidates or name not in candidates[tag]:
            raise SystemExit(f"--tag-name {item!r} is not a source tag candidate")
        if tag in names and "|" not in names[tag] and names[tag] != name:
            raise SystemExit(f"conflicting --tag-name for {tag}")
        names[tag] = name
    return names


def validate_tags(recs, names):
    used = set(map(int, recs["tag"]))
    unknown = sorted(used - names.keys())
    if unknown:
        raise SystemExit(f"unknown SMlit tags {unknown}: source must match the instrumented build")
    ambiguous = {tag: names[tag] for tag in sorted(used) if "|" in names[tag]}
    if ambiguous:
        raise SystemExit(f"ambiguous SM-lit tags {ambiguous}: use --tag-name TAG=KERNEL from the captured run configuration")


def load(path):
    b = Path(path).read_bytes()
    magic, first, nb = struct.unpack_from("<3Q", b, 0)
    if magic != 0x544C4D53:
        raise SystemExit("bad magic")
    bounds = np.frombuffer(b, "<u8", nb, 24)
    recs = np.frombuffer(b, REC, offset=24 + 8 * nb)
    if len(recs) != bounds[-1]:
        raise SystemExit(f"record count {len(recs)} != bound {bounds[-1]}")
    return int(first), bounds, recs


def launches(r):
    # CTA-level reduction
    key = np.stack([r["tag"].astype(np.int64), r["grid"].astype(np.int64), r["cta"].astype(np.int64)], 1)
    order = np.lexsort((r["t0"], r["cta"], r["grid"], r["tag"]))
    # group warps of the same CTA instance: same (tag,grid,cta) and entries within the CTA lifetime.
    rs = r[order]
    ctas = []  # (entry, exit, tag, grid, cta, sm, threads, aux, wait_end_max, warps)
    i = 0
    n = len(rs)
    while i < n:
        j = i + 1
        while j < n and rs["tag"][j] == rs["tag"][i] and rs["grid"][j] == rs["grid"][i] and rs["cta"][j] == rs["cta"][i] \
                and rs["sm"][j] == rs["sm"][i] and rs["t0"][j] <= rs["t1"][i:j].max():
            j += 1
        g = rs[i:j]
        ctas.append(g)
        i = j
    ctas.sort(key=lambda g: int(g["t0"].min()))
    out, open_ = [], {}
    for g in ctas:
        k = (int(g["tag"][0]), int(g["grid"][0]))
        L = open_.get(k)
        c = int(g["cta"][0])
        if L is None or c in L["seen"] or len(L["seen"]) == k[1]:
            L = {"tag": k[0], "grid": k[1], "seen": set(), "ctas": []}
            open_[k] = L
            out.append(L)
        L["seen"].add(c)
        L["ctas"].append(g)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("source")
    parser.add_argument("outdir")
    parser.add_argument("--sms", type=int, default=170)
    parser.add_argument("--tag-name", action="append", default=[], metavar="TAG=KERNEL")
    args = parser.parse_args()
    dump, src, outdir, sms = args.dump, args.source, args.outdir, args.sms
    names = tag_names(src, args.tag_name)
    first, bounds, recs = load(dump)
    validate_tags(recs, names)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # ns per cycle from long records
    m = recs["cx"] > 20000
    ns_per_cyc = float(np.median((recs["t1"][m] - recs["t0"][m]) / recs["cx"][m]))
    report = {"first_round": first, "rounds": [], "ns_per_cycle": ns_per_cyc, "clock_ghz": 1 / ns_per_cyc}
    lo = 0
    allrows = []
    for ri, hi in enumerate(bounds):
        r = recs[lo:int(hi)]; lo = int(hi)
        T0 = int(r["t0"].min()); T1 = int(r["t1"].max())
        L = launches(r)
        incomplete = [(names.get(x["tag"]), x["grid"], len(x["seen"])) for x in L if len(x["seen"]) != x["grid"]]
        rows = []
        # per-SM busy intervals
        res = [[] for _ in range(sms)]; act = [[] for _ in range(sms)]
        for idx, x in enumerate(L):
            e = min(int(g["t0"].min()) - T0 for g in x["ctas"]); f = max(int(g["t1"].max()) - T0 for g in x["ctas"])
            wends = []
            for g in x["ctas"]:
                for w in g:
                    # Subtract the integer origin before float clock conversion: absolute
                    # globaltimer values near 1.8e18 ns have a 256 ns float ULP.
                    entry = int(w["t0"]) - T0; exit_ = int(w["t1"]) - T0
                    we = entry + int(w["cw"]) * ns_per_cyc
                    wends.append(we)
                    res[w["sm"]].append((entry, exit_))
                    act[w["sm"]].append((we, exit_))
            g0 = x["ctas"][0]
            name = names.get(x["tag"], str(x["tag"]))
            aux = int(g0["aux"][0])
            wmax = max(wends); wmin = min(wends)
            ctas_first_exit = min(int(g["t1"].max()) - T0 for g in x["ctas"])
            row = {"i": idx, "name": name, "tag": x["tag"], "grid": x["grid"], "threads": int(g0["threads"][0]),
                   "aux_blocks": aux, "entry_us": e / 1e3, "exit_us": f / 1e3, "span_us": (f - e) / 1e3,
                   "wait_release_first_us": wmin / 1e3, "wait_release_last_us": wmax / 1e3,
                   "post_release_us": (f - wmin) / 1e3, "tail_us": (f - ctas_first_exit) / 1e3,
                   "sms_used": len(set(int(g["sm"][0]) for g in x["ctas"]))}
            if aux:
                row["weight_bytes_dense5"] = aux * 26
                row["GBps_post_release"] = aux * 26 / max(f - wmin, 1)
            rows.append(row)

        def union(iv):
            if not iv: return 0
            iv.sort(); tot = 0; s, t = iv[0]
            for a_, b_ in iv[1:]:
                if a_ > t: tot += t - s; s, t = a_, b_
                else: t = max(t, b_)
            return tot + t - s
        span = T1 - T0
        res_f = [union(v) / span for v in res]; act_f = [union(v) / span for v in act]
        # SM-count-active histogram over time (post-wait work)
        ev = []
        for smi, v in enumerate(act):
            v = sorted(v); merged = []
            for a_, b_ in v:
                if merged and a_ <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], b_)
                else: merged.append([a_, b_])
            for a_, b_ in merged: ev += [(a_, 1), (b_, -1)]
        ev.sort(); cur = 0; last = 0; hist = np.zeros(sms + 1)
        for t, d in ev:
            hist[cur] += t - last; last = t; cur += d
        hist[cur] += span - last
        full = hist[sms] / span
        dark_sm_time = float(sum((sms - k) * hist[k] for k in range(sms + 1)) / (sms * span))
        rr = {"round": first + ri, "span_us": span / 1e3, "launches": len(L), "incomplete_launches": incomplete[:20],
              "sm_resident_mean": float(np.mean(res_f)), "sm_postwait_active_mean": float(np.mean(act_f)),
              "time_all_sms_active_frac": float(full), "dark_sm_time_frac": dark_sm_time,
              "time_zero_sms_active_us": float(hist[0] / 1e3)}
        report["rounds"].append(rr)
        for row in rows: row["round"] = first + ri
        allrows += rows
    (outdir / "summary.json").write_text(json.dumps(report, indent=1))
    keys = sorted({k for row in allrows for k in row})
    with open(outdir / "launches.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for row in allrows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    # per-name aggregate for the first round
    agg = {}
    for row in allrows:
        if row["round"] != first: continue
        a_ = agg.setdefault(row["name"] + f"/g{row['grid']}x{row['threads']}", [0, 0.0, 0.0, 0.0, 0])
        a_[0] += 1; a_[1] += row["span_us"]; a_[2] += row["post_release_us"]; a_[3] += row["tail_us"]; a_[4] += row.get("weight_bytes_dense5", 0)
    print(json.dumps(report["rounds"], indent=1))
    print(f"clock {report['clock_ghz']:.3f} GHz")
    print(f"{'kernel/grid x threads':48s} {'n':>4s} {'span':>9s} {'postrel':>9s} {'tail':>8s} {'MB':>8s} {'GB/s':>7s}")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        bw = v[4] / (v[2] * 1e3) if v[4] else 0
        print(f"{k:48s} {v[0]:4d} {v[1]:9.1f} {v[2]:9.1f} {v[3]:8.1f} {v[4]/1e6:8.1f} {bw:7.0f}")


if __name__ == "__main__":
    main()
