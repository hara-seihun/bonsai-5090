#!/usr/bin/env python3
"""Summarise benchmark runs: bench_table.py RAW_DIR > table.md

Reads ENGINE-BENCH-MODE.bin.score.json and .bin.timing.json written by tools/evals.py and the pod run scripts
(ENGINE is h5 for halo-5090 with DFlash2 speculation, h5gsm for halo-5090 without it, fork for the PrismML fork).
"""
import glob, json, math, os, sys

d = sys.argv[1]
ENG = {"h5": "halo-5090", "h5gsm": "halo-5090 (no spec)", "fork": "PrismML fork"}
NAMES = {"mmlupro": "MMLU-Pro (12,032)", "mmlupro2k": "MMLU-Pro (stratified 2k)", "gpqa": "GPQA Diamond", "ifeval": "IFEval",
         "math500": "MATH-500", "humaneval": "HumanEval+", "mbpp": "MBPP+", "lcb": "LiveCodeBench v6 (2025-01..04)"}
runs = {}
for f in sorted(glob.glob(f"{d}/*.bin.score.json")):
    tag = os.path.basename(f)[: -len(".bin.score.json")]
    eng, rest = tag.split("-", 1)
    bench, mode = rest.rsplit("-", 1)
    s = json.load(open(f))
    t = json.load(open(f[: -len(".score.json")] + ".timing.json")) if os.path.exists(f[: -len(".score.json")] + ".timing.json") else {}
    runs[(bench, mode, eng)] = (s, t)


def cell(s):
    return f"{s['score']:.1f} ± {s['se']:.1f}"


def paired(a, b):
    pa, pb = a["per_item"], b["per_item"]
    if len(pa) != len(pb):
        return ""
    dl = [x - y for x, y in zip(pa, pb)]
    n = len(dl); mu = sum(dl) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in dl) / (n - 1)) if n > 1 else 0
    return f"{100 * mu:+.1f} ± {100 * sd / math.sqrt(n):.1f}"


print("| Benchmark | Thinking | n | halo-5090 | PrismML fork | halo − fork (paired) | halo tok/s (tok/verify) | fork tok/s | trunc h5/fork |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for mode in ("off", "on"):
    for bench in NAMES:
        h = runs.get((bench, mode, "h5")); f = runs.get((bench, mode, "fork"))
        if not h and not f:
            continue
        n = (h or f)[0]["n"]
        row = [NAMES[bench], mode, str(n), cell(h[0]) if h else "—", cell(f[0]) if f else "—",
               paired(h[0], f[0]) if h and f else "—",
               f"{h[1].get('tok_s', 0):.0f} ({h[1].get('tokens_per_verify') or 0:.2f})" if h and h[1] else "—",
               f"{f[1].get('tok_s', 0):.0f}" if f and f[1] else "—",
               f"{h[0].get('truncated', '—') if h else '—'}/{f[0].get('truncated', '—') if f else '—'}"]
        print("| " + " | ".join(row) + " |")
print()
for (bench, mode, eng), (s, t) in sorted(runs.items()):
    extra = {k: v for k, v in s.items() if k in ("prompt_strict", "prompt_loose", "inst_strict", "inst_loose", "base", "plus", "by_difficulty", "no_answer", "gen_tokens_mean")}
    if extra:
        print(f"- {ENG.get(eng, eng)} {bench} thinking-{mode}: " + "; ".join(
            f"{k} {v['score']:.1f} ± {v['se']:.1f}" if isinstance(v, dict) and "score" in v else f"{k} {v if not isinstance(v, float) else round(v)}" for k, v in extra.items()))
