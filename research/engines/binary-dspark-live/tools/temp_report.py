#!/usr/bin/env python3
"""Aggregate the temperature sweep: think-T{T}-s{S}/ (think_eval) and gate-T{T}-s{S}/ (quality gate) run dirs plus
speed/speed.json (temp_speed.py) under ROOT into one table (markdown) and JSON.

  temp_report.py ROOT [--json OUT]

Accuracy per temperature is the mean over seeds; spread is the min-max range and the sample sd across seeds.
Pooled counts (truncations, loops, circling) are summed over seeds and shown with the item-run total.
"""
import argparse, json, re, statistics
from pathlib import Path


def runs(root, kind):
    out = {}
    for d in sorted(Path(root).glob(f"{kind}-T*-s*")):
        m = re.fullmatch(rf"{kind}-T([0-9.]+)-s(\d+)", d.name)
        if m and (d / "summary.json").exists():
            out.setdefault(float(m.group(1)), {})[int(m.group(2))] = json.loads((d / "summary.json").read_text())
    return out


def spread(xs):
    if not xs:
        return None
    return {"mean": statistics.mean(xs), "min": min(xs), "max": max(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}


def fmt(s, scale=100):
    if not s:
        return "–"
    if s["n"] == 1:
        return f"{scale * s['mean']:.1f}"
    return f"{scale * s['mean']:.1f} [{scale * s['min']:.1f}–{scale * s['max']:.1f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--json")
    a = ap.parse_args()
    think, gate = runs(a.root, "think"), runs(a.root, "gate")
    sp = Path(a.root) / "speed" / "speed.json"
    speed = json.loads(sp.read_text())["summary"] if sp.exists() else {}
    temps = sorted(set(think) | set(gate) | {float(k) for k in speed if k != "instruct"})
    res = {}
    for t in temps:
        r = {}
        sk = next((k for k in speed if k != "instruct" and float(k) == t), None)
        if sk:
            r["speed"] = speed[sk]
        for b in ("aime", "math500", "gsm8k"):
            per = [s[b] for s in think.get(t, {}).values() if b in s]
            if not per:
                continue
            r[b] = {"accuracy": spread([x["accuracy"] for x in per]),
                    "seeds": len(per), "item_runs": sum(x["n"] for x in per),
                    "truncated": sum(x["truncated"] for x in per), "loops": sum(x["loops"] for x in per),
                    "circling": sum(x.get("circling", 0) for x in per), "degenerate": sum(x.get("degenerate", 0) for x in per),
                    "think_tokens_mean": statistics.mean(x["think_tokens_mean"] for x in per),
                    "reconsider_per_kword": statistics.mean(x.get("reconsider_per_kword_mean", 0) for x in per)}
        g = gate.get(t, {})
        if g:
            metrics = {}
            for s in g.values():
                for bench, v in s["benchmarks"].items():
                    for m, mv in v["metrics"].items():
                        metrics.setdefault(f"{bench}.{m}", []).append(mv["score"])
            r["gate"] = {k: spread(v) for k, v in metrics.items()}
            r["gate_truncated"] = sum(v.get("truncated", 0) for s in g.values() for v in s["benchmarks"].values())
        res[str(t)] = r
    if "instruct" in speed:
        res["instruct-preset"] = {"speed": speed["instruct"]}
    print("| T | mixed tok/s | mixed tok/round | think tok/s | think tok/round | TTFT ms | AIME % | AIME trunc/loops/circ | MATH % | GSM8K % | gate GSM8K | HE+ | IFEval strict | MMLU-R |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t, r in res.items():
        s = r.get("speed", {})
        mx, th = s.get("mixed", {}), s.get("think", {})
        ai = r.get("aime", {})
        g = r.get("gate", {})
        cells = [t, f"{mx.get('decode_tok_s') or 0:.0f}" if mx else "–", f"{mx.get('tokens_per_round') or 0:.2f}" if mx else "–",
                 f"{th.get('decode_tok_s') or 0:.0f}" if th else "–", f"{th.get('tokens_per_round') or 0:.2f}" if th else "–",
                 f"{mx.get('ttft_ms_median') or 0:.0f}" if mx else "–",
                 fmt(ai.get("accuracy")) + (f" (×{ai['seeds']})" if ai else ""),
                 f"{ai['truncated']}/{ai['loops']}/{ai['circling']} of {ai['item_runs']}" if ai else "–",
                 fmt(r.get("math500", {}).get("accuracy")), fmt(r.get("gsm8k", {}).get("accuracy")),
                 fmt(g.get("gsm8k.accuracy")), fmt(g.get("humaneval_plus.plus")), fmt(g.get("ifeval.prompt_strict")), fmt(g.get("mmlu_redux.accuracy"))]
        print("| " + " | ".join(cells) + " |")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
