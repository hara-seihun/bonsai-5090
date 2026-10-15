#!/usr/bin/env python3
"""Single-stream speed across sampling temperatures on one resident server (the chat's serving configuration).

  temp_speed.py BASE_URL OUT.json [--temps 0,0.3,0.6,0.8,1.0] [--reps 1]

Workload (fixed): tools/demo/speedbench.py's 35 mixed prompts with thinking off (max 512 new tokens), plus thinking-on
GSM8K (4), code (3) and AIME (3) prompts (max 2048 new tokens). T>0 sends top_p 0.95, top_k 20, min_p 0.05 (the Bonsai
Chat thinking preset apart from temperature) and a fixed per-prompt seed; T=0 is strict greedy. An extra arm
"instruct" sends the chat's thinking-off preset (T 0.7, top_p 0.8, top_k 20, presence 1.5) on the thinking-off prompts.
Temperatures are interleaved per prompt (rotating order) so drift hits all arms equally. Rates come from the server's
own timings: decode tok/s = generated tokens / decode time; tokens per round = generated tokens / verify rounds.
"""
import argparse, hashlib, json, sys, time, urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / "demo"))
from speedbench import P as MIXED  # noqa: E402
from gsm8k import SUFFIX  # noqa: E402

CODE = ["Write a Python function that returns the longest palindromic substring of a string, with tests.",
        "Implement a thread-safe bounded blocking queue in C++17 with condition variables.",
        "Write a function that merges overlapping intervals, explain its complexity, and give examples."]


def prompts():
    out = []
    for cat, ps in MIXED.items():
        out += [("mixed", cat, p, False, 512) for p in ps]
    q = TOOLS / "qgate" / "data"
    gsm = [json.loads(l) for l in (q / "gsm8k256.jsonl").read_text().splitlines()][:4]
    out += [("think", "gsm8k", r["question"] + SUFFIX, True, 2048) for r in gsm]
    out += [("think", "code", p, True, 2048) for p in CODE]
    aime = [json.loads(l) for l in (q / "thinking" / "aime2025.jsonl").read_text().splitlines()][:3]
    out += [("think", "aime", r["problem"] + SUFFIX, True, 2048) for r in aime]
    return out


def sampling(arm, prompt):
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    if arm == "instruct":
        return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "seed": seed}
    t = float(arm)
    if t == 0:
        return {"temperature": 0}
    return {"temperature": t, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "seed": seed}


def call(base, prompt, think, max_tokens, samp):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think}, **samp}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read())
    wall = time.perf_counter() - t0
    tm = j.get("timings", {})
    return {"tokens": j["usage"]["completion_tokens"], "prompt_tokens": j["usage"]["prompt_tokens"], "finish": j["choices"][0]["finish_reason"],
            "wall_s": wall, "decode_tok_s": tm.get("decode_tok_per_s"), "ttft_s": tm.get("ttft_s"), "verify_rounds": tm.get("verify_rounds"),
            "accepted_drafts": tm.get("accepted_drafts"), "verify_s": tm.get("verify_s"), "draft_s": tm.get("draft_s"),
            "sampling_s": tm.get("sampling_s"), "prefix_histogram": tm.get("prefix_histogram")}


def agg(rs):
    toks = sum(r["tokens"] for r in rs)
    dec = sum(r["tokens"] / r["decode_tok_s"] for r in rs if r["decode_tok_s"])
    rounds = sum(r["verify_rounds"] or 0 for r in rs)
    ttft = sorted(r["ttft_s"] for r in rs if r["ttft_s"] is not None)
    return {"n": len(rs), "tokens": toks, "decode_tok_s": toks / dec if dec else None,
            "tokens_per_round": toks / rounds if rounds else None, "ttft_ms_median": 1000 * ttft[len(ttft) // 2] if ttft else None,
            "verify_s": sum(r["verify_s"] or 0 for r in rs), "draft_s": sum(r["draft_s"] or 0 for r in rs),
            "sampling_s": sum(r["sampling_s"] or 0 for r in rs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_url"); ap.add_argument("output")
    ap.add_argument("--temps", default="0,0.3,0.6,0.8,1.0"); ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--no-instruct", action="store_true")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    arms = a.temps.split(",") + ([] if a.no_instruct else ["instruct"])
    call(base, "Say hi.", False, 16, {"temperature": 0})
    res = {arm: [] for arm in arms}
    ps = prompts()
    for rep in range(a.reps):
        for k, (group, cat, p, think, mt) in enumerate(ps):
            order = arms[(k + rep) % len(arms):] + arms[:(k + rep) % len(arms)]
            for arm in order:
                if arm == "instruct" and think:
                    continue
                r = call(base, p, think, mt, sampling(arm, p))
                res[arm].append({"group": group, "cat": cat, "think": think, "rep": rep, "prompt": p[:60], **r})
            print(f"{k + 1}/{len(ps)} {cat}: " + " ".join(f"{arm}={res[arm][-1]['decode_tok_s']:.0f}" for arm in order
                                                           if res[arm] and res[arm][-1]["prompt"] == p[:60]), file=sys.stderr, flush=True)
    summary = {arm: {g: agg([r for r in rs if r["group"] == g]) for g in ("mixed", "think") if any(r["group"] == g for r in rs)}
               for arm, rs in res.items()}
    for arm, rs in res.items():
        for cat in sorted({r["cat"] for r in rs if r["group"] == "think"}):
            summary[arm]["think_" + cat] = agg([r for r in rs if r["cat"] == cat and r["group"] == "think"])
    print(f"{'arm':9} {'group':12} {'tok/s':>7} {'tok/rnd':>7} {'ttft ms':>7} {'tokens':>7}")
    for arm, g in summary.items():
        for name, v in g.items():
            print(f"{arm:9} {name:12} {v['decode_tok_s'] or 0:7.1f} {v['tokens_per_round'] or 0:7.2f} {v['ttft_ms_median'] or 0:7.0f} {v['tokens']:7d}")
    Path(a.output).write_text(json.dumps({"summary": summary, "runs": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
