#!/usr/bin/env python3
"""Multi-user throughput curve for an OpenAI-compatible endpoint (h5-server).

  batch_bench.py URL --tag TAG --out FILE.jsonl --prompts mixed|gsm8k --concurrency 1,4,8,16,32,64
                 [--gsm-file gsm8k_test.jsonl] [--requests N] [--max-tokens T]

Closed loop: C workers each send their next streamed request as soon as the previous one finishes.
Per point it reports
  aggregate tok/s   completion tokens / wall time of the whole point (includes ramp-up and the drain tail)
  steady tok/s      slope of the server's generated-token counter while >= C requests were active
                    (sampled from /metrics every 0.25 s), i.e. the rate a sustained C-user load gets
  per-user tok/s    (tokens - 1) / (last token - first token) per request: median and p10
  TTFT, latency     request start -> first content / last byte: p50 and p95
  server deltas     steps, rows/step, verify and drafter ms per step, emitted tokens per verify round
GSM8K points also score accuracy with tools/gsm8k.py's extractor (same prompt suffix, 768 tokens).
Mixed points cycle tools/demo/speedbench.py's 35 prompts (creative, code, explain, chat, structured,
long input, math).
"""
import argparse, json, statistics, sys, threading, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k import SUFFIX, extract, num  # noqa: E402


def mixed_prompts():
    from demo.speedbench import P
    return [p for ps in P.values() for p in ps]


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def open_response(req):
    try:
        return urllib.request.urlopen(req, timeout=7200)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors='replace')
        raise RuntimeError(f'HTTP {error.code}: {detail}') from error


def one(url, msg, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = tl = None; ntok = 0; text = []
    with open_response(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                ntok = j["usage"]["completion_tokens"]
            for c in j.get("choices") or []:
                d = (c.get("delta") or {}).get("content")
                if d:
                    now = time.time(); tf = tf or now; tl = now; text.append(d)
    t1 = time.time()
    return dict(t0=t0, tf=tf or t1, tl=tl or t1, t1=t1, ntok=ntok, text="".join(text))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))] if v else 0.0


def run_point(a, url, prompts, golds, c, n):
    m0 = get_json(url + "/metrics")
    res = [None] * n; errors = []; lock = threading.Lock(); it = iter(range(n))
    samples = []; stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                m = get_json(url + "/metrics", 5); samples.append((time.time(), m["generated_tokens"], m["active"]))
            except Exception:
                pass
            stop.wait(0.25)

    # mixed: stride 7 over the 30 prompts so short points still span every category
    order = (lambda i: (7 * i) % len(prompts)) if golds is None else (lambda i: i % len(prompts))

    def worker():
        while True:
            with lock:
                i = next(it, None)
            if i is None:
                return
            try:
                res[i] = one(url, prompts[order(i)], a.max_tokens)
            except Exception as e:
                with lock:
                    errors.append((i, repr(e)))

    st = threading.Thread(target=sampler); st.start()
    t0 = time.time()
    th = [threading.Thread(target=worker) for _ in range(c)]
    [t.start() for t in th]; [t.join() for t in th]
    t1 = time.time(); stop.set(); st.join()
    m1 = get_json(url + "/metrics")
    if errors:
        raise RuntimeError(f"{len(errors)} failed requests, first {errors[0]}")
    tok = sum(r["ntok"] for r in res)
    per = [(r["ntok"] - 1) / (r["tl"] - r["tf"]) for r in res if r["ntok"] > 1 and r["tl"] > r["tf"]]
    ttft = [r["tf"] - r["t0"] for r in res]; lat = [r["t1"] - r["t0"] for r in res]
    # steady window: samples with active >= c (all workers busy); slope of generated tokens
    full = [s for s in samples if s[2] >= c]
    steady = (full[-1][1] - full[0][1]) / (full[-1][0] - full[0][0]) if len(full) >= 3 and full[-1][0] > full[0][0] else None
    d = {k: m1[k] - m0[k] for k in ("generated_tokens", "steps", "busy_s", "verify_s", "draft_s", "verify_rounds", "accepted_drafts", "proposed_drafts", "prefill_rows") if k in m1}
    if "verifier_rows" in m1:
        d["verifier_rows"] = m1["verifier_rows"] - m0["verifier_rows"]
    out = dict(tag=a.tag, prompts=a.prompts, concurrency=c, requests=n, max_tokens=a.max_tokens, completion_tokens=tok,
               wall_s=t1 - t0, aggregate_tok_s=tok / (t1 - t0), steady_tok_s=steady, steady_window_s=(full[-1][0] - full[0][0]) if len(full) >= 2 else 0,
               per_user_tok_s_median=statistics.median(per) if per else 0, per_user_tok_s_p10=pct(per, 0.10),
               ttft_s_p50=pct(ttft, 0.5), ttft_s_p95=pct(ttft, 0.95), latency_s_p50=pct(lat, 0.5), latency_s_p95=pct(lat, 0.95),
               usd_per_m_at_069=0.69 * 1e6 / 3600 / (steady or tok / (t1 - t0)), server=d)
    if d.get("steps"):
        out["step_ms"] = 1e3 * d["busy_s"] / d["steps"]
        out["verify_ms_per_step"] = 1e3 * d["verify_s"] / d["steps"]
        out["draft_ms_per_step"] = 1e3 * d["draft_s"] / d["steps"]
        out["tokens_per_step"] = d["generated_tokens"] / d["steps"]
    if d.get("verify_rounds"):
        out["tokens_per_round"] = 1 + d["accepted_drafts"] / d["verify_rounds"]
        if d.get("verifier_rows"):
            out["verifier_rows_per_round"] = d["verifier_rows"] / d["verify_rounds"]
    if golds is not None:
        ok = 0
        for i, r in enumerate(res):
            g = golds[i % len(golds)]; v = extract(r["text"])
            ok += v is not None and g is not None and abs(v - g) < 1e-6
        out["correct"] = ok; out["accuracy"] = ok / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", choices=["mixed", "gsm8k"], default="mixed")
    ap.add_argument("--concurrency", default="1,4,8,16,32,64")
    ap.add_argument("--gsm-file", default="/root/models/gsm8k_test.jsonl")
    ap.add_argument("--requests", type=int, default=0, help="per point; 0 = automatic")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = 512 mixed / 768 GSM8K")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    golds = None
    if a.prompts == "gsm8k":
        rows = [json.loads(l) for l in open(a.gsm_file)]
        prompts = [r["question"] + SUFFIX for r in rows]; golds = [num(r["answer"].split("####")[-1]) for r in rows]
        a.max_tokens = a.max_tokens or 768
    else:
        prompts = mixed_prompts(); a.max_tokens = a.max_tokens or 512
    for c in [int(x) for x in a.concurrency.split(",") if x]:
        if a.requests:
            n = a.requests
        elif a.prompts == "gsm8k":
            n = min(len(prompts), max(32, 6 * c))
        else:
            n = max(12, 3 * c)
        r = run_point(a, url, prompts, golds, c, n)
        print(json.dumps(r), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
