#!/usr/bin/env python3
"""End-to-end load test for an OpenAI-compatible endpoint (h5-server or anything else).

  loadtest.py URL --concurrency N --requests M [--max-tokens T] [--stream] [--prompts gsm8k|mixed] [--think]

Each worker thread sends chat completions back to back. Reports aggregate output tok/s over the
busy window, per-stream decode tok/s (completion tokens after the first, over time after the first
token), time to first token, and request latency. Token counts come from the server's usage field.
"""
import argparse, json, random, statistics, threading, time, urllib.request

MIXED = [
    "Write a short story about a lighthouse keeper who discovers a strange signal.",
    "Explain how a refrigerator works, step by step.",
    "Write a Python function that returns the n-th prime number, with comments.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "What are the differences between TCP and UDP? Give examples of when to use each.",
    "Write a haiku sequence about autumn in Montreal.",
    "Describe the water cycle to a ten-year-old.",
    "Write a SQL query that finds the top 5 customers by total order value, and explain it.",
]


def gsm8k_questions(n):
    import datasets
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    return [x["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}." for x in ds.select(range(n))]


def one(url, msg, max_tokens, stream, think):
    body = {"model": "ternary-bonsai-2-27b", "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
            "temperature": 0, "stream": stream}
    if think:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); tfirst = None; ntok = 0; text = ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        if not stream:
            j = json.loads(r.read())
            tfirst = t0 + j.get("timings", {}).get("ttft_s", 0)
            ntok = j["usage"]["completion_tokens"]; text = j["choices"][0]["message"]["content"]
        else:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                j = json.loads(line[6:])
                if j.get("usage"):
                    ntok = j["usage"]["completion_tokens"]
                for c in j.get("choices", []):
                    d = c.get("delta", {}).get("content")
                    if d:
                        if tfirst is None:
                            tfirst = time.time()
                        text += d
    t1 = time.time()
    return dict(t0=t0, tfirst=tfirst or t1, t1=t1, ntok=ntok, text=text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--prompts", default="mixed")
    ap.add_argument("--price", type=float, default=0.99, help="$/h of the GPU, for $/M")
    ap.add_argument("--json", default=None)
    ap.add_argument("--arrival-rate", type=float, default=0, help="Seeded Poisson requests/s; zero means closed-loop workers")
    ap.add_argument("--details", default=None, help="Per-request timing JSON, excluding generated text")
    a = ap.parse_args()
    qs = gsm8k_questions(a.requests) if a.prompts == "gsm8k" else None
    if not qs:
        qs = [MIXED[i % len(MIXED)] for i in range(a.requests)]
    res = []; errors = []; lock = threading.Lock(); it = iter(range(a.requests))

    def run_one(i):
        try:
            r = one(a.url, qs[i], a.max_tokens, a.stream, a.think)
            r['index'] = i
            with lock:
                res.append(r)
        except Exception as e:
            with lock:
                errors.append({'index': i, 'error': repr(e)})

    def worker():
        while True:
            with lock:
                i = next(it, None)
            if i is None:
                return
            run_one(i)

    t0 = time.time()
    if a.arrival_rate > 0:
        rng = random.Random(17); due = time.monotonic(); th = []
        for i in range(a.requests):
            due += rng.expovariate(a.arrival_rate)
            time.sleep(max(0, due - time.monotonic()))
            t = threading.Thread(target=run_one, args=(i,)); t.start(); th.append(t)
    else:
        th = [threading.Thread(target=worker) for _ in range(a.concurrency)]
        [t.start() for t in th]
    [t.join() for t in th]
    t1 = time.time()
    if a.details:
        with open(a.details, 'w') as f:
            json.dump({'requests': [{k: v for k, v in r.items() if k != 'text'} for r in res], 'errors': errors}, f)
    if errors or len(res) != a.requests:
        raise RuntimeError(f'Incomplete benchmark: {len(res)}/{a.requests}; {errors}')
    tok = sum(r["ntok"] for r in res)
    per = [(r["ntok"] - 1) / (r["t1"] - r["tfirst"]) for r in res if r["ntok"] > 1 and r["t1"] > r["tfirst"]]
    ttft = [r["tfirst"] - r["t0"] for r in res]
    lat = [r["t1"] - r["t0"] for r in res]
    agg = tok / (t1 - t0)
    out = dict(concurrency=a.concurrency if not a.arrival_rate else None, arrival_rate=a.arrival_rate, requests=len(res), stream=a.stream, think=a.think, max_tokens=a.max_tokens,
               completion_tokens=tok, wall_s=t1 - t0, aggregate_tok_s=agg,
               per_stream_tok_s_median=statistics.median(per) if per else 0, per_stream_tok_s_mean=statistics.mean(per) if per else 0,
               ttft_s_median=statistics.median(ttft), ttft_s_p90=sorted(ttft)[int(0.9 * (len(ttft) - 1))],
               latency_s_median=statistics.median(lat), usd_per_m=a.price * 1e6 / 3600 / agg)
    print(json.dumps(out))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out) + "\n")
    if len(res):
        print("sample:", res[0]["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
