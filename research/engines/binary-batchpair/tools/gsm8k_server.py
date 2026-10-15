#!/usr/bin/env python3
"""GSM8K through an OpenAI-compatible endpoint (h5-server), same prompt and scorer as tools/gsm8k.py.

  gsm8k_server.py URL TEST.jsonl [N] [CONCURRENCY]

Chat template is applied by the server (thinking off); greedy; max 768 new tokens. Prints accuracy,
aggregate tok/s over the run and per-request decode tok/s.
"""
import json, sys, time, threading, statistics, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gsm8k import SUFFIX, extract, num  # noqa: E402

url, path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1319
conc = int(sys.argv[4]) if len(sys.argv) > 4 else 64
rows = [json.loads(l) for l in open(path)][:n]
res = [None] * len(rows); it = iter(range(len(rows))); lock = threading.Lock(); toks = [0] * len(rows); rates = []


def work():
    while True:
        with lock:
            i = next(it, None)
        if i is None:
            return
        body = {"model": "x", "messages": [{"role": "user", "content": rows[i]["question"] + SUFFIX}], "max_tokens": 768, "temperature": 0}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        j = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        res[i] = j["choices"][0]["message"]["content"]; toks[i] = j["usage"]["completion_tokens"]
        r = j.get("timings", {}).get("decode_tok_per_s", 0)
        if r:
            with lock:
                rates.append(r)


t0 = time.time()
th = [threading.Thread(target=work) for _ in range(conc)]
[t.start() for t in th]; [t.join() for t in th]
dt = time.time() - t0
ok = 0
for r, o in zip(rows, res):
    gold = num(r["answer"].split("####")[-1])
    v = extract(o)
    ok += v is not None and gold is not None and abs(v - gold) < 1e-6
print(json.dumps({"n": len(rows), "correct": ok, "accuracy": ok / len(rows), "concurrency": conc, "tokens": sum(toks), "wall_s": dt,
                  "aggregate_tok_s": sum(toks) / dt, "per_request_decode_tok_s_median": statistics.median(rates) if rates else 0}))
