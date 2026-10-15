#!/usr/bin/env python3
"""GSM8K harness shared by halo-5090 and the PrismML fork.

  gsm8k.py prep  TEST.jsonl N PROMPTS.bin          templated prompts (thinking off), length-prefixed utf-8
  gsm8k.py fork  PROMPTS.bin OUT.bin URL [PAR]     greedy completions from a llama-server
  gsm8k.py score TEST.jsonl OUT.bin [OTHER.bin]    accuracy (and agreement with OTHER)

The prompt is identical for both engines; both are greedy, max 768 new tokens, stop at <|im_end|>.
"""
import json, re, struct, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_NEW = 768


def template(q):
    return f"<|im_start|>user\n{q}{SUFFIX}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def write_bin(path, items):
    with open(path, "wb") as f:
        for s in items:
            b = s.encode()
            f.write(struct.pack("<i", len(b)))
            f.write(b)


def read_bin(path):
    out = []
    with open(path, "rb") as f:
        while True:
            h = f.read(4)
            if not h:
                return out
            (n,) = struct.unpack("<i", h)
            out.append(f.read(n).decode("utf-8", "replace"))


def load(path, n=None):
    rows = [json.loads(l) for l in open(path)]
    return rows[:n] if n else rows


def num(s):
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
    if not m:
        return None
    v = m[-1]
    try:
        if "/" in v:
            a, b = v.split("/")
            return float(a) / float(b)
        return float(v)
    except Exception:
        return None


def extract(text):
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxed:
        v = num(boxed[-1])
        if v is not None:
            return v
    return num(text)


def main():
    cmd = sys.argv[1]
    if cmd == "prep":
        a = sys.argv[3]
        rows = load(sys.argv[2])
        if ":" in a:
            lo, hi = a.split(":"); rows = rows[int(lo):int(hi)]
        elif int(a):
            rows = rows[:int(a)]
        write_bin(sys.argv[4], [template(r["question"]) for r in rows])
    elif cmd == "fork":
        prompts = read_bin(sys.argv[2])
        url = sys.argv[4]
        par = int(sys.argv[5]) if len(sys.argv) > 5 else 64

        def one(p):
            body = json.dumps({"prompt": p, "n_predict": MAX_NEW, "temperature": 0, "top_k": 1, "cache_prompt": False,
                               "stop": ["<|im_end|>"], "special": False}).encode()
            for _ in range(5):
                try:
                    r = urllib.request.urlopen(urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"}), timeout=3600)
                    j = json.loads(r.read())
                    return j["content"], j.get("tokens_predicted", 0)
                except Exception as e:
                    err = e
                    time.sleep(2)
            raise err

        t0 = time.time()
        with ThreadPoolExecutor(par) as ex:
            res = list(ex.map(one, prompts))
        dt = time.time() - t0
        ntok = sum(r[1] for r in res)
        print(f"fork: {len(prompts)} prompts, {ntok} tokens in {dt:.1f} s = {ntok / dt:.0f} tok/s at {par} parallel")
        write_bin(sys.argv[3], [r[0] for r in res])
    elif cmd == "fewshot":   # fewshot TRAIN.jsonl TEST.jsonl NSHOT RANGE PREFIX.txt PROMPTS.bin
        train = load(sys.argv[2])[: int(sys.argv[4])]
        shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in train)
        prefix = "<|im_start|>user\nHere are some solved examples.\n\n" + shots
        a = sys.argv[5]
        rows = load(sys.argv[3])
        if ":" in a:
            lo, hi = a.split(":"); rows = rows[int(lo):int(hi)]
        open(sys.argv[6], "w").write(prefix)
        write_bin(sys.argv[7], [f"Now solve this one.\nQuestion: {r['question']}{SUFFIX}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" for r in rows])
    elif cmd == "cat":
        out = []
        for p in sys.argv[3:]:
            out += read_bin(p)
        write_bin(sys.argv[2], out)
    elif cmd == "score":
        rows = load(sys.argv[2])
        outs = read_bin(sys.argv[3])
        other = read_bin(sys.argv[4]) if len(sys.argv) > 4 else None
        ok = agree = 0
        for i, o in enumerate(outs):
            gold = float(rows[i]["answer"].split("####")[-1].replace(",", "").strip())
            v = extract(o)
            ok += v is not None and abs(v - gold) < 1e-4 * max(1, abs(gold))
            if other:
                w = extract(other[i])
                agree += v is not None and w is not None and abs(v - w) < 1e-6
        n = len(outs)
        print(f"GSM8K {sys.argv[3]}: {ok}/{n} = {100 * ok / n:.2f}%" + (f"  answer agreement with {sys.argv[4]}: {agree}/{n}" if other else ""))


if __name__ == "__main__":
    main()
