#!/usr/bin/env python3
"""Thinking-mode reasoning evaluation (AIME 2025+2026, MATH-500 subset, GSM8K-256) against an h5-server.

  think_eval.py run --server URL --out DIR [--concurrency N] [--benches aime,math500,gsm8k] [--limit N] [--max-tokens T] [--resume]
  think_eval.py rescore DIR                      re-score saved responses (no server)
  think_eval.py report DIR [DIR...] [--ref DIR] [--json]
                                                 per-run table; with --ref, paired deltas and trace divergence vs the reference
  think_eval.py diverge REF DIR [--bench B]      per-item first divergence from the reference trace

Decoding: greedy by default (temperature 0); --temperature T>0 sends native sampling fields (--top-p/--top-k/--min-p)
with per-item seeds derived from --seed; enable_thinking=true. h5-server renders
thinking-on as "<|im_start|>assistant\\n<think>\\n" with no system message, which is the model's own chat template
at reasoning_effort=medium (the template's default, xhigh, adds a system instruction). Default cap 32768 new
tokens; start the server with --ctx >= 34816. Details: tools/THINK_EVAL.md.
"""
import argparse, hashlib, json, math, re, statistics, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from gsm8k import SUFFIX, extract as gsm8k_extract, num as gsm8k_num  # noqa: E402
from qgate.gate import GateError, http_json, sha256  # noqa: E402

SCHEMA = "halo-think-eval-v1"
QDATA = TOOLS / "qgate" / "data"
TDATA = QDATA / "thinking"
BENCHES = ["aime", "math500", "gsm8k"]          # submission order: longest first so the tail overlaps
MAX_TOKENS = 32768
TIMEOUT_S = 3600


class EvalError(Exception):
    pass


# ---------------- data ----------------

def load_data():
    man = json.loads((TDATA / "MANIFEST.json").read_text())
    for name, meta in man["files"].items():
        if sha256((TDATA / name).read_bytes()) != meta["sha256"]:
            raise EvalError(f"vendored {name} does not match thinking/MANIFEST.json")
    qman = json.loads((QDATA / "MANIFEST.json").read_text())
    g = (QDATA / "gsm8k256.jsonl").read_bytes()
    if sha256(g) != qman["files"]["gsm8k256.jsonl"]["sha256"]:
        raise EvalError("vendored gsm8k256.jsonl does not match qgate MANIFEST.json")
    rows = lambda p: [json.loads(l) for l in p.read_text().splitlines()]
    aime = rows(TDATA / "aime2025.jsonl") + rows(TDATA / "aime2026.jsonl")
    return {"manifest_sha256": sha256((TDATA / "MANIFEST.json").read_bytes() + g),
            "aime": {r["id"]: r for r in aime},
            "math500": {r["id"]: r for r in rows(TDATA / "math500_175.jsonl")},
            "gsm8k": {r["id"]: r for r in rows(QDATA / "gsm8k256.jsonl")}}


def items(bench, data):
    if bench in ("aime", "math500"):
        return [(i, [{"role": "user", "content": r["problem"] + SUFFIX}]) for i, r in data[bench].items()]
    if bench == "gsm8k":
        return [(i, [{"role": "user", "content": r["question"] + SUFFIX}]) for i, r in data[bench].items()]
    raise EvalError(f"unknown benchmark {bench}")


SAMPLING = None   # set by run(): None = greedy; else dict of temperature/top_p/top_k/min_p/seed/draft_verify_limit
EXTRA = {}        # set by run(): further request fields sent with every item (--lz-penalty), part of the prompt hash


def prompt_hash(messages, max_tokens, sampling=None):
    d = {"messages": messages, "max_tokens": max_tokens, "thinking": True}
    if sampling:
        d["sampling"] = sampling
    return sha256(json.dumps(d, sort_keys=True).encode())


def item_sampling(bench, i):
    """Per-item request sampling: the run's sampler settings plus a seed derived from (run seed, bench, item) so items
    draw independent noise and a (seed, item) pair always replays the same realization."""
    if not SAMPLING:
        return {"temperature": 0, "top_p": 1, **EXTRA} if EXTRA else None
    s = {k: v for k, v in SAMPLING.items() if k != "seed" and v is not None}
    s["seed"] = int(sha256(f"{SAMPLING['seed']}:{bench}:{i}".encode())[:8], 16)
    return s | EXTRA


def sampling_body(sampling):
    if not sampling:
        return {"temperature": 0, "top_p": 1}
    return dict(sampling)


# ---------------- client ----------------

TIMING_KEYS = ("verify_rounds", "accepted_drafts", "proposed_drafts", "verifier_rows", "decode_tok_per_s", "ttft_s", "total_s",
               "verify_s", "draft_s", "lookup_proposed", "lookup_accepted")


def complete(server, messages, max_tokens, sampling=None):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, **sampling_body(sampling),
            "chat_template_kwargs": {"enable_thinking": True}}
    t = time.time()
    try:
        j = http_json(server + "/v1/chat/completions", body, timeout=TIMEOUT_S)
    except GateError as e:
        raise EvalError(str(e))
    wall = time.time() - t
    try:
        ch = j["choices"][0]
        content, finish, usage = ch["message"]["content"], ch["finish_reason"], j["usage"]
    except (KeyError, IndexError, TypeError):
        raise EvalError(f"malformed completion response: {json.dumps(j)[:500]}")
    if finish not in ("stop", "length") or content is None:
        raise EvalError(f"unexpected finish_reason {finish!r} or null content")
    tm = j.get("timings", {})
    return {"content": content, "finish_reason": finish, "completion_tokens": usage["completion_tokens"],
            "prompt_tokens": usage["prompt_tokens"], "wall_s": round(wall, 3), "t_end": round(time.time(), 3),
            "timings": {k: tm[k] for k in TIMING_KEYS if k in tm}}


LOOP_SPAN = 16000   # characters of exact periodic suffix (~5K tokens) that count as a never-ending greedy loop
LOOP_MAX_PERIOD = 4000


def loop_period(text):
    """Smallest period p <= LOOP_MAX_PERIOD such that the last LOOP_SPAN characters repeat with period p, else None."""
    n = len(text)
    if n < LOOP_SPAN + 64:
        return None
    key = text[-64:]
    lo = max(0, n - LOOP_SPAN - LOOP_MAX_PERIOD - 64)
    end = n - 1
    while True:
        k = text.rfind(key, lo, end + 63)
        if k < 0 or k + 64 >= n:
            if k < 0:
                return None
            end = k - 1 + 0
            if end <= lo:
                return None
            continue
        p = n - (k + 64)
        if p > LOOP_MAX_PERIOD:
            return None
        if text[n - LOOP_SPAN:] == text[n - LOOP_SPAN - p:n - p]:
            return p
        end = k - 1
        if end <= lo:
            return None


def complete_stream(server, messages, max_tokens, abort_loops, sampling=None):
    """Streaming request; with abort_loops, closes the connection (server cancels) once the output has an exact
    periodic suffix of LOOP_SPAN characters, recording finish_reason "loop"."""
    import http.client, urllib.parse
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, **sampling_body(sampling), "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": True}}
    u = urllib.parse.urlparse(server)
    t = time.time()
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=TIMEOUT_S)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise EvalError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    parts, finish, usage, nchars, checked, period = [], None, None, 0, 0, None
    t_first = None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        d = line[6:]
        if d == "[DONE]":
            break
        j = json.loads(d)
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices", []):
            piece = ch.get("delta", {}).get("content")
            if piece:
                if t_first is None:
                    t_first = time.time()
                parts.append(piece); nchars += len(piece)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
        if abort_loops and nchars - checked >= 2000:
            checked = nchars
            text = "".join(parts); parts = [text]
            period = loop_period(text)
            if period:
                finish = "loop"
                break
    conn.close()
    content = "".join(parts)
    wall = time.time() - t
    if finish not in ("stop", "length", "loop"):
        raise EvalError(f"unexpected finish_reason {finish!r}")
    if finish == "loop" or usage is None:
        ntok, est = round(len(content) / 3.3), True      # cancelled streams carry no usage; estimate
    else:
        ntok, est = usage["completion_tokens"], False
    return {"content": content, "finish_reason": finish, "completion_tokens": ntok, "completion_tokens_estimated": est,
            "prompt_tokens": usage["prompt_tokens"] if usage else None, "wall_s": round(wall, 3), "t_end": round(time.time(), 3),
            "loop_period_chars": period,
            "timings": {"decode_tok_per_s": (ntok - 1) / (time.time() - t_first) if t_first and ntok > 1 else None}}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def run(args):
    global SAMPLING, EXTRA
    if args.lz_penalty is not None:
        EXTRA["lz_penalty"] = args.lz_penalty
    if args.marker_penalty is not None:
        EXTRA["marker_penalty"] = args.marker_penalty
    if args.lz_draft is not None:
        EXTRA["lz_draft"] = args.lz_draft == "on"
    if args.loop_close is not None:
        EXTRA["loop_close"] = args.loop_close == "on"
    if args.temperature and args.temperature > 0:
        SAMPLING = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
                    "seed": args.seed, "draft_verify_limit": args.draft_verify_limit}
    server = args.server.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data()
    benches = args.benches.split(",")
    for b in benches:
        if b not in BENCHES:
            raise EvalError(f"unknown benchmark {b}")
    benches = [b for b in BENCHES if b in benches]
    rpath = out / "responses.jsonl"
    have = {}
    if rpath.exists():
        if not args.resume:
            raise EvalError(f"{rpath} exists; use a fresh --out or --resume")
        for l in rpath.read_text().splitlines():
            r = json.loads(l)
            have[(r["bench"], r["id"])] = r
    todo = []
    for b in benches:
        its = items(b, data)[:args.limit] if args.limit else items(b, data)
        if args.only:
            want = {tuple(l.split("\t")[:2]) for l in Path(args.only).read_text().splitlines() if l.strip()}
            its = [(i, m) for i, m in its if (b, i) in want]
        todo += [(b, i, m) for i, m in its if (b, i) not in have or have[(b, i)]["prompt_sha256"] != prompt_hash(m, args.max_tokens, item_sampling(b, i))]
    models = http_json(server + "/v1/models", timeout=30)
    metrics0 = http_json(server + "/metrics", timeout=30)
    meta = {"schema": SCHEMA, "server": server, "server_models": models, "server_metrics_start": metrics0, "label": args.label,
            "data_manifest_sha256": data["manifest_sha256"], "benches": benches, "limit": args.limit,
            "decoding": {"greedy": not SAMPLING, "temperature": SAMPLING["temperature"] if SAMPLING else 0, "sampling": SAMPLING, "request_extra": EXTRA, "enable_thinking": True, "reasoning_effort_equivalent": "medium",
                         "max_tokens": args.max_tokens, "concurrency": args.concurrency,
                         "stream": args.stream, "abort_loops": args.abort_loops, "loop_span_chars": LOOP_SPAN},
            "harness": git_identity(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"{len(todo)} requests ({len(have)} reused), concurrency {args.concurrency}, max_tokens {args.max_tokens}")
    lock = threading.Lock()
    t0, toks, done = time.time(), 0, 0
    with open(rpath, "a") as f, ThreadPoolExecutor(args.concurrency) as ex:
        call = (lambda m, sp: complete_stream(server, m, args.max_tokens, args.abort_loops, sp)) if args.stream else (lambda m, sp: complete(server, m, args.max_tokens, sp))
        futs = {ex.submit(call, m, item_sampling(b, i)): (b, i, m) for b, i, m in todo}
        last = t0
        for fu in as_completed(futs):
            b, i, m = futs[fu]
            r = fu.result()
            sp = item_sampling(b, i)
            rec = {"bench": b, "id": i, "prompt_sha256": prompt_hash(m, args.max_tokens, sp), "max_tokens": args.max_tokens, "sampling": sp, **r}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                toks += r["completion_tokens"]; done += 1
                if time.time() - last >= 60 or done == len(todo):
                    last = time.time(); dt = last - t0
                    log(f"{done}/{len(todo)}  {toks / dt:.0f} tok/s aggregate  {dt:.0f} s")
    wall = time.time() - t0
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["generation_wall_s"] = round(wall, 1)
    meta["generated_tokens_this_session"] = toks
    meta["aggregate_tok_per_s"] = round(toks / wall, 1) if wall > 0 and toks else None
    meta["server_metrics_end"] = http_json(server + "/metrics", timeout=30)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return rescore_dir(out, data)


def git_identity():
    import subprocess
    g = lambda *a: subprocess.run(["git", "-C", str(TOOLS), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD") or None, "dirty": bool(g("status", "--porcelain", "--", "tools/think_eval.py", "tools/qgate"))}


# ---------------- scoring ----------------

def split_think(content):
    """(thinking, answer, closed). h5-server's thinking-on prompt ends inside <think>, so content starts with the thoughts."""
    k = content.find("</think>")
    if k < 0:
        return content, "", False
    return content[:k], content[k + len("</think>"):], True


def last_boxed(text):
    k = text.rfind("\\boxed")
    if k < 0:
        return None
    j = text.find("{", k)
    if j < 0:
        return None
    depth, s = 0, j
    for p in range(j, len(text)):
        if text[p] == "{":
            depth += 1
        elif text[p] == "}":
            depth -= 1
            if depth == 0:
                return text[s + 1:p]
    return None


def aime_int(s):
    if s is None:
        return None
    s = re.sub(r"\\(text|mathrm|textbf)\{([^}]*)\}", r"\2", s).replace("$", "").replace(",", "").strip()
    m = re.fullmatch(r"0*(\d{1,3})(?:\^\{?\\circ\}?)?", s)
    return int(m.group(1)) if m else None


_mv = None


def math_verify(gold, answer):
    global _mv
    if _mv is None:
        try:
            from math_verify import parse, verify
        except ImportError:
            raise EvalError("math500 scoring needs math-verify: python3 -m pip install 'math-verify[antlr4_13_2]==0.9.0'")
        _mv = (parse, verify)
    parse, verify = _mv
    try:
        return bool(verify(parse("$" + gold + "$"), parse(answer)))
    except Exception:
        return False


def loop_score(content):
    """zlib compression ratio of the last 8000 characters: a repetition loop compresses far below normal prose (~0.3)."""
    tail = content[-8000:].encode()
    return round(len(zlib.compress(tail, 9)) / max(1, len(tail)), 4) if len(tail) >= 2000 else None


DEGEN_WINDOW, DEGEN_STRIDE, DEGEN_RATIO = 4000, 1000, 0.10


def degenerate_onset(content):
    """Start (characters) of the first 4000-character window whose zlib ratio is below 0.10 (short-period repetition or
    token salad; normal reasoning, even repetitive enumeration, stays above ~0.16), else None."""
    for i in range(0, max(0, len(content) - DEGEN_WINDOW) + 1, DEGEN_STRIDE):
        t = content[i:i + DEGEN_WINDOW].encode()
        if len(t) >= DEGEN_WINDOW and len(zlib.compress(t, 6)) / len(t) < DEGEN_RATIO:
            return i
    return None


_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_RECONSIDER = re.compile(r"\b(?:wait|actually|hmm+|hold on|on second thought|let me (?:re\w+|double[- ]check|check again|verify again)|"
                         r"i made (?:a|an) (?:mistake|error)|that's not right|that doesn't (?:seem right|work))\b", re.I)
CIRCLE_MIN_TAIL_CHARS, CIRCLE_MAX_NEW_NUMBERS = 6000, 3


def circling(think):
    """Semantic-circling heuristic for one thinking trace: reconsideration phrases per 1000 words, and how many distinct
    numeric literals first appear in the final quarter (a trace that keeps re-checking without progress introduces almost
    none). Flag: final quarter >= 6000 characters (~1.8K tokens) with <= 3 new numbers."""
    words = max(1, len(think.split()))
    cut = len(think) * 3 // 4
    seen = set(_NUM.findall(think[:cut]))
    tail = set(_NUM.findall(think[cut:]))
    new_tail = len(tail - seen)
    return {"reconsider_per_kword": round(1000 * len(_RECONSIDER.findall(think)) / words, 2), "tail_new_numbers": new_tail,
            "circling": len(think) - cut >= CIRCLE_MIN_TAIL_CHARS and new_tail <= CIRCLE_MAX_NEW_NUMBERS}


def score_record(bench, row, rec):
    think, ans, closed = split_think(rec["content"])
    if bench == "aime":
        got = aime_int(last_boxed(ans)) if closed else None
        ok = got is not None and got == int(row["answer"])
        gold = row["answer"]
    elif bench == "math500":
        got = last_boxed(ans) if closed else None
        ok = closed and math_verify(row["answer"], ans)
        gold = row["answer"]
    else:
        gold = gsm8k_num(row["answer"].split("####")[-1])
        got = gsm8k_extract(ans) if closed else None
        ok = got is not None and gold is not None and abs(got - gold) < 1e-6
    n = rec["completion_tokens"]
    tm = rec.get("timings", {})
    dg = degenerate_onset(rec["content"])
    think_tok = n if not closed else round(n * len(think) / max(1, len(rec["content"])))
    loop = rec["finish_reason"] == "loop" or (rec["finish_reason"] == "length" and loop_period(rec["content"]) is not None)
    circ = circling(think)
    circ["circling"] = circ["circling"] and not loop
    return {**circ, "seed": (rec.get("sampling") or {}).get("seed"),"bench": bench, "id": rec["id"], "correct": int(bool(ok)), "extracted": got, "gold": gold, "think_closed": closed,
            "loop_detected": rec["finish_reason"] == "loop" or (rec["finish_reason"] == "length" and loop_period(rec["content"]) is not None),
            "loop_period_chars": rec.get("loop_period_chars") or (loop_period(rec["content"]) if rec["finish_reason"] == "length" else None),
            "finish_reason": rec["finish_reason"], "completion_tokens": n, "think_tokens_est": think_tok,
            "degenerate": dg is not None, "degenerate_onset_tokens_est": round(n * dg / max(1, len(rec["content"]))) if dg is not None else None,
            "think_chars": len(think), "answer_chars": len(ans), "loop_zlib": loop_score(rec["content"]),
            "tokens_per_round": round(n / tm["verify_rounds"], 3) if tm.get("verify_rounds") else None,
            "decode_tok_per_s": tm.get("decode_tok_per_s")}


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(scored, recs):
    out = {}
    for b in BENCHES:
        s = [x for x in scored if x["bench"] == b]
        if not s:
            continue
        n = len(s)
        p = sum(x["correct"] for x in s) / n
        tt = [x["think_tokens_est"] for x in s]
        ct = [x["completion_tokens"] for x in s]
        rounds = sum(r.get("timings", {}).get("verify_rounds", 0) for r in recs if r["bench"] == b)
        dec = [x["decode_tok_per_s"] for x in s if x["decode_tok_per_s"]]
        out[b] = {"n": n, "correct": sum(x["correct"] for x in s), "accuracy": p, "se": math.sqrt(p * (1 - p) / n),
                  "think_tokens_mean": statistics.mean(tt), "think_tokens_p50": pct(tt, .5), "think_tokens_p95": pct(tt, .95),
                  "completion_tokens_mean": statistics.mean(ct), "completion_tokens_total": sum(ct),
                  "truncated": sum(x["finish_reason"] in ("length", "loop") for x in s), "think_unclosed": sum(not x["think_closed"] for x in s),
                  "think_tokens_p99": pct(tt, .99), "think_tokens_max": max(tt), "finished_think_tokens_max": max([x["think_tokens_est"] for x in s if x["finish_reason"] == "stop"] or [0]),
                  "loops": sum(1 for x in s if x["loop_detected"]),
                  "degenerate": sum(1 for x in s if x["degenerate"]),
                  "length_nonloop": sum(1 for x in s if x["finish_reason"] == "length" and not x["loop_detected"]),
                  "low_entropy_tail": sum(1 for x in s if x["loop_zlib"] is not None and x["loop_zlib"] < 0.12),
                  "tokens_per_round": round(sum(ct) / rounds, 3) if rounds else None,
                  "circling": sum(1 for x in s if x.get("circling")),
                  "circling_truncated": sum(1 for x in s if x.get("circling") and x["finish_reason"] != "stop"),
                  "reconsider_per_kword_mean": round(statistics.mean(x.get("reconsider_per_kword", 0) for x in s), 2),
                  "decode_tok_per_s_mean": round(statistics.mean(dec), 1) if dec else None}
    return out


def rescore_dir(out, data=None):
    data = data or load_data()
    recs = [json.loads(l) for l in (out / "responses.jsonl").read_text().splitlines()]
    last = {}
    for r in recs:            # resumed runs may append a newer record for the same item
        last[(r["bench"], r["id"])] = r
    recs = list(last.values())
    scored = [score_record(r["bench"], data[r["bench"]][r["id"]], r) for r in recs]
    with open(out / "items.jsonl", "w") as f:
        for x in sorted(scored, key=lambda x: (BENCHES.index(x["bench"]), x["id"])):
            f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
    summ = summarize(scored, recs)
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps({b: f"{v['correct']}/{v['n']} trunc {v['truncated']} think_mean {v['think_tokens_mean']:.0f}" for b, v in summ.items()}))
    return 0


# ---------------- comparison ----------------

def load_run(d):
    d = Path(d)
    its = {(x["bench"], x["id"]): x for x in map(json.loads, (d / "items.jsonl").read_text().splitlines())}
    recs = {}
    for l in (d / "responses.jsonl").read_text().splitlines():
        r = json.loads(l)
        recs[(r["bench"], r["id"])] = r
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return its, recs, meta


def common_prefix(a, b):
    n = min(len(a), len(b))
    lo = 0
    step = 4096
    while lo < n:                      # coarse then fine
        hi = min(n, lo + step)
        if a[lo:hi] == b[lo:hi]:
            lo = hi
        else:
            for p in range(lo, hi):
                if a[p] != b[p]:
                    return p
    return n


def char_class(c):
    if c == "":
        return "end"
    if c.isdigit():
        return "digit"
    if c.isspace():
        return "space"
    if c.isalpha():
        return "letter"
    if c in "\\$^_{}()[]=+-*/<>|&":
        return "latex/math"
    return "punct"


def divergence(ref_recs, recs, key):
    a, b = ref_recs[key]["content"], recs[key]["content"]
    p = common_prefix(a, b)
    if p == len(a) == len(b):
        return {"identical": True, "pos_chars": p}
    # back up to the token-ish boundary (start of the word) to show context
    w = p
    while w > 0 and a[w - 1].isalnum() and p - w < 20:
        w -= 1
    frac = p / max(1, len(a))
    est_tok = round(ref_recs[key]["completion_tokens"] * frac)
    return {"identical": False, "pos_chars": p, "pos_tokens_est": est_tok, "frac_of_ref": round(frac, 4),
            "ref_next": a[w:p + 12], "cand_next": b[w:p + 12], "class_ref": char_class(a[p:p + 1]), "class_cand": char_class(b[p:p + 1]),
            "in_think": p < (a.find("</think>") if "</think>" in a else len(a))}


def paired(base, cand):
    n = len(base)
    d = [c - b for b, c in zip(base, cand)]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    return {"n": n, "delta": mean, "se_diff": math.sqrt(var / n), "gains": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d)}


def report(args):
    ref = load_run(args.ref) if args.ref else None
    rows = []
    for d in args.dirs:
        its, recs, meta = load_run(d)
        summ = json.loads((Path(d) / "summary.json").read_text())
        row = {"run": str(d), "label": meta.get("label"), "aggregate_tok_per_s": meta.get("aggregate_tok_per_s"),
               "concurrency": meta.get("decoding", {}).get("concurrency"), "benches": summ}
        if ref:
            ri, rr, _ = ref
            for b in summ:
                keys = [k for k in its if k[0] == b and k in ri]
                pr = paired([ri[k]["correct"] for k in keys], [its[k]["correct"] for k in keys])
                dv = [divergence(rr, recs, k) for k in keys]
                pos = [x["pos_tokens_est"] for x in dv if not x["identical"]]
                pr.update({"identical_traces": sum(x["identical"] for x in dv),
                           "diverge_tok_p50": pct(pos, .5), "diverge_tok_p10": pct(pos, .1),
                           "diverge_class": dict(sorted(_count(x["class_ref"] + ">" + x["class_cand"] for x in dv if not x["identical"]).items(), key=lambda kv: -kv[1])[:6])})
                row["benches"][b]["vs_ref"] = pr
        rows.append(row)
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    hdr = f"{'run':34} {'bench':8} {'acc':>13} {'think mean/p50/p95':>22} {'trunc':>5} {'unclosed':>8} {'loops':>5} {'degen':>5} {'cap':>4} {'circ':>4} {'rc/kw':>5} {'tok/rnd':>7} {'dec t/s':>7}"
    if ref:
        hdr += f" {'Δ vs ref (±se, +g/-l)':>26} {'same':>5} {'div p50':>7}"
    print(hdr)
    for row in rows:
        for b, v in row["benches"].items():
            line = (f"{(row['label'] or Path(row['run']).name)[:34]:34} {b:8} {v['correct']:>3}/{v['n']:<3} {100*v['accuracy']:5.1f}±{100*v['se']:3.1f}"
                    f" {v['think_tokens_mean']:7.0f}/{v['think_tokens_p50']:6.0f}/{v['think_tokens_p95']:6.0f} {v['truncated']:5d} {v['think_unclosed']:8d} {v['loops']:5d} {v.get('degenerate', 0):5d} {v.get('length_nonloop', 0):4d} {v.get('circling', 0):4d} {v.get('reconsider_per_kword_mean', 0):5.1f}"
                    f" {v['tokens_per_round'] or 0:7.2f} {v['decode_tok_per_s_mean'] or 0:7.0f}")
            if "vs_ref" in v:
                r = v["vs_ref"]
                line += f" {100*r['delta']:+6.1f}±{100*r['se_diff']:4.1f} +{r['gains']}/-{r['losses']:<3}       {r['identical_traces']:5d} {r['diverge_tok_p50'] or 0:7.0f}"
            print(line)
        print(f"{'':34} aggregate {row['aggregate_tok_per_s']} tok/s at concurrency {row['concurrency']}")
    return 0


def _count(xs):
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def diverge_cmd(args):
    ri, rr, _ = load_run(args.ref)
    ci, cr, _ = load_run(args.dir)
    for k in sorted(ci):
        if args.bench and k[0] != args.bench or k not in ri:
            continue
        dv = divergence(rr, cr, k)
        print(json.dumps({"bench": k[0], "id": k[1], "ref_correct": ri[k]["correct"], "cand_correct": ci[k]["correct"],
                          "ref_tokens": ri[k]["completion_tokens"], "cand_tokens": ci[k]["completion_tokens"], **dv}, ensure_ascii=False))
    return 0


def main(argv):
    p = argparse.ArgumentParser(prog="think_eval.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--server", required=True); r.add_argument("--out", required=True)
    r.add_argument("--concurrency", type=int, default=1); r.add_argument("--benches", default=",".join(BENCHES))
    r.add_argument("--limit", type=int); r.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    r.add_argument("--resume", action="store_true"); r.add_argument("--label")
    r.add_argument("--stream", action="store_true", help="stream responses (needed for --abort-loops)")
    r.add_argument("--abort-loops", action="store_true", help="cancel a request once its output is exactly periodic over the last %d characters" % LOOP_SPAN)
    r.add_argument("--only", help="file of bench<TAB>id lines: run only these items")
    r.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default); >0 sends native sampling fields")
    r.add_argument("--top-p", type=float, default=0.95); r.add_argument("--top-k", type=int, default=20)
    r.add_argument("--min-p", type=float, default=0.05, help="Bonsai Chat thinking preset uses 0.05")
    r.add_argument("--seed", type=int, default=0, help="run seed; per-item seeds derive from it")
    r.add_argument("--lz-penalty", type=float, help="send lz_penalty (LZ penalty strength; the server needs H5_LZ, tools/ANTILOOP.md)")
    r.add_argument("--marker-penalty", type=float, help="send marker_penalty (the server needs H5_MARKERS)")
    r.add_argument("--lz-draft", choices=("on", "off"), help="send lz_draft (drafter-side LZ; the server needs H5_LZ_DRAFT=on)")
    r.add_argument("--loop-close", choices=("on", "off"), help="send loop_close (the server needs H5_LOOP_CLOSE)")
    r.add_argument("--draft-verify-limit", type=int, help="send draft_verify_limit (0 = no drafting; required for multi-slot sampling with a drafter loaded)")
    s = sub.add_parser("rescore"); s.add_argument("dir")
    q = sub.add_parser("report"); q.add_argument("dirs", nargs="+"); q.add_argument("--ref"); q.add_argument("--json", action="store_true")
    v = sub.add_parser("diverge"); v.add_argument("ref"); v.add_argument("dir"); v.add_argument("--bench")
    a = p.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "rescore":
        return rescore_dir(Path(a.dir))
    if a.cmd == "report":
        return report(a)
    return diverge_cmd(a)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except EvalError as e:
        print(f"think_eval: error: {e}", file=sys.stderr)
        sys.exit(2)
