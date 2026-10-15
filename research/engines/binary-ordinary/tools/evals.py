#!/usr/bin/env python3
"""Benchmark harness shared by halo-5090 and the PrismML llama.cpp fork (same prompts, greedy).

  evals.py prep  BENCH MODE DIR             write DIR/BENCH-MODE.prompts.bin (length-prefixed utf-8, chat template applied)
                                            and DIR/BENCH-MODE.items.jsonl (gold answers and scoring data, one row per prompt)
  evals.py fork  PROMPTS OUT URL PAR MAXNEW greedy completions from a llama-server; writes OUT and OUT.meta
  evals.py score BENCH MODE DIR OUT [OTHER] score OUT; writes OUT.score.json (and paired difference vs OTHER)

BENCH: mmlupro mmlupro2k gpqa ifeval math500 humaneval mbpp lcb.  MODE: off | on.
Thinking off renders the model's own chat template with enable_thinking=false (the prompt ends in an empty
<think></think>); thinking on renders it with reasoning_effort=medium (prompt ends in "<think>\\n").
OUT.meta (and h5's H5_META) holds one line per prompt: prompt_tokens generated_tokens stopped(1/0).
Data roots are under $EVAL_DATA (default /root/w/data); see results/benchmarks/README.md for versions and hashes.
"""
import csv, json, math, os, random, re, struct, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA = os.environ.get("EVAL_DATA", "/root/w/data")
TEMPLATE = os.environ.get("CHAT_TEMPLATE", "/root/w/bonsai_template.jinja")
EFFORT = os.environ.get("REASONING_EFFORT", "medium")


# ---------------------------------------------------------------- io
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


def jl(path):
    return [json.loads(l) for l in open(path)]


# ---------------------------------------------------------------- chat template
_tmpl = None


def chat(messages, mode, prefix=""):
    """Render the GGUF's own chat template (HF semantics), then append an optional assistant prefix."""
    global _tmpl
    if _tmpl is None:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(m):
            raise ValueError(m)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        _tmpl = env.from_string(open(TEMPLATE).read())
    kw = {"enable_thinking": False} if mode == "off" else {"reasoning_effort": EFFORT}
    return _tmpl.render(messages=messages, add_generation_prompt=True, **kw) + prefix


def answer_part(text):
    """Text after the reasoning block (thinking-on outputs start inside <think>)."""
    i = text.rfind("</think>")
    return text[i + len("</think>"):] if i >= 0 else text


# ---------------------------------------------------------------- benchmarks
LETTERS = "ABCDEFGHIJ"


def mmlupro_items(mode, sub=None):
    import pandas as pd

    df = pd.read_parquet(f"{DATA}/mmlu_pro/data/test-00000-of-00001.parquet")
    rows = df.to_dict("records")
    if sub == "2k":    # fixed stratified 2,000: categories in proportion to their size (largest remainder), seed 0
        rng = random.Random(0)
        by = {}
        for r in rows:
            by.setdefault(r["category"], []).append(r)
        quota = {c: 2000 * len(v) / len(rows) for c, v in by.items()}
        take = {c: int(q) for c, q in quota.items()}
        for c in sorted(quota, key=lambda c: quota[c] - take[c], reverse=True)[: 2000 - sum(take.values())]:
            take[c] += 1
        keep = {r["question_id"] for c in sorted(by) for r in rng.sample(by[c], take[c])}
        rows = [r for r in rows if r["question_id"] in keep]
    elif mode == "on":   # fixed stratified subset: 20 per category, seed 0
        rng = random.Random(0)
        by = {}
        for r in rows:
            by.setdefault(r["category"], []).append(r)
        rows = [r for c in sorted(by) for r in sorted(rng.sample(by[c], 20), key=lambda r: r["question_id"])]
    items = []
    for r in rows:
        # TIGER-Lab MMLU-Pro evaluate_from_api.py instruction and format_example, zero-shot
        q = (f"The following are multiple choice questions (with answers) about {r['category']}. Think step by step and then "
             f"output the answer in the format of \"The answer is (X)\" at the end.\n\n")
        q += f"Question: {r['question']}\nOptions: " + "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(r["options"]))
        q += "Answer: Let's think step by step."
        items.append({"id": int(r["question_id"]), "category": r["category"], "gold": r["answer"], "prompt": chat([{"role": "user", "content": q}], mode)})
    return items


def mmlupro_extract(text):
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if m:
        return m.group(1)
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return m.group(0) if m else None


GPQA_T = ("Answer the following multiple choice question. The last line of your response should be of the following format: "
          "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
          "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")


def gpqa_items(mode):
    rows = list(csv.DictReader(open(f"{DATA}/gpqa_diamond.csv")))
    rng = random.Random(0)   # openai/simple-evals gpqa_eval.py: one permutation per row from Random(0)
    items = []
    for i, r in enumerate(rows):
        perm = rng.sample(range(4), 4)
        ch = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        ch = [ch[j] for j in perm]
        gold = "ABCD"[ch.index(r["Correct Answer"])]
        q = GPQA_T.format(Question=r["Question"], A=ch[0], B=ch[1], C=ch[2], D=ch[3])
        items.append({"id": i, "gold": gold, "prompt": chat([{"role": "user", "content": q}], mode)})
    return items


def gpqa_extract(text):
    m = re.search(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?", text)
    return m.group(1) if m else None


def ifeval_items(mode):
    rows = jl(f"{DATA}/ifeval/ifeval_input_data.jsonl")
    return [dict(r, id=r["key"], prompt_text=r["prompt"], prompt=chat([{"role": "user", "content": r["prompt"]}], mode)) for r in rows]


MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def math500_items(mode):
    rows = jl(f"{DATA}/math500/test.jsonl")
    return [{"id": r["unique_id"], "gold": r["answer"], "subject": r["subject"], "level": r["level"],
             "prompt": chat([{"role": "user", "content": r["problem"] + MATH_SUFFIX}], mode)} for r in rows]


# EvalPlus codegen instruct prompt (evalplus/provider/utility.py make_raw_chat_prompt)
EP_INSTR = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
EP_RESP = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"


def evalplus_items(bench, mode):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    ds = get_human_eval_plus() if bench == "humaneval" else get_mbpp_plus()
    items = []
    for tid, t in ds.items():
        user = f"{EP_INSTR}\n```\n{t['prompt'].strip()}\n```\n"
        items.append({"id": tid, "entry_point": t["entry_point"], "prompt": chat([{"role": "user", "content": user}], mode, f"{EP_RESP}\n```" if mode == "off" else "")})
    return items


LCB = "/root/w/lcb-repo"   # LiveCodeBench repo; its prompt module reads few-shot files relative to the repo root


def lcb_problems():
    sys.path.insert(0, LCB)
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    return [CodeGenerationProblem(**r) for r in jl(f"{DATA}/lcb/test6.jsonl")]


def lcb_items(mode):
    sys.path.insert(0, LCB)
    os.chdir(LCB)
    from lcb_runner.prompts.code_generation import format_prompt_generation
    from lcb_runner.lm_styles import LMStyle

    items = []
    for p in lcb_problems():
        msgs = format_prompt_generation(p, LMStyle.OpenAIChat)   # generic chat: system + user
        items.append({"id": p.question_id, "difficulty": p.difficulty.value, "date": str(p.contest_date)[:10], "prompt": chat(msgs, mode)})
    return items


def mmlupro2k_items(mode):
    return mmlupro_items(mode, "2k")


def items_for(bench, mode):
    if bench in ("humaneval", "mbpp"):
        return evalplus_items(bench, mode)
    return globals()[f"{bench}_items"](mode)


# ---------------------------------------------------------------- scoring
def se(p, n):
    return math.sqrt(max(p * (1 - p), 0) / n) if n else float("nan")


def summarize(name, correct, extra=None):
    n = len(correct)
    p = sum(correct) / n
    out = {"metric": name, "n": n, "score": 100 * p, "se": 100 * se(p, n)}
    out.update(extra or {})
    return out


def score(bench, mode, items, outs, workdir, tag):
    texts = [answer_part(o) for o in outs]
    per = []      # per-item correctness (primary metric) for paired comparison
    res = {}
    if bench in ("mmlupro", "mmlupro2k", "gpqa"):
        ex = mmlupro_extract if bench.startswith("mmlupro") else gpqa_extract
        preds = [ex(t) for t in texts]
        per = [int(p == it["gold"]) for p, it in zip(preds, items)]
        res = summarize("accuracy", per, {"no_answer": sum(p is None for p in preds)})
        if bench.startswith("mmlupro"):
            cats = {}
            for c, it in zip(per, items):
                cats.setdefault(it["category"], []).append(c)
            res["by_category"] = {k: round(100 * sum(v) / len(v), 2) for k, v in sorted(cats.items())}
    elif bench == "math500":
        from math_verify import parse, verify

        for t, it in zip(texts, items):
            try:
                ok = verify(parse("$" + it["gold"] + "$"), parse(t))
            except Exception:
                ok = False
            per.append(int(bool(ok)))
        res = summarize("accuracy", per)
    elif bench == "ifeval":
        from lm_eval.tasks.ifeval.utils import process_results

        ps, pl, is_, il = [], [], [], []
        for t, it in zip(texts, items):
            doc = {"key": it["key"], "prompt": it["prompt_text"], "instruction_id_list": it["instruction_id_list"], "kwargs": it["kwargs"]}
            r = process_results(doc, [t])
            ps.append(int(r["prompt_level_strict_acc"])); pl.append(int(r["prompt_level_loose_acc"]))
            is_ += r["inst_level_strict_acc"]; il += r["inst_level_loose_acc"]
        per = ps
        res = {"prompt_strict": summarize("prompt_strict", ps), "prompt_loose": summarize("prompt_loose", pl),
               "inst_strict": summarize("inst_strict", [int(x) for x in is_]), "inst_loose": summarize("inst_loose", [int(x) for x in il]),
               "n": len(ps), "score": 100 * sum(ps) / len(ps), "se": 100 * se(sum(ps) / len(ps), len(ps)), "metric": "prompt_strict"}
    elif bench in ("humaneval", "mbpp"):
        from evalplus.sanitize import sanitize

        samples = os.path.join(workdir, f"{tag}.samples.jsonl")
        with open(samples, "w") as f:
            for o, it in zip(outs, items):
                code = sanitize("```" + o if mode == "off" else answer_part(o), entrypoint=it["entry_point"])
                f.write(json.dumps({"task_id": it["id"], "solution": code}) + "\n")
        rj = samples.replace(".jsonl", "_eval_results.json")
        if os.path.exists(rj):
            os.remove(rj)
        subprocess.run(["evalplus.evaluate", "--dataset", bench, "--samples", samples, "--parallel", str(os.cpu_count()), "--i-just-wanna-run"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ev = json.load(open(rj))["eval"]
        base = [int(ev[it["id"]][0]["base_status"] == "pass") for it in items]
        plus = [int(ev[it["id"]][0]["base_status"] == "pass" and ev[it["id"]][0]["plus_status"] == "pass") for it in items]
        per = plus
        res = {"plus": summarize("pass@1 (plus)", plus), "base": summarize("pass@1 (base)", base),
               "n": len(plus), "score": 100 * sum(plus) / len(plus), "se": 100 * se(sum(plus) / len(plus), len(plus)), "metric": "pass@1 plus tests"}
    elif bench == "lcb":
        sys.path.insert(0, LCB)
        from lcb_runner.utils.extraction_utils import extract_code
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

        probs = {p.question_id: p for p in lcb_problems()}
        samples = [probs[it["id"]].get_evaluation_sample() for it in items]
        gens = [[extract_code(t, LMStyle.OpenAIChat)] for t in texts]
        m, results, _ = codegen_metrics(samples, gens, k_list=[1], num_process_evaluate=os.cpu_count(), timeout=6)
        per = [int(all(x is True or x == 1 for x in results[i][0])) for i in range(len(items))]
        res = summarize("pass@1", per)
        bd = {}
        for c, it in zip(per, items):
            bd.setdefault(it["difficulty"], []).append(c)
        res["by_difficulty"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(bd.items())}
    return res, per


def main():
    cmd = sys.argv[1]
    if cmd == "prep":
        bench, mode, d = sys.argv[2:5]
        os.makedirs(d, exist_ok=True)
        items = items_for(bench, mode)
        write_bin(f"{d}/{bench}-{mode}.prompts.bin", [it["prompt"] for it in items])
        with open(f"{d}/{bench}-{mode}.items.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps({k: v for k, v in it.items() if k != "prompt"}, default=str) + "\n")
        print(f"{bench}-{mode}: {len(items)} prompts")
    elif cmd == "fork":
        prompts = read_bin(sys.argv[2])
        out, url, par, maxnew = sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])

        def one(p):
            body = json.dumps({"prompt": p, "n_predict": maxnew, "temperature": 0, "top_k": 1, "cache_prompt": False, "special": False}).encode()
            err = None
            for _ in range(5):
                try:
                    r = urllib.request.urlopen(urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"}), timeout=36000)
                    j = json.loads(r.read())
                    return j["content"], j.get("tokens_evaluated", 0), j.get("tokens_predicted", 0), int(j.get("stop_type", "") == "eos" or j.get("stopped_eos", False))
                except Exception as e:
                    err = e
                    time.sleep(2)
            raise err

        t0 = time.time()
        # longest prompts first would bias timing; keep file order like h5
        with ThreadPoolExecutor(par) as ex:
            res = list(ex.map(one, prompts))
        dt = time.time() - t0
        ntok = sum(r[2] for r in res)
        print(f"fork: {len(prompts)} prompts, {ntok} tokens in {dt:.1f} s = {ntok / dt:.0f} tok/s at {par} parallel")
        write_bin(out, [r[0] for r in res])
        with open(out + ".meta", "w") as f:
            for r in res:
                f.write(f"{r[1]} {r[2]} {r[3]}\n")
        with open(out + ".timing.json", "w") as f:
            json.dump({"engine": "fork", "prompts": len(prompts), "gen_tokens": ntok, "wall_s": dt, "tok_s": ntok / dt, "parallel": par, "maxnew": maxnew}, f)
    elif cmd == "score":
        bench, mode, d, out = sys.argv[2:6]
        items = jl(f"{d}/{bench}-{mode}.items.jsonl")
        outs = read_bin(out)
        assert len(outs) == len(items), (len(outs), len(items))
        tag = os.path.basename(out).replace(".bin", "")
        res, per = score(bench, mode, items, outs, d, tag)
        if os.path.exists(out + ".meta"):
            meta = [list(map(int, l.split())) for l in open(out + ".meta")]
            res["gen_tokens_mean"] = sum(m[1] for m in meta) / len(meta)
            res["truncated"] = sum(1 - m[2] for m in meta)
        if len(sys.argv) > 6:
            other_per = json.load(open(sys.argv[6] + ".score.json"))["per_item"]
            dlt = [a - b for a, b in zip(per, other_per)]
            n = len(dlt); mu = sum(dlt) / n
            sd = math.sqrt(sum((x - mu) ** 2 for x in dlt) / (n - 1))
            res["vs_other"] = {"other": sys.argv[6], "diff": 100 * mu, "paired_se": 100 * sd / math.sqrt(n), "same_correctness": sum(x == 0 for x in dlt)}
        res["bench"], res["mode"], res["file"] = bench, mode, out
        json.dump(dict(res, per_item=per), open(out + ".score.json", "w"))
        print(json.dumps({k: v for k, v in res.items() if k != "per_item"}))


if __name__ == "__main__":
    main()
