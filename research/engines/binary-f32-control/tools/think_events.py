#!/usr/bin/env python3
"""think_events.py RESPONSES.jsonl OUTDIR ID[,ID...]: retokenize saved think_eval responses into gdn-state-drift STEP histories
(RESET 0; prompt in 512-token STEPs; one STEP per output token). Needs `tokenizers` and the Qwen3.8-27B tokenizer.json at /tmp/qwen38-tokenizer.json."""
import json, sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file("/tmp/qwen38-tokenizer.json")
src, outdir, ids = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
import os; os.makedirs(outdir, exist_ok=True)
for l in open(src):
    r = json.loads(l)
    if r["id"] not in ids: continue
    # prompt rendering: src/chat.h (no system message), thinking on
    msgs = r.get("messages")
    if msgs is None:
        sys.path.insert(0, "~/work/clones/eval-thinking/tools"); import think_eval as te
        d = te.load_data(); msgs = dict(te.items(r["bench"], d))[r["id"]]
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n<think>\n"
    p = tk.encode(prompt, add_special_tokens=False).ids
    c = tk.encode(r["content"], add_special_tokens=False).ids
    full = tk.encode(prompt + r["content"], add_special_tokens=False).ids
    ok = full == p + c
    with open(f"{outdir}/{r['id']}.events.txt", "w") as f:
        f.write("RESET 0\n")
        for s in range(0, len(p), 512):
            ch = p[s:s+512]
            f.write(f"STEP {len(ch)} " + " ".join(f"0 {t} {s+i}" for i, t in enumerate(ch)) + "\n")
        for i, t in enumerate(c):
            f.write(f"STEP 1 0 {t} {len(p)+i}\n")
    print(json.dumps({"id": r["id"], "prompt_tokens": len(p), "server_prompt_tokens": r["prompt_tokens"], "completion_tokens": len(c),
                      "server_completion_tokens": r["completion_tokens"], "concat_equals_joint": ok, "finish": r["finish_reason"]}))
