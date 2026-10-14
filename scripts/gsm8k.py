#!/usr/bin/env python3
"""GSM8K (all 1319 test items) against a running engine/serve.py, with the measured protocol.

Protocol (identical request bytes to the measured 1276/1319 run):
  * dataset: openai/gsm8k, config "main", split "test", revision 740312add88f781978c0658806c59bc2815b9866
  * prompt: Bonsai 2 chat template rendered with thinking on and reasoning effort "xhigh"; user turn is
    the question + "\\nPlease reason step by step, and put your final answer within \\boxed{}."
  * sampling: temperature 0.7, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1; max_tokens 16384
  * per-item seed: derived from the request bytes and campaign seed 20261014 (see request_seed)
  * scoring: text after </think>, parsed and compared with Math-Verify against the "####" gold answer.
    Unclosed reasoning or an empty final answer is wrong.

  python engine/serve.py --model MODEL --profile profiles/serving.json --slots 8 --ctx 20480 --port 18322 &
  python scripts/gsm8k.py --url http://127.0.0.1:18322 --out results/gsm8k
  python scripts/gsm8k.py --out results/gsm8k --score-only          # rescore saved outputs
  python scripts/gsm8k.py --print-request 0                         # show one exact request

Requires: pip install -r requirements.txt  (pyarrow, math-verify)
"""
import argparse, hashlib, json, sys, time, urllib.request
from pathlib import Path

DATASET_URL = ('https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/'
               'main/test-00000-of-00001.parquet')
DATASET_SHA256 = 'ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59'
SYSTEM = ('Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, '
          'consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.')
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
SAMPLING = {'temperature': 0.7, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0}
MAX_TOKENS = 16384
TEMPLATE_SHA256 = 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'  # GGUF chat template bytes
CAMPAIGN_SEED = 20261014


def canonical(v): return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
def digest(v): return hashlib.sha256(canonical(v).encode()).hexdigest()


def render(question):
    return (f'<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}{SUFFIX}<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n')


def request_seed(prompt):
    key = digest({'prompt': prompt, 'sampling': SAMPLING, 'max_tokens': MAX_TOKENS, 'template_sha256': TEMPLATE_SHA256})
    return int(digest({'request': key, 'repeat': 0, 'campaign_seed': CAMPAIGN_SEED})[:8], 16), digest({'request': key, 'repeat': 0})


def load_dataset(cache):
    path = cache / 'gsm8k-test.parquet'
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATASET_SHA256:
        raise SystemExit(f'{path}: unexpected sha256')
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    return [dict(index=i, question=r['question'], gold=r['answer'].split('####')[-1].strip()) for i, r in enumerate(rows)]


def body(item):
    prompt = render(item['question'])
    seed, rid = request_seed(prompt)
    return rid, {'model': 'bonsai', 'prompt': prompt, **SAMPLING, 'seed': seed, 'max_tokens': MAX_TOKENS, 'stream': False,
                 'special': True, 'stop': ['<|im_end|>', '<|endoftext|>'], 'cache_prompt': False}


def score(item, text):
    from math_verify import parse, verify
    if '</think>' not in text:
        return False
    final = text.split('</think>', 1)[1].strip()
    return bool(final) and bool(verify(parse(item['gold']), parse(final)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:18322')
    ap.add_argument('--out', type=Path, default=Path('results/gsm8k'))
    ap.add_argument('--batch', type=int, default=8, help='requests per /v1/batch/completions call (<= server --slots)')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--print-request', type=int)
    a = ap.parse_args()
    items = load_dataset(a.out / 'cache')
    if a.print_request is not None:
        print(json.dumps(body(items[a.print_request])[1], indent=2, ensure_ascii=False))
        return
    items = items[:a.limit] if a.limit else items
    outputs = a.out / 'outputs.jsonl'
    a.out.mkdir(parents=True, exist_ok=True)
    done = {}
    if outputs.exists():
        for line in outputs.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            done[r['index']] = r
    todo = [x for x in items if x['index'] not in done]
    if todo and not a.score_only:
        with open(outputs, 'a', encoding='utf-8') as f:
            for off in range(0, len(todo), a.batch):
                group = todo[off:off + a.batch]
                req = urllib.request.Request(a.url.rstrip('/') + '/v1/batch/completions',
                                             json.dumps({'requests': [body(x)[1] for x in group]}).encode(),
                                             {'Content-Type': 'application/json'})
                started = time.time()
                with urllib.request.urlopen(req, timeout=24 * 3600) as r:
                    res = json.load(r)
                if 'error' in res:
                    raise SystemExit('server error: ' + json.dumps(res['error']))
                for x, out in zip(group, res['results']):
                    choice = out['choices'][0]
                    rec = dict(index=x['index'], request_id=body(x)[0], seed=out['quality_campaign']['seed'],
                               text=choice['text'], finish_reason=choice['finish_reason'], usage=out['usage'])
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    done[x['index']] = rec
                f.flush()
                print(f'{len(done)}/{len(items)}  batch {time.time() - started:.0f}s', file=sys.stderr)
    scored = [dict(index=x['index'], correct=score(x, done[x['index']]['text']), finish=done[x['index']]['finish_reason'])
              for x in items if x['index'] in done]
    summary = dict(covered=len(scored), expected=len(items), correct=sum(r['correct'] for r in scored),
                   length_capped=sum(r['finish'] == 'length' for r in scored))
    summary['accuracy'] = summary['correct'] / summary['covered'] if scored else None
    (a.out / 'score.json').write_text(json.dumps(dict(summary=summary, rows=scored), indent=1) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
