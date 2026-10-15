#!/usr/bin/env python3
"""Summarise loop-safe lane thinking runs: loopsafe_summary.py DIR... (each DIR = loopsafe_think.py output).

Per run and bench: items, correct, cut (length or aborted loop), loops (aborted or periodic), degenerate,
completion tokens, tokens/round, decode tok/s (token-weighted), and the policy's veto counters from metrics-end.json.
"""
import json, sys
from collections import defaultdict
from pathlib import Path


def summarise(d):
    d = Path(d)
    items = [json.loads(l) for l in open(d / 'items.jsonl')] if (d / 'items.jsonl').exists() else []
    by = defaultdict(lambda: defaultdict(float))
    for i in items:
        for b in (i['bench'], 'ALL'):
            s = by[b]
            s['n'] += 1; s['correct'] += i['correct']
            s['cut'] += i['finish_reason'] in ('length', 'loop')
            s['loops'] += bool(i['loop_detected']); s['degen'] += bool(i['degenerate'])
            s['tokens'] += i['completion_tokens']
            s['secs'] += i['completion_tokens'] / max(i.get('decode_tok_per_s') or 1, 1)
            s['rounds'] += i['completion_tokens'] / max(i.get('tokens_per_round') or 1, 1e-9)
    m = {}
    if (d / 'metrics-end.json').exists():
        j = json.loads((d / 'metrics-end.json').read_text())
        m = {'tok/round': round(j['generated_tokens'] / max(j['verify_rounds'], 1), 2),
             'offered': j.get('relax_offered'), **{k: v for k, v in (j.get('relax_by_reason') or {}).items() if v}}
    return items, by, m


def main():
    for d in sys.argv[1:]:
        items, by, m = summarise(d)
        print(f'== {Path(d).name}  veto {m}')
        for b, s in sorted(by.items()):
            print(f'  {b:8s} n {int(s["n"]):3d} correct {int(s["correct"]):3d} cut {int(s["cut"]):2d} loops {int(s["loops"]):2d} '
                  f'degen {int(s["degen"]):2d} tokens {int(s["tokens"]):8d} '
                  f'tok/s {s["tokens"] / max(s["secs"], 1e-9):6.0f}')


if __name__ == '__main__':
    main()
