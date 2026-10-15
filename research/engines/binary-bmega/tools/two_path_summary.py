#!/usr/bin/env python3
"""Summarize complete paid matrices, including warm-up and repeated-baseline controls."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('jsonl', type=Path)
p.add_argument('--prefix', type=Path, help='pathmatrix output prefix, for token divergence comparison')
p.add_argument('--cap', type=int, default=192)
a = p.parse_args()
rows = [json.loads(x) for x in a.jsonl.read_text().splitlines() if x.startswith('{')]
cases = {}
for r in rows:
    key = (r.get('run', 'single'), r['paths'], r['depth'])
    cases.setdefault(key, {'requests': [], 'summary': None})
    if r.get('summary'):
        cases[key]['summary'] = r
    else:
        cases[key]['requests'].append(r)
legacy = all(k[0] == 'single' for k in cases)
expected = {('single',1,8),('single',2,4),('single',2,8)} if legacy else {
    ('base-before',1,8),('eight',2,4),('sixteen',2,8),('base-after',1,8)}
measured = {k for k in cases if not k[0].startswith('warm')}
if measured != expected or any(cases[k]['summary'] is None for k in measured):
    raise SystemExit('Incomplete matrix: no paid decision from partial output')
bk = ('single',1,8) if legacy else ('base-before',1,8)
base = {r['request']:r for r in cases[bk]['requests']}

def token_file(key):
    run,paths,depth=key
    return Path(str(a.prefix)+(f'-{run}' if run!='single' else '')+f'-p{paths}-d{depth}.bin.tokens')

def tokens(key):
    return [[int(t) for t in line.split()] for line in token_file(key).read_text().splitlines()]

base_tokens = tokens(bk) if a.prefix else None
result=[]
for key,case in sorted(cases.items()):
    if key[0].startswith('warm'):continue
    reqs=case['requests']
    if {r['request'] for r in reqs} != set(base):raise SystemExit('Request sets do not match')
    matched=all(r['decode_tokens']==base[r['request']]['decode_tokens'] for r in reqs)
    item=dict(case['summary'])
    item.update(run=key[0],requests=len(reqs),matched_output_lengths=matched,
                capped_requests=sum(r['generated']>=a.cap for r in reqs),
                clone_s=sum(r['clone_s'] for r in reqs),
                clone_ms_per_round=1000*sum(r['clone_s'] for r in reqs)/max(case['summary']['rounds'],1),
                alternate_rescues=sum(r['alternate_rescues'] for r in reqs),
                measured_tok_s_ratio=case['summary']['tok_s']/cases[bk]['summary']['tok_s'])
    if matched:
        item['matched_length_paid_speedup_vs_base_before']=cases[bk]['summary']['decode_s']/case['summary']['decode_s']
        item['per_request_time_speedups']=[base[r['request']]['decode_s']/r['decode_s'] for r in reqs]
    else:
        item['warning']='Output lengths differ; token-rate ratio is not a matched-output latency win'
        same=[r for r in reqs if r['decode_tokens']==base[r['request']]['decode_tokens']]
        if same:
            candidate_s=sum(r['decode_s'] for r in same)
            item['matched_length_subset']={
                'requests':[r['request'] for r in same],
                'decode_tokens':sum(r['decode_tokens'] for r in same),
                'candidate_decode_s':candidate_s,
                'baseline_before_decode_s':sum(base[r['request']]['decode_s'] for r in same),
                'paid_speedup_vs_base_before':sum(base[r['request']]['decode_s'] for r in same)/candidate_s,
                'note':'Equal lengths, not equal generated content or quality.'}
    ak=('base-after',1,8)
    if ak in cases:
        after={r['request']:r for r in cases[ak]['requests']}
        if all(r['decode_tokens']==after[r['request']]['decode_tokens'] for r in reqs):
            item['matched_length_paid_speedup_vs_base_after']=cases[ak]['summary']['decode_s']/case['summary']['decode_s']
        elif 'matched_length_subset' in item:
            subset=item['matched_length_subset']
            if all(base[i]['decode_tokens']==after[i]['decode_tokens'] for i in subset['requests']):
                subset['baseline_after_decode_s']=sum(after[i]['decode_s'] for i in subset['requests'])
                subset['paid_speedup_vs_base_after']=subset['baseline_after_decode_s']/subset['candidate_decode_s']
    if base_tokens is not None:
        got=tokens(key)
        if len(got)!=len(base_tokens):raise SystemExit('Token dump request count does not match')
        divergence=[]
        for ref,cand in zip(base_tokens,got):
            same=0
            while same<min(len(ref),len(cand)) and ref[same]==cand[same]:same+=1
            divergence.append({'common_prefix':same,'baseline_length':len(ref),'candidate_length':len(cand),
                               'position_agreement':sum(x==y for x,y in zip(ref,cand))/max(len(ref),len(cand),1)})
        item['token_comparison_vs_base_before']=divergence
    result.append(item)
print(json.dumps(result,indent=2))
