#!/usr/bin/env python3
"""Score retained single_bench JSONL against a frozen GSM8K selection."""
import argparse
import json
from pathlib import Path
from gsm8k import extract, num

p = argparse.ArgumentParser()
p.add_argument('selection', help='JSON array with question, answer and source_index')
p.add_argument('requests', help='single_bench requests.jsonl')
p.add_argument('--output', required=True)
a = p.parse_args()
gold = json.loads(Path(a.selection).read_text())
rows = [json.loads(line) for line in Path(a.requests).read_text().splitlines() if line]
seen = set(); scored = []
for row in rows:
    index = row['prompt_index']; repeat = row['repeat']
    if (repeat, index) in seen: raise ValueError('duplicate request index')
    seen.add((repeat, index))
    target = num(gold[index]['answer'].split('####')[-1])
    # Do not score hidden reasoning as the delivered answer.
    text = row['text'].split('</think>')[-1]
    if '<think>' in text: text = text.split('<think>')[0]
    answer = extract(text)
    scored.append(dict(repeat=repeat, prompt_index=index, source_index=gold[index].get('source_index'),
                       expected=target, predicted=answer,
                       correct=target is not None and answer is not None and abs(answer-target)<1e-6))
for repeat in {row['repeat'] for row in rows}:
    if {i for r,i in seen if r==repeat} != set(range(len(gold))): raise ValueError('incomplete selection')
if not scored: raise ValueError('no requests')
result = dict(correct=sum(row['correct'] for row in scored), total=len(scored),
              accuracy=sum(row['correct'] for row in scored)/len(scored), per_item=scored)
Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('correct','total','accuracy')}))
