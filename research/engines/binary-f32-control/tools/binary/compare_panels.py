#!/usr/bin/env python3
"""Paired score/throughput receipts for scoped binary panels, not a full gate."""
import argparse
import json
import math
from pathlib import Path


def rows(path):
    values = [json.loads(line) for line in path.read_text().splitlines()]
    result = {r['id']: r for r in values}
    if len(result) != len(values):
        raise ValueError('DUPLICATE_ITEM_ID:' + str(path))
    return result


def timing(summary):
    t = summary['generation_timing']['session']
    if t['status'] != 'measured' or t['reused_n'] != 0:
        raise ValueError('FRESH_MEASURED_TIMING_REQUIRED')
    tokens, seconds = t['completion_tokens'], t['generation_s']
    if tokens <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('TIMING_INVALID')
    return {'completion_tokens': tokens, 'seconds': seconds, 'http_tok_s': tokens / seconds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reference', type=Path)
    ap.add_argument('candidate', type=Path)
    a = ap.parse_args()
    summaries = [json.loads((p / 'summary.json').read_text()) for p in (a.reference, a.candidate)]
    if summaries[0]['data_manifest_sha256'] != summaries[1]['data_manifest_sha256']:
        raise ValueError('DATA_MANIFEST_MISMATCH')
    reference_time, candidate_time = map(timing, summaries)
    record = {'scope': 'same-item scoped panel, not full quality gate',
              'reference': str(a.reference), 'candidate': str(a.candidate),
              'reference_timing': reference_time, 'candidate_timing': candidate_time,
              'http_ratio': candidate_time['http_tok_s'] / reference_time['http_tok_s'], 'benchmarks': {}}
    if set(summaries[0]['benchmarks']) != set(summaries[1]['benchmarks']):
        raise ValueError('BENCHMARK_SET_MISMATCH')
    for benchmark in summaries[0]['benchmarks']:
        left, right = [rows(p / benchmark / 'items.jsonl') for p in (a.reference, a.candidate)]
        if set(left) != set(right) or not left:
            raise ValueError('ITEM_SET_MISMATCH:' + benchmark)
        metrics = set(summaries[0]['benchmarks'][benchmark]['metrics'])
        if metrics != set(summaries[1]['benchmarks'][benchmark]['metrics']):
            raise ValueError('METRIC_SET_MISMATCH:' + benchmark)
        paired = {}
        for metric in metrics:
            lvalues, rvalues = [], []
            for key, l in left.items():
                r = right[key]
                if l['prompt_sha256'] != r['prompt_sha256']:
                    raise ValueError('PROMPT_MISMATCH:' + key)
                lv, rv = l['metrics'][metric], r['metrics'][metric]
                if lv not in (0, 1) or rv not in (0, 1):
                    raise ValueError('BINARY_SCORE_REQUIRED:' + key)
                lvalues.append(lv); rvalues.append(rv)
            paired[metric] = {'n': len(left), 'reference_correct': sum(lvalues), 'candidate_correct': sum(rvalues),
                              'gains': sum(r > l for l, r in zip(lvalues, rvalues)),
                              'losses': sum(r < l for l, r in zip(lvalues, rvalues))}
        record['benchmarks'][benchmark] = {'paired': paired, 'reference_truncated': summaries[0]['benchmarks'][benchmark]['truncated'],
                                          'candidate_truncated': summaries[1]['benchmarks'][benchmark]['truncated']}
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
