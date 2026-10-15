#!/usr/bin/env python3
"""Compare fixed-work replay outputs and native host intervals, excluding prefill."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(prefix):
    predictions, seconds = [], []
    with Path(str(prefix) + '.csv').open() as stream:
        for row in csv.reader(stream):
            round_id, rows = int(row[0]), int(row[1])
            if len(row) != rows + 3 or rows <= 0:
                raise ValueError('invalid prediction row')
            if round_id == 0:
                continue
            predictions.append((round_id, rows, tuple(map(int, row[3:]))))
            seconds.append(float(row[2]))
    if not seconds:
        raise ValueError('no decode samples')
    result = {'decode_samples': len(seconds), 'decode_mean_ms': statistics.mean(seconds) * 1000,
              'decode_median_ms': statistics.median(seconds) * 1000}
    host = Path(str(prefix) + '-host.csv')
    if host.exists():
        with host.open() as stream:
            samples = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        decode = [s for s in samples if s['round'] != 0]
        if len(decode) != len(seconds):
            raise ValueError('host/prediction sample count mismatch')
        for s in samples:
            if s['total_ns'] != sum(s[k] for k in ('prepare_ns', 'submit_ns', 'sync_ns', 'finish_ns')):
                raise ValueError('host decomposition does not sum')
            if s['launch_wall_ns'] > s['submit_ns']:
                raise ValueError('launch is not contained in submission')
        result['route_counts'] = {str(r): sum(s['route'] == r for s in decode) for r in range(4)}
        cached = [s for s in decode if s['route'] == 2]
        if cached:
            result['cached_shapes'] = sorted({tuple(s[k] for k in ('kind','a','b','c','d','e','f','g','nodes','roots')) for s in cached})
            memory_fields = ('rewrite_ns', 'memory_copies', 'memory_clears', 'copy_bytes', 'clear_bytes', 'host_read_bytes', 'host_write_bytes')
            presence = [field in cached[0] for field in memory_fields]
            if any(presence) and not all(presence):
                raise ValueError('incomplete graph-memory metadata schema')
            if all(presence):
                result['memory_shapes'] = sorted({tuple(s[k] for k in ('kind','a','b','c','d','e','f','g') + memory_fields) for s in cached})
            for metric in ('prepare_ns', 'submit_ns', 'launch_wall_ns', 'launch_thread_ns', 'sync_ns', 'finish_ns', 'total_ns'):
                values = [s[metric] for s in cached]
                result[metric.replace('_ns', '_us')] = {'mean': statistics.mean(values) / 1000,
                                                        'median': statistics.median(values) / 1000,
                                                        'max': max(values) / 1000}
    return result, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', nargs='+', type=Path)
    args = parser.parse_args()
    reference = None
    report = {}
    for prefix in args.prefixes:
        summary, predictions = summarize(prefix)
        if reference is None:
            reference = predictions
        summary['predictions_equal_first'] = predictions == reference
        summary['work_equal_first'] = [(r, n) for r, n, _ in predictions] == [(r, n) for r, n, _ in reference]
        report[str(prefix)] = summary
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
