#!/usr/bin/env python3
"""Target norm-prep publication warp tails without changing the SM-lit record ABI.

Usage: smlit_norm_publication.py DUMP EXACT_BINARY_SOURCE_DIR OUTPUT_JSON
Marks0..3: row loaded, norm barrier, Hadamard exchange, amax barrier.
Scope exit minus mark3 includes quantization/stores and any publication work.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from smlit_analyze import load, tag_names


def stats(values):
    if not len(values):
        raise ValueError('empty warp cohort')
    return {'n': len(values), 'mean_us': float(np.mean(values)), 'p50_us': float(np.median(values)),
            'p95_us': float(np.percentile(values, 95)), 'p99_us': float(np.percentile(values, 99)),
            'max_us': float(np.max(values))}


def analyze(dump, source):
    tags = [tag for tag, name in tag_names(source).items() if name == 'k_prep2w']
    if len(tags) != 1:
        raise ValueError('exact source must identify one k_prep2w tag')
    first, bounds, all_records = load(dump)
    long = all_records['cx'] > 20000
    if not np.any(long):
        raise ValueError('no long records for clock calibration')
    ns = float(np.median((all_records['t1'][long] - all_records['t0'][long]) / all_records['cx'][long]))
    if not np.isfinite(ns) or ns <= 0:
        raise ValueError('invalid clock calibration')
    records = all_records[all_records['tag'] == tags[0]]
    if not len(records):
        raise ValueError('no norm records for this exact source tag; wrong source or workload')
    marks = records['mk'].astype(np.int64)
    if np.any(marks == 0) or np.any(np.diff(marks, axis=1) < 0):
        raise ValueError('missing/nonmonotonic norm marks')
    cw, cx = records['cw'].astype(np.int64), records['cx'].astype(np.int64)
    segments = np.array([marks[:, 0] - cw, marks[:, 1] - marks[:, 0], marks[:, 2] - marks[:, 1],
                         marks[:, 3] - marks[:, 2], cx - marks[:, 3]]).T * ns / 1000
    if np.any(segments < 0):
        raise ValueError('negative norm phase interval')
    names = ['wait_to_row_loaded', 'norm_barrier', 'hadamard_exchange', 'amax_barrier', 'quant_store_publication_tail']
    tail = segments[:, 4]
    result = {
        'norm_tag': tags[0], 'ns_per_cycle': ns, 'clock_ghz': 1 / ns,
        'norm_source_sha256': hashlib.sha256((Path(source) / 'kernels.cu').read_bytes()).hexdigest(),
        'segments': {name: stats(segments[:, i]) for i, name in enumerate(names)},
        'publication_tail': {'warp0': stats(tail[records['warp'] == 0]),
                             'other_warps': stats(tail[records['warp'] != 0])},
        'rounds': [],
    }
    lo = 0
    for index, hi in enumerate(bounds):
        round_records = all_records[lo:int(hi)]
        lo = int(hi)
        r = round_records[round_records['tag'] == tags[0]]
        t = (r['cx'].astype(np.int64) - r['mk'][:, 3].astype(np.int64)) * ns / 1000
        result['rounds'].append({'round': first + index,
                                 'warp0': stats(t[r['warp'] == 0]),
                                 'other_warps': stats(t[r['warp'] != 0])})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        args.output.write_text(json.dumps(analyze(args.dump, args.source), indent=2) + '\n')
    except (ValueError, OSError) as error:
        parser.exit(1, f'norm publication: {error}\n')


if __name__ == '__main__':
    main()
