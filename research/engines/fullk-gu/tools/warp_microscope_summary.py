#!/usr/bin/env python3
"""Distributions and the sampled critical end from one bounded microscope snapshot."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        raise ValueError('empty distribution')
    def quantile(p):
        at = (len(values)-1)*p
        lo = int(at)
        hi = min(lo+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(at-lo)
    return {'count': len(values), 'min': values[0], 'p50': quantile(.5),
            'p90': quantile(.9), 'p99': quantile(.99), 'max': values[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    args = parser.parse_args()
    prefix = str(args.prefix)
    meta = json.loads(Path(prefix+'-meta.json').read_text())
    with open(prefix+'-spans.csv', newline='') as file:
        spans = list(csv.DictReader(file))
    with open(prefix+'-phases.csv', newline='') as file:
        phases = list(csv.DictReader(file))
    if not spans or not phases:
        raise ValueError('missing sampled warp data')
    groups = defaultdict(list)
    by_warp = defaultdict(list)
    for phase in phases:
        if int(phase['end_ns']) < int(phase['begin_ns']):
            raise ValueError('invalid interval')
        groups[(phase['role'], phase['phase'])].append(phase)
        by_warp[(phase['cta'], phase['warp'])].append(phase)
    critical = max(spans, key=lambda span: int(span['end_ns']))
    origin = min(int(span['begin_ns']) for span in spans)
    latest_consumers = sorted((span for span in spans if span['role']=='consumer'),
                              key=lambda span: int(span['end_ns']), reverse=True)[:8]
    end_details = []
    for span in latest_consumers:
        intervals = by_warp[(span['cta'], span['warp'])]
        wait_ns = sum(int(p['end_ns'])-int(p['begin_ns']) for p in intervals if p['kind']=='wait_region')
        work_ns = sum(int(p['end_ns'])-int(p['begin_ns']) for p in intervals if p['kind']=='work_region')
        duration_ns = int(span['end_ns'])-int(span['begin_ns'])
        end_details.append({'cta': int(span['cta']), 'warp': int(span['warp']), 'sm': int(span['sm']),
                            'split': int(span['z']), 'duration_ns': duration_ns,
                            'end_from_sample_origin_ns': int(span['end_ns'])-origin,
                            'wait_region_ns': wait_ns, 'work_region_ns': work_ns,
                            'uncovered_ns': duration_ns-wait_ns-work_ns,
                            'largest_waits': sorted((p for p in intervals if p['kind']=='wait_region'),
                                                    key=lambda p: int(p['cycles']), reverse=True)[:4]})
    result = {'timing_and_scope': meta,
              'sampled_latest_warp': critical,
              'sampled_end_from_origin_ns': int(critical['end_ns'])-origin,
              'latest_sampled_consumers': end_details,
              'distributions': [{'role': role, 'phase': phase,
                                 'cycles': distribution([int(p['cycles']) for p in rows]),
                                 'globaltimer_ns': distribution([int(p['end_ns'])-int(p['begin_ns']) for p in rows])}
                                for (role, phase), rows in sorted(groups.items())],
              'interpretation': 'Wait regions include successful poll/acquire overhead. Work regions include scheduling/memory stalls. Uncovered time includes observer stores and loop bookkeeping. Latest sampled end is not necessarily the grid critical end. Combined full-barrier wait does not separate weight from activation completion.'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
