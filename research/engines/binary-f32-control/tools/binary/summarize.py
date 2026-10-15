#!/usr/bin/env python3
import json
import math
from pathlib import Path
import re
import statistics
import sys

QUALITY = re.compile(r'^(prefill|decode):\s+(\d+) rows\s+KLD mean (\S+) max (\S+)\s+cos (\S+)\s+top1 (\d+)/(\d+)\s+ref_ppl (\S+) got_ppl (\S+) max_logit_error (\S+)$')
ARMS = ('reference', 'int8', 'fp4', 'packed1', 'binary1', 'binary1-two', 'packed1-f16', 'binary1-two-f16', 'packed1-lut8', 'packed1-acc4')


def collect(root):
    result = {'protocol': '128-token WikiText prefix; positions128..255 teacher-forced; five repetitions after one warmup', 'arms': {}}
    for arm in ARMS:
        record = {'state': 'not-measured'}
        p = root / f'{arm}-decodebench.json'
        if p.exists() and p.stat().st_size:
            data = json.loads(p.read_text())
            times = data['seconds']
            count = data['decode_tokens']
            if count <= 0 or not times or any(not math.isfinite(x) or x <= 0 for x in times):
                return {'status': 'error', 'reason': 'TIMING_INVALID', 'file': str(p)}
            rates = [count / x for x in times]
            record = {'state': 'measured', 'median_tps': count / statistics.median(times), 'mean_tps': statistics.mean(rates), 'sample_tps': rates, 'source': p.name}
        for panel in ('wiki', 'chat'):
            p = root / f'{arm}-{panel}-quality.log'
            if not p.exists():
                continue
            records = {}
            for line in p.read_text().splitlines():
                m = QUALITY.fullmatch(line)
                if m is None:
                    continue
                route, rows, kl, maxkl, cosine, correct, total, reference, candidate, error = m.groups()
                values = list(map(float, (kl, maxkl, cosine, reference, candidate, error)))
                if any(not math.isfinite(v) for v in values) or int(rows) != int(total) or int(rows) < 1 or not 0 <= int(correct) <= int(total):
                    return {'status': 'error', 'reason': 'QUALITY_INVALID', 'file': str(p)}
                records[route] = {'tokens': int(rows), 'kl': float(kl), 'max_kl': float(maxkl), 'top1_count': int(correct), 'top1_percent': 100 * int(correct) / int(total), 'reference_ppl': float(reference), 'candidate_ppl': float(candidate), 'ppl_relative_percent': 100 * (float(candidate) / float(reference) - 1), 'max_logit_error': float(error)}
            record[panel] = {'state': 'measured' if set(records) == {'prefill', 'decode'} else 'incomplete', 'routes': records, 'source': p.name}
        result['arms'][arm] = record
    ref = result['arms']['reference']
    if ref['state'] == 'measured':
        for record in result['arms'].values():
            if record['state'] == 'measured':
                record['vs_reference_ratio'] = record['median_tps'] / ref['median_tps']
    return {'status': 'ok', **result}


def main():
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
        print(json.dumps({'status': 'error', 'reason': 'RESULT_DIRECTORY_REQUIRED'}))
        return 1
    try:
        result = collect(Path(sys.argv[1]))
    except (OSError, ValueError, TypeError, KeyError) as e:
        result = {'status': 'error', 'reason': 'RESULT_READ_FAILED', 'detail': str(e)}
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
