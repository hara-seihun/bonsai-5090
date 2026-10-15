#!/usr/bin/env python3
"""Recover analysis from a complete recorded panel after JSON numeric overflow.
No model calls, source-manifest edits, or overwritten inference records.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--point', type=Path, required=True)
    ap.add_argument('--failure-log', type=Path, required=True)
    ap.add_argument('--runner', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    point, runner, out = a.point.resolve(), a.runner.resolve(), a.output.resolve()
    sys.path.insert(0, str(runner.parent))
    spec = importlib.util.spec_from_file_location('recovery_runner', runner)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    initial = json.loads((point/'summary.json').read_text())
    failure = json.loads(a.failure_log.read_text().splitlines()[-1])
    if initial['state'] != 'running' or failure.get('error', {}).get('code') != 'invalid_input_or_io' or 'Out of range float values' not in failure['error']['detail']:
        raise ValueError('This recovery requires the explicit recorded serialization failure, not arbitrary incomplete inference')
    gold = json.loads((point.parent/'selection.json').read_text())
    rows = [json.loads(s) for s in (point/'requests.jsonl').read_text().splitlines()]
    if len(rows) != mod.N or len(gold) != mod.N or [r['prompt_index'] for r in rows] != list(range(mod.N)):
        raise ValueError('No full sequential panel recorded: cannot recover complete score')
    if [r['source_index'] for r in rows] != [g['source_index'] for g in gold]:
        raise ValueError('Question pairing mismatch')
    scores = [mod.score(r, g) for r, g in zip(rows, gold)]
    overflow = [s['source_index'] for s in scores if s['prediction']['status'] == 'nonfinite']
    if not overflow:
        raise ValueError('No numeric overflow witness: this is not the claimed repair')
    if rows[-1]['metrics_after'].get('active') != 0:
        raise ValueError('Recorded final metric did not establish completed server request')
    identity = {'schema': 1, 'kind': 'posthoc-serialization-recovery',
                'inference_repeated': False, 'recovered_at_unix': time.time(),
                'original_summary_sha256': mod.sha(point/'summary.json'),
                'request_records_sha256': mod.sha(point/'requests.jsonl'),
                'failure_log_sha256': mod.sha(a.failure_log), 'repair_runner_sha256': mod.sha(runner),
                'nonfinite_prediction_source_ids': overflow,
                'score_semantics': 'Original numeric scorer decision retained; nonfinite predictions remain wrong',
                'missing_fields': ['original cleanup exit status', 'original final GPU receipt', 'exact final task_wall_s']}
    result = {**initial, **mod.aggregate(rows, scores, True),
              'state': 'recovered-from-complete-recorded-panel', 'recovery': identity,
              'metrics_before': rows[0]['metrics_before'], 'metrics_after': rows[-1]['metrics_after']}
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Recovery output must be a new empty directory')
    mod.dump(out/'analysis.json', result)
    mod.dump(out/'score.json', {'correct': result['correct'], 'total': mod.N, 'accuracy': result['accuracy'], 'per_item': scores, 'recovery': identity})
    # Terminalize locally for future consumers, retaining the initial receipt verbatim.
    (out/'original-running-summary.json').write_bytes((point/'summary.json').read_bytes())
    terminal = {**initial, **mod.aggregate(rows, scores, False), 'state': 'failed',
                'error': {'code': 'numeric_prediction_serialization', 'detail': failure['error']['detail']},
                'inference_requests_completed': mod.N, 'recovery_analysis': str(out/'analysis.json'),
                'failed_question_ids': [], 'unattempted_question_ids': [], 'recovery': identity}
    mod.dump(point/'summary.json', terminal)
    print(json.dumps({'state': 'recovered', 'correct': result['correct'], 'total': mod.N, 'overflow_ids': overflow}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(json.dumps({'state': 'error', 'error': {'code': 'recovery_invalid', 'detail': str(e)}}), file=sys.stderr)
        sys.exit(1)
