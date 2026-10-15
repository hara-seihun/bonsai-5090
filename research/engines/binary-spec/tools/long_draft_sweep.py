#!/usr/bin/env python3
"""Bounded, foreground native 7/16/32/64-proposal extrapolation screen."""
import argparse
import csv
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

from gsm8k import SUFFIX, extract, num

SOURCE: str  # Required explicit CLI provenance, set before any action.
SELECTION_SHA = 'ed454c8f72749157463d8b203ec32f7248854bdaab85273a1c71e0385d64aec9'
N = 32
GRID = ['0', '4', '8', '32']
SEED = 20261009
ROOT = Path(__file__).resolve().parents[1]
COUNTERS = ['requests', 'generated_tokens', 'steps', 'busy_s', 'verify_s', 'draft_s',
            'verify_rounds', 'accepted_drafts', 'proposed_drafts', 'graphs_captured', 'graph_replays']
TIMINGS = ['native_prefill_s', 'verify_s', 'draft_s', 'ttft_s', 'total_s', 'decode_tok_per_s',
           'proposed_drafts', 'accepted_drafts']


class SweepError(Exception):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail
        super().__init__(detail)

    def data(self):
        return {'code': self.code, 'detail': self.detail}


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def schedule():
    entries = [(7, '0', 'baseline-before'), (7, '16', 'seven-reference')]
    entries += [(b, '32', 'long-block') for b in (64, 32, 16)]
    entries += [(b, '16', 'long-block') for b in (64, 32, 16)]
    entries += [(7, '32', 'seven-reference')]
    for d in ('8', '4'):
        entries += [(b, d, 'long-block') for b in (64, 32, 16)]
        entries += [(7, d, 'seven-reference')]
    entries += [(7, '0', 'baseline-after')]
    # Separate v2 continuation: v1 panels0..6 are complete and panel7 retained all32
    # responses but failed serialization of an overflowing numeric prediction.
    entries = [(7, '0', 'baseline-before-repair')] + entries[8:]
    return [{'index': i, 'delta': d, 'draft_proposals': b, 'query_rows': b + 1,
             'kind': k, 'name': f'{i:02d}-proposals-{b}-delta-{d}'}
            for i, (b, d, k) in enumerate(entries)]


def init(out):
    selection = ROOT / 'results/long-draft/selection32.json'
    prompts = ROOT / 'results/long-draft/prompts32.json'
    if sha(selection) != SELECTION_SHA:
        raise SweepError('corpus_mismatch', str(selection))
    rows, strings = json.loads(selection.read_text()), json.loads(prompts.read_text())
    if len(rows) != N or strings != [r['question'] + SUFFIX for r in rows]:
        raise SweepError('prompt_mismatch', 'Frozen question subset and prompt template must match')
    manifest = {
        'schema': 1, 'protocol_group': 'continuation-v2-nonfinite-reporting',
        'prior_runner_commit': '955ab3818c6d35d53b3e8458a99b3b9748cb61a1',
        'prior_runner_sha256': 'f760dc8b6c83c8771f9045e50a4917104bcb8a69fbed547768eaf56536902909',
        'server_source_commit': SOURCE, 'selection_sha256': SELECTION_SHA,
        'prompts_sha256': sha(prompts), 'scorer_sha256': sha(ROOT / 'tools/gsm8k.py'),
        'score_policy': 'score_single_gsm.py numeric extraction, strip hidden thinking, tolerance 1e-6',
        'runner_sha256': sha(Path(__file__)), 'order_seed': SEED, 'unique_delta_grid': GRID,
        'schedule': schedule(), 'requests_per_point': N, 'warmup_tokens': 128,
        'warmup_prompt_index': 0, 'max_new_tokens': 768, 'slots': 1, 'ctx': 4096, 'prefill_budget': 128,
        'temperature': 0, 'thinking': False, 'stream': False, 'compact': 'dense5',
        'panel': 'Seeded32 subset of reused exploratory GSM8K screen; not untouched holdout',
        'checkpoint_extrapolation': 'Active DFlash block_size8; query lengths17/33/65 are extrapolation',
        'accuracy_gating': False, 'proposal_lengths': [7, 16, 32, 64],
        'transport': 'Nonstream exposes native timings. All new points/controls use it; old SSE points are unmatched historical references.'}
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        if json.loads((out / 'manifest.json').read_text()) != manifest:
            raise SweepError('manifest_conflict', 'Choose a new output directory; an existing sweep is immutable')
    else:
        dump(out / 'manifest.json', manifest)
        (out / 'selection.json').write_bytes(selection.read_bytes())
        (out / 'prompts.json').write_bytes(prompts.read_bytes())
    report(out)
    print(json.dumps(manifest), flush=True)


def remaining(deadline, cap):
    left = deadline - time.monotonic()
    if left <= 0:
        raise SweepError('deadline', 'Foreground chunk deadline reached')
    return min(cap, left)


def http_json(url, timeout, body=None):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode('utf-8')
        return json.loads(raw), raw
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read(4096).decode('utf-8', errors='replace')
        raise SweepError('http_error', f'{error.code}: {detail}') from error
    except (OSError, urllib.error.URLError, ValueError) as error:
        raise SweepError('http_transport_or_json', str(error)) from error


def valid_numbers(obj, keys, label):
    for key in keys:
        value = obj.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise SweepError('invalid_response', f'{label}.{key} is missing or invalid: {value!r}')


def metrics(url, deadline):
    value, _ = http_json(url + '/metrics', remaining(deadline, 3))
    valid_numbers(value, COUNTERS, 'metrics')
    if value.get('active') != 0:
        raise SweepError('server_not_idle', f'active={value.get("active")}')
    return value


def delta_metrics(before, after):
    values = {k: after[k] - before[k] for k in COUNTERS}
    if any(v < 0 for v in values.values()):
        raise SweepError('counter_reset', str(values))
    return values


def request(url, prompt, max_tokens, deadline, raw_path, proposals):
    body = {'model': 'ternary-bonsai-2-27b', 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': 0, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}}
    t0_epoch, t0 = time.time(), time.monotonic()
    response, raw = http_json(url + '/v1/chat/completions', remaining(deadline, 45), body)
    t1 = time.monotonic()
    raw_path.write_text(raw + '\n')
    try:
        text = response['choices'][0]['message']['content']
        finish = response['choices'][0]['finish_reason']
        usage, timings = response['usage'], response['timings']
    except (KeyError, IndexError, TypeError) as error:
        raise SweepError('invalid_response', str(error)) from error
    if not isinstance(text, str) or finish not in ('stop', 'length'):
        raise SweepError('invalid_response', f'Unexpected text/finish_reason: {finish!r}')
    valid_numbers(usage, ['prompt_tokens', 'completion_tokens', 'total_tokens'], 'usage')
    valid_numbers(timings, TIMINGS, 'timings')
    if timings.get('draft_proposals') != proposals or timings.get('draft_query_rows') != proposals + 1 or timings.get('verifier_rows_per_full_round') != proposals + 1:
        raise SweepError('draft_shape_mismatch', 'Response length metadata differs from configured arm')
    if timings.get('bonus_policy') != 'greedy' or timings.get('checkpoint_length_extrapolation') is not (proposals != 7):
        raise SweepError('draft_policy_mismatch', 'Response bonus/extrapolation metadata differs from protocol')
    arrays = [('prefix_histogram', proposals + 1), ('position_proposed', proposals),
              ('position_reached', proposals), ('position_accepted', proposals),
              ('position_emitted', proposals + 1), ('position_reject_target_eos', proposals),
              ('position_reject_proposed_eos', proposals), ('position_reject_other', proposals)]
    for key, length in arrays:
        value = timings.get(key)
        if not isinstance(value, list) or len(value) != length or any(type(v) is not int or v < 0 for v in value):
            raise SweepError('invalid_prefix_evidence', key)
    valid_numbers(timings, ['verify_rounds', 'verifier_rows'], 'timings')
    if sum(timings['prefix_histogram']) != timings['verify_rounds']:
        raise SweepError('invalid_prefix_evidence', 'Histogram total differs from verification rounds')
    if sum(i*v for i,v in enumerate(timings['prefix_histogram'])) != timings['accepted_drafts']:
        raise SweepError('invalid_prefix_evidence', 'Histogram accepted total differs from accepted drafts')
    for p in range(proposals):
        if not timings['position_accepted'][p] <= timings['position_reached'][p] <= timings['position_proposed'][p]:
            raise SweepError('invalid_prefix_evidence', f'Invalid reached/accepted position {p}')
    tokens = usage['completion_tokens']
    if not isinstance(tokens, int) or tokens < 0 or tokens > max_tokens:
        raise SweepError('invalid_response', f'completion_tokens={tokens!r}')
    ttft, server_total = timings['ttft_s'], timings['total_s']
    if server_total < ttft or (tokens > 1 and (server_total == ttft or timings['decode_tok_per_s'] <= 0)):
        raise SweepError('invalid_response', 'Invalid server decode interval')
    return {'state': 'ok', 't0': t0_epoch, 't1': t0_epoch + t1 - t0,
            'request_s': t1 - t0, 'ttft_s': ttft, 'decode_s': server_total - ttft if tokens else 0,
            'server_total_s': server_total, 'native_prefill_s': timings['native_prefill_s'],
            'native_verify_decode_s': timings['verify_s'], 'native_draft_s': timings['draft_s'],
            'decode_tok_s': timings['decode_tok_per_s'], 'ntok': tokens, 'usage': usage,
            'finish_reason': finish, 'eos': finish == 'stop', 'cap': finish == 'length',
            'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'timings': timings, 'raw_response': str(raw_path.name)}


def score(row, gold):
    target = num(gold['answer'].split('####')[-1])
    text = row['text'].split('</think>')[-1]
    if '<think>' in text:
        text = text.split('<think>')[0]
    answer = extract(text)
    if target is None or not math.isfinite(target):
        raise SweepError('invalid_gold_number', str(gold['source_index']))
    correct = answer is not None and abs(answer - target) < 1e-6
    if answer is None:
        prediction = {'status': 'missing', 'value': None}
    elif not math.isfinite(answer):
        prediction = {'status': 'nonfinite', 'value': None, 'parsed_representation': repr(answer),
                      'error': {'code': 'numeric_prediction_overflow', 'detail': 'Raw response retained; original score is wrong'}}
    else:
        prediction = {'status': 'finite', 'value': answer}
    return {'repeat': 0, 'prompt_index': row['prompt_index'], 'source_index': gold['source_index'],
            'expected': target, 'predicted': prediction['value'], 'prediction': prediction,
            'correct': correct}


def gpu_receipt():
    command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw', '--format=csv']
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'state': 'unavailable', 'error': str(error)}
    return {'state': 'ok' if result.returncode == 0 else 'error', 'command': command,
            'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}


def aggregate(rows, scores, complete):
    correct = sum(r['correct'] for r in scores)
    result = {'scored_requests': len(scores), 'partial_correct': correct,
              'correct': correct if complete else None, 'total': N,
              'accuracy': correct / N if complete else None,
              'wrong_question_ids': [r['source_index'] for r in scores if not r['correct']]}
    if not rows:
        return result
    tokens = sum(r['ntok'] for r in rows)
    request_s = sum(r['request_s'] for r in rows)
    decode_s = sum(r['decode_s'] for r in rows)
    native_prefill_s = sum(r['native_prefill_s'] for r in rows)
    counters = {k: sum(r['metrics_delta'][k] for r in rows) for k in COUNTERS}
    proposed, accepted = counters['proposed_drafts'], counters['accepted_drafts']
    rounds = counters['verify_rounds']
    result.update(requests=len(rows), tokens=tokens, request_wall_s=request_s,
                  end_to_end_tok_s=tokens / request_s,
                  weighted_decode_tok_s=sum(max(0, r['ntok'] - 1) for r in rows) / decode_s if decode_s else None,
                  native_prefill_s=native_prefill_s, decode_s=decode_s,
                  native_verify_decode_s=sum(r['native_verify_decode_s'] for r in rows),
                  native_draft_s=sum(r['native_draft_s'] for r in rows),
                  median_decode_tok_s=statistics.median(r['decode_tok_s'] for r in rows),
                  median_ttft_s=statistics.median(r['ttft_s'] for r in rows),
                  median_request_s=statistics.median(r['request_s'] for r in rows),
                  p90_request_s=sorted(r['request_s'] for r in rows)[int(.9 * (len(rows) - 1))],
                  eos_count=sum(r['eos'] for r in rows), cap_count=sum(r['cap'] for r in rows),
                  proposed_drafts=proposed, accepted_drafts=accepted, verify_rounds=rounds,
                  draft_acceptance=accepted / proposed if proposed else None,
                  emitted_per_round=tokens / rounds if rounds else None, metrics_delta=counters)
    return result


def run(a):
    out = Path(a.output).resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['schedule'] != schedule() or manifest['server_source_commit'] != SOURCE:
        raise SweepError('manifest_mismatch', 'Schedule/source is not this frozen sweep')
    if sha(out / 'selection.json') != SELECTION_SHA or sha(out / 'prompts.json') != manifest['prompts_sha256']:
        raise SweepError('corpus_mismatch', str(out))
    if sha(Path(__file__)) != manifest['runner_sha256'] or sha(ROOT / 'tools/gsm8k.py') != manifest['scorer_sha256']:
        raise SweepError('runner_mismatch', 'Runner/scorer changed after sweep initialization')
    entry = manifest['schedule'][a.index]
    for prior in manifest['schedule'][:a.index]:
        receipt = out / prior['name'] / 'summary.json'
        if not receipt.exists() or json.loads(receipt.read_text())['state'] not in ('complete', 'failed'):
            raise SweepError('order_violation', f'Run index {prior["index"]} first, or explicitly record its failure')
    server = Path(a.server).resolve()
    model, drafter = Path(a.model).resolve(), Path(a.drafter).resolve()
    for path in (server, model, drafter):
        if not path.is_file():
            raise SweepError('missing_input', str(path))
    binary_sha = sha(server)
    for prior in manifest['schedule'][:a.index]:
        receipt = json.loads((out / prior['name'] / 'summary.json').read_text())
        if receipt['server_sha256'] != binary_sha:
            raise SweepError('binary_mismatch', 'Every point must use the identical frozen binary')
        for key, path in [('model', model), ('drafter', drafter)]:
            expected = {'path': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            if receipt[key] != expected:
                raise SweepError('asset_mismatch', f'{key} path/size/mtime changed during the sweep')
    point = out / entry['name']
    if point.exists():
        raise SweepError('point_already_exists', f'{point}; retained results are never overwritten')
    point.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith('H5_')}
    env.update(H5_COMPACT='dense5', H5_DRAFT_MARGIN=entry['delta'])
    command = [str(server), str(model), str(drafter), '--slots', '1', '--ctx', '4096', '--prefill', '128',
               '--draft-proposals', str(entry['draft_proposals']), '--port', str(a.port)]
    summary = {'state': 'running', **entry, 'server_source_commit': SOURCE,
               'server_sha256': binary_sha, 'runner_sha256': sha(Path(__file__)),
               'command': command, 'environment': {k: v for k, v in env.items() if k.startswith('H5_')},
               'host': socket.gethostname(), 'platform': platform.platform(),
               'model': {'path': str(model), 'bytes': model.stat().st_size, 'mtime_ns': model.stat().st_mtime_ns},
               'drafter': {'path': str(drafter), 'bytes': drafter.stat().st_size, 'mtime_ns': drafter.stat().st_mtime_ns},
               'started_at_unix': time.time(), 'deadline_s': a.deadline_s,
               'gpu_before': gpu_receipt()}
    dump(point / 'summary.json', summary)
    prompts, gold = json.loads((out / 'prompts.json').read_text()), json.loads((out / 'selection.json').read_text())
    url = f'http://127.0.0.1:{a.port}'
    deadline = time.monotonic() + a.deadline_s
    rows, scores, child, error, attempted_index = [], [], None, None, None
    signal.alarm(a.deadline_s)
    try:
        with socket.socket() as connection:
            connection.settimeout(1)
            status = connection.connect_ex(('127.0.0.1', a.port))
            if status == 0:
                raise SweepError('port_occupied', 'A live TCP listener owns the requested port')
            if status != errno.ECONNREFUSED:
                raise SweepError('port_probe_error', f'Unexpected TCP admission status: {status}')
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            try:
                probe.bind(('127.0.0.1', a.port))
            except OSError as failure:
                raise SweepError('port_occupied', str(failure)) from failure
        with (point / 'server.log').open('w') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            startup_deadline = min(deadline, time.monotonic() + 90)
            while True:
                if child.poll() is not None:
                    raise SweepError('server_exit', f'{child.returncode}; see server.log')
                try:
                    health, _ = http_json(url + '/health', remaining(startup_deadline, 1))
                    if health.get('status') not in ('ok',):
                        raise SweepError('health_response', str(health))
                    if (health.get('draft_proposals'), health.get('draft_query_rows'), health.get('verifier_rows')) != (entry['draft_proposals'], entry['query_rows'], entry['query_rows']):
                        raise SweepError('draft_shape_mismatch', str(health))
                    summary['health'] = health
                    break
                except SweepError as readiness:
                    if readiness.code != 'http_transport_or_json':
                        raise
                    remaining(startup_deadline, 1)
                    time.sleep(.1)
            warm = request(url, prompts[0], 128, deadline, point / 'warmup-response.json', entry['draft_proposals'])
            dump(point / 'warmup.json', warm)
            summary['metrics_before'] = metrics(url, deadline)
            task_start = time.monotonic()
            before = summary['metrics_before']
            with (point / 'requests.jsonl').open('w') as records:
                for i, prompt in enumerate(prompts):
                    attempted_index = i
                    row = request(url, prompt, 768, deadline, point / f'response-{i:03d}.json', entry['draft_proposals'])
                    after = metrics(url, deadline)
                    row.update(repeat=0, prompt_index=i, source_index=gold[i]['source_index'],
                               metrics_before=before, metrics_after=after,
                               metrics_delta=delta_metrics(before, after))
                    for key in ('verify_rounds', 'accepted_drafts', 'proposed_drafts'):
                        if row['timings'][key] != row['metrics_delta'][key]:
                            raise SweepError('counter_evidence_mismatch', key)
                    rows.append(row)
                    scores.append(score(row, gold[i]))
                    records.write(json.dumps(row, allow_nan=False) + '\n')
                    records.flush()
                    before = after
                    print(json.dumps({'index': a.index, 'delta': entry['delta'], 'request': i,
                                      'tokens': row['ntok'], 'request_s': row['request_s'],
                                      'correct': scores[-1]['correct'], 'finish': row['finish_reason']}), flush=True)
            summary['task_wall_s'] = time.monotonic() - task_start
            summary['metrics_after'] = before
    except SweepError as failure:
        error = failure.data()
    except (OSError, subprocess.SubprocessError, InterruptedError, KeyboardInterrupt) as failure:
        error = {'code': 'process_or_io_error', 'detail': str(failure)}
    except Exception as defect:
        error = {'code': 'runner_defect', 'detail': repr(defect)}
        raise
    finally:
        signal.alarm(0)
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            summary['server_exit_code'] = child.returncode
        failed_ids = [gold[attempted_index]['source_index']] if error is not None and attempted_index is not None and attempted_index >= len(rows) else []
        attempted_count = attempted_index + 1 if attempted_index is not None else 0
        summary.update(state='complete' if error is None else 'failed', error=error,
                       ended_at_unix=time.time(), gpu_after=gpu_receipt(), failed_question_ids=failed_ids,
                       unattempted_question_ids=[r['source_index'] for r in gold[attempted_count:]])
        dump(point / 'errors.json', [] if error is None else [{'error': error, 'prompt_index': attempted_index,
                                                            'failed_question_ids': failed_ids}])
        summary.update(aggregate(rows, scores, error is None and len(rows) == N))
        dump(point / 'score.json', {'state': summary['state'], 'correct': summary['correct'],
                                  'total': N, 'accuracy': summary['accuracy'], 'per_item': scores,
                                  'failed_question_ids': summary['failed_question_ids'],
                                  'unattempted_question_ids': summary['unattempted_question_ids']})
        dump(point / 'summary.json', summary)
        report(out)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if summary['state'] == 'complete' else 1


COLUMNS = ['index', 'draft_proposals', 'query_rows', 'delta', 'kind', 'state', 'correct', 'total', 'accuracy', 'requests', 'tokens',
           'request_wall_s', 'task_wall_s', 'end_to_end_tok_s', 'weighted_decode_tok_s',
           'native_prefill_s', 'decode_s', 'native_verify_decode_s', 'native_draft_s',
           'median_ttft_s', 'median_request_s', 'p90_request_s', 'eos_count', 'cap_count',
           'accepted_drafts', 'proposed_drafts', 'verify_rounds', 'draft_acceptance',
           'emitted_per_round', 'wrong_question_ids', 'failed_question_ids', 'unattempted_question_ids', 'error']


def report(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    rows = []
    for entry in manifest['schedule']:
        path = out / entry['name'] / 'summary.json'
        row = json.loads(path.read_text()) if path.exists() else {**entry, 'state': 'not_run', 'total': N}
        rows.append(row)
    dump(out / 'table.json', rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    (out / 'table.csv').write_text(buffer.getvalue())
    def fmt(row, key, digits=3):
        value = row.get(key)
        return '—' if value is None or row['state'] != 'complete' else f'{value:.{digits}f}'
    lines = ['# Long-block native draft extrapolation screen', '',
             'Separate v2 continuation of seeded32 screen after numeric serialization repair; no accuracy gating or imputation.',
             'All new points use identical nonstream transport and frozen server source. Startup/warmup excluded.',
             'E2E = completion tokens / sum client request durations; task wall includes metric reads.',
             'Decode = server completion-to-first-token interval; native prefill is measured separately.',
             'Failed/incomplete runs have no reported panel score; retained partial data is not a curve point.', '',
             '| Order | Proposals | δ | Kind | State | Correct /32 | E2E tok/s | Decode tok/s | Median answer s | EOS / cap |',
             '|---:|---:|---:|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        score_text = '—' if row.get('correct') is None else str(row['correct'])
        eos = '—' if 'eos_count' not in row else f'{row["eos_count"]} / {row["cap_count"]}'
        lines.append(f'| {row["index"]} | {row["draft_proposals"]} | {row["delta"]} | {row["kind"]} | {row["state"]} | {score_text} | '
                     f'{fmt(row, "end_to_end_tok_s", 2)} | {fmt(row, "weighted_decode_tok_s", 2)} | '
                     f'{fmt(row, "median_request_s")} | {eos} |')
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('init', 'report', 'run'):
        item = sub.add_parser(name)
        item.add_argument('--output', required=True)
        item.add_argument('--server-source-commit', required=True)
        if name == 'run':
            item.add_argument('--index', type=int, choices=range(len(schedule())), required=True)
            item.add_argument('--server', required=True)
            item.add_argument('--model', required=True)
            item.add_argument('--drafter', required=True)
            item.add_argument('--port', type=int, required=True)
            item.add_argument('--deadline-s', type=int, choices=range(1, 601), required=True)
    args = parser.parse_args()
    global SOURCE
    if re.fullmatch(r'[0-9a-f]{40}', args.server_source_commit) is None:
        raise SweepError('invalid_source_commit', 'Explicit40-character engine commit required')
    SOURCE = args.server_source_commit
    if args.action == 'init':
        init(Path(args.output).resolve())
        return 0
    if args.action == 'report':
        report(Path(args.output).resolve())
        return 0
    return run(args)


def interrupted(signum, frame):
    raise InterruptedError(f'signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    try:
        sys.exit(main())
    except SweepError as error:
        print(json.dumps({'state': 'error', 'error': error.data()}), file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({'state': 'error', 'error': {'code': 'invalid_input_or_io', 'detail': str(error)}}), file=sys.stderr)
        sys.exit(1)
