#!/usr/bin/env python3
"""One bounded, matched-prompt commitment-policy panel. Python stdlib only."""
import argparse
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import struct
import sys
import time

from gsm8k import SUFFIX, extract, num
import commit_schedule

ROOT = Path(__file__).resolve().parents[1]
SELECTION_SHA = 'ed454c8f72749157463d8b203ec32f7248854bdaab85273a1c71e0385d64aec9'
ROUND_INTS = ('ordinal', 'proposal_count', 'selected_prefix', 'accepted_prefix',
              'selection_limit', 'next_feedback_limit', 'emitted_tokens', 'emitted_drafts',
              'emitted_greedy_tokens', 'verifier_rows', 'committed_inputs')
TRIGGERS = {'full-block', 'periodic', 'low-confidence', 'prefix-limit', 'feedback-limit', 'frozen-schedule'}
ENGINE_ENUMS = {
    'H5_GRAPH_NODES': {'native', 'kernel-memory'},
    'H5_STATE_FUSION': {'0', '1'}, 'H5_ATTN_DIRECT': {'0', '1'}, 'H5_PDL': {'0', '1'},
    'H5_GDN_CG': {'4', '8', '16'}, 'H5_PREP_CHUNK': {'0', '1', '2'},
    'H5_PREP_CLEAR': {'0', '1'}, 'H5_SCALE_CARRIER': {'0', '1'},
    'H5_CARRIER_WF0': {'0'}, 'H5_LOAD_THREADS': {'4'}, 'H5_DF_WINDOW': {'512', '1024'},
    'H5_TREE2': {'0', '1'}, 'H5_CLONE_FULL_KV': {'0', '1'},
    'H5_HEAD_ROWS': {'all', 'demand'}, 'H5_SERVING_FAIR': {'0', '1'},
    'H5_SERVING_RESERVE': {'0'}, 'H5_SERVING_CHECK': {'0'},
    'H5_STATE': {'f16', 'bf16', 'f32', 'q8'}, 'H5_KV': {'f16', 'fp8'},
    'H5_WEIGHTS': {'int8', 'fp4', 'both'}, 'H5_ACT': {'int8', 'fp4'},
    'H5_CARVEOUT': {'100'}, 'H5_CARVEOUT_SCOPE': {'prep'}, 'H5_GEMM_PREWAIT': {'4', '5', '6'}, 'H5_TOPK_SPLIT': {'0', '1'}, 'H5_PREP_WIDE': {'0', '1', 'multi'}, 'H5_CONV_AB': {'0', '1'}, 'H5_DRAFT_PRECLEAR': {'0', '1'}, 'H5_HEAD_KS': {'1', '2', '3', '4', '6'}, 'H5_AB_FIRST': {'0', '1'}, 'H5_AB_PARALLEL': {'0', '1'},  'H5_SK_AB': {'0', '1'}, 'H5_DRAFT_DCONV_PREP': {'0', '1'}, 'H5_DRAFT_INGEST_FUSE': {'0', '1'},
    'H5_PREP_CHUNK_WIDE': {'0', '1'}, 'H5_DOWN_CLEAR': {'0', '1'}, 'H5_TAIL_PREP': {'0'}, 'H5_FLAG_WAIT': {'0', '1'}, 'H5_GEMM_FLAG': {'0', '1'},
    'H5_PREP_NORM_CHUNK': {'0', '1'}, 'H5_ARGMAX_SPLIT': {'0', '1'}, 'H5_PREP_EARLY': {'0', '4', '8', '12'}, 'H5_EARLY_K': {'0'},
    'H5_DRAFT_NEST': {'0', '8'}, 'H5_LOOKUP_EXTRA': {'0', '8'},
}
# Free-form engine keys: the engine parses and rejects malformed values itself.
ENGINE_KEYS = set(ENGINE_ENUMS) | {'H5_DRAFT_WFMT', 'H5_PREP_G4', 'H5_PLAN_READ', 'H5_GU_SPLIT', 'H5_PLAN_OVERRIDE', 'H5_SK', 'H5_SK_OVERRIDE', 'H5_PF_OUT_MB', 'H5_PF_GU_MB'}


@dataclass(frozen=True)
class Error:
    code: str
    detail: str

    def data(self):
        return {'code': self.code, 'detail': self.detail}


class SignalInterrupt(KeyboardInterrupt):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'signal {signum}')


@dataclass(frozen=True)
class Wire:
    status: int
    raw: bytes
    elapsed_s: float


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def integer(value):
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class EngineEnvironment:
    values: dict
    normalized: dict
    source: dict
    raw: bytes
    plan_raw: bytes | None


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate environment key: ' + key)
        value[key] = item
    return value


def engine_environment(path):
    if path is None:
        return None
    try:
        source = Path(path).resolve()
        raw = source.read_bytes()
        values = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(values, dict):
            return Error('invalid_engine_environment', 'JSON root must be an object')
        for key, value in values.items():
            if key not in ENGINE_KEYS:
                return Error('invalid_engine_environment', 'Unknown/non-engine key: ' + key)
            if type(value) is not str or not value or value != value.strip() or '\x00' in value:
                return Error('invalid_engine_environment', key + ': use a nonempty canonical JSON string')
            if key in ENGINE_ENUMS and value not in ENGINE_ENUMS[key]:
                return Error('invalid_engine_environment', key + ': expected ' + '|'.join(sorted(ENGINE_ENUMS[key])))
        chunk = values.get('H5_PREP_CHUNK')
        if (chunk in ('1', '2')) != ('H5_PREP_G4' in values):
            return Error('engine_environment_conflict', 'PREP_CHUNK=1|2 requires explicit PREP_G4; scale requires one of these modes')
        if 'H5_PREP_G4' in values:
            scale = struct.unpack('f', struct.pack('f', float(values['H5_PREP_G4'])))[0]
            if scale <= 0 or not math.isfinite(scale):
                return Error('invalid_engine_environment', 'PREP_G4 must be finite positive float32')
            reciprocal = struct.unpack('f', struct.pack('f', 1 / scale))[0]
            if not math.isfinite(reciprocal):
                return Error('invalid_engine_environment', 'PREP_G4 reciprocal must be finite float32')
            values['H5_PREP_G4'] = format(scale, '.9g')
        if values.get('H5_ACT') == 'int8' and values.get('H5_WEIGHTS') not in ('int8', 'both'):
            return Error('engine_environment_conflict', 'ACT=int8 requires explicit WEIGHTS=int8|both')
        if values.get('H5_ACT') == 'fp4' and values.get('H5_WEIGHTS') == 'int8':
            return Error('engine_environment_conflict', 'ACT=fp4 conflicts with WEIGHTS=int8')
        for key in ('H5_PREP_CLEAR', 'H5_SCALE_CARRIER'):
            if values.get(key) == '1' and (values.get('H5_ACT') != 'fp4' or values.get('H5_WEIGHTS') not in ('fp4', 'both')):
                return Error('engine_environment_conflict', key + '=1 requires explicit ACT=fp4 and WEIGHTS=fp4|both')
        if ('H5_TREE2' in values) != ('H5_CLONE_FULL_KV' in values):
            return Error('engine_environment_conflict', 'TREE2 and CLONE_FULL_KV must be explicitly paired')
        if 'H5_TREE2' in values and values['H5_TREE2'] != values['H5_CLONE_FULL_KV']:
            return Error('engine_environment_conflict', 'TREE2 requires full KV clone; this single-slot panel has no other clone route')
        plan = None
        plan_raw = None
        if 'H5_PLAN_READ' in values:
            plan_path = Path(values['H5_PLAN_READ'])
            if not plan_path.is_absolute():
                plan_path = source.parent / plan_path
            plan_path = plan_path.resolve()
            plan_raw = plan_path.read_bytes()
            if not plan_raw:
                return Error('invalid_engine_environment', 'PLAN_READ file is empty')
            values['H5_PLAN_READ'] = str(plan_path)
            plan = {'path': str(plan_path), 'sha256': hashlib.sha256(plan_raw).hexdigest(), 'bytes': len(plan_raw)}
        values = dict(sorted(values.items()))
        return EngineEnvironment(values, {'values': values, 'plan': plan},
                                 {'path': str(source), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}, raw, plan_raw)
    except (OSError, ValueError, TypeError, OverflowError, struct.error) as error:
        return Error('invalid_engine_environment', str(error))


def config(a):
    b = a.draft_proposals
    if a.commit_policy not in ('fixed', 'periodic', 'confidence', 'feedback', 'scheduled'):
        return Error('invalid_policy_parameters', 'Unknown commit policy')
    seed = a.commit_schedule_seed
    if (a.commit_policy == 'scheduled') != (seed is not None):
        return Error('invalid_policy_parameters', 'commit_schedule_seed: required iff scheduled')
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 0x7fffffff):
        return Error('invalid_schedule_seed', 'Require integer seed in 0..INT_MAX')
    if a.commit_policy == 'scheduled' and b != 7:
        return Error('invalid_policy_range', 'Scheduled requires exactly seven proposals')
    used = {'fixed': (), 'scheduled': (), 'periodic': ('commit_max',),
            'confidence': ('commit_min', 'commit_max', 'commit_confidence'),
            'feedback': ('commit_min', 'commit_max', 'commit_confidence', 'commit_growth')}[a.commit_policy]
    for key in ('commit_min', 'commit_max', 'commit_confidence', 'commit_growth'):
        if (key in used) != (getattr(a, key) is not None):
            return Error('invalid_policy_parameters', f'{key}: required iff used by {a.commit_policy}')
    if a.commit_policy == 'scheduled':
        lo, hi = 0, 4
    elif a.commit_policy == 'fixed':
        lo = hi = b
    elif a.commit_policy == 'periodic':
        lo = hi = a.commit_max
    else:
        lo, hi = a.commit_min, a.commit_max
    if not (0 <= lo <= hi <= b):
        return Error('invalid_policy_range', 'require 0 <= min <= max <= proposals')
    gap = a.commit_confidence if a.commit_confidence is not None else 0.0
    growth = a.commit_growth if a.commit_growth is not None else 0
    if not finite(gap) or (a.commit_policy == 'feedback' and growth <= 0):
        return Error('invalid_policy_range', 'finite nonnegative gap; positive feedback growth required')
    confidence = a.commit_policy in ('confidence', 'feedback')
    normalized = {'mode': a.commit_policy, 'minimum': lo, 'maximum': hi,
                  'confidence_gap_threshold': gap if confidence else None,
                  'confidence_signal': 'draft-selector-best-minus-second-raw-score' if confidence else None,
                  'feedback_growth': growth if a.commit_policy == 'feedback' else None,
                  'target_regret_observed': False, 'selection_timing': 'before-target-verification'}
    if seed is not None:
        normalized['schedule'] = commit_schedule.descriptor(seed)
    return normalized


def same_config(observed, expected):
    if not isinstance(observed, dict) or set(expected) - set(observed):
        return False
    for key, value in expected.items():
        actual = observed[key]
        if key == 'confidence_gap_threshold' and value is not None:
            if not finite(actual) or not math.isclose(actual, value, rel_tol=1e-6, abs_tol=1e-8):
                return False
        elif key == 'schedule':
            if not isinstance(actual, dict) or json.dumps(actual, sort_keys=True) != json.dumps(value, sort_keys=True):
                return False
        elif type(actual) is not type(value) or actual != value:
            return False
    return True


def corpus(a):
    selection, prompts = Path(a.selection).resolve(), Path(a.prompts).resolve()
    if sha(selection) != SELECTION_SHA:
        return Error('corpus_mismatch', 'Use the declared results/long-draft/selection32.json without modification')
    gold, texts = json.loads(selection.read_text()), json.loads(prompts.read_text())
    if len(gold) != 32 or texts != [row['question'] + SUFFIX for row in gold]:
        return Error('prompt_mismatch', 'Declared prompts must exactly match frozen questions plus tools/gsm8k.py SUFFIX')
    if not (0 <= a.start < 32 and 1 <= a.count <= 32 - a.start):
        return Error('invalid_subset', 'Explicit contiguous start/count must lie in the frozen 32 items')
    selected = gold[a.start:a.start + a.count]
    for row in selected:
        target = num(row['answer'].split('####')[-1])
        if target is None or not math.isfinite(target):
            return Error('invalid_gold', str(row['source_index']))
    return (selected, texts[a.start:a.start + a.count],
            {'selection_sha256': sha(selection), 'prompts_sha256': sha(prompts),
             'scorer_sha256': sha(ROOT / 'tools/gsm8k.py'), 'start': a.start, 'count': a.count,
             'source_ids': [row['source_index'] for row in selected]})


def listeners(port):
    inodes = set()
    for path in ('/proc/net/tcp', '/proc/net/tcp6'):
        for line in Path(path).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == '0A' and int(fields[1].split(':')[1], 16) == port:
                inodes.add(fields[9])
    return inodes


def port_admission(port):
    if listeners(port):
        return Error('port_occupied', 'Linux LISTEN entry exists; httplib SO_REUSEPORT must not share an existing listener')
    with socket.socket() as probe:
        probe.settimeout(.5)
        status = probe.connect_ex(('127.0.0.1', port))
        if status == 0:
            return Error('port_occupied', 'Live TCP listener')
        if status != errno.ECONNREFUSED:
            return Error('port_probe_error', str(status))
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        try:
            probe.bind(('127.0.0.1', port))
        except OSError as error:
            return Error('port_occupied', str(error))
    return None


def owns_listener(pid, port):
    owned = set()
    for fd in Path(f'/proc/{pid}/fd').iterdir():
        try:
            link = os.readlink(fd)
        except FileNotFoundError:
            continue
        if link.startswith('socket:['):
            owned.add(link[8:-1])
    active = listeners(port)
    return bool(active) and active <= owned


def http_request(port, path, timeout, body=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    started = time.monotonic()
    try:
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        connection.request('GET' if body is None else 'POST', path, body=data,
                           headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        return Wire(response.status, response.read(), time.monotonic() - started)
    except (OSError, http.client.HTTPException) as error:
        return Error('http_transport', str(error))
    finally:
        connection.close()


def decode(wire):
    if wire.status != 200:
        return Error('http_status', f'{wire.status}: {wire.raw[:4096].decode(errors="replace")}')
    try:
        value = json.loads(wire.raw)
    except (ValueError, UnicodeError) as error:
        return Error('invalid_json', str(error))
    return value if isinstance(value, dict) else Error('invalid_response', 'Expected JSON object')


def validate_response(response, expected, proposals, cap):
    try:
        choice = response['choices'][0]
        text, finish = choice['message']['content'], choice['finish_reason']
        usage, timing = response['usage'], response['timings']
        commitment = timing['commitment']
    except (KeyError, IndexError, TypeError) as error:
        return Error('invalid_response', str(error))
    if not isinstance(text, str) or finish not in ('stop', 'length'):
        return Error('invalid_response', 'Text or finish_reason missing/invalid')
    if not isinstance(usage, dict) or not all(integer(usage.get(k)) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
        return Error('invalid_usage', 'Nonnegative integer token counts required')
    tokens = usage['completion_tokens']
    if tokens > cap or usage['total_tokens'] != usage['prompt_tokens'] + tokens:
        return Error('invalid_usage', 'Token counts inconsistent with request')
    for key in ('native_prefill_s', 'verify_s', 'draft_s', 'ttft_s', 'total_s', 'decode_tok_per_s'):
        if not finite(timing.get(key)):
            return Error('invalid_timing', key)
    if timing['ttft_s'] > timing['total_s']:
        return Error('invalid_timing', 'TTFT exceeds total')
    if timing.get('draft_proposals') != proposals:
        return Error('proposal_mismatch', str(timing.get('draft_proposals')))
    if not isinstance(commitment, dict) or not same_config(commitment.get('config'), expected):
        return Error('commitment_config_mismatch', repr(commitment))
    if 'error' not in commitment or not (commitment['error'] is None or isinstance(commitment['error'], str)):
        return Error('invalid_commitment', 'error must be null|string')
    if commitment['error'] is not None:
        return Error('commitment_error', commitment['error'])
    rounds = commitment.get('rounds')
    if not isinstance(rounds, list):
        return Error('invalid_commitment', 'rounds must be an array')
    if not integer(timing.get('verify_rounds')) or (expected['mode'] == 'fixed' and rounds) or (expected['mode'] != 'fixed' and len(rounds) != timing['verify_rounds']):
        return Error('invalid_commitment', 'Fixed has no profiling rounds; adaptive rounds must equal verify_rounds')
    previous_ordinal = None
    schedule_offset = None
    sequence = bytes.fromhex(expected['schedule']['sequence_hex']) if expected['mode'] == 'scheduled' else None
    for event in rounds:
        if not isinstance(event, dict) or not all(integer(event.get(k)) for k in ROUND_INTS):
            return Error('invalid_round', 'Missing/noninteger round accounting')
        if previous_ordinal is not None and event['ordinal'] != previous_ordinal + 1:
            return Error('invalid_round', 'Noncontiguous ordinal')
        if sequence is not None:
            if previous_ordinal is None and event['ordinal'] != 0:
                return Error('invalid_round', 'Scheduled ordinal must start at zero')
            if not all(integer(event.get(k)) for k in ('schedule_offset', 'schedule_index', 'schedule_cycle')):
                return Error('invalid_round', 'Scheduled offset/index/cycle must be nonnegative integers')
            if schedule_offset is None:
                schedule_offset = event['schedule_offset']
            position = schedule_offset + event['ordinal']
            cycle, index = divmod(position, len(sequence))
            if (schedule_offset >= len(sequence) or event['schedule_offset'] != schedule_offset
                    or event['schedule_index'] != index or event['schedule_cycle'] != cycle
                    or event['selected_prefix'] != sequence[index]
                    or event.get('selection_trigger') != 'frozen-schedule'
                    or event['selection_limit'] != 4 or event['next_feedback_limit'] != 4
                    or event.get('selected_gap') is not None):
                return Error('invalid_round', 'Frozen schedule consumption mismatch')
        elif event.get('selection_trigger') == 'frozen-schedule':
            return Error('invalid_round', 'Frozen schedule trigger outside scheduled policy')
        previous_ordinal = event['ordinal']
        if event['proposal_count'] != proposals or not (0 <= event['accepted_prefix'] <= event['selected_prefix'] <= event['selection_limit'] <= proposals):
            return Error('invalid_round', 'Prefix order/proposal count mismatch')
        if event.get('selection_trigger') not in TRIGGERS:
            return Error('invalid_round', 'Unknown or missing selection_trigger')
        for key in ('natural_gate_rejection', 'matched_eos'):
            if type(event.get(key)) is not bool:
                return Error('invalid_round', key + ' must be boolean')
        if event['natural_gate_rejection'] != (event['accepted_prefix'] < event['selected_prefix'] and not event['matched_eos']):
            return Error('invalid_round', 'Gate rejection must exclude matched EOS')
        if not integer(event.get('input_position')) or not integer(event.get('output_start')) or event.get('greedy_kind') not in ('selected-prefix-bonus', 'gate-correction', 'none'):
            return Error('invalid_round', 'Position or greedy_kind missing/invalid')
        if 'selected_gap' not in event or (event['selected_gap'] is not None and not finite(event['selected_gap'])):
            return Error('invalid_round', 'selected_gap must be null or finite nonnegative raw selector gap')
        if (event['selection_trigger'] == 'low-confidence') != (event['selected_gap'] is not None):
            return Error('invalid_round', 'selected_gap must be set only for confidence cuts')
        if event['emitted_tokens'] != event['emitted_drafts'] + event['emitted_greedy_tokens']:
            return Error('invalid_round', 'Emission accounting mismatch')
        if event['verifier_rows'] != event['selected_prefix'] + 1 or event['committed_inputs'] != event['accepted_prefix'] + 1:
            return Error('invalid_round', 'Paid verifier rows or retained input prefix mismatch')
        if event['emitted_drafts'] > event['accepted_prefix'] or event['emitted_greedy_tokens'] > 1:
            return Error('invalid_round', 'Visible emission exceeds accepted prefix/one greedy token')
        for key, length in (('proposal_token_ids', proposals), ('verified_policy_output_ids', event['verifier_rows']),
                            ('emitted_token_ids', event['emitted_tokens'])):
            if key in event and (not isinstance(event[key], list) or len(event[key]) != length or not all(integer(t) for t in event[key])):
                return Error('invalid_round', key + ' has invalid token provenance')
    return {'text': text, 'finish_reason': finish, 'eos': finish == 'stop', 'cap': finish == 'length',
            'completion_tokens': tokens, 'usage': usage, 'timings': timing, 'commitment': commitment}


def score(text, gold):
    target = num(gold['answer'].split('####')[-1])
    visible = text.split('</think>')[-1]
    if '<think>' in visible:
        visible = visible.split('<think>')[0]
    prediction = extract(visible)
    status = 'missing' if prediction is None else 'finite' if math.isfinite(prediction) else 'nonfinite'
    return {'expected': target, 'prediction_state': status,
            'predicted': prediction if status == 'finite' else None,
            'correct': status == 'finite' and abs(prediction - target) < 1e-6}


def command(a):
    cmd = [str(Path(a.server).resolve()), str(Path(a.model).resolve()), str(Path(a.drafter).resolve()),
           '--host', '127.0.0.1', '--port', str(a.port), '--slots', '1', '--ctx', '4096', '--prefill', '128',
           '--draft-proposals', str(a.draft_proposals), '--commit-policy', a.commit_policy]
    for key in ('commit_min', 'commit_max', 'commit_confidence', 'commit_growth', 'commit_schedule_seed'):
        value = getattr(a, key)
        if value is not None:
            cmd += ['--' + key.replace('_', '-'), str(value)]
    return cmd


def paired(rows, baseline):
    return [{'source_index': row['source_index'],
             'baseline_request_s': base['request_s'], 'candidate_request_s': row['request_s'],
             'request_ratio': row['request_s'] / base['request_s'] if base['request_s'] else None,
             'baseline_total_s': base['timings']['total_s'], 'candidate_total_s': row['timings']['total_s'],
             'baseline_tokens': base['completion_tokens'], 'candidate_tokens': row['completion_tokens'],
             'baseline_correct': base['score']['correct'], 'candidate_correct': row['score']['correct'],
             'baseline_eos': base['eos'], 'candidate_eos': row['eos'],
             'baseline_cap': base['cap'], 'candidate_cap': row['cap'],
             'same_text': base['text'] == row['text']}
            for row, base in zip(rows, baseline)]


def aggregate(rows, total, complete):
    partial_correct = sum(r['score']['correct'] for r in rows)
    return {'requests': len(rows), 'total': total, 'partial_correct': partial_correct,
            'correct': partial_correct if complete else None,
            'accuracy': partial_correct / total if complete else None,
            'tokens': sum(r['completion_tokens'] for r in rows),
            'request_wall_s': sum(r['request_s'] for r in rows),
            'eos_count': sum(r['eos'] for r in rows), 'cap_count': sum(r['cap'] for r in rows),
            'wrong_source_ids': [r['source_index'] for r in rows if not r['score']['correct']],
            'selected_drafts': sum(e['selected_prefix'] for r in rows for e in r['commitment']['rounds']),
            'accepted_drafts': sum(e['accepted_prefix'] for r in rows for e in r['commitment']['rounds'])}


def validate_schedule_artifact(out, identity, descriptor):
    try:
        artifact = out / identity['artifact']
        raw = artifact.read_bytes()
        manifest = json.loads((out / 'schedule-manifest.json').read_text())
        if (raw != bytes.fromhex(descriptor['sequence_hex'])
                or hashlib.sha256(raw).hexdigest() != identity['sequence_sha256']
                or len(raw) != identity['bytes']
                or json.dumps(manifest, sort_keys=True) != json.dumps({**identity, 'schedule': descriptor}, sort_keys=True)
                or identity['source_sha256'] != sha(Path(commit_schedule.__file__))):
            return Error('schedule_artifact_mismatch', 'Frozen schedule bytes/source/hash/manifest changed')
        return None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return Error('schedule_artifact_mismatch', str(error))


def run(a):
    expected = config(a)
    if isinstance(expected, Error):
        return expected
    engine = engine_environment(a.engine_environment)
    if isinstance(engine, Error):
        return engine
    if a.role == 'baseline' and (a.commit_policy != 'fixed' or a.baseline is not None):
        return Error('invalid_baseline', 'Baseline is fixed and must not reference another baseline')
    if a.role == 'candidate' and a.baseline is None:
        return Error('missing_baseline', 'Candidate requires --baseline DIR with complete matched fixed results')
    if re.fullmatch('[0-9a-f]{40}', a.server_source_commit) is None:
        return Error('invalid_source', 'Explicit 40-character source commit required')
    if not finite(a.draft_margin) or not 1 <= a.port <= 65535 or not 0 < a.deadline_s <= 600 or not 0 < a.request_timeout_s <= 45:
        return Error('invalid_settings', 'Nonnegative finite margin; port 1..65535; deadline (0,600]; HTTP timeout (0,45]')
    if not 1 <= a.max_tokens <= 3072 or not 1 <= a.warmup_tokens <= 3072 or a.compact != 'dense5':
        return Error('invalid_settings', 'Token budgets 1..3072 and compact=dense5 required')
    if engine is not None and engine.values.get('H5_TREE2') == '1' and (a.draft_margin != 0 or a.commit_policy != 'fixed' or a.draft_proposals != 7):
        return Error('engine_policy_conflict', 'Strict tree requires margin0, fixed commitment and native7 proposals')
    loaded = corpus(a)
    if isinstance(loaded, Error):
        return loaded
    gold, prompts, fingerprint = loaded
    for key in ('server', 'model', 'drafter'):
        path = Path(getattr(a, key)).resolve()
        if not path.is_file():
            return Error('missing_input', str(path))
        fingerprint[key] = {'path': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    fingerprint.update(server_sha256=sha(Path(a.server)), server_source_commit=a.server_source_commit,
                       runner_sha256=sha(Path(__file__)), draft_margin=a.draft_margin, compact=a.compact,
                       max_tokens=a.max_tokens, warmup_tokens=a.warmup_tokens,
                       request_timeout_s=a.request_timeout_s,
                       slots=1, ctx=4096, prefill=128, temperature=0, thinking=False, stream=False,
                       score_policy='tools/gsm8k.py num/extract; hidden-thinking stripping; absolute tolerance 1e-6')
    if engine is not None:
        fingerprint['engine_environment'] = engine.normalized
    baseline_rows = None
    baseline_info = None
    if a.baseline is not None:
        base_path = Path(a.baseline).resolve()
        base = json.loads((base_path / 'summary.json').read_text())
        if base['state'] != 'complete' or base['role'] != 'baseline' or base['config']['mode'] != 'fixed' or base['fingerprint'] != fingerprint:
            return Error('baseline_mismatch', 'Baseline must be complete, fixed and match corpus, binary/assets, request settings and runner')
        baseline_rows = [json.loads(line) for line in (base_path / 'requests.jsonl').read_text().splitlines()]
        if len(baseline_rows) != a.count or [r['source_index'] for r in baseline_rows] != fingerprint['source_ids']:
            return Error('baseline_mismatch', 'Baseline response source IDs/count differ')
        baseline_info = {'draft_proposals': base['draft_proposals'], 'config': base['config'],
                         'summary_sha256': sha(base_path / 'summary.json'),
                         'requests_sha256': sha(base_path / 'requests.jsonl')}
    out = Path(a.output).resolve()
    if out.exists():
        return Error('output_exists', 'Retained results are immutable; choose a new directory: ' + str(out))
    out.mkdir(parents=True)
    dump(out / 'selection.json', gold)
    dump(out / 'prompts.json', prompts)
    env = {k: v for k, v in os.environ.items() if not k.startswith('H5_')}
    env.update(H5_COMPACT=a.compact, H5_DRAFT_MARGIN=str(a.draft_margin))
    if engine is not None:
        env.update(engine.values)
        (out / 'engine-environment-input.json').write_bytes(engine.raw)
        dump(out / 'engine-environment-normalized.json', engine.normalized)
        if engine.plan_raw is not None:
            frozen_plan = out / 'engine-plan.txt'
            frozen_plan.write_bytes(engine.plan_raw)
            env['H5_PLAN_READ'] = str(frozen_plan)
    summary = {'state': 'running', 'role': a.role, 'config': expected, 'draft_proposals': a.draft_proposals,
               'fingerprint': fingerprint, 'baseline': None if a.baseline is None else str(Path(a.baseline).resolve()),
               'baseline_info': baseline_info,
               'command': command(a), 'environment': {k: v for k, v in env.items() if k.startswith('H5_')},
               'deadline_s': a.deadline_s, 'started_at_unix': time.time(), 'host': socket.gethostname()}
    if engine is not None:
        summary['engine_environment_source'] = engine.source
        summary['run_config'] = {'commitment': expected, 'engine_environment': engine.normalized}
    if expected['mode'] == 'scheduled':
        sequence = bytes.fromhex(expected['schedule']['sequence_hex'])
        artifact = out / 'schedule.bin'
        artifact.write_bytes(sequence)
        artifact.chmod(0o444)
        identity = {'id': expected['schedule']['id'], 'seed': expected['schedule']['seed'],
                    'sequence_sha256': hashlib.sha256(sequence).hexdigest(), 'bytes': len(sequence),
                    'artifact': artifact.name, 'source_sha256': sha(Path(commit_schedule.__file__))}
        dump(out / 'schedule-manifest.json', {**identity, 'schedule': expected['schedule']})
        (out / 'schedule-manifest.json').chmod(0o444)
        summary['schedule'] = identity
        summary.setdefault('run_config', {'commitment': expected})['schedule'] = identity
    dump(out / 'summary.json', summary)
    rows, child, failure, attempted = [], None, None, None
    lock = log = records = None
    deadline = time.monotonic() + a.deadline_s
    previous_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, a.deadline_s)

    def timeout():
        left = deadline - time.monotonic()
        return Error('deadline', 'Panel deadline exhausted') if left <= 0 else min(left, a.request_timeout_s)

    def request(prompt, cap, stem):
        limit = timeout()
        if isinstance(limit, Error):
            return limit
        body = {'model': 'ternary-bonsai-2-27b', 'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': cap, 'temperature': 0, 'stream': False,
                'chat_template_kwargs': {'enable_thinking': False}}
        dump(out / (stem + '-request.json'), body)
        wire = http_request(a.port, '/v1/chat/completions', limit, body)
        if isinstance(wire, Error):
            return wire
        (out / (stem + '-response.json')).write_bytes(wire.raw)
        dump(out / (stem + '-wire.json'), {'status': wire.status, 'request_s': wire.elapsed_s})
        value = decode(wire)
        if isinstance(value, Error):
            return value
        choices = value.get('choices')
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get('message')
            if isinstance(message, dict) and isinstance(message.get('content'), str):
                (out / (stem + '-text.txt')).write_text(message['content'])
        checked = validate_response(value, expected, a.draft_proposals, cap)
        if isinstance(checked, Error):
            return checked
        return {**checked, 'request_s': wire.elapsed_s, 'raw_response': stem + '-response.json'}

    try:
        lock = Path(f'/tmp/halo-adaptive-panel-port-{a.port}.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            failure = Error('port_locked', 'Another panel owns this port')
        if failure is None:
            failure = port_admission(a.port)
        if failure is None and expected['mode'] == 'scheduled':
            failure = validate_schedule_artifact(out, summary['schedule'], expected['schedule'])
        if failure is None:
            log = (out / 'server.log').open('w')
            child = subprocess.Popen(summary['command'], env=env, stdout=log, stderr=log)
            summary['server_pid'] = child.pid
            dump(out / 'summary.json', summary)
            while failure is None:
                if child.poll() is not None:
                    failure = Error('server_exit', f'{child.returncode}; see server.log')
                    break
                limit = timeout()
                if isinstance(limit, Error):
                    failure = limit
                    break
                wire = http_request(a.port, '/health', min(limit, 1))
                if isinstance(wire, Error):
                    time.sleep(min(.05, max(0, deadline - time.monotonic())))
                    continue
                (out / 'health-response.json').write_bytes(wire.raw)
                health = decode(wire)
                if isinstance(health, Error):
                    failure = health
                elif health.get('status') != 'ok' or health.get('draft_proposals') != a.draft_proposals or not same_config(health.get('commitment_policy'), expected):
                    failure = Error('health_config_mismatch', repr(health))
                elif not owns_listener(child.pid, a.port):
                    failure = Error('listener_ownership', 'Health response is not exclusively from the owned child')
                else:
                    summary['health'] = health
                break
        if failure is None:
            warm = request(prompts[0], a.warmup_tokens, 'warmup')
            if isinstance(warm, Error):
                failure = warm
            else:
                dump(out / 'warmup.json', warm)
        records = (out / 'requests.jsonl').open('w')
        for i, (prompt, target) in enumerate(zip(prompts, gold)):
            if failure is not None:
                break
            if time.monotonic() >= deadline:
                failure = Error('deadline', 'Deadline reached before next request')
                break
            attempted = i
            row = request(prompt, a.max_tokens, f'item-{i:03d}')
            if isinstance(row, Error):
                failure = row
                break
            row.update(prompt_index=a.start + i, source_index=target['source_index'], score=score(row['text'], target))
            rows.append(row)
            records.write(json.dumps(row, allow_nan=False) + '\n')
            records.flush()
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        failure = Error('io_or_external_data', str(error))
    except (KeyboardInterrupt, InterruptedError) as error:
        failure = Error('deadline' if isinstance(error, SignalInterrupt) and error.signum == signal.SIGALRM else 'interrupted', str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if records is not None:
            records.close()
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
            summary['server_exit_code'] = child.returncode
        if log is not None:
            log.close()
        if lock is not None:
            lock.close()
        if failure is None and expected['mode'] == 'scheduled':
            failure = validate_schedule_artifact(out, summary['schedule'], expected['schedule'])
        if expected['mode'] == 'scheduled':
            events = [event for row in rows for event in row['commitment']['rounds']]
            summary['selected_histogram'] = [sum(e['selected_prefix'] == k for e in events) for k in range(5)]
            summary['schedule_used_positions'] = [
                {'source_index': row['source_index'], 'ordinal': e['ordinal'],
                 **{key: e[key] for key in ('schedule_offset', 'schedule_index', 'schedule_cycle')}}
                for row in rows for e in row['commitment']['rounds']]
        complete = failure is None and len(rows) == a.count
        if failure is None and not complete:
            failure = Error('incomplete_panel', 'Not all declared requests completed')
        failed = [gold[attempted]['source_index']] if attempted is not None and attempted >= len(rows) else []
        attempted_count = attempted + 1 if attempted is not None else 0
        summary.update(state='complete' if complete else 'failed', error=None if failure is None else failure.data(),
                       ended_at_unix=time.time(), failed_source_ids=failed,
                       unattempted_source_ids=[r['source_index'] for r in gold[attempted_count:]],
                       **aggregate(rows, a.count, complete))
        dump(out / 'summary.json', summary)
        dump(out / 'score.json', {'state': summary['state'], 'accuracy': summary['accuracy'],
                                 'items': [{'source_index': r['source_index'], **r['score']} for r in rows]})
        if baseline_rows is not None:
            dump(out / 'paired.json', {'state': summary['state'], 'baseline': summary['baseline'],
                                      'baseline_info': baseline_info, 'candidate_config': expected,
                                      'candidate_draft_proposals': a.draft_proposals,
                                      'items': paired(rows, baseline_rows)})
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'server', 'model', 'drafter', 'server-source-commit', 'selection', 'prompts', 'compact'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--role', choices=('baseline', 'candidate'), required=True)
    p.add_argument('--baseline')
    p.add_argument('--engine-environment', help='Explicit allowlisted runtime environment JSON; omitted retains original launch behavior')
    p.add_argument('--draft-proposals', type=int, choices=(7, 16, 32, 64), required=True)
    p.add_argument('--draft-margin', type=float, required=True)
    p.add_argument('--commit-policy', choices=('fixed', 'periodic', 'confidence', 'feedback', 'scheduled'), required=True)
    p.add_argument('--commit-schedule-seed', type=int)
    p.add_argument('--commit-min', type=int)
    p.add_argument('--commit-max', type=int)
    p.add_argument('--commit-confidence', type=float)
    p.add_argument('--commit-growth', type=int)
    for name in ('port', 'start', 'count', 'max-tokens', 'warmup-tokens'):
        p.add_argument('--' + name, type=int, required=True)
    for name in ('deadline-s', 'request-timeout-s'):
        p.add_argument('--' + name, type=float, required=True)
    return p


def interrupted(signum, frame):
    raise SignalInterrupt(signum)


def main():
    a = parser().parse_args()
    try:
        value = run(a)
    except (OSError, ValueError, KeyError, TypeError) as error:
        value = Error('invalid_input_or_io', str(error))
    if isinstance(value, Error):
        print(json.dumps({'state': 'error', 'error': value.data()}), file=sys.stderr)
        return 1
    print(json.dumps(value, allow_nan=False), flush=True)
    return 0 if value['state'] == 'complete' else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    sys.exit(main())
