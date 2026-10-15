#!/usr/bin/env python3
"""Analyze captured strict bonus interventions; never run inference or infer hidden logits."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import sys


@dataclass(frozen=True)
class Failure:
    code: str
    detail: str


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_panel(path, mode):
    summary_path, requests_path = path / 'summary.json', path / 'requests.jsonl'
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in requests_path.read_text().splitlines()]
    if summary['state'] != 'complete' or summary['error'] is not None:
        return Failure('incomplete_panel', str(path))
    if summary['config']['mode'] != mode or summary['config']['minimum'] != 0 or summary['config']['maximum'] != 4:
        return Failure('policy_mismatch', str(path))
    if len(rows) != len(summary['fingerprint']['source_ids']) or [r['source_index'] for r in rows] != summary['fingerprint']['source_ids']:
        return Failure('source_mismatch', str(path))
    if sum(r['score']['correct'] for r in rows) != summary['correct']:
        return Failure('score_total_mismatch', str(path))
    if sum(r['completion_tokens'] for r in rows) != summary['tokens']:
        return Failure('token_total_mismatch', str(path))
    if abs(sum(r['request_s'] for r in rows) - summary['request_wall_s']) > 1e-8:
        return Failure('wall_total_mismatch', str(path))
    for row in rows:
        failure = validate_row(row, mode)
        if failure is not None:
            return failure
    return {'summary': summary, 'rows': rows, 'evidence': {
        'path': str(path), 'summary_sha256': digest(summary_path),
        'requests_sha256': digest(requests_path)}}


def validate_row(row, mode):
    events = row['commitment']['rounds']
    if row['commitment']['error'] is not None or len(events) != row['timings']['verify_rounds']:
        return Failure('round_total_mismatch', str(row['source_index']))
    if sum(e['emitted_tokens'] for e in events) + 1 != row['completion_tokens']:
        return Failure('emission_total_mismatch', str(row['source_index']))
    input_position, output_position = row['usage']['prompt_tokens'], 1
    for ordinal, event in enumerate(events):
        k, a = event['selected_prefix'], event['accepted_prefix']
        if not (type(k) is int and type(a) is int and 0 <= a <= k <= 4):
            return Failure('invalid_prefix', str(row['source_index']))
        if event['ordinal'] != ordinal or event['proposal_count'] != 7:
            return Failure('invalid_ordinal', str(row['source_index']))
        if event['input_position'] != input_position or event['output_start'] != output_position:
            return Failure('position_mismatch', str(row['source_index']))
        if event['verifier_rows'] != k + 1 or event['committed_inputs'] != a + 1:
            return Failure('state_ledger_mismatch', str(row['source_index']))
        proposed, verified, emitted = (event[key] for key in (
            'proposal_token_ids', 'verified_policy_output_ids', 'emitted_token_ids'))
        if len(proposed) != 7 or len(verified) != k + 1 or len(emitted) != event['emitted_tokens']:
            return Failure('token_ledger_mismatch', str(row['source_index']))
        if verified[:a] != proposed[:a] or emitted != verified[:len(emitted)]:
            return Failure('token_identity_mismatch', str(row['source_index']))
        d, g = event['emitted_drafts'], event['emitted_greedy_tokens']
        if g not in (0, 1) or d < 0 or d > a or d + g != len(emitted) or (g and d != a):
            return Failure('role_ledger_mismatch', str(row['source_index']))
        expected_kind = 'none' if not g else 'selected-prefix-bonus' if a == k else 'gate-correction'
        if event['greedy_kind'] != expected_kind:
            return Failure('greedy_role_mismatch', str(row['source_index']))
        trigger = event['selection_trigger']
        if mode == 'confidence':
            if (k < 4 and trigger != 'low-confidence') or (k == 4 and trigger != 'prefix-limit'):
                return Failure('trigger_mismatch', str(row['source_index']))
            gap = event['selected_gap']
            if k < 4 and not (type(gap) in (float, int) and 0 <= gap < 1):
                return Failure('gap_mismatch', str(row['source_index']))
            if k == 4 and gap is not None:
                return Failure('gap_mismatch', str(row['source_index']))
        elif mode == 'scheduled' and (trigger != 'frozen-schedule' or event['selected_gap'] is not None):
            return Failure('trigger_mismatch', str(row['source_index']))
        input_position += a + 1
        output_position += len(emitted)
    return None


def rate(counts):
    n, changed = counts
    return {'eligible': n, 'disagreements': changed,
            'disagreement_rate': changed / n if n else None}


def describe(rows):
    by_k, by_trigger, by_score = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    questions, replacements = [], Counter()
    for row in rows:
        question = {'source_index': row['source_index'], 'correct': row['score']['correct'],
                    'completion_tokens': row['completion_tokens'], 'request_s': row['request_s'],
                    'eligible': 0, 'disagreements': 0, 'gate_corrections': 0}
        for event in row['commitment']['rounds']:
            if event['greedy_kind'] == 'gate-correction':
                question['gate_corrections'] += event['emitted_greedy_tokens']
            if event['greedy_kind'] != 'selected-prefix-bonus' or not event['emitted_greedy_tokens']:
                continue
            k = event['selected_prefix']
            proposed, strict = event['proposal_token_ids'][k], event['verified_policy_output_ids'][k]
            changed = int(proposed != strict)
            for bucket in (by_k[k], by_trigger[event['selection_trigger']], by_score[str(row['score']['correct'])]):
                bucket[0] += 1
                bucket[1] += changed
            question['eligible'] += 1
            question['disagreements'] += changed
            if changed:
                replacements[(proposed, strict)] += 1
        question['disagreement_rate'] = question['disagreements'] / question['eligible'] if question['eligible'] else None
        questions.append(question)
    return {'by_selected_prefix': {str(k): rate(by_k[k]) for k in range(5)},
            'by_trigger': {key: rate(value) for key, value in sorted(by_trigger.items())},
            'by_answer_score': {key: rate(value) for key, value in sorted(by_score.items())},
            'questions': questions,
            'top_replacements': [{'proposed_id': p, 'strict_id': s, 'count': n}
                                 for (p, s), n in replacements.most_common(20)]}


def analyze(confidence, scheduled):
    if confidence['summary']['fingerprint'] != scheduled['summary']['fingerprint']:
        return Failure('fingerprint_mismatch', 'confidence and scheduled must have identical common fingerprints')
    if confidence['summary']['config']['confidence_gap_threshold'] != 1:
        return Failure('threshold_mismatch', 'analysis requires frozen gap1/max4')
    c, s = describe(confidence['rows']), describe(scheduled['rows'])
    # Standardize both policies to the same observed blind prefix mix, separately below cap.
    totals = [s['by_selected_prefix'][str(k)]['eligible'] for k in range(4)]
    if not all(c['by_selected_prefix'][str(k)]['eligible'] and totals[k] for k in range(4)):
        return Failure('empty_stratum', 'all four low-gap positions require observed bonuses')
    weights = [n / sum(totals) for n in totals]
    standardized = {name: sum(weights[k] * description['by_selected_prefix'][str(k)]['disagreement_rate'] for k in range(4))
                    for name, description in [('confidence', c), ('scheduled', s)]}
    paired = [(x, y) for x, y in zip(c['questions'], s['questions']) if x['eligible'] and y['eligible']]
    return {'state': 'complete', 'scope': 'captured emitted strict-bonus versus same-row draft ID; not hidden-prefix regret or causal answer improvement',
            'evidence': {'confidence': confidence['evidence'], 'scheduled': scheduled['evidence']},
            'fingerprint': confidence['summary']['fingerprint'], 'confidence': c, 'scheduled': s,
            'low_position_standardization': {'weights_from_scheduled_k0_to_k3': weights, **standardized},
            'paired_question_disagreement_rate_difference_median': statistics.median(x['disagreement_rate'] - y['disagreement_rate'] for x, y in paired) if paired else None,
            'confidence_higher_disagreement_questions': sum(x['disagreement_rate'] > y['disagreement_rate'] for x, y in paired),
            'paired_questions': len(paired),
            'missing_observations': ['all-position selector gaps', 'strict IDs before relaxed substitution on accepted draft rows', 'target draft logit regrets', 'counterfactual outcomes after changed interventions']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confidence', type=Path, required=True)
    parser.add_argument('--scheduled', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            result = Failure('output_exists', str(args.output))
        else:
            c, s = read_panel(args.confidence, 'confidence'), read_panel(args.scheduled, 'scheduled')
            result = c if isinstance(c, Failure) else s if isinstance(s, Failure) else analyze(c, s)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = Failure('invalid_input', str(error))
    if isinstance(result, Failure):
        print(json.dumps({'state': 'error', 'error': {'code': result.code, 'detail': result.detail}}))
        return 1
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write('\n')
    except (OSError, ValueError) as error:
        print(json.dumps({'state': 'error', 'error': {'code': 'write_failed', 'detail': str(error)}}))
        return 1
    print(json.dumps({'state': 'complete', 'output': str(args.output),
                      'standardized': result['low_position_standardization'],
                      'confidence_higher_questions': result['confidence_higher_disagreement_questions'],
                      'paired_questions': result['paired_questions']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
