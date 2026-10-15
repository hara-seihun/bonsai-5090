#!/usr/bin/env python3
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from commit_intervention_analysis import Failure, analyze, describe, validate_row


def row(k, strict, proposed, trigger='low-confidence', emitted=True):
    ids = list(range(100, 107))
    ids[k] = proposed
    verified = ids[:k] + [strict]
    count = k + int(emitted)
    e = {'ordinal': 0, 'proposal_count': 7, 'selected_prefix': k, 'accepted_prefix': k,
         'input_position': 12, 'output_start': 1, 'verifier_rows': k + 1,
         'committed_inputs': k + 1, 'proposal_token_ids': ids,
         'verified_policy_output_ids': verified, 'emitted_token_ids': verified[:count],
         'emitted_tokens': count, 'emitted_drafts': k,
         'emitted_greedy_tokens': int(emitted),
         'greedy_kind': 'selected-prefix-bonus' if emitted else 'none',
         'selection_trigger': trigger, 'selected_gap': .5 if k < 4 else None}
    return {'source_index': k, 'score': {'correct': True}, 'completion_tokens': count + 1,
            'request_s': .1, 'usage': {'prompt_tokens': 12},
            'commitment': {'error': None, 'rounds': [e]}, 'timings': {'verify_rounds': 1}}


class AnalysisTest(unittest.TestCase):
    def test_bonus_positions_and_nonemitted_tail(self):
        rows = [row(k, 900, 800) for k in range(4)] + [row(4, 900, 800, 'prefix-limit'), row(0, 900, 800, emitted=False)]
        for r in rows:
            self.assertIsNone(validate_row(r, 'confidence'))
        result = describe(rows)
        for k in range(5):
            self.assertEqual(result['by_selected_prefix'][str(k)]['eligible'], 1)
            self.assertEqual(result['by_selected_prefix'][str(k)]['disagreements'], 1)

    def test_exact_agreement(self):
        self.assertEqual(describe([row(0, 800, 800)])['by_selected_prefix']['0']['disagreement_rate'], 0)

    def test_corrupt_ledgers(self):
        for key, value in [('accepted_prefix', 5), ('ordinal', 2), ('input_position', 13),
                           ('committed_inputs', 0), ('verified_policy_output_ids', []),
                           ('emitted_token_ids', [999]), ('emitted_greedy_tokens', 2),
                           ('greedy_kind', 'gate-correction'), ('selected_gap', float('nan')),
                           ('selected_gap', False)]:
            r = row(0, 900, 800)
            r['commitment']['rounds'][0][key] = value
            self.assertIsInstance(validate_row(r, 'confidence'), Failure, key)

    def test_standardization_and_pair_rejection(self):
        crows = [row(k, 900, 800) for k in range(4)]
        srows = [row(k, 800, 800, 'frozen-schedule') for k in range(4)]
        for r in srows:
            r['commitment']['rounds'][0]['selected_gap'] = None
            self.assertIsNone(validate_row(r, 'scheduled'))
        summary = {'fingerprint': {'common': 1}, 'config': {'confidence_gap_threshold': 1}}
        c = {'summary': summary, 'rows': crows, 'evidence': {}}
        s = {'summary': copy.deepcopy(summary), 'rows': srows, 'evidence': {}}
        result = analyze(c, s)
        self.assertEqual(result['low_position_standardization']['confidence'], 1)
        self.assertEqual(result['low_position_standardization']['scheduled'], 0)
        s['summary']['fingerprint']['common'] = 2
        self.assertEqual(analyze(c, s).code, 'fingerprint_mismatch')
        s['summary'] = copy.deepcopy(summary)
        s['rows'] = []
        self.assertEqual(analyze(c, s).code, 'empty_stratum')


if __name__ == '__main__':
    unittest.main()
