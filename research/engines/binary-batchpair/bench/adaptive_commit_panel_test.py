#!/usr/bin/env python3
"""Seconds-only CPU contract proof. This file is also the owned fake server."""
import argparse
import copy
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('panel', ROOT / 'tools/adaptive_commit_panel.py')
panel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = panel
spec.loader.exec_module(panel)


def fake_config():
    def option(name, conversion=str):
        return conversion(sys.argv[sys.argv.index('--' + name) + 1]) if '--' + name in sys.argv else None
    args = argparse.Namespace(draft_proposals=option('draft-proposals', int), commit_policy=option('commit-policy'),
                              commit_min=option('commit-min', int), commit_max=option('commit-max', int),
                              commit_confidence=option('commit-confidence', float), commit_growth=option('commit-growth', int),
                              commit_schedule_seed=option('commit-schedule-seed', int))
    return args, panel.config(args)


def serve():
    port = int(sys.argv[sys.argv.index('--port') + 1])
    args, config = fake_config()
    assert not isinstance(config, panel.Error)
    gold = json.loads(Path(os.environ['PANEL_TEST_SELECTION']).read_text())
    b = args.draft_proposals
    count = 0
    fault = os.environ.get('PANEL_TEST_FAULT', 'none')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            observed = dict(config)
            if fault == 'health':
                observed['maximum'] = 999
            if fault == 'schedule-health':
                observed = copy.deepcopy(config)
                observed['schedule']['seed'] += 1
            self.send({'status': 'ok', 'draft_proposals': b, 'commitment_policy': observed,
                       'observed_engine_environment': {k: v for k, v in os.environ.items() if k.startswith('H5_')}})

        def do_POST(self):
            nonlocal count
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body['stream'] is False and body['temperature'] == 0
            assert body['chat_template_kwargs']['enable_thinking'] is False
            index = max(0, count - 1)
            count += 1
            if count == 3 and fault == 'http':
                self.send({'error': 'deliberate HTTP failure'}, 500)
                return
            if count == 2 and fault == 'timeout':
                time.sleep(2)
            target = panel.num(gold[index]['answer'].split('####')[-1])
            text = '\\boxed{' + str(target if index != 1 else target + 1) + '}'
            if count == 4:
                text = '9' * 1000
            gap = .1 if config['mode'] in ('confidence', 'feedback') else None
            offset = panel.commit_schedule.request_offset([20, 42, index], args.commit_schedule_seed) if config['mode'] == 'scheduled' else None
            selected = bytes.fromhex(config['schedule']['sequence_hex'])[offset] if offset is not None else min(2, config['maximum'])
            event = {'ordinal': 0, 'proposal_count': b, 'selected_prefix': selected,
                     'accepted_prefix': selected, 'natural_gate_rejection': False, 'matched_eos': False,
                     'selection_trigger': 'low-confidence' if gap is not None else 'periodic',
                     'selection_limit': config['maximum'], 'selected_gap': gap,
                     'next_feedback_limit': config['maximum'], 'emitted_tokens': selected + 1,
                     'emitted_drafts': selected, 'emitted_greedy_tokens': 1,
                     'verifier_rows': selected + 1, 'committed_inputs': selected + 1,
                     'input_position': 20, 'output_start': 0, 'greedy_kind': 'selected-prefix-bonus'}
            if offset is not None:
                event.update(selection_trigger='frozen-schedule', schedule_offset=offset,
                             schedule_index=offset, schedule_cycle=0)
                if count == 2 and fault == 'schedule-index':
                    event['schedule_index'] = (offset + 1) % panel.commit_schedule.ARRAY_LENGTH
                if count == 2 and fault == 'schedule-config':
                    config['schedule']['sequence_hex'] = 'ff' + config['schedule']['sequence_hex'][2:]
            if count == 2 and fault == 'round':
                event['accepted_prefix'] = b + 10
            commitment = {'config': config, 'error': 'deliberate policy failure' if fault == 'policy' else None,
                          'rounds': [] if config['mode'] == 'fixed' else [event]}
            tokens = selected + 1
            self.send({'choices': [{'message': {'content': text}, 'finish_reason': 'length' if index == 1 else 'stop'}],
                       'usage': {'prompt_tokens': 20, 'completion_tokens': tokens, 'total_tokens': 20 + tokens},
                       'timings': {'native_prefill_s': .001, 'verify_s': .003, 'draft_s': .002,
                                   'ttft_s': .002, 'total_s': .01, 'decode_tok_per_s': 200,
                                   'draft_proposals': b, 'verify_rounds': 1, 'commitment': commitment}})

    HTTPServer(('127.0.0.1', port), Handler).serve_forever()


class Contract(unittest.TestCase):
    def args(self, out, model, policy='fixed', baseline=None, count=3, proposals=7):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        options = ['--output', str(out), '--server', __file__, '--model', str(model), '--drafter', str(model),
                   '--server-source-commit', 'a' * 40,
                   '--selection', str(ROOT / 'results/long-draft/selection32.json'),
                   '--prompts', str(ROOT / 'results/long-draft/prompts32.json'), '--compact', 'dense5',
                   '--role', 'candidate' if baseline else 'baseline', '--draft-proposals', str(proposals),
                   '--draft-margin', '16', '--commit-policy', policy, '--port', str(port),
                   '--start', '0', '--count', str(count), '--max-tokens', '768', '--warmup-tokens', '128',
                   '--deadline-s', '4', '--request-timeout-s', '1']
        if baseline:
            options += ['--baseline', str(baseline)]
        if policy == 'scheduled':
            options += ['--commit-schedule-seed', '17']
        elif policy != 'fixed':
            options += ['--commit-max', '4']
        if policy in ('confidence', 'feedback'):
            options += ['--commit-min', '0', '--commit-confidence', '0.25']
        if policy == 'feedback':
            options += ['--commit-growth', '2']
        return panel.parser().parse_args(options)

    def invoke(self, args, fault='none'):
        with patch.dict(os.environ, PANEL_TEST_SELECTION=args.selection, PANEL_TEST_FAULT=fault):
            result = panel.run(args)
        if not isinstance(result, panel.Error):
            with socket.socket() as probe:
                self.assertNotEqual(probe.connect_ex(('127.0.0.1', args.port)), 0)
            if 'server_pid' in result:
                self.assertFalse(Path(f'/proc/{result["server_pid"]}').exists())
        return result

    def test_matrix_pairing_score_lengths_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model = temp / 'model'; model.write_bytes(b'CPU only; not weights')
            base = temp / 'base'
            result = self.invoke(self.args(base, model))
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['correct'], 1)
            self.assertEqual((result['eos_count'], result['cap_count']), (2, 1))
            self.assertEqual(len(list(base.glob('item-*-response.json'))), 3)
            for policy, b in [('fixed', 16), ('periodic', 32), ('confidence', 64), ('feedback', 7)]:
                out = temp / policy
                result = self.invoke(self.args(out, model, policy, base, proposals=b))
                self.assertEqual(result['state'], 'complete')
                pairs = json.loads((out / 'paired.json').read_text())['items']
                self.assertEqual(len(pairs), 3)
                self.assertEqual([r['source_index'] for r in pairs], result['fingerprint']['source_ids'])
                self.assertTrue(all(r['baseline_tokens'] > 0 and r['candidate_tokens'] > 0 for r in pairs))
                self.assertTrue(all(r['same_text'] for r in pairs))
                self.assertEqual(result['correct'], 1)  # Wrong and overflow outputs are observations, not a gate.
                rows = [json.loads(line) for line in (out / 'requests.jsonl').read_text().splitlines()]
                self.assertEqual(rows[2]['score']['prediction_state'], 'nonfinite')
                self.assertEqual(rows[0]['commitment']['rounds'] == [], policy == 'fixed')
                self.assertEqual(result['draft_proposals'], b)
                self.assertEqual(pairs[0]['baseline_tokens'], 3)
                self.assertEqual(result['baseline_info']['draft_proposals'], 7)
                if policy == 'confidence':
                    raw = json.loads((out / 'item-000-response.json').read_text())
                    event = raw['timings']['commitment']['rounds'][0]
                    event['accepted_prefix'] = 1
                    event['committed_inputs'] = 2
                    event['matched_eos'] = True
                    event['emitted_tokens'] = event['emitted_drafts'] = event['emitted_greedy_tokens'] = 0
                    event['greedy_kind'] = 'none'
                    self.assertIsInstance(panel.validate_response(raw, result['config'], b, 768), dict)
                    event['natural_gate_rejection'] = True
                    self.assertEqual(panel.validate_response(raw, result['config'], b, 768).code, 'invalid_round')
                    event['natural_gate_rejection'] = False
                    del event['selection_trigger']
                    self.assertEqual(panel.validate_response(raw, result['config'], b, 768).code, 'invalid_round')

    def test_failures_retain_raw_and_unattempted_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model = temp / 'model'; model.write_bytes(b'CPU')
            base = temp / 'base'
            self.invoke(self.args(base, model))
            for fault, code, requests in [('http', 'http_status', 1), ('round', 'invalid_round', 0),
                                          ('policy', 'commitment_error', 0), ('health', 'health_config_mismatch', 0),
                                          ('timeout', 'http_transport', 0)]:
                out = temp / fault
                result = self.invoke(self.args(out, model, 'confidence', base), fault)
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['error']['code'], code)
                self.assertIsNone(result['accuracy'])
                self.assertEqual(result['requests'], requests)
                expected_failed = 0 if fault in ('health', 'policy') else 1
                self.assertEqual(len(result['failed_source_ids']), expected_failed)
                self.assertEqual(len(result['unattempted_source_ids']), 3 - requests - expected_failed)
                if fault == 'http':
                    self.assertIn('deliberate HTTP failure', (out / 'item-001-response.json').read_text())
                if fault == 'round':
                    self.assertTrue((out / 'item-000-text.txt').is_file())

    def test_deadline_and_existing_listener_no_foreign_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model = temp / 'model'; model.write_bytes(b'CPU')
            args = self.args(temp / 'deadline', model)
            args.deadline_s = .001
            result = self.invoke(args)
            self.assertEqual(result['state'], 'failed')
            self.assertEqual(result['error']['code'], 'deadline')
            self.assertEqual(len(result['unattempted_source_ids']), 3)
            args = self.args(temp / 'inflight-deadline', model)
            args.deadline_s = .3
            result = self.invoke(args, 'timeout')
            self.assertEqual(result['error']['code'], 'deadline')
            self.assertEqual(result['requests'], 0)
            self.assertEqual(len(result['failed_source_ids']), 1)
            with socket.socket() as foreign:
                foreign.bind(('127.0.0.1', 0)); foreign.listen()
                args = self.args(temp / 'occupied', model)
                args.port = foreign.getsockname()[1]
                result = panel.run(args)
                self.assertEqual(result['error']['code'], 'port_occupied')
                self.assertNotIn('server_pid', result)
                self.assertGreaterEqual(foreign.fileno(), 0)

    def test_engine_environment_launch_fingerprint_and_plan_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model = temp / 'model'; model.write_bytes(b'CPU')
            omitted = self.args(temp / 'omitted', model)
            with patch.dict(os.environ, H5_STATE_FUSION='1', H5_DRAFT_MARGIN='99',
                            H5_PREP_CLEAR='1', H5_SCALE_CARRIER='1', H5_WSCALE='e4m3', H5_MEGA='1'):
                result = self.invoke(omitted)
            self.assertNotIn('engine_environment', result['fingerprint'])
            self.assertNotIn('run_config', result)
            self.assertEqual(result['health']['observed_engine_environment'], {'H5_COMPACT': 'dense5', 'H5_DRAFT_MARGIN': '16.0'})
            plan = temp / 'launch plan.txt'; plan.write_text('5120 5120 8 0 0 1 1\n')
            values = {'H5_GRAPH_NODES': 'kernel-memory', 'H5_STATE_FUSION': '1', 'H5_GDN_CG': '8',
                      'H5_PREP_CHUNK': '1', 'H5_PREP_G4': '1.5000', 'H5_ATTN_DIRECT': '0',
                      'H5_PLAN_READ': plan.name, 'H5_PDL': '1', 'H5_STATE': 'f16', 'H5_KV': 'fp8',
                      'H5_WEIGHTS': 'both', 'H5_ACT': 'fp4', 'H5_PREP_CLEAR': '1', 'H5_SCALE_CARRIER': '1'}
            declaration = temp / 'engine.json'; declaration.write_text(json.dumps(values))
            base = self.args(temp / 'engine-base', model); base.engine_environment = str(declaration)
            result = self.invoke(base)
            self.assertEqual(result['state'], 'complete')
            normalized = result['fingerprint']['engine_environment']
            self.assertEqual(normalized, result['run_config']['engine_environment'])
            self.assertEqual(normalized['values']['H5_PREP_G4'], '1.5')
            for key in ('H5_PREP_CLEAR', 'H5_SCALE_CARRIER'):
                self.assertEqual(normalized['values'][key], '1')
                self.assertEqual(result['health']['observed_engine_environment'][key], '1')
            self.assertEqual(normalized['plan']['sha256'], panel.sha(plan))
            self.assertEqual(normalized['plan']['bytes'], plan.stat().st_size)
            self.assertEqual(result['engine_environment_source']['sha256'], panel.sha(declaration))
            frozen = Path(result['environment']['H5_PLAN_READ'])
            self.assertEqual(frozen.read_bytes(), plan.read_bytes())
            self.assertNotEqual(frozen, plan)
            self.assertEqual(result['environment'], result['health']['observed_engine_environment'])
            self.assertEqual((Path(base.output) / 'engine-environment-input.json').read_bytes(), declaration.read_bytes())
            candidate = self.args(temp / 'engine-candidate', model, 'confidence', base.output)
            candidate.engine_environment = str(declaration)
            paired = self.invoke(candidate)
            self.assertEqual(paired['state'], 'complete')
            self.assertEqual(result['fingerprint'], paired['fingerprint'])
            self.assertNotEqual(result['environment']['H5_PLAN_READ'], paired['environment']['H5_PLAN_READ'])
            omitted_candidate = self.args(temp / 'unset-mismatch', model, 'periodic', base.output)
            self.assertEqual(self.invoke(omitted_candidate).code, 'baseline_mismatch')
            values['H5_GDN_CG'] = '4'; declaration.write_text(json.dumps(values))
            wrong = self.args(temp / 'engine-mismatch', model, 'periodic', base.output)
            wrong.engine_environment = str(declaration)
            self.assertEqual(self.invoke(wrong).code, 'baseline_mismatch')
            self.assertFalse(Path(wrong.output).exists())
            values['H5_GDN_CG'] = '8'
            for key in ('H5_PREP_CLEAR', 'H5_SCALE_CARRIER'):
                for value in ('0', None):
                    changed = dict(values)
                    if value is None:
                        del changed[key]
                    else:
                        changed[key] = value
                    declaration.write_text(json.dumps(changed))
                    wrong.output = str(temp / (key + '-' + str(value)))
                    self.assertEqual(self.invoke(wrong).code, 'baseline_mismatch')
                    self.assertFalse(Path(wrong.output).exists())
            declaration.write_text(json.dumps(values))
            plan.write_text('5120 5120 8 0 0 2 1\n')
            wrong.output = str(temp / 'plan-content-mismatch')
            self.assertEqual(self.invoke(wrong).code, 'baseline_mismatch')
            self.assertFalse(Path(wrong.output).exists())
            self.assertNotEqual(panel.sha(plan), panel.sha(frozen))

    def test_engine_environment_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            declaration = temp / 'engine.json'
            cases = [[], {'H5_COMPACT': 'dense5'}, {'H5_DRAFT_MARGIN': '8'}, {'UNKNOWN': '1'},
                     {'H5_STATE_FUSION': True}, {'H5_PDL': 1}, {'H5_GDN_CG': '12'},
                     {'H5_PREP_CHUNK': '1'}, {'H5_PREP_G4': '2'},
                     {'H5_PREP_CHUNK': '0', 'H5_PREP_G4': '2'},
                     {'H5_PREP_CHUNK': '1', 'H5_PREP_G4': 'nan'},
                     {'H5_PREP_CHUNK': '1', 'H5_PREP_G4': '1e-45'},
                     {'H5_PREP_CHUNK': '1', 'H5_PREP_G4': '1e99'},
                     {'H5_ACT': 'int8'}, {'H5_ACT': 'fp4', 'H5_WEIGHTS': 'int8'},
                     {'H5_PLAN_READ': []}, {'H5_PLAN_READ': 'not-present'}, {'H5_STATE': 'f64'}]
            for values in cases:
                declaration.write_text(json.dumps(values))
                self.assertIsInstance(panel.engine_environment(declaration), panel.Error, values)
            declaration.write_text('{"H5_PDL":"0","H5_PDL":"1"}')
            self.assertIsInstance(panel.engine_environment(declaration), panel.Error)
            self.assertIsInstance(panel.engine_environment(temp / 'unset.json'), panel.Error)
            declaration.write_text('{}')
            loaded = panel.engine_environment(declaration)
            self.assertEqual(loaded.normalized, {'values': {}, 'plan': None})
            self.assertIsNone(panel.engine_environment(None))
            model = temp / 'model'; model.write_bytes(b'CPU')
            args = self.args(temp / 'not-launched', model)
            args.engine_environment = str(declaration)
            declaration.write_text('{"H5_PDL":false}')
            self.assertEqual(self.invoke(args).code, 'invalid_engine_environment')
            self.assertFalse(Path(args.output).exists())

    def test_clear_carrier_admission_geometry_and_disabled_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            declaration = temp / 'engine.json'
            model = temp / 'model'; model.write_bytes(b'CPU')
            args = self.args(temp / 'not-launched', model)
            args.engine_environment = str(declaration)
            for key in ('H5_PREP_CLEAR', 'H5_SCALE_CARRIER'):
                for value in (True, False, 0, 1, None, [], {}, '', 'true', '2', '01', ' 1', '1 ', '1\x00'):
                    with self.subTest(key=key, value=value):
                        declaration.write_text(json.dumps({key: value}))
                        self.assertEqual(self.invoke(args).code, 'invalid_engine_environment')
                        self.assertFalse(Path(args.output).exists())
                for geometry in ({}, {'H5_ACT': 'fp4'}, {'H5_WEIGHTS': 'fp4'},
                                 {'H5_WEIGHTS': 'both'}, {'H5_WEIGHTS': 'int8'},
                                 {'H5_ACT': 'int8', 'H5_WEIGHTS': 'both'},
                                 {'H5_ACT': 'fp4', 'H5_WEIGHTS': 'int8'}):
                    with self.subTest(key=key, geometry=geometry):
                        declaration.write_text(json.dumps({key: '1', **geometry}))
                        self.assertEqual(self.invoke(args).code, 'engine_environment_conflict')
                        self.assertFalse(Path(args.output).exists())
                for weights in ('fp4', 'both'):
                    declaration.write_text(json.dumps({key: '1', 'H5_ACT': 'fp4', 'H5_WEIGHTS': weights}))
                    self.assertIsInstance(panel.engine_environment(declaration), panel.EngineEnvironment)
                declaration.write_text(json.dumps({key: '0'}))
                self.assertEqual(panel.engine_environment(declaration).values, {key: '0'})
            for key in ('H5_WSCALE', 'H5_MEGA', 'H5_COMPACT'):
                declaration.write_text(json.dumps({'H5_SCALE_CARRIER': '1', 'H5_ACT': 'fp4',
                                                  'H5_WEIGHTS': 'fp4', key: '1'}))
                self.assertEqual(self.invoke(args).code, 'invalid_engine_environment')
                self.assertFalse(Path(args.output).exists())
            values = {'H5_PREP_CLEAR': '0', 'H5_SCALE_CARRIER': '0', 'H5_ACT': 'int8', 'H5_WEIGHTS': 'both'}
            declaration.write_text(json.dumps(values))
            args.output = str(temp / 'disabled')
            with patch.dict(os.environ, H5_PREP_CLEAR='1', H5_SCALE_CARRIER='1', H5_MEGA='1', H5_WSCALE='e4m3'):
                result = self.invoke(args)
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['fingerprint']['engine_environment']['values'], values)
            self.assertEqual(result['health']['observed_engine_environment'],
                             {**values, 'H5_COMPACT': 'dense5', 'H5_DRAFT_MARGIN': '16.0'})
            values = {'H5_PREP_CLEAR': '1', 'H5_SCALE_CARRIER': '1', 'H5_ACT': 'fp4', 'H5_WEIGHTS': 'fp4'}
            declaration.write_text(json.dumps(values))
            args.output = str(temp / 'enabled')
            with patch.dict(os.environ, H5_MEGA='1', H5_WSCALE='e4m3'):
                result = self.invoke(args)
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['health']['observed_engine_environment'],
                             {**values, 'H5_COMPACT': 'dense5', 'H5_DRAFT_MARGIN': '16.0'})

    def test_frozen_schedule_generator_and_parameter_contract(self):
        schedule = panel.commit_schedule
        vectors = [(0, '5cd133f53f3d588035613533fae6999557496faf344eff8a4b88b86a682cfbce', 2494),
                   (17, 'fbdd6f3a566d38c653069afcf13383627c70a7fe4968316cea49824f47ae78d5', 1445),
                   (0x7fffffff, '4257338adc794a59c6146f7f56d31ab8de2e9780c7340d0f85f3f31df361e089', 2558)]
        for seed, digest, offset in vectors:
            descriptor = schedule.descriptor(seed)
            sequence = bytes.fromhex(descriptor['sequence_hex'])
            self.assertEqual(len(sequence), 2843)
            self.assertEqual([sequence.count(k) for k in range(5)], [394, 338, 232, 239, 1640])
            self.assertEqual(descriptor, schedule.descriptor(seed))
            self.assertEqual(hashlib.sha256(sequence).hexdigest(), digest)
            self.assertEqual(schedule.request_offset([20, 42, 0], seed), offset)
        rng = schedule.SplitMix64(0)
        with patch.object(rng, 'next', side_effect=[0, (1 << 63) + 2]) as draw:
            self.assertEqual(rng.bounded((1 << 63) + 1), 1)
            self.assertEqual(draw.call_count, 2)
        self.assertNotEqual(schedule.descriptor(0), schedule.descriptor(17))
        self.assertEqual(schedule.SplitMix64(0).next(), 0xe220a8397b1dcdaf)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model = temp / 'model'; model.write_bytes(b'CPU')
            args = self.args(temp / 'out', model, 'scheduled', temp / 'base')
            expected = panel.config(args)
            self.assertEqual((expected['minimum'], expected['maximum']), (0, 4))
            self.assertIsNone(expected['confidence_gap_threshold'])
            for seed in (None, -1, 0x80000000, True, 1.5, '17'):
                args.commit_schedule_seed = seed
                self.assertIsInstance(panel.config(args), panel.Error, seed)
            args.commit_schedule_seed = 17
            for key in ('commit_min', 'commit_max', 'commit_confidence', 'commit_growth'):
                setattr(args, key, 0)
                self.assertEqual(panel.config(args).code, 'invalid_policy_parameters')
                setattr(args, key, None)
            for proposals in (16, 32, 64):
                args.draft_proposals = proposals
                self.assertEqual(panel.config(args).code, 'invalid_policy_range')
            for policy in ('fixed', 'periodic', 'confidence', 'feedback'):
                unused = self.args(temp / policy, model, policy)
                unused.commit_schedule_seed = 0
                self.assertEqual(panel.config(unused).code, 'invalid_policy_parameters')
            with open(os.devnull, 'w') as sink, contextlib.redirect_stderr(sink):
                with self.assertRaises(SystemExit):
                    panel.parser().parse_args(['--commit-schedule-seed', '1.5'])

    def test_scheduled_settlement_pairing_evidence_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model = temp / 'model'; model.write_bytes(b'CPU')
            base = temp / 'base'
            baseline = self.invoke(self.args(base, model))
            out = temp / 'scheduled'
            result = self.invoke(self.args(out, model, 'scheduled', base))
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['fingerprint'], baseline['fingerprint'])
            zero = self.args(temp / 'seed-zero', model, 'scheduled', base)
            zero.commit_schedule_seed = 0
            alternate = self.invoke(zero)
            self.assertEqual(alternate['state'], 'complete')
            self.assertEqual(alternate['fingerprint'], result['fingerprint'])
            self.assertNotEqual(alternate['schedule']['sequence_sha256'], result['schedule']['sequence_sha256'])
            self.assertNotIn('schedule', baseline)
            self.assertEqual(result['schedule'], result['run_config']['schedule'])
            sequence = (out / 'schedule.bin').read_bytes()
            self.assertEqual(result['schedule']['sequence_sha256'], hashlib.sha256(sequence).hexdigest())
            self.assertEqual(result['schedule']['source_sha256'], panel.sha(Path(panel.commit_schedule.__file__)))
            self.assertEqual((out / 'schedule.bin').stat().st_mode & 0o222, 0)
            self.assertEqual(sum(result['selected_histogram']), 3)
            self.assertEqual(len(result['schedule_used_positions']), 3)
            pairs = json.loads((out / 'paired.json').read_text())['items']
            self.assertEqual([r['source_index'] for r in pairs], baseline['fingerprint']['source_ids'])
            raw = json.loads((out / 'item-000-response.json').read_text())
            commitment = raw['timings']['commitment']
            event = commitment['rounds'][0]
            for field, value in [('schedule_offset', 2843), ('schedule_index', -1), ('schedule_cycle', 1),
                                 ('selected_prefix', 7), ('selection_limit', 7), ('next_feedback_limit', 7),
                                 ('selected_gap', 0), ('selection_trigger', 'periodic'), ('ordinal', 1)]:
                broken = copy.deepcopy(raw)
                broken['timings']['commitment']['rounds'][0][field] = value
                self.assertEqual(panel.validate_response(broken, result['config'], 7, 768).code, 'invalid_round', field)
            repeated = copy.deepcopy(raw)
            repeated['timings']['verify_rounds'] = 2
            second = copy.deepcopy(event)
            second.update(ordinal=1, schedule_offset=2842, schedule_index=0, schedule_cycle=1)
            repeated['timings']['commitment']['rounds'].append(second)
            self.assertEqual(panel.validate_response(repeated, result['config'], 7, 768).code, 'invalid_round')
            # Force wraparound; emitted/verifier accounting is still independently checked.
            first = repeated['timings']['commitment']['rounds'][0]
            first.update(schedule_offset=2842, schedule_index=2842)
            for e in (first, second):
                selected = sequence[e['schedule_index']]
                e.update(selected_prefix=selected, accepted_prefix=selected, emitted_tokens=selected + 1,
                         emitted_drafts=selected, verifier_rows=selected + 1, committed_inputs=selected + 1)
            self.assertIsInstance(panel.validate_response(repeated, result['config'], 7, 768), dict)
            changed = copy.deepcopy(result['config'])
            changed['schedule']['counts'][0] = True
            self.assertFalse(panel.same_config(changed, result['config']))
            for fault, code in [('schedule-health', 'health_config_mismatch'),
                                ('schedule-index', 'invalid_round'), ('schedule-config', 'commitment_config_mismatch')]:
                failure = self.invoke(self.args(temp / fault, model, 'scheduled', base), fault)
                self.assertEqual(failure['state'], 'failed')
                self.assertEqual(failure['error']['code'], code)
            identity = result['schedule']
            self.assertIsNone(panel.validate_schedule_artifact(out, identity, result['config']['schedule']))
            bad = {**identity, 'sequence_sha256': '0' * 64}
            self.assertEqual(panel.validate_schedule_artifact(out, bad, result['config']['schedule']).code, 'schedule_artifact_mismatch')
            manifest_path = out / 'schedule-manifest.json'
            manifest_path.chmod(0o644)
            manifest = json.loads(manifest_path.read_text())
            manifest['source_sha256'] = '0' * 64
            panel.dump(manifest_path, manifest)
            self.assertEqual(panel.validate_schedule_artifact(out, identity, result['config']['schedule']).code, 'schedule_artifact_mismatch')
            panel.dump(manifest_path, {**identity, 'schedule': result['config']['schedule']})
            (out / 'schedule.bin').chmod(0o644)
            (out / 'schedule.bin').write_bytes(b'corrupt')
            self.assertEqual(panel.validate_schedule_artifact(out, identity, result['config']['schedule']).code, 'schedule_artifact_mismatch')
            wrong = self.args(temp / 'source-mismatch', model, 'scheduled', base)
            wrong.server_source_commit = 'b' * 40
            self.assertEqual(self.invoke(wrong).code, 'baseline_mismatch')
            self.assertFalse(Path(wrong.output).exists())

    def test_invalid_parameters_and_baseline_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model = temp / 'model'; model.write_bytes(b'CPU')
            args = self.args(temp / 'unused', model)
            args.commit_max = 1
            self.assertEqual(panel.run(args).code, 'invalid_policy_parameters')
            args = self.args(temp / 'missing', model, 'confidence', temp / 'no-baseline')
            args.commit_min = None
            self.assertEqual(panel.run(args).code, 'invalid_policy_parameters')
            base = temp / 'base'
            self.invoke(self.args(base, model))
            args = self.args(temp / 'mismatched', model, 'periodic', base)
            args.max_tokens = 64
            self.assertEqual(panel.run(args).code, 'baseline_mismatch')
            self.assertFalse(Path(args.output).exists())
            args = self.args(base, model)
            self.assertEqual(panel.run(args).code, 'output_exists')


if __name__ == '__main__':
    if '--slots' in sys.argv:
        serve()
    else:
        unittest.main()
