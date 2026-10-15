#!/usr/bin/env python3
"""CPU-only contract proof, including owned server cleanup on HTTP failure."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('sweep', ROOT / 'tools/long_draft_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
sweep.SOURCE = 'f91d574754ffcffa20989e5c5ea567c5f0eb8cef'  # Explicit CPU mock provenance, not a GPU claim.


def serve():
    port = int(sys.argv[sys.argv.index('--port') + 1])
    proposals = int(sys.argv[sys.argv.index('--draft-proposals') + 1])
    assert proposals == int(os.environ['SWEEP_TEST_EXPECT_PROPOSALS'])
    gold = json.loads(Path(os.environ['SWEEP_TEST_GOLD']).read_text())
    state = {'requests': 0, 'generated_tokens': 0, 'steps': 0, 'busy_s': 0,
             'verify_s': 0, 'draft_s': 0, 'verify_rounds': 0, 'accepted_drafts': 0,
             'proposed_drafts': 0, 'graphs_captured': 0, 'graph_replays': 0, 'active': 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, code=200):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send({'status': 'ok', 'draft_proposals': proposals, 'draft_query_rows': proposals + 1,
                       'verifier_rows': proposals + 1} if self.path == '/health' else state)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body['temperature'] == 0 and body['chat_template_kwargs']['enable_thinking'] is False
            assert body['stream'] is False
            index = state['requests'] - 1
            if index == 3 and os.environ.get('SWEEP_TEST_FAILURE') == 'yes':
                self.send({'error': 'deliberate CPU test HTTP failure'}, 500)
                return
            expected = sweep.num(gold[max(index, 0)]['answer'].split('####')[-1])
            text = '\\boxed{' + str(expected if index != 2 else expected + 1) + '}'
            if index == 6:
                text = '9' * 1000  # Real lossy-output overflow, must be recorded as wrong.
            state['requests'] += 1
            for key in ['steps', 'verify_rounds', 'graph_replays']:
                state[key] += 1
            state['generated_tokens'] += 8
            state['accepted_drafts'] += proposals
            state['proposed_drafts'] += proposals
            for key in ['busy_s', 'verify_s', 'draft_s']:
                state[key] += .01
            tokens = 0 if index == 5 else 8
            self.send({'choices': [{'message': {'content': '' if tokens == 0 else text}, 'finish_reason': 'length' if index == 4 else 'stop'}],
                       'usage': {'prompt_tokens': 20, 'completion_tokens': tokens, 'total_tokens': 20 + tokens},
                       'timings': {'native_prefill_s': .001, 'verify_s': .003, 'draft_s': .002,
                                   'ttft_s': .002 if tokens else 0, 'total_s': .01, 'decode_tok_per_s': 875 if tokens else 0,
                                   'proposed_drafts': proposals, 'accepted_drafts': proposals,
                                   'draft_proposals': proposals, 'draft_query_rows': proposals + 1,
                                   'verifier_rows_per_full_round': proposals + 1,
                                   'checkpoint_length_extrapolation': proposals != 7, 'bonus_policy': 'greedy',
                                   'verify_rounds': 1, 'verifier_rows': proposals + 1,
                                   'prefix_histogram': [0]*proposals + [1],
                                   'position_proposed': [1]*proposals, 'position_reached': [1]*proposals,
                                   'position_accepted': [1]*proposals, 'position_emitted': [0]*(proposals+1),
                                   'position_reject_target_eos': [0]*proposals,
                                   'position_reject_proposed_eos': [0]*proposals,
                                   'position_reject_other': [0]*proposals}})

    HTTPServer(('127.0.0.1', port), Handler).serve_forever()


class Contract(unittest.TestCase):
    def test_order(self):
        plan = sweep.schedule()
        self.assertEqual(len(plan), 11)
        self.assertEqual({row['draft_proposals'] for row in plan}, {7, 16, 32, 64})
        self.assertTrue(all(row['query_rows'] == row['draft_proposals'] + 1 for row in plan))
        self.assertEqual({row['delta'] for row in plan}, set(sweep.GRID))
        self.assertEqual([row['index'] for row in plan if row['delta'] == '0'], [0, 10])
        self.assertEqual(plan, sweep.schedule())

    def test_complete_and_failed_point_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model = temp / 'model.gguf'
            model.write_bytes(b'cpu-test-not-a-model')
            for failure in (False, True):
                out = temp / ('failure' if failure else 'complete')
                with contextlib.redirect_stdout(io.StringIO()):
                    sweep.init(out)
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', 0))
                    port = probe.getsockname()[1]
                os.environ['SWEEP_TEST_GOLD'] = str(out / 'selection.json')
                os.environ['SWEEP_TEST_FAILURE'] = 'yes' if failure else 'no'
                os.environ['SWEEP_TEST_EXPECT_PROPOSALS'] = '7'
                args = argparse.Namespace(output=str(out), index=0, server=__file__, model=str(model),
                                          drafter=str(model), port=port, deadline_s=15)
                with contextlib.redirect_stdout(io.StringIO()):
                    code = sweep.run(args)
                point = out / sweep.schedule()[0]['name']
                summary = json.loads((point / 'summary.json').read_text())
                self.assertEqual(code, 1 if failure else 0)
                self.assertEqual(summary['state'], 'failed' if failure else 'complete')
                self.assertEqual(summary['server_exit_code'], -15)
                if failure:
                    self.assertIsNone(summary['accuracy'])
                    self.assertIsNone(summary['correct'])
                    self.assertEqual(summary['requests'], 3)
                    self.assertEqual(summary['error']['code'], 'http_error')
                    self.assertEqual(len(summary['failed_question_ids']), 1)
                    self.assertEqual(len(summary['unattempted_question_ids']), 28)
                else:
                    self.assertEqual(summary['correct'], 29)
                    self.assertEqual(summary['eos_count'], 31)
                    self.assertEqual(summary['cap_count'], 1)
                    self.assertEqual(len(summary['wrong_question_ids']), 3)
                    self.assertEqual(len(list(point.glob('response-*.json'))), 32)
                    self.assertEqual(summary['tokens'], 248)
                with socket.socket() as probe:
                    self.assertNotEqual(probe.connect_ex(('127.0.0.1', port)), 0)
                table = json.loads((out / 'table.json').read_text())
                self.assertEqual(len(table), 11)
                self.assertTrue(all(row['state'] == 'not_run' for row in table[1:]))

    def test_all_lengths_same_server_source_and_binary(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model = temp / 'model'; model.write_bytes(b'cpu-mock')
            out = temp / 'all-lengths'
            with contextlib.redirect_stdout(io.StringIO()):
                sweep.init(out)
            os.environ['SWEEP_TEST_GOLD'] = str(out / 'selection.json')
            os.environ['SWEEP_TEST_FAILURE'] = 'no'
            for entry in sweep.schedule()[:5]:
                os.environ['SWEEP_TEST_EXPECT_PROPOSALS'] = str(entry['draft_proposals'])
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
                args = argparse.Namespace(output=str(out), index=entry['index'], server=__file__, model=str(model),
                                          drafter=str(model), port=port, deadline_s=15)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(sweep.run(args), 0)
                receipt = json.loads((out / entry['name'] / 'summary.json').read_text())
                self.assertEqual(receipt['draft_proposals'], entry['draft_proposals'])
                self.assertEqual(receipt['query_rows'], entry['draft_proposals'] + 1)
                self.assertEqual(receipt['correct'], 29)
                self.assertIn('--draft-proposals', receipt['command'])


if __name__ == '__main__':
    if '--slots' in sys.argv:
        serve()
    else:
        unittest.main()
