#!/usr/bin/env python3
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import adaptive_commit_panel as panel
import serve_integrated as serve
import selected_build_defines as selected_build


class PresetTest(unittest.TestCase):
    def test_explicit_tree_route_conflicts(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'engine.json'
            for values in [{'H5_TREE2': '1'}, {'H5_TREE2': '1', 'H5_CLONE_FULL_KV': '0'},
                           {'H5_TREE2': '0', 'H5_CLONE_FULL_KV': '1'}, {'H5_CARRIER_WF0': '1'}]:
                p.write_text(json.dumps(values))
                self.assertIsInstance(panel.engine_environment(p), panel.Error)
            for mode, margin, proposals in [('confidence', 0, 7), ('fixed', 16, 7), ('fixed', 0, 32)]:
                p.write_text(json.dumps({'H5_TREE2': '1', 'H5_CLONE_FULL_KV': '1'}))
                args = panel.parser().parse_args(['--server', '/missing', '--model', '/missing', '--drafter', '/missing',
                    '--server-source-commit', '0' * 40, '--selection', '/missing', '--prompts', '/missing',
                    '--compact', 'dense5', '--draft-margin', str(margin), '--port', '18080',
                    '--request-timeout-s', '45', '--warmup-tokens', '128', '--deadline-s', '50',
                    '--start', '0', '--count', '1', '--max-tokens', '64', '--draft-proposals', str(proposals),
                    '--role', 'baseline' if mode == 'fixed' else 'candidate', '--commit-policy', mode,
                    '--output', str(Path(folder) / 'out'), '--engine-environment', str(p)] +
                    (['--baseline', '/missing', '--commit-min', '0', '--commit-max', '4', '--commit-confidence', '1'] if mode == 'confidence' else []))
                self.assertEqual(panel.run(args).code, 'engine_policy_conflict')
            p.write_text(json.dumps({'H5_TREE2': '0', 'H5_CLONE_FULL_KV': '0', 'H5_CARRIER_WF0': '0', 'H5_LOAD_THREADS': '4'}))
            self.assertIsInstance(panel.engine_environment(p), panel.EngineEnvironment)

    def test_launcher_identity_and_no_inherited_h5(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            server, model, drafter, plan = [folder / n for n in ['server', 'model', 'drafter', 'plan']]
            for p in [server, model, drafter, plan]:
                p.write_bytes(b'fixture')
            preset = json.loads((Path(__file__).resolve().parents[1] / 'results/integrated/selected-preset.json').read_text())
            preset['server_sha256'] = hashlib.sha256(b'fixture').hexdigest()
            preset['plan_sha256'] = preset['server_sha256']; preset['plan_file'] = 'plan'
            for d in preset['drafters'].values(): d['sha256'] = preset['server_sha256']
            p = folder / 'preset.json'; p.write_text(json.dumps(preset))
            with patch.dict(os.environ, {'H5_MEGA': '1', 'H5_WSCALE': 'e4m3', 'H5_CARRIER_WF0': '1'}):
                result = serve.command(p, 'strict-tree', server, model, drafter, 18080, 4096, 1, None, None, None)
            self.assertEqual(result['state'], 'ready')
            self.assertNotIn('H5_MEGA', result['environment']); self.assertNotIn('H5_WSCALE', result['environment'])
            self.assertEqual(result['selected_h5']['H5_CARRIER_WF0'], '0')
            self.assertEqual(result['selected_h5']['H5_TREE2'], '1')
            self.assertEqual(result['selected_h5']['H5_DRAFT_MARGIN'], '0')
            self.assertEqual(serve.command(p, 'unknown', server, model, drafter, 18080, 4096, 1, None, None, None)['code'], 'unknown_preset')
            server.write_bytes(b'corrupt')
            self.assertEqual(serve.command(p, 'adaptive7', server, model, drafter, 18080, 4096, 1, None, None, None)['code'], 'server_hash_mismatch')
            server.write_bytes(b'fixture'); plan.write_bytes(b'corrupt')
            self.assertEqual(serve.command(p, 'adaptive7', server, model, drafter, 18080, 4096, 1, None, None, None)['code'], 'plan_hash_mismatch')
            plan.write_bytes(b'fixture')
            preset['drafters']['procreations-r2']['sha256'] = hashlib.sha256(b'other drafter').hexdigest(); p.write_text(json.dumps(preset))
            self.assertEqual(serve.command(p, 'strict-native-procreations', server, model, drafter, 18080, 4096, 1, None, None, None)['code'], 'drafter_hash_mismatch')
            self.assertEqual(serve.command(p, 'relaxed15-vocab64', server, model, drafter, 18080, 4096, 1, None, None, None)['state'], 'ready')

    def test_high_row_route_plan_custody(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); fake = folder / 'fixture'; fake.write_bytes(b'fixture')
            for filename, mode, scale in [('selected-batch-preset.json', 'f02-h32', 'e4m3'), ('selected-degree8-preset.json', 'degree8', None)]:
                spec = json.loads((root / 'results/integrated' / filename).read_text())
                spec['server_sha256'] = hashlib.sha256(b'fixture').hexdigest(); spec['unsupported_modes'] = []
                for d in spec['drafters'].values(): d['sha256'] = spec['server_sha256']
                preset = folder / 'preset.json'; preset.write_text(json.dumps(spec))
                args = [preset, mode, fake, fake, fake, 18080, 4096, 64, None, None, None]
                new_plan = folder / 'new-plan'
                result = serve.command(*args, plan_write=new_plan)
                self.assertEqual(result['state'], 'ready')
                self.assertEqual(result['selected_h5'].get('H5_WSCALE'), scale)
                self.assertNotIn('H5_COMPACT', result['selected_h5'])
                self.assertEqual(result['selected_h5']['H5_PLAN_WRITE'], str(new_plan))
                self.assertEqual(serve.command(*args, plan_read=fake)['code'], 'plan_hash_mismatch')
                frozen = serve.command(*args, plan_read=fake, plan_sha256=spec['server_sha256'])
                self.assertEqual(frozen['state'], 'ready')
                self.assertNotIn('H5_PLAN_WRITE', frozen['selected_h5'])
                self.assertEqual(serve.command(*args, plan_read=fake, plan_write=new_plan)['code'], 'conflicting_plan_selection')

    def test_selected_flags_and_vision_route(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(selected_build.flags(root / 'results/integrated/selected-preset.json')['state'], 'ready')
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            fake = folder / 'fixture'; fake.write_bytes(b'fixture')
            spec = json.loads((root / 'results/integrated/selected-vision-preset.json').read_text())
            spec['server_sha256'] = hashlib.sha256(b'fixture').hexdigest()
            spec['plan_sha256'] = spec['server_sha256']; spec['plan_file'] = fake.name
            for d in spec['drafters'].values(): d['sha256'] = spec['server_sha256']
            preset = folder / 'vision.json'; preset.write_text(json.dumps(spec))
            self.assertEqual(serve.command(preset, 'vision-cuda', fake, fake, fake, 18080, 4096, 1, None, None, None)['code'], 'vision_inputs_required')
            backend = folder / 'backend'; backend.mkdir()
            self.assertEqual(serve.command(preset, 'vision-cuda', fake, fake, fake, 18080, 4096, 1, None, fake, backend)['code'], 'cuda_encoder_backend_missing')
            for library in ['libggml-cuda.so', 'libggml.so', 'libllama.so', 'libmtmd.so']:
                (backend / library).write_bytes(b'fixture')
            result = serve.command(preset, 'vision-cuda', fake, fake, fake, 18080, 4096, 1, None, fake, backend)
            self.assertEqual(result['state'], 'ready')
            self.assertIn('--vision-gpu', result['argv']); self.assertIn('--mmproj', result['argv'])
            self.assertEqual(result['selected_h5']['H5_DRAFT_MARGIN'], '0')
            self.assertEqual(result['argv'][-1], str(fake.resolve()))
            spec = json.loads((root / 'results/integrated/selected-preset.json').read_text())
            spec['build_defines'].pop(); preset.write_text(json.dumps(spec))
            self.assertEqual(selected_build.flags(preset)['code'], 'incomplete_or_unknown_build_contract')


if __name__ == '__main__':
    unittest.main()
