#!/usr/bin/env python3
"""Emit explicit binary target/strict speculation panel configurations."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine-root', required=True)
    ap.add_argument('--result-root', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    common = {'H5_STATE': 'f16', 'H5_KV': 'f16', 'H5_DRAFT_WFMT': 'nvfp4', 'H5_DRAFT_MARGIN': '0',
              'H5_DETERMINISTIC_GEMM': '1', 'H5_ORDERED_PREP': 'off'}
    fp4 = {'H5_WEIGHTS': 'fp4', 'H5_ACT': 'fp4', 'H5_COMPACT': 'binary1', 'H5_FP4_PASSES': '2'}
    packed = {'H5_WEIGHTS': 'packed1', 'H5_ACT': 'f32', 'H5_CHUNK': '8'}
    hybrid = {'H5_WEIGHTS': 'packed1', 'H5_ACT': 'f32-r1-a8', 'H5_CHUNK': '16'}
    vocab = '/workspace/binary/models/vocab-65536.txt'
    nested = ['--draft-proposals', '15', '--commit-policy', 'nested', '--commit-min', '7', '--commit-max', '15', '--commit-confidence', '0.15']
    definitions = [
        ('int8-plain', {'H5_WEIGHTS': 'int8', 'H5_ACT': 'int8', 'H5_NO_SPEC': '1'}, [], 512),
        ('packed1-plain', {**packed, 'H5_NO_SPEC': '1'}, [], 8),
        ('packed1-s7', packed, ['--draft-proposals', '7'], 8),
        ('packed1-s7-vocab', {**packed, 'H5_DRAFT_VOCAB': vocab}, ['--draft-proposals', '7'], 8),
        ('packed1-n16-vocab', {**packed, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '0'}, nested, 8),
        ('packed1-n16-lookup', {**packed, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '8'}, nested, 8),
        ('packed1-a8-plain', {'H5_WEIGHTS': 'packed1', 'H5_ACT': 'a8', 'H5_CHUNK': '16', 'H5_NO_SPEC': '1'}, [], 16),
        ('packed1-hybrid-plain', {**hybrid, 'H5_NO_SPEC': '1'}, [], 16),
        ('packed1-hybrid-s7', hybrid, ['--draft-proposals', '7'], 16),
        ('packed1-hybrid-s7-vocab', {**hybrid, 'H5_DRAFT_VOCAB': vocab}, ['--draft-proposals', '7'], 16),
        ('packed1-hybrid-n16-vocab', {**hybrid, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '0'}, nested, 16),
        ('packed1-hybrid-n16-lookup', {**hybrid, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '8'}, nested, 16),
        ('binary1-two-plain', {**fp4, 'H5_NO_SPEC': '1'}, [], 512),
        ('binary1-two-s7', fp4, ['--draft-proposals', '7'], 512),
        ('binary1-two-s7-vocab', {**fp4, 'H5_DRAFT_VOCAB': vocab}, ['--draft-proposals', '7'], 512),
        ('binary1-two-n16-vocab', {**fp4, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '0'}, nested, 512),
        ('binary1-two-n16-lookup', {**fp4, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '8'}, nested, 512),
    ]
    definitions += [
        (name.replace('binary1-two-', 'binary1-joint-'), {**env, 'H5_FP4_PAIR': 'joint-r16'}, extra, prefill)
        for name, env, extra, prefill in definitions if name.startswith('binary1-two-')
    ]
    for name, env, extra, prefill in definitions:
        config = {
            'command': [a.engine_root + '/h5-server', '/workspace/binary/models/Bonsai-27B-Q1_0.gguf',
                        '/workspace/binary/models/procreations-r2-gptq2.safetensors',
                        '--host', '127.0.0.1', '--port', '8088', '--slots', '1', '--ctx', '8192',
                        '--prefill', str(prefill), *extra],
            'env': {**common, **env}, 'url': 'http://127.0.0.1:8088',
            'out': a.result_root + '/tasks-modern-' + name,
            'gate_args': ['--quick', '--limit', '32'],
            'startup_seconds': 180, 'run_seconds': 1800, 'metrics_path': '/metrics'
        }
        (a.out / (name + '.json')).write_text(json.dumps(config, indent=2) + '\n')
        if name in ('binary1-two-s7-vocab', 'binary1-joint-s7-vocab', 'binary1-joint-plain',
                    'packed1-plain', 'packed1-hybrid-plain', 'packed1-s7-vocab', 'packed1-hybrid-s7-vocab'):
            for slots in (2, 4, 5, 8, 16, 32):
                batch = json.loads(json.dumps(config))
                batch['command'][batch['command'].index('--slots') + 1] = str(slots)
                batch['out'] = a.result_root + f'/tasks-modern-{name}-b{slots}'
                batch['gate_args'] += ['--concurrency', str(slots)]
                (a.out / f'{name}-b{slots}.json').write_text(json.dumps(batch, indent=2) + '\n')


if __name__ == '__main__':
    main()
