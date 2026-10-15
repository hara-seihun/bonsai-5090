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
    common = {'H5_STATE': 'f16', 'H5_KV': 'f16', 'H5_DRAFT_WFMT': 'gu=b2,down=b3,fc=b3,attn=b3,conv=b3', 'H5_DRAFT_MARGIN': '0',
              'H5_DETERMINISTIC_GEMM': '1', 'H5_ORDERED_PREP': 'off'}
    fp4 = {'H5_WEIGHTS': 'fp4', 'H5_ACT': 'fp4', 'H5_COMPACT': 'binary1', 'H5_FP4_PASSES': '2'}
    packed = {'H5_WEIGHTS': 'packed1', 'H5_ACT': 'f32', 'H5_CHUNK': '8'}
    hybrid = {'H5_WEIGHTS': 'packed1', 'H5_ACT': 'f32-r1-a8', 'H5_CHUNK': '16'}
    vocab = '/workspace/binary/models/vocab-65536.txt'
    nested = ['--draft-proposals', '15', '--commit-policy', 'nested', '--commit-min', '7', '--commit-max', '15', '--commit-confidence', '0.15', '--commit-calibration', '-0.9292,1.1550,-0.0298']
    definitions = [
        ('int8-plain', {'H5_WEIGHTS': 'int8', 'H5_ACT': 'int8', 'H5_NO_SPEC': '1'}, [], 512),
        ('packed1-plain', {**packed, 'H5_NO_SPEC': '1'}, [], 8),
        ('packed1-s7', packed, ['--draft-proposals', '7'], 8),
        ('packed1-s7-vocab', {**packed, 'H5_DRAFT_VOCAB': vocab}, ['--draft-proposals', '7'], 8),
        ('packed1-n16-vocab', {**packed, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '0'}, nested, 8),
        ('packed1-n16-lookup', {**packed, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '8'}, nested, 8),
        ('packed1-a8-plain', {'H5_WEIGHTS': 'packed1', 'H5_ACT': 'a8', 'H5_CHUNK': '16', 'H5_NO_SPEC': '1'}, [], 16),
        ('packed1-hybrid-plain', {**hybrid, 'H5_NO_SPEC': '1'}, [], 16),
        ('packed1-dp4a-pf32-plain', {**hybrid, 'H5_A8_PROJECT': 'dp4a', 'H5_A8_GU': 'off', 'H5_NO_SPEC': '1'}, [], 32),
        ('packed1-digits-plain', {**hybrid, 'H5_A8_PROJECT': 'fp4-digits', 'H5_A8_GU': 'off', 'H5_NO_SPEC': '1'}, [], 32),
        ('packed1-digits-ragged-plain', {**hybrid, 'H5_A8_PROJECT': 'fp4-digits-ragged', 'H5_A8_GU': 'off', 'H5_NO_SPEC': '1'}, [], 32),
        ('packed1-three-digits-ragged-plain', {**hybrid, 'H5_A8_PROJECT': 'fp4-three-digits-ragged', 'H5_A8_GU': 'off', 'H5_NO_SPEC': '1'}, [], 32),
        ('packed1-whole-ffn-plain', {**hybrid, 'H5_A8_PROJECT': 'fp4-three-digits-ragged', 'H5_A8_GU': 'off', 'H5_A8_FFN': 'three-direct-r16', 'H5_NO_SPEC': '1'}, [], 32),
        ('packed1-batchpair-plain', {**hybrid, 'H5_ACT': 'f32-r1-a8-batchpair', 'H5_NO_SPEC': '1'}, [], 16),
        ('packed1-fused-plain', {**hybrid, 'H5_A8_GU': 'fused', 'H5_NO_SPEC': '1'}, [], 16),
        ('packed1-a8-fused-plain', {**hybrid, 'H5_ACT': 'a8', 'H5_A8_GU': 'fused', 'H5_NO_SPEC': '1'}, [], 16),
        ('packed1-hybrid-s7', hybrid, ['--draft-proposals', '7'], 16),
        ('packed1-hybrid-s7-vocab', {**hybrid, 'H5_DRAFT_VOCAB': vocab}, ['--draft-proposals', '7'], 16),
        ('packed1-hybrid-n16-vocab', {**hybrid, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '0'}, nested, 16),
        ('packed1-hybrid-n16-lookup', {**hybrid, 'H5_DRAFT_VOCAB': vocab, 'H5_DRAFT_NEST': '8', 'H5_LOOKUP_EXTRA': '8'}, nested, 16),
        ('binary1-two-plain', {**fp4, 'H5_NO_SPEC': '1'}, [], 512),
        ('binary1-sf-off-plain', {**fp4, 'H5_FP4_PAIR': 'joint-r16', 'H5_NATIVE_SF': 'off', 'H5_NO_SPEC': '1'}, [], 32),
        ('binary1-sf1-plain', {**fp4, 'H5_FP4_PAIR': 'joint-r16', 'H5_NATIVE_SF': 'sf1-r32', 'H5_NO_SPEC': '1'}, [], 32),
        ('binary1-sf2-plain', {**fp4, 'H5_FP4_PAIR': 'joint-r16', 'H5_NATIVE_SF': 'sf2-r32', 'H5_NO_SPEC': '1'}, [], 32),
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
            'env': {**common, **env, **({'H5_TOPK_SPLIT': '1'} if 'H5_DRAFT_VOCAB' in env else {})},
            'url': 'http://127.0.0.1:8088',
            'out': a.result_root + '/tasks-modern-' + name,
            'gate_args': ['--quick', '--limit', '32'],
            'startup_seconds': 180, 'run_seconds': 1800, 'metrics_path': '/metrics'
        }
        (a.out / (name + '.json')).write_text(json.dumps(config, indent=2) + '\n')
        if name in ('binary1-two-s7-vocab', 'binary1-joint-s7-vocab', 'binary1-joint-plain',
                    'packed1-plain', 'packed1-hybrid-plain', 'packed1-batchpair-plain', 'packed1-fused-plain',
                    'packed1-dp4a-pf32-plain', 'packed1-digits-plain', 'packed1-digits-ragged-plain',
                    'packed1-three-digits-ragged-plain', 'packed1-whole-ffn-plain',
                    'binary1-sf-off-plain', 'binary1-sf1-plain', 'binary1-sf2-plain',
                    'packed1-s7-vocab', 'packed1-hybrid-s7-vocab'):
            for slots in (2, 4, 5, 8, 16, 32):
                batch = json.loads(json.dumps(config))
                batch['command'][batch['command'].index('--slots') + 1] = str(slots)
                batch['out'] = a.result_root + f'/tasks-modern-{name}-b{slots}'
                batch['gate_args'] += ['--concurrency', str(slots)]
                (a.out / f'{name}-b{slots}.json').write_text(json.dumps(batch, indent=2) + '\n')


if __name__ == '__main__':
    main()
