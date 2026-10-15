#!/usr/bin/env python3
"""Run one named integrated single-stream preset on the exact retained engine binary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def command(preset, mode, server, model, drafter, port, context, slots, prefix, mmproj, backend, plan_read=None, plan_sha256=None, plan_write=None):
    try:
        spec = json.loads(preset.read_text())
        if spec['schema'] != 'halo-integrated-serving-v1' or mode not in spec['modes']:
            return {'state': 'error', 'code': 'unknown_preset'}
        if mode in spec['unsupported_modes']:
            return {'state': 'error', 'code': 'missing_engine_feature', 'mode': mode}
        if not (1 <= port <= 65535 and 1024 <= context <= 16384):
            return {'state': 'error', 'code': 'invalid_port_or_context'}
        for path in (server, model, drafter):
            if not path.is_file():
                return {'state': 'error', 'code': 'missing_input', 'path': str(path)}
        if hashlib.sha256(server.read_bytes()).hexdigest() != spec['server_sha256']:
            return {'state': 'error', 'code': 'server_hash_mismatch'}
        # drafter checkpoint: the mode's "drafter" (else the preset's) names an entry of "drafters"
        want = spec['drafters'][spec['modes'][mode].get('drafter', spec['drafter'])]
        h = hashlib.sha256()
        with open(drafter, 'rb') as f:
            for chunk in iter(lambda: f.read(64 << 20), b''): h.update(chunk)
        if h.hexdigest() != want['sha256']:
            return {'state': 'error', 'code': 'drafter_hash_mismatch', 'expected': want}
        if slots not in spec['allowed_slots']:
            return {'state': 'error', 'code': 'invalid_slots_for_preset'}
        if spec['planning'] == 'frozen':
            plan = preset.parent / spec['plan_file']
            if hashlib.sha256(plan.read_bytes()).hexdigest() != spec['plan_sha256']:
                return {'state': 'error', 'code': 'plan_hash_mismatch'}
        elif spec['planning'] == 'autotune' and spec['plan_file'] is None and spec['plan_sha256'] is None:
            plan = None
        else:
            return {'state': 'error', 'code': 'unknown_planning'}
        if plan_read is not None or plan_write is not None or plan_sha256 is not None:
            if spec['planning'] != 'autotune' or (plan_read is not None and plan_write is not None):
                return {'state': 'error', 'code': 'conflicting_plan_selection'}
            if plan_read is not None:
                if plan_sha256 is None or hashlib.sha256(plan_read.read_bytes()).hexdigest() != plan_sha256:
                    return {'state': 'error', 'code': 'plan_hash_mismatch'}
                plan = plan_read
            elif plan_sha256 is not None:
                return {'state': 'error', 'code': 'unused_plan_hash'}
            elif plan_write.exists() or not plan_write.parent.is_dir():
                return {'state': 'error', 'code': 'plan_output_must_be_new'}
        selected = spec['modes'][mode]
        environment = {k: v for k, v in os.environ.items() if not k.startswith('H5_')}
        environment.update(spec['environment'])
        environment.update(selected['environment'])
        if 'H5_DRAFT_VOCAB' in environment and not os.path.isabs(environment['H5_DRAFT_VOCAB']):
            environment['H5_DRAFT_VOCAB'] = str((preset.parent.parent.parent / environment['H5_DRAFT_VOCAB']).resolve())   # repository-relative
        if plan is not None:
            environment['H5_PLAN_READ'] = str(plan.resolve())
        if plan_write is not None:
            environment['H5_PLAN_WRITE'] = str(plan_write.resolve())
        if mode == 'http-prefix':
            if prefix is None or not prefix.is_file() or not prefix.stat().st_size:
                return {'state': 'error', 'code': 'prefix_file_required'}
            environment['H5_SERVING_PREFIX'] = str(prefix.resolve())
        elif prefix is not None:
            return {'state': 'error', 'code': 'unused_prefix_file'}
        args = [str(server.resolve()), str(model.resolve()), str(drafter.resolve()),
                '--slots', str(slots), '--ctx', str(context), '--prefill', str(spec['prefill_budget']), '--port', str(port), *selected['arguments']]
        if mode == 'vision-cuda':
            if mmproj is None or not mmproj.is_file() or not mmproj.stat().st_size or backend is None or not backend.is_dir():
                return {'state': 'error', 'code': 'vision_inputs_required'}
            for name in ['libggml-cuda.so', 'libggml.so', 'libllama.so', 'libmtmd.so']:
                if not (backend / name).is_file():
                    return {'state': 'error', 'code': 'cuda_encoder_backend_missing', 'library': name}
            library_path = str(backend.resolve())
            if 'LD_LIBRARY_PATH' in environment:
                library_path += os.pathsep + environment['LD_LIBRARY_PATH']
            environment['LD_LIBRARY_PATH'] = library_path
            environment['GGML_BACKEND_PATH'] = str(backend.resolve())
            args += ['--mmproj', str(mmproj.resolve())]
        elif mmproj is not None or backend is not None:
            return {'state': 'error', 'code': 'unused_vision_inputs'}
        return {'state': 'ready', 'argv': args, 'environment': environment,
                'source': spec['engine_source'], 'selected_h5': {k: v for k, v in environment.items() if k.startswith('H5_')}}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'state': 'error', 'code': 'preset_input_or_io', 'detail': str(error)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', type=Path, required=True)
    p.add_argument('--mode', required=True)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--drafter', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--context', type=int, required=True)
    p.add_argument('--slots', type=int, required=True)
    p.add_argument('--prefix-file', type=Path)
    p.add_argument('--mmproj', type=Path)
    p.add_argument('--backend-libraries', type=Path)
    p.add_argument('--plan', type=Path, help='Explicit newly generated complete batch/original-scale plan')
    p.add_argument('--plan-sha256', help='Required frozen plan identity with --plan')
    p.add_argument('--write-plan', type=Path, help='New complete startup plan output; distinct from --plan')
    p.add_argument('--describe', action='store_true')
    a = p.parse_args()
    result = command(a.preset, a.mode, a.server, a.model, a.drafter, a.port, a.context, a.slots, a.prefix_file, a.mmproj, a.backend_libraries, a.plan, a.plan_sha256, a.write_plan)
    if result['state'] == 'error':
        print(json.dumps(result), file=sys.stderr); return 1
    print(json.dumps({k: v for k, v in result.items() if k != 'environment'}), flush=True)
    if a.describe:
        return 0
    try:
        os.execve(result['argv'][0], result['argv'], result['environment'])
    except OSError as error:
        print(json.dumps({'state': 'error', 'code': 'server_exec_failed', 'detail': str(error)}), file=sys.stderr); return 1


if __name__ == '__main__':
    sys.exit(main())
