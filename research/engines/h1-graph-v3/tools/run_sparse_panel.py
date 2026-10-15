#!/usr/bin/env python3
"""One foreground server, one immutable panel, and unconditional process cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--depth', required=True, type=int)
    p.add_argument('--threshold', required=True, type=int)
    p.add_argument('--stage', choices=('both','target','draft'), default='both')
    p.add_argument('--check-logits', action='store_true', help='Diagnostic full-head oracle; exclude from performance panels')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('bench_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not 1 <= a.depth <= 64 or not 2 <= a.threshold <= 248320:
        p.error('depth must be 1..64 and threshold 2..248320')
    if a.out.exists():
        p.error('output path already exists')
    a.out.mkdir(parents=True)
    root = Path(__file__).resolve().parent.parent
    preset = subprocess.run([sys.executable, str(root/'tools/pod/preset_env.py'),
        str(root/'results/integrated/selected-preset.json'), '--mode', 'strict-native-procreations'],
        capture_output=True, text=True)
    if preset.returncode:
        (a.out/'error.json').write_text(json.dumps({'stage':'preset','returncode':preset.returncode,'stderr':preset.stderr}))
        return 1
    env = {k:v for k,v in os.environ.items() if not k.startswith('H5_')}
    for line in preset.stdout.splitlines():
        fields = shlex.split(line)
        if not fields:
            continue
        if fields[0] == 'unset':
            for key in fields[1:]:
                env.pop(key,None)
            continue
        assignments = fields[1:] if fields[0] == 'export' else fields
        if not assignments or any('=' not in field for field in assignments):
            (a.out/'error.json').write_text(json.dumps({'stage':'preset','error':'invalid assignment','line':line}))
            return 1
        for field in assignments:
            k,v = field.split('=',1)
            if k.startswith('H5_') or k == 'M':
                env[k] = v
            else:
                (a.out/'error.json').write_text(json.dumps({'stage':'preset','error':'unknown key','key':k}))
                return 1
    env.update(H5_PLAN_READ=str(root/'results/integrated/plan-b973-ingest16.txt'),
        H5_PLAN_FILL='1', H5_DETERMINISTIC_GEMM='1', H5_ORDERED_PREP='out8',
        H5_DRAFT_MARGIN='0', H5_DRAFT_NEST='0', H5_LOOKUP_EXTRA='0', H5_NO_SPEC='0',
        H5_GRAMMAR_SPARSE_THRESHOLD=str(a.threshold), H5_GRAMMAR_SPARSE_STAGE=a.stage,
        H5_GRAMMAR_SPARSE_LOGIT_CHECK='1' if a.check_logits else '0')
    cmd = [str(a.binary), '/root/models/bonsai.gguf', '/root/models/drafters/procreations-r2.safetensors',
        '--host','127.0.0.1','--slots','1','--ctx','8192','--prefill','128', '--port',str(a.port),
        '--draft-proposals',str(a.depth),'--commit-policy','fixed']
    receipt = {'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'binary':str(a.binary), 'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        'command':cmd, 'environment':{k:v for k,v in env.items() if k.startswith('H5_')},
        'depth':a.depth,'threshold':a.threshold, 'started_at':time.time()}
    (a.out/'config.json').write_text(json.dumps(receipt,indent=2))
    server = None
    try:
        with (a.out/'server.log').open('w') as log:
            server = subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            deadline = time.monotonic()+150
            while True:
                if server.poll() is not None:
                    raise RuntimeError('server exited before ready: '+str(server.returncode))
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError,TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('server readiness deadline')
                    time.sleep(.2)
            receipt['ready_at'] = time.time()
            (a.out/'config.json').write_text(json.dumps(receipt,indent=2))
            rest = a.bench_args[1:] if a.bench_args[:1] == ['--'] else a.bench_args
            result = subprocess.run([sys.executable,str(root/'tools/grammar_bench.py'),
                '--server',f'http://127.0.0.1:{a.port}','--out',str(a.out/'panel'),
                '--draft-limit',str(a.depth),*rest],cwd=root,env=env)
            receipt['panel_returncode'] = result.returncode
            return result.returncode
    except (OSError,RuntimeError) as exc:
        (a.out/'error.json').write_text(json.dumps({'stage':'server-panel','error':str(exc)}))
        print(str(exc),file=sys.stderr)
        return 1
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        receipt['finished_at'] = time.time()
        receipt['server_returncode'] = server.returncode if server is not None else None
        (a.out/'config.json').write_text(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    sys.exit(main())
