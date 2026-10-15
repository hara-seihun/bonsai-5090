#!/usr/bin/env python3
"""Durable benchmark driver; GPU custody is released between <=90s chunks."""
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path('/workspace/halo-6000')
OUT = ROOT / 'bench/matmul-tiles-results'
BINARY = ROOT / 'build/matmul-tiles-v1'
SNAPSHOT = ROOT / 'build/matmul-tiles-v1-src'
FIXTURE = ROOT / 'build/matmul-fixture'
CASES = ['v1-k', 'v1-qkv', 'v1-gdn_out', 'v1-gate', 'v1-down',
         'hybrid-shared_down', 'hybrid-qkv', 'hybrid-gdn_out', 'v1-head']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity():
    files = list(SNAPSHOT.iterdir()) + [BINARY, pathlib.Path(__file__)]
    return {str(p): sha(p) for p in files if p.is_file()}


def chunk(key, columns, subset):
    name = f'{key}-s{columns}' + ('-subset' if subset else '')
    start = time.monotonic()
    receipt = {'case': key, 'columns': columns, 'subset': subset,
               'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    code = 1
    try:
        expected = json.loads((OUT / 'identities.json').read_text())
        if identity() != expected:
            raise RuntimeError('immutable build/source identity changed')
        with open(OUT / (name + '-before.txt'), 'w') as context:
            subprocess.run(['nvidia-smi', '-q'], stdout=context, check=True)
            subprocess.run(['ps', '-eo', 'pid,ppid,pgid,args'], stdout=context, check=True)
        command = ['timeout', '85', str(BINARY), str(FIXTURE), key, str(columns)]
        if subset:
            command.append('subset')
        with open(OUT / (name + '.jsonl'), 'w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        receipt['command'] = command
    except Exception as error:
        receipt['error'] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        with open(OUT / (name + '-after.txt'), 'w') as context:
            subprocess.run(['nvidia-smi', '-q'], stdout=context)
        receipt.update(exit_code=code, state='succeeded' if code == 0 else 'failed',
                       wall_seconds=time.monotonic() - start)
        (OUT / (name + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    return code


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    if len(sys.argv) > 1:
        return chunk(sys.argv[2], int(sys.argv[3]), sys.argv[4] == '1')
    ids = identity()
    (OUT / 'identities.json').write_text(json.dumps(ids, indent=2) + '\n')
    tasks = [(key, col, False) for key in CASES for col in (13, 16, 29, 32, 65)]
    tasks += [(key, 29, True) for key in ('v1-gate', 'v1-down')]
    for key, columns, subset in tasks:
        command = ['flock', str(ROOT / 'gpu.lock'), 'timeout', '90', sys.executable,
                   str(pathlib.Path(__file__).resolve()), '--chunk', key, str(columns), str(int(subset))]
        code = subprocess.run(command).returncode
        print(json.dumps({'case': key, 'columns': columns, 'subset': subset, 'exit_code': code}), flush=True)
        if code:
            return code
    return 0


if __name__ == '__main__':
    sys.exit(main())
