#!/usr/bin/env python3
"""Own a server for a bounded benchmark; always reap it before returning.

serve_bench.py TAG --env H5_STATE=f16 --env H5_GDN_DELTA=1 --gsm 256 --loads 1,8,32,64
Run on the GPU pod from the engine root. Outputs include environment and raw server log.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('--env', action='append', default=[])
p.add_argument('--gsm', type=int, default=0)
p.add_argument('--loads', default='')
p.add_argument('--rates', default='', help='Poisson arrival rates, requests/s')
p.add_argument('--prefill', type=int, default=512)
p.add_argument('--slots', type=int, default=64)
p.add_argument('--ctx', type=int, default=4096)
p.add_argument('--tokens', type=int, default=512)
p.add_argument('--port', type=int, default=8000)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
os.chdir(root)
out = root / 'results' / 'combined'
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = '/root/llama/build/bin:' + env.get('LD_LIBRARY_PATH', '')
for value in a.env:
    key, val = value.split('=', 1)
    env[key] = val
meta = {'args': vars(a), 'environment': {k:v for k,v in env.items() if k.startswith('H5_')}}
(out / (a.tag + '.meta.json')).write_text(json.dumps(meta, indent=2))
cmd = ['./h5-server', '/root/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf', '/root/models/model.safetensors', '--port', str(a.port), '--slots', str(a.slots), '--ctx', str(a.ctx), '--prefill', str(a.prefill)]
url = f'http://127.0.0.1:{a.port}'
with (out / (a.tag + '.server.log')).open('w') as log:
    server = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 100
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'server exited {server.returncode}; see {log.name}')
            try:
                with urllib.request.urlopen(url + '/health', timeout=1) as r:
                    if r.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('server startup')
            time.sleep(.1)
        def run(command, name):
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
            (out / (a.tag + '.' + name)).write_text(result.stdout)
            print(result.stdout, flush=True)
            result.check_returncode()
        for c in filter(None, a.loads.split(',')):
            run([sys.executable, 'tools/loadtest.py', url, '--concurrency', c, '--requests', str(max(8, 3*int(c))), '--max-tokens', str(a.tokens), '--stream', '--price', '.69'], 'load' + c + '.txt')
        for rate in filter(None, a.rates.split(',')):
            run([sys.executable, 'tools/loadtest.py', url, '--arrival-rate', rate, '--requests', '128', '--max-tokens', str(a.tokens), '--stream', '--price', '.69', '--details', str(out / (a.tag + '.rate' + rate + '.json'))], 'rate' + rate + '.txt')
        if a.gsm:
            run([sys.executable, 'tools/gsm8k_server.py', url, '/root/models/gsm8k_test.jsonl', str(a.gsm), str(a.slots)], 'gsm.txt')
        with urllib.request.urlopen(url + '/metrics') as r:
            (out / (a.tag + '.metrics')).write_bytes(r.read())
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
