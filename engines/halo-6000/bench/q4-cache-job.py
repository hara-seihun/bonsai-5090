#!/usr/bin/env python3
"""Durable dependency owner: consume source-copy receipt, then build/prove cache."""
import json,pathlib,subprocess,time
receipt=pathlib.Path('/workspace/ternary-bench/jobs/halo-q4-source.json')
deadline=time.monotonic()+1800
print('Waiting for the one existing halo-q4-source transfer receipt',flush=True)
while True:
 try:r=json.loads(receipt.read_text())
 except (FileNotFoundError,json.JSONDecodeError):r={}
 if r.get('state')=='succeeded':break
 if r.get('state') in ('failed','cancelled','interrupted'):
  raise SystemExit('Source transfer terminated: '+json.dumps(r))
 if time.monotonic()>deadline:raise SystemExit('Source receipt still pending after30min; no duplicate transfer submitted')
 time.sleep(5)
print('Source transfer succeeded; building/validating nativeQ4 model cache',flush=True)
raise SystemExit(subprocess.call(['/workspace/halo-6000/bench/run-q4-loader.sh','--model']))
