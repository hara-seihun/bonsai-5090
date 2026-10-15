#!/usr/bin/env python3
"""Bounded-memory CPU publication proof, with no GNU-time dependency."""
import json,os,resource,subprocess,time
from pathlib import Path
E=Path('/workspace/halo-6000');os.chdir(E)
os.environ['OMP_NUM_THREADS']='8'
R=E/'docs/results';R.mkdir(parents=True,exist_ok=True)
source='/workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf'
cache='/workspace/models/halo6000-cache'
cases=[('loader-unit',['bench/loader-test']),('loader-v1-inspect',['./inspect','inspect',source]),('loader-v1-repack',['./inspect','repack',source,cache,'--verify']),('loader-v1-reload',['./inspect','check',source,cache])]
receipts=[]
for name,command in cases:
 before=time.monotonic()
 with (R/f'{name}.json' if name!='loader-unit' else R/f'{name}.txt').open('w') as out,(R/f'{name}.log').open('w') as log:
  p=subprocess.run(command,stdout=out,stderr=log)
 usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 receipts.append({'name':name,'command':command,'exit_code':p.returncode,'seconds':time.monotonic()-before,'max_rss_kib':usage.ru_maxrss})
 (R/'loader-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
 if p.returncode:raise SystemExit(f'{name} exit {p.returncode}; see {R}/{name}.log')
 print(receipts[-1],flush=True)
