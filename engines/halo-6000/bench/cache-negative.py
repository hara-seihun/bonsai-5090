#!/usr/bin/env python3
"""Corrupt disposable sparse copies; never modify the published cache."""
import json,subprocess,tempfile
from pathlib import Path
E=Path('/workspace/halo-6000');source='/workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf'
cache=Path(json.loads((E/'docs/results/loader-v1-repack.json').read_text())['cache'])
with cache.open('rb') as f:prefix=f.read(1024*1024)
records=[]
for mode in ['version','truncated','checksum']:
 with tempfile.TemporaryDirectory(prefix='halo6000-cache-negative-') as folder:
  copy=Path(folder)/cache.name
  data=bytearray(prefix)
  if mode=='version':data[8:12]=(999).to_bytes(4,'little')
  if mode=='truncated':data=data[:1024]
  with copy.open('wb') as f:
   f.write(data)
   if mode=='checksum':f.truncate(cache.stat().st_size)
  p=subprocess.run([str(E/'inspect'),'check',source,folder],capture_output=True,text=True)
  assert p.returncode!=0,(mode,p.stdout)
  expected={'version':'cache version/geometry','truncated':'cache','checksum':'cache checksum'}[mode]
  assert expected in p.stderr,(mode,p.stderr)
  records.append({'case':mode,'exit_code':p.returncode,'error':p.stderr.strip()})
(E/'docs/results/loader-cache-negative.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
