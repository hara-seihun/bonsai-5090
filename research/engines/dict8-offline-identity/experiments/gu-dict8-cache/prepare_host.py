from pathlib import Path
import json,time,subprocess,hashlib,shutil
owner=Path('C:/work/models');dest=owner/'experiments/bonsai-quality-20261013/gu-dict8-identity-build-v2'
assert json.loads((dest/'build-receipt.json').read_text())['state']=='built'
cache_dir=dest/'cache';cache_dir.mkdir(exist_ok=True)
name='weights.d8'
cache=cache_dir/name;receipt=dest/'prepare-v2-receipt.json'
if receipt.exists() or cache.exists():raise RuntimeError('existing preparation; inspect without replay')
command=[str(dest/'cache_tool.exe'),'prepare',str(owner/'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'),str(cache),'8',str(3*1024**3),str(20*1024**3)]
result={'command':command,'state':'started','gpu_work':False,'native_engine_execution':False,'budget_bytes':3*1024**3,'headroom_bytes':20*1024**3,'free_before':shutil.disk_usage(dest).free};receipt.write_text(json.dumps(result,indent=2));start=time.perf_counter()
try:
 r=subprocess.run(command,cwd=dest,capture_output=True,text=True,timeout=240)
 (dest/'prepare.log').write_text(r.stdout+r.stderr);result.update(state='prepared' if r.returncode==0 else 'failed',exit_code=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr)
 if r.returncode==0:
  t=time.perf_counter();h=hashlib.sha256()
  with cache.open('rb') as f:
   while b:=f.read(8*1024**2):h.update(b)
  result.update(cache_sha256=h.hexdigest(),cache_bytes=cache.stat().st_size,custody_full_file_hash_s=time.perf_counter()-t)
except subprocess.TimeoutExpired as e:result.update(state='timeout',seconds=time.perf_counter()-start,exit_code=124)
finally:
 result['free_after']=shutil.disk_usage(dest).free;receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result))
