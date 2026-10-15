from pathlib import Path
import json,subprocess,time
owner=Path('C:/work/models');dest=owner/'experiments/bonsai-quality-20261013/gu-dict8-identity-build-v2';receipt=dest/'compare-receipt.json'
if receipt.exists():raise RuntimeError('existing comparison; inspect receipt')
command=[str(dest/'cache_tool.exe'),'compare',str(owner/'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'),str(dest/'cache/weights.d8'),'8',str(3*1024**3),str(20*1024**3)]
r={'state':'started','command':command,'gpu_work':False};receipt.write_text(json.dumps(r));t=time.perf_counter()
try:
 p=subprocess.run(command,capture_output=True,text=True,timeout=180,cwd=dest);r.update(state='passed' if p.returncode==0 else 'failed',exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.perf_counter()-t)
except subprocess.TimeoutExpired:r.update(state='timeout',seconds=time.perf_counter()-t)
finally:receipt.write_text(json.dumps(r,indent=2));print(json.dumps(r))
