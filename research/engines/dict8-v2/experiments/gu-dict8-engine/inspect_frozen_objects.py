from pathlib import Path
import hashlib,json
owner=Path('C:/work/models');b=owner/'experiments/bonsai-quality-20261013/custom-wide256'
rows=[]
for p in (b/'build').rglob('*.obj'):
 rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
print(json.dumps({'gpu_work':False,'native_model_execution':False,'objects':rows,'control_dll_sha256':hashlib.sha256((b/'build/quality_custom.dll').read_bytes()).hexdigest()},indent=2))
