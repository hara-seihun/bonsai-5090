from pathlib import Path
import hashlib,json
base=Path('C:/work/models/experiments/bonsai-quality-20261013');b=base/'custom-wide256'
r={'gpu_work':False,'native_model_execution':False,'dispatch_source':(b/'dispatch.cu').read_text(),'objects':[]}
for p in (base/'custom/build/CMakeFiles/quality_custom.dir').rglob('*.obj'):
 r['objects'].append({'path':str(p),'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
p=b/'build/CMakeFiles/quality_custom.dir/dispatch.cu.obj';r['wide_dispatch']={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(r,indent=2))
