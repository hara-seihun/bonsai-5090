from pathlib import Path
import hashlib,json
b=Path('C:/work/models/experiments/bonsai-quality-20261013/custom-wide256')
r={'gpu_work':False,'native_model_execution':False,'files':{}}
for name in ['CMakeLists.txt','inherited-objects.json','src/dispatch.cu']:
 p=b/name
 if p.exists():r['files'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
r['inherited']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (b/'inherited').glob('*.obj')]
print(json.dumps(r,indent=2))
