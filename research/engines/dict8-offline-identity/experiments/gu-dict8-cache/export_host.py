from pathlib import Path
import subprocess,hashlib,json,zipfile,io,base64,time
owner=Path('C:/work/models');dest=owner/'experiments/bonsai-quality-20261013/gu-dict8-identity-build-v2';base=owner/'experiments/bonsai-quality-20261013'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
assert json.loads((dest/'prepare-v2-receipt.json').read_text())['state']=='prepared'
assert json.loads((dest/'compare-receipt.json').read_text())['state']=='passed'
assert json.loads((dest/'fault-receipt.json').read_text())['state']=='passed'
cmd='@echo off\ncall "'+str(owner/'tools/msvc/setup_x64.bat')+'"\ncd /d "%~dp0"\ndumpbin /exports build\\quality_custom.dll\ndumpbin /imports cache_tool.exe\ndumpbin /imports fault_test.exe\n'
(dest/'inspect.cmd').write_text(cmd);r=subprocess.run(['cmd.exe','/c',str(dest/'inspect.cmd')],capture_output=True,text=True,timeout=15,cwd=dest);(dest/'inspect.log').write_text(r.stdout+r.stderr);assert r.returncode==0
assert all(symbol in r.stdout for symbol in ['h5_dict8_cache_last_error','h5_dict8_cache_stats','h5_dict8_identity_stats']) and 'cudart' not in r.stdout.lower()
cache=dest/'cache/weights.d8';t=time.perf_counter();digest=sha(cache);cost=time.perf_counter()-t
identity={'cache_sha256':digest,'cache_bytes':cache.stat().st_size,'cache_readonly':not bool(cache.stat().st_mode&0o200),'cache_path':str(cache),'custody_hash_seconds':cost,'gpu_work':False,'native_cached_startup_ns':None,'device_upload_ns':None,'original_v2_replayed':False,'serving_sha256':sha(base/'custom-wide256/build/quality_custom.dll'),'plan_sha256':sha(base/'custom/frozen-production-plan.txt')}
assert identity['serving_sha256']=='a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c';assert identity['plan_sha256']=='713c060b627ef9955d68edfef17e12d75f07093c68b74c3fd8ecf4d19b3507bd';assert identity['cache_readonly']
(dest/'cache-identity.json').write_text(json.dumps(identity,indent=2))
files=[f for f in dest.rglob('*') if f.is_file() and 'cache' not in f.relative_to(dest).parts and f.name not in ['build-custody.zip','custody-manifest.json']]
manifest={str(f.relative_to(dest)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in files};(dest/'custody-manifest.json').write_text(json.dumps(manifest,indent=2));files.append(dest/'custody-manifest.json')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,str(f.relative_to(dest)))
b=buf.getvalue();(dest/'build-custody.zip').write_bytes(b)
print(json.dumps({'identity':identity,'files':len(files),'archive_bytes':len(b),'archive_sha256':hashlib.sha256(b).hexdigest(),'archive_base64':base64.b64encode(b).decode()}))
