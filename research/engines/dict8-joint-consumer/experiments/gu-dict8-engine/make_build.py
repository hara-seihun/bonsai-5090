from pathlib import Path
import json,hashlib,base64,zipfile,io
p=Path(__file__).parent;src=p.parents[1]/'src'
r=json.loads(Path('~/work/five-wide-native/dict8-frozen-inputs-v2.json').read_text())
objects=[o for o in r['objects'] if o['name'] not in ['engine.cu.obj','dispatch.cu.obj']]
objects.append({'name':'dispatch.cu.obj',**r['wide_dispatch']});assert len(objects)==12
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for f in src.iterdir():
  if f.is_file() and (f.suffix in ['.h','.hpp','.cuh','.inc'] or f.name in ['engine.cu','dict8.cu','dict8_checks.cu']):z.write(f,'src/'+f.name)
 z.write(p/'CMakeLists.txt','CMakeLists.txt')
payload=buf.getvalue()
s='''from pathlib import Path
import os,subprocess,hashlib,json,zipfile,io,base64,time,shutil
owner=Path('C:/work/models');base=owner/'experiments/bonsai-quality-20261013';dest=base/'gu-dict8-engine-build-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(base/'custom-wide256/build/quality_custom.dll')=='a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c'
assert sha(base/'custom/frozen-production-plan.txt')=='713c060b627ef9955d68edfef17e12d75f07093c68b74c3fd8ecf4d19b3507bd'
if dest.exists():raise RuntimeError('existing build generation: inspect retained receipt, never overwrite')
dest.mkdir();payload=base64.b64decode(PAYLOAD);zipfile.ZipFile(io.BytesIO(payload)).extractall(dest)
objects=OBJECTS;inherited=dest/'inherited';inherited.mkdir()
for o in objects:
 source=Path(o['path']);assert sha(source)==o['sha256'];shutil.copy2(source,inherited/o['name']);assert sha(inherited/o['name'])==o['sha256']
(dest/'inherited-objects.json').write_text(json.dumps(objects,indent=2))
cmd='@echo off\\ncall "'+str(owner/'tools/msvc/setup_x64.bat')+'"\\nset "CUDA_PATH='+str(owner/'tools/cuda')+'"\\nset "PATH='+str(owner/'tools/cuda/bin')+';'+str(owner/'tools/ninja')+';'+str(owner/'tools/cmake/cmake-4.1.2-windows-x86_64/bin')+';%PATH%"\\ncd /d "%~dp0"\\ncmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=OFF\\nif errorlevel 1 exit /b 1\\ncmake --build build --parallel 2\\nexit /b %ERRORLEVEL%\\n'
(dest/'build.cmd').write_text(cmd);start=time.time()
result={'gpu_work':False,'native_model_execution':False,'state':'started','source_payload_sha256':hashlib.sha256(payload).hexdigest(),'source':{str(f.relative_to(dest)):sha(f) for f in (dest/'src').iterdir()},'inherited':objects,'command':cmd}
try:
 r=subprocess.run(['cmd.exe','/c',str(dest/'build.cmd')],capture_output=True,text=True,timeout=50,cwd=dest)
 (dest/'build.log').write_text(r.stdout+r.stderr);result.update(state='built' if r.returncode==0 else 'build_failed',exit_code=r.returncode,seconds=time.time()-start)
 if r.returncode==0:
  dll=dest/'build/quality_custom.dll';result['candidate_dll_sha256']=sha(dll)
  bench=base/'custom-wide256/build/quality_length_bench.exe';shutil.copy2(bench,dest/'build/quality_length_bench.exe');result['bench_sha256']=sha(bench)
  nv=owner/'tools/cuda_nvdisasm-windows-x86_64-13.0.85-archive/bin';env=os.environ.copy();env['NVDISASM_PATH']=str(nv);env['PATH']=str(nv)+';'+env['PATH']
  for flag,name in [('--dump-resource-usage','candidate.resources'),('--dump-elf-symbols','candidate.symbols')]:
   o=subprocess.run([str(owner/'tools/cuda/bin/cuobjdump.exe'),flag,str(dll)],capture_output=True,timeout=35,env=env);(dest/name).write_bytes(o.stdout+o.stderr);assert o.returncode==0
except subprocess.TimeoutExpired as e:
 result.update(state='build_timeout',seconds=time.time()-start,exit_code=124)
 (dest/'timeout.log').write_text(str(e.stdout)+'\\n'+str(e.stderr))
finally:
 result['control_dll_final_sha256']=sha(base/'custom-wide256/build/quality_custom.dll')
 result['control_plan_final_sha256']=sha(base/'custom/frozen-production-plan.txt')
 (dest/'build-receipt.json').write_text(json.dumps(result,indent=2))
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for f in dest.rglob('*'):
   if f.is_file() and (f.parent==dest or f.parent.name in ['src','inherited'] or f.name.endswith(('.obj','.dll','.exe','.lib'))):z.write(f,str(f.relative_to(dest)))
 b=buf.getvalue();(dest/'build-custody.zip').write_bytes(b)
 print(json.dumps({'archive_sha256':hashlib.sha256(b).hexdigest(),'archive_base64':base64.b64encode(b).decode(),'receipt':result}))
raise SystemExit(result['exit_code'])
'''
s=s.replace('PAYLOAD',repr(base64.b64encode(payload).decode())).replace('OBJECTS',repr(objects))
(p/'build-v2.py').write_text(s)
(p/'build-source-manifest.json').write_text(json.dumps({'payload_sha256':hashlib.sha256(payload).hexdigest(),'files':{str(f.relative_to(src)):hashlib.sha256(f.read_bytes()).hexdigest() for f in src.iterdir() if f.is_file() and (f.suffix in ['.h','.hpp','.cuh','.inc'] or f.name in ['engine.cu','dict8.cu','dict8_checks.cu'])},'inherited':objects},indent=2))
print('CPU build payload',len(payload))
