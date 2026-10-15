"""CPU-only native build in a new immutable generation. Run output through gpu_host.py."""
from pathlib import Path
import json,hashlib,base64,zipfile,io
p=Path(__file__).resolve().parent;root=p.parents[1];src=root/'src'
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for f in src.iterdir():
  if f.is_file() and (f.suffix in ['.h','.hpp','.cuh','.inc'] or f.name in ['engine.cu','dict8.cu','dict8_checks.cu','gguf.cpp']):z.write(f,'src/'+f.name)
 z.write(p/'cache_tool.cpp','cache_tool.cpp');z.write(p/'fault_test.cpp','fault_test.cpp')
 for name in ['test_identity.py','identity_test.cpp','fault_test.cpp','test_identity_abi.py']:
  z.write(p/name,'experiments/gu-dict8-cache/'+name)
 cmake=(root/'experiments/gu-dict8-engine/CMakeLists.txt').read_text();z.writestr('CMakeLists.txt',cmake)
payload=buf.getvalue()
script=r'''from pathlib import Path
import os,subprocess,hashlib,json,zipfile,io,base64,time,shutil
owner=Path('C:/work/models');base=owner/'experiments/bonsai-quality-20261013';old=base/'gu-dict8-engine-build-v2';dest=base/'gu-dict8-identity-build-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if dest.exists():raise RuntimeError('existing generation; inspect receipt')
assert sha(base/'custom-wide256/build/quality_custom.dll')=='a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c'
assert sha(base/'custom/frozen-production-plan.txt')=='713c060b627ef9955d68edfef17e12d75f07093c68b74c3fd8ecf4d19b3507bd'
dest.mkdir();payload=base64.b64decode(PAYLOAD);zipfile.ZipFile(io.BytesIO(payload)).extractall(dest)
objects=json.loads((old/'inherited-objects.json').read_text());(dest/'inherited').mkdir()
for o in objects:
 f=old/'inherited'/o['name'];assert sha(f)==o['sha256'];shutil.copy2(f,dest/'inherited'/o['name'])
# All existing shared headers remain byte-identical; only engine consumes new headers.
shared={}
for f in (old/'src').iterdir():
 if f.suffix in ['.h','.hpp','.cuh','.inc']:
  assert sha(f)==sha(dest/'src'/f.name),f.name;shared[f.name]=sha(f)
cmd='@echo off\ncall "'+str(owner/'tools/msvc/setup_x64.bat')+'"\nset "CUDA_PATH='+str(owner/'tools/cuda')+'"\nset "PATH='+str(owner/'tools/cuda/bin')+';'+str(owner/'tools/ninja')+';'+str(owner/'tools/cmake/cmake-4.1.2-windows-x86_64/bin')+';%PATH%"\ncd /d "%~dp0"\ncl /nologo /O2 /EHsc /std:c++17 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Isrc cache_tool.cpp src/gguf.cpp /Fe:cache_tool.exe /link bcrypt.lib\nif errorlevel 1 exit /b 1\ncl /nologo /O2 /EHsc /std:c++17 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Isrc fault_test.cpp src/gguf.cpp /Fe:fault_test.exe /link bcrypt.lib\nif errorlevel 1 exit /b 1\n"'+str(owner/'tools/python/python.exe')+'" experiments/gu-dict8-cache/test_identity.py\nif errorlevel 1 exit /b 1\n"'+str(owner/'tools/python/python.exe')+'" experiments/gu-dict8-cache/test_identity_abi.py\nif errorlevel 1 exit /b 1\ncmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=OFF\nif errorlevel 1 exit /b 1\ncmake --build build --parallel 2\nexit /b %ERRORLEVEL%\n'
(dest/'build.cmd').write_text(cmd);result={'gpu_work':False,'native_engine_execution':False,'state':'started','source_payload_sha256':hashlib.sha256(payload).hexdigest(),'source':{str(f.relative_to(dest)):sha(f) for f in (dest/'src').iterdir()},'inherited':objects,'shared_headers':shared,'command':cmd};start=time.time()
try:
 r=subprocess.run(['cmd.exe','/c',str(dest/'build.cmd')],capture_output=True,text=True,timeout=50,cwd=dest)
 (dest/'build.log').write_text(r.stdout+r.stderr);result.update(state='built' if r.returncode==0 else 'build_failed',exit_code=r.returncode,seconds=time.time()-start)
 if r.returncode==0:
  result['candidate_dll_sha256']=sha(dest/'build/quality_custom.dll');result['cpu_tool_sha256']=sha(dest/'cache_tool.exe')
except subprocess.TimeoutExpired as e:
 result.update(state='build_timeout',exit_code=124,seconds=time.time()-start);(dest/'timeout.log').write_text(str(e))
finally:
 result['control_dll_final_sha256']=sha(base/'custom-wide256/build/quality_custom.dll');result['control_plan_final_sha256']=sha(base/'custom/frozen-production-plan.txt')
 (dest/'build-receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
raise SystemExit(result['exit_code'])
'''
print(script.replace('PAYLOAD',repr(base64.b64encode(payload).decode())))
