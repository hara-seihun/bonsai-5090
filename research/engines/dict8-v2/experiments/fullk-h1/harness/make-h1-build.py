from pathlib import Path
import base64,json
r=Path('~/work/five-wide-native');checkout=Path('~/work/clones/fullk-h1-tail')
files={n:base64.b64encode((checkout/'src'/n).read_bytes()).decode() for n in ['engine.cu','fullk_gu.cuh','fullk_profile.h','fp4.cu','fullk_ring.cu','fullk_tail.cu']}
cm=(r/'retained-engine-v2/fullk-fp4-tail-v3/CMakeLists.txt').read_text().replace('EQUAL 15','EQUAL 13').replace('Expected15','Expected13').replace('src/fullk_ring.cu ${OBJECTS}','src/engine.cu src/fullk_ring.cu src/fullk_tail.cu ${OBJECTS}')
s='''from pathlib import Path
import base64,hashlib,json,shutil,subprocess,time
r=Path('C:/work/models');base=r/'experiments/bonsai-quality-20261013';old=base/'fullk-fp4-tail-v3';wide=base/'custom-wide256';d=base/'fullk-h1-tail-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if (d/'build-receipt.json').exists():
 print((d/'build-receipt.json').read_text());raise SystemExit(0)
assert sha(wide/'build/quality_custom.dll')=='a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c'
assert sha(base/'custom/frozen-production-plan.txt')=='713c060b627ef9955d68edfef17e12d75f07093c68b74c3fd8ecf4d19b3507bd'
assert sha(old/'build/quality_custom.dll')=='3572a8e6aecd89bf96b26b755440ac1e6b5ded7a49b602fe710ff8279f2e3512'
d.mkdir(exist_ok=True);shutil.copytree(old/'src',d/'src',dirs_exist_ok=True)
files=FILES
for n,b in files.items():(d/'src'/n).write_bytes(base64.b64decode(b))
prior=json.loads((old/'inherited-objects.json').read_text());manifest={};inputs=d/'inherited';inputs.mkdir(exist_ok=True)
for n,m in prior.items():
 if n in ['engine.cu.obj','fullk_ring.cu.obj','fullk_tail.cu.obj']:continue
 p=old/'inherited'/n;assert sha(p)==m['sha256'];shutil.copy2(p,inputs/n);manifest[n]=m
assert len(manifest)==13
(d/'inherited-objects.json').write_text(json.dumps(manifest,indent=2));(d/'CMakeLists.txt').write_text(CMAKE_SOURCE)
cmd='@echo off\\ncall "'+str(r/'tools/msvc/setup_x64.bat')+'"\\nset "CUDA_PATH='+str(r/'tools/cuda')+'"\\nset "PATH='+str(r/'tools/cuda/bin')+';'+str(r/'tools/ninja')+';'+str(r/'tools/cmake/cmake-4.1.2-windows-x86_64/bin')+';%PATH%"\\ncd /d "%~dp0"\\ncmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=OFF\\nif errorlevel 1 exit /b 1\\ncmake --build build --parallel 2\\nexit /b %errorlevel%\\n'
(d/'build.cmd').write_text(cmd);t=time.time()
try:
 a=subprocess.run(['cmd.exe','/c',str(d/'build.cmd')],cwd=d,capture_output=True,text=True,timeout=55)
 (d/'build.log').write_text(a.stdout+a.stderr);print(a.stdout+a.stderr,flush=True)
 result={'state':'built' if a.returncode==0 else 'build_failed','exit':a.returncode,'wall_s':time.time()-t,'source':{n:sha(d/'src'/n) for n in files},'inherited':manifest,'baseline_dll':sha(wide/'build/quality_custom.dll'),'frozen_plan':sha(base/'custom/frozen-production-plan.txt'),'gpu_work':False}
 if a.returncode==0:
  shutil.copy2(wide/'build/quality_length_bench.exe',d/'build/quality_length_bench.exe');result['binary_sha256']=sha(d/'build/quality_custom.dll')
 (d/'build-receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 raise SystemExit(a.returncode)
except subprocess.TimeoutExpired as e:
 (d/'build-timeout.json').write_text(json.dumps({'state':'timeout','wall_s':time.time()-t,'stdout':str(e.stdout),'stderr':str(e.stderr)}));raise
'''.replace('FILES',repr(files)).replace('CMAKE_SOURCE',repr(cm))
(r/'build-h1-tail-v1.py').write_text(s)
print('CPU build script prepared')
