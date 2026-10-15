from pathlib import Path
import json,base64,io,zipfile
p=Path(__file__).parent;src=p.parents[1]/'src';buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for f in src.iterdir():
  if f.is_file() and f.suffix in ['.h','.hpp','.cuh','.inc']:z.write(f,'src/'+f.name)
 z.write(p/'probe-dict8-ring4.cu','src/probe.cu')
s="""from pathlib import Path
import os,subprocess,hashlib,json,zipfile,io,base64,time
owner=Path('C:/work/models');dest=owner/'experiments/bonsai-quality-20261013/gu-joint9-dict8-ring4-compile-v1'
if dest.exists():raise RuntimeError('existing compiler generation; inspect')
dest.mkdir()
payload=base64.b64decode(PAYLOAD)
zipfile.ZipFile(io.BytesIO(payload)).extractall(dest)
cmd='@echo off\\ncall "'+str(owner/'tools/msvc/setup_x64.bat')+'"\\ncd /d "%~dp0"\\n"'+str(owner/'tools/cuda/bin/nvcc.exe')+'" --cubin -std=c++17 -O3 --expt-relaxed-constexpr --generate-code=arch=compute_120a,code=sm_120a -lineinfo -Xptxas=-v -DH5_PROFILE_GEMM -DH5_PDL_EARLY_GEMM_RELEASE -DH5_PDL_GDN_STATE_OVERLAP -DH5_NO_L2PF -DH5_PROFILE_SCALE_CARRIER -DH5_WEIGHT_TMA_CACHE_POLICY=2 -DNOMINMAX -Isrc -I"'+str(owner/'source/llama/include')+'" -I"'+str(owner/'source/llama/vendor')+'" -I"'+str(owner/'source/llama/ggml/include')+'" src/probe.cu -o probe.cubin\\nexit /b %ERRORLEVEL%\\n'
(dest/'compile.cmd').write_text(cmd);before=time.time()
r=subprocess.run(['cmd.exe','/c',str(dest/'compile.cmd')],capture_output=True,text=True,timeout=50,cwd=dest)
(dest/'compile.log').write_text(r.stdout+r.stderr)
receipt={'gpu_work':False,'native_model_execution':False,'source_payload_sha256':hashlib.sha256(payload).hexdigest(),'probe_source_sha256':hashlib.sha256((dest/'src/probe.cu').read_bytes()).hexdigest(),'exit_code':r.returncode,'seconds':time.time()-before,'stdout':r.stdout,'stderr':r.stderr,'compiler_sha256':hashlib.sha256((owner/'tools/cuda/bin/nvcc.exe').read_bytes()).hexdigest(),'command':cmd}
if r.returncode==0:
 nv=owner/'tools/cuda_nvdisasm-windows-x86_64-13.0.85-archive/bin';env=os.environ.copy();env['NVDISASM_PATH']=str(nv);env['PATH']=str(nv)+';'+env['PATH']
 for mode,name in [('--dump-sass','probe.sass'),('--dump-resource-usage','probe.resources')]:
  o=subprocess.run([str(owner/'tools/cuda/bin/cuobjdump.exe'),mode,str(dest/'probe.cubin')],capture_output=True,timeout=40,env=env);(dest/name).write_bytes(o.stdout+o.stderr);assert o.returncode==0
 receipt['cubin_sha256']=hashlib.sha256((dest/'probe.cubin').read_bytes()).hexdigest()
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2))
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for f in dest.iterdir():
  if f.is_file():z.write(f,f.name)
b=buf.getvalue();(dest/'compiler-custody.zip').write_bytes(b)
print(json.dumps({'archive_sha256':hashlib.sha256(b).hexdigest(),'archive_base64':base64.b64encode(b).decode(),'exit_code':r.returncode,'gpu_work':False}))
raise SystemExit(r.returncode)
"""
s=s.replace('PAYLOAD',repr(base64.b64encode(buf.getvalue()).decode()))
Path('~/work/five-wide-native/gu-joint9-dict8-ring4-compile-v1.py').write_text(s)
print('payload',len(buf.getvalue()))
