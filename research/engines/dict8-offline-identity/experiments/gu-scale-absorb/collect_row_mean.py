from pathlib import Path
import os,subprocess,hashlib,json,base64
root=Path('C:/work/models');base=root/'experiments/bonsai-quality-20261013'
image=base/'custom-wide256/build/quality_custom.dll';tool=root/'tools/cuda/bin/cuobjdump.exe';dest=base/'gu-scale-absorb-row-mean-v1'
if dest.exists():raise RuntimeError('existing CPU generation; inspect')
dest.mkdir();assert hashlib.sha256(image.read_bytes()).hexdigest()=='a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c'
env=os.environ.copy();nv=root/'tools/cuda_nvdisasm-windows-x86_64-13.0.85-archive/bin';env['NVDISASM_PATH']=str(nv);env['PATH']=str(nv)+';'+env['PATH']
r=subprocess.run([str(tool),'--dump-resource-usage',str(image)],capture_output=True,env=env,timeout=40);assert r.returncode==0
lines=r.stdout.decode().splitlines();matches=[l.strip().removeprefix('Function ').removesuffix(':') for l in lines if 'Function ' in l and 'k_row_scale_mean' in l];assert len(matches)==1,matches
r2=subprocess.run([str(tool),'--dump-sass','--function',matches[0],str(image)],capture_output=True,env=env,timeout=40);assert r2.returncode==0 and ('Function : '+matches[0]) in r2.stdout.decode()
(dest/'row-mean.sass').write_bytes(r2.stdout);(dest/'stderr').write_bytes(r2.stderr)
print(json.dumps({'gpu_work':False,'native_execution':False,'symbol':matches[0],'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'tool_sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),'sass_sha256':hashlib.sha256(r2.stdout).hexdigest(),'sass_base64':base64.b64encode(r2.stdout).decode(),'returncode':r2.returncode}))
