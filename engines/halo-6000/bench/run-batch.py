#!/usr/bin/env python3
"""Freeze the owner source, memoize its build, then own a bounded GPU receipt."""
import argparse, fcntl, hashlib, json, os, pathlib, re, shlex, shutil, subprocess, time
p=argparse.ArgumentParser()
p.add_argument('model');p.add_argument('rows',type=int);p.add_argument('--steps',type=int,default=16);p.add_argument('--context',type=int,default=128)
p.add_argument('--bucketed',action='store_true')
a=p.parse_args();root=pathlib.Path('/workspace/halo-6000')
files=sorted([*root.glob('include/*'),*root.glob('src/*')]+[root/'bench/batch.cu',root/'bench/run-batch.py'])
files=[f for f in files if f.is_file()]
inputs=[(f.relative_to(root),f.read_bytes())for f in files];h=hashlib.sha256()
for relative,content in inputs:h.update(str(relative).encode());h.update(content)
key=h.hexdigest();snap=root/'build/batch'/key;snap.mkdir(parents=True,exist_ok=True)
build_lock=(snap/'build.lock').open('a');fcntl.flock(build_lock,fcntl.LOCK_EX)
for relative,content in inputs:
 dst=snap/relative;dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert dst.read_bytes()==content
 else:dst.write_bytes(content)
(snap/'source.json').write_text(json.dumps({'sha256':key,'files':[str(f.relative_to(root)) for f in files]},indent=2)+'\n')
nvcc='/usr/local/cuda/bin/nvcc';flags=['-O3','-std=c++20','-arch=sm_120a','-lineinfo','-Xptxas=-v','-Iinclude','-Isrc']
with (snap/'build.log').open('a') as log:
 def build(cmd):
  print('BUILD',*cmd,flush=True);subprocess.run(cmd,cwd=snap,stdout=log,stderr=subprocess.STDOUT,check=True)
 for source in ['src/runtime.cu','src/decode.cu','src/batch.cu','bench/batch.cu','src/gguf.cpp','src/cache.cpp']:
  obj=snap/(source.replace('/','-')+'.o')
  if obj.exists():continue
  # Source snapshots are complete, but an unrelated owner's file should not
  # force a fresh CUDA compilation. Reuse only identical source/header inputs.
  pending=[pathlib.Path(source)];dependencies=set()
  while pending:
   d=pending.pop()
   if d in dependencies:continue
   dependencies.add(d)
   for header in re.findall(r'^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]',(snap/d).read_text(),re.M):
    choices=[d.parent/header,pathlib.Path('include')/header,pathlib.Path('src')/header]
    found=next((q for q in choices if (snap/q).is_file()),None)
    if found is not None:pending.append(found)
  for previous in sorted((root/'build/batch').iterdir()):
   prior=previous/obj.name
   if previous==snap or not prior.is_file():continue
   old_runner=previous/'bench/run-batch.py'
   if not old_runner.is_file() or next((line for line in old_runner.read_text().splitlines() if line.startswith('nvcc=')),None)!=next(line for line in (snap/'bench/run-batch.py').read_text().splitlines() if line.startswith('nvcc=')):continue
   if all((previous/d).is_file() and (previous/d).read_bytes()==(snap/d).read_bytes() for d in dependencies):
    shutil.copy2(prior,obj);print('REUSE',prior,flush=True);break
  if obj.exists():continue
  tmp=str(obj)+'.tmp'
  cmd=([nvcc,*flags] if source.endswith('.cu') else ['g++','-O3','-std=c++20','-Iinclude','-Isrc','-fopenmp'])+['-c',source,'-o',tmp]
  build(cmd);os.rename(tmp,obj)
 exe=snap/'batch'
 if not exe.exists():
  build([nvcc,*flags,'-Xcompiler=-fopenmp',*[str(snap/(s.replace('/','-')+'.o')) for s in ['src/runtime.cu','src/decode.cu','src/batch.cu','bench/batch.cu','src/gguf.cpp','src/cache.cpp']],'-o',str(exe)+'.tmp']);os.rename(str(exe)+'.tmp',exe)
fcntl.flock(build_lock,fcntl.LOCK_UN);build_lock.close()
result=snap/(f'rows{a.rows}-{pathlib.Path(a.model).stem}-{time.time_ns()}.jsonl')
print(json.dumps({'snapshot':str(snap),'result':str(result),'sha256':key}),flush=True)
# Lock wait is outside the GPU timeout. Every invocation yields within 90s.
context=result.with_suffix('.context.txt');processes=result.with_suffix('.processes.txt');after=result.with_suffix('.after.txt')
gpu_command=shlex.join(['timeout','90',str(exe),a.model,str(a.rows),str(a.steps),str(a.context),str(int(a.bucketed))])
locked=f'nvidia-smi -q > {shlex.quote(str(context))}; ps -eo pid,ppid,pgid,args > {shlex.quote(str(processes))}; {gpu_command}; rc=$?; nvidia-smi -q > {shlex.quote(str(after))}; exit $rc'
command=['flock',str(root/'gpu.lock'),'bash','-c',locked]
with result.open('w') as log:subprocess.run(command,cwd=snap,stdout=log,stderr=subprocess.STDOUT,check=True)
print(result.read_text(),end='',flush=True)
