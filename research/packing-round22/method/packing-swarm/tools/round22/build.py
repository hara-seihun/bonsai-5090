#!/usr/bin/env python3
import hashlib,json,pathlib,re,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[4]
FLAGS=['-O3','-std=c++20','-gencode','arch=compute_120a,code=sm_120a','-lineinfo','-Xcompiler','-fPIC,-O1','--expt-relaxed-constexpr','--ptxas-options=-v','-I~/work/packing-mma-round9/results/packing-swarm/round6/source','-I~/work/packing-mma-round8/method/packing-swarm/tools','-I~/work/packing-mma-round8/method/packing-swarm/tools/round8/producer']
SOURCE='method/packing-swarm/tools/round22/native/worker.cu'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call(args,log):
 with log.open('w') as f:
  r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=52)
 if r.returncode:raise ValueError('compiler_failed:'+str(r.returncode)+':'+str(log))
def main():
 if len(sys.argv)<3:raise ValueError('required: plan FRESH_DIRECTORY or phase DIRECTORY INDEX')
 mode=sys.argv[1];out=pathlib.Path(sys.argv[2]);out=out if out.is_absolute() else ROOT/out
 if mode=='plan':
  if len(sys.argv)!=3 or out.exists():raise ValueError('fresh_plan_required')
  out.mkdir(parents=True)
  call(['nvcc','-M',*FLAGS,SOURCE,'-o',str(out/'dependencies.mk')],out/'dependency.log')
  call(['python3','method/packing-swarm/tools/round22/closure.py',str(out/'dependencies.mk')],out/'closure.log')
  call(['nvcc','--dryrun','--keep','--keep-dir',str(out),*FLAGS,SOURCE,'-lz','-o',str(out/'native22')],out/'dryrun.log')
  commands=[];env=[]
  for line in (out/'dryrun.log').read_text().splitlines():
   if not line.startswith('#$ '):continue
   s=line[3:]
   if s.startswith(('PATH=','CICC_PATH=','LD_LIBRARY_PATH=')):env.append('export '+s)
   elif re.match(r'^[A-Za-z_][A-Za-z_0-9]*=',s):continue
   elif s.startswith('rm '):continue
   else:
    if s.startswith('gcc ') and ' -c ' in s:s+=' -O1'
    commands.append(s)
  if len(commands)!=11:raise ValueError('unknown_nvcc_phase_contract:'+str(len(commands)))
  plan={'state':'planned_source_compile','source_closure_SHA256':sha(out/'sources.json'),'environment':env,'commands':commands,'device_optimization':'O3/defaultcontraction/no_fastmath','host_optimization':'O1 appended after nvcc injected O3','GPU_execution':False}
  (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'state':plan['state'],'phases':len(commands),'output':str(out)}));return
 if mode!='phase' or len(sys.argv)!=4:raise ValueError('unknown_mode')
 plan=json.loads((out/'plan.json').read_text());index=int(sys.argv[3])
 if index<0 or index>=len(plan['commands']) or (out/f'phase-{index}.json').exists():raise ValueError('fresh_phase_required')
 if index and not (out/f'phase-{index-1}.json').exists():raise ValueError('previous_phase_unset')
 if sha(out/'sources.json')!=plan['source_closure_SHA256']:raise ValueError('changed_closure')
 for name,digest in json.loads((out/'sources.json').read_text())['sources'].items():
  if sha(pathlib.Path(name))!=digest:raise ValueError('changed_source:'+name)
 start=time.monotonic();cmd='set -euo pipefail\n'+'\n'.join(plan['environment'])+'\n'+plan['commands'][index]+'\n';(out/f'phase-{index}.sh').write_text(cmd)
 call(['bash',str(out/f'phase-{index}.sh')],out/f'phase-{index}.log')
 r={'state':'compiled_phase','index':index,'elapsed_seconds':time.monotonic()-start,'command_SHA256':sha(out/f'phase-{index}.sh'),'log_SHA256':sha(out/f'phase-{index}.log'),'GPU_execution':False}
 if index==10:
  if not (out/'native22').is_file():raise ValueError('binary_unset')
  r['binary_SHA256']=sha(out/'native22')
 (out/f'phase-{index}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:raise SystemExit('typed_build_error:'+str(e))
