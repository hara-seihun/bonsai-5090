#!/usr/bin/env python3
import collections,hashlib,json,pathlib,re,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[4]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if len(sys.argv)!=2:raise ValueError('required compiler directory')
 out=pathlib.Path(sys.argv[1]);out=out if out.is_absolute() else ROOT/out
 phases=[json.loads((out/f'phase-{i}.json').read_text()) for i in range(11)]
 for name,digest in json.loads((out/'sources.json').read_text())['sources'].items():
  if sha(pathlib.Path(name))!=digest:raise ValueError('changed compiler source:'+name)
 kernels={}
 for part in (out/'paired.sass').read_text().split('Function : ')[1:]:
  name=part.splitlines()[0].strip();ops=collections.Counter()
  for line in part.splitlines()[1:]:
   m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?(?:U?P\d+|U?PT)\s+)?([A-Z][A-Z0-9_.]*)(?:\s|;)',line)
   if m:ops[m.group(1)]+=1
  if not ops:raise ValueError('empty sass:'+name)
  kernels[name]={'static_SASS_instructions':sum(ops.values()),'opcodes':dict(sorted(ops.items())),'static_local_load_sites':sum(n for op,n in ops.items() if op.startswith('LDL')),'static_local_store_sites':sum(n for op,n in ops.items() if op.startswith('STL')),'static_shared_load_sites':sum(n for op,n in ops.items() if op.startswith(('LDS','LDSM'))),'static_barrier_sites':sum(n for op,n in ops.items() if op.startswith(('BAR','WARPSYNC','MBARRIER','SYNCS')))}
 if len(kernels)!=2:raise ValueError('paired kernel set')
 guards=[]
 for i,args in enumerate([[],['--describe-native23'],['--serve-native23'],['--authorized-live-GPU-context'],['--serve-native21']]):
  r=subprocess.run([str(out/'native23'),*args],capture_output=True,text=True,timeout=5)
  (out/f'guard-{i}.log').write_text(r.stdout+r.stderr)
  value=json.loads(r.stdout)
  if value.get('GPU_calls')!=0:raise ValueError('zero GPU guard failed')
  if r.returncode not in (0,2):raise ValueError('unknown guard return')
  guards.append({'argv':args,'exit':r.returncode,'observation':value,'log_SHA256':sha(out/f'guard-{i}.log')})
 resources={}
 for name in kernels:
  text=(out/'phase-4.log').read_text();marker='Function properties for '+name
  if marker not in text:raise ValueError('missing ptxas')
  block=text.split(marker,1)[1].split('ptxas info    : Compile time',1)[0]
  m=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block);reg=re.search(r'Used (\d+) registers',block)
  if not m or not reg:raise ValueError('unknown ptxas resource format')
  resources[name]={'registers_per_thread':int(reg[1]),'stack_frame_bytes':int(m[1]),'ptxas_spill_store_bytes':int(m[2]),'ptxas_spill_load_bytes':int(m[3]),'static_shared_bytes':0,'dynamic_shared_source_bytes':74112}
 report={'state':'compiled_source_only_zero_GPU_guards','compiler':'CUDA13.1.115/sm120a/deviceO3/actualhostO1/defaultcontraction/no-fastmath','phases':phases,'sum_seconds':sum(p['elapsed_seconds'] for p in phases),'maximum_phase_seconds':max(p['elapsed_seconds'] for p in phases),'binary_SHA256':sha(out/'native23'),'cubin_SHA256':sha(out/'worker.cubin'),'SASS_SHA256':sha(out/'paired.sass'),'source_closure_SHA256':sha(out/'sources.json'),'resources':resources,'kernels':kernels,'guards':guards,'GPU_execution':False,'unsets':['native CUDA words','native time','occupancy','dynamic executed SASS counts','physical allocation/residency','cache-effective traffic']}
 dest=ROOT/'results/packing-swarm/round23/compiler.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'state':report['state'],'resources':resources,'kernels':{n:{k:v for k,v in d.items() if k!='opcodes'} for n,d in kernels.items()},'guards':len(guards),'sum_seconds':report['sum_seconds']}))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:raise SystemExit('typed_compiler_report_error:'+str(e))
