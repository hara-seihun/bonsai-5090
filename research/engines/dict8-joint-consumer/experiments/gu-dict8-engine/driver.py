from pathlib import Path
import ctypes as C,hashlib,json,os,subprocess,sys,time,statistics,math,traceback
root=Path('C:/work/models');base=root/'experiments/bonsai-quality-20261013';d=base/'gu-dict8-engine-build-v2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=json.loads((d/'harness-plan.json').read_text())
for name,digest in plan['files'].items():
 if sha(d/name)!=digest:raise RuntimeError('changed staged harness: '+name)
for name,digest in plan['immutable_inputs'].items():
 if sha(Path(name))!=digest:raise RuntimeError('changed immutable input: '+name)
handback=json.loads((d/'gpu-handback-v1.json').read_text())
if handback['experiment']!='gu-dict8-engine-v2' or handback['candidate_dll_sha256']!=plan['candidate_dll_sha256'] or handback['all_current_and_queued_reservations_cleared'] is not True or handback['remaining_reservations']!=[] or handback['children_ceased'] is not True or handback['released_by_conversation']!='00000000-0000-0000-0000-000000000000' or not handback['authorization_operation_id'] or not handback['handback_operation_id'] or not handback['nonce'] or not 0<=time.time()-handback['issued_epoch_s']<=120:
 raise RuntimeError('fresh explicit PrismML handback clearing all current and queued reservations, plus separate run authorization required')
record=d/'gpu-driver-v1-receipt.json'
if record.exists():raise RuntimeError('original driver receipt retained; no replay')
k=C.WinDLL('kernel32',use_last_error=True)
k.CreateMutexW.argtypes=[C.c_void_p,C.c_bool,C.c_wchar_p];k.CreateMutexW.restype=C.c_void_p
k.WaitForSingleObject.argtypes=[C.c_void_p,C.c_uint32];k.ReleaseMutex.argtypes=[C.c_void_p];k.CloseHandle.argtypes=[C.c_void_p]
m=k.CreateMutexW(None,False,'Local\\BonsaiGPU')
if not m:raise C.WinError(C.get_last_error())
wait=k.WaitForSingleObject(m,0)
if wait not in (0,128):k.CloseHandle(m);raise RuntimeError('GPU mutex busy; no child started')
start=time.time();deadline=start+240
rec={'state':'started','start_epoch_s':start,'deadline_epoch_s':deadline,'window_bound_s':240,'child_bound_s':50,'handback':handback,'mutex_wait':wait,'steps':[],'source_identity':plan['source_identity'],'candidate_dll_sha256':plan['candidate_dll_sha256'],'control_dll_sha256':plan['control_dll_sha256'],'matched_AB':{},'strict_warmup_steps':3,'strict_cached_observations':2}
record.write_text(json.dumps(rec,indent=2))
def budget():
 remaining=deadline-time.time()-2
 if remaining<=2:raise RuntimeError('window exhausted; preserve prefix without new work')
 return min(50,remaining)
def environment(profile,candidate,target):
 e={a:b for a,b in os.environ.items() if not a.startswith(('H5_','P12_','EXPLORER_','SWEEP_','GGML_','QC_'))}
 e.update(json.loads((d/'environment.json').read_text()))
 e['H5_DETERMINISTIC_GEMM']='1' if profile=='deterministic' else '0';e['H5_DOWN_CLEAR']='0';e['H5_PLAN_WRITE']=str(target/'private-plan.txt')
 e['EXPLORER_PASSES']='3'
 if candidate:e['H5_GU_DICT8']='1'
 e['PATH']=';'.join(str(p) for p in [root/'tools/cuda/bin',root/'tools/cuda/bin/x64',root/'build/llama/bin'])+';'+e.get('PATH','')
 return e
def child(label,cmd,cwd,env,target):
 target.mkdir(parents=True,exist_ok=False);bound=budget();before=time.time()
 with (target/'stdout.log').open('w') as out,(target/'stderr.log').open('w') as err:
  try:
   r=subprocess.run(cmd,cwd=cwd,env=env,stdout=out,stderr=err,timeout=bound)
  except subprocess.TimeoutExpired:
   failure={'state':'timeout','label':label,'accepted':False,'native_timeout_s':bound,'process':'subprocess.run killed and waited','elapsed_s':time.time()-before}
   (target/'failure-receipt.json').write_text(json.dumps(failure,indent=2));rec['steps'].append(failure);raise
 step={'label':label,'exit_code':r.returncode,'elapsed_s':time.time()-before,'native_timeout_s':bound,'child_returned':True}
 rec['steps'].append(step);record.write_text(json.dumps(rec,indent=2))
 if r.returncode:raise RuntimeError('child failed: '+label)
try:
 for mode in ['serving-reference','serving-repeat','dict8']:
  target=d/'strict'/mode
  child('strict-'+mode,[sys.executable,str(d/'strict-worker.py'),mode,str(target)],d,environment('deterministic',mode=='dict8',target),target)
  result=json.loads((target/'receipt.json').read_text());rec.setdefault('strict',{})[mode]=result
  if result['accepted'] is not True:raise RuntimeError('strict acceptance failed: '+mode)
 for profile in ['ordinary','deterministic']:
  pair={}
  for name,candidate in [('serving',False),('dict8',True)]:
   cohort=d if candidate else base/'custom-wide256';target=d/'timing'/profile/name
   env=environment(profile,candidate,target)
   command=[str(cohort/'build/quality_length_bench.exe'),str(root/'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'),str(base/'inputs/video-gsm-prompts256.bin'),'64','128']
   child('timing-'+profile+'-'+name,command,cohort/'build',env,target)
   samples=[]
   for line in (target/'stdout.log').read_text().splitlines():
    try:row=json.loads(line)
    except json.JSONDecodeError:continue
    if 'generation_tps' in row:samples.append(row)
   if len(samples)!=3 or any(row['pass']!=i or row['streams']!=64 or row['tokens']!=8192 or row['warmup_tokens_per_stream']!=8 or not math.isfinite(row['generation_tps']) or not math.isfinite(row['generation_s']) or row['generation_s']<=0 or row['generation_tps']<=0 or abs(row['generation_tps']*row['generation_s']/8192-1)>1e-6 for i,row in enumerate(samples)):
    raise RuntimeError('incomplete or invalid matched whole-model samples')
   result={'state':'measured','profile':profile,'candidate':candidate,'samples':samples,'median_tps':statistics.median(row['generation_tps'] for row in samples),'command':command,'dll_sha256':sha(cohort/'build/quality_custom.dll'),'benchmark_sha256':sha(cohort/'build/quality_length_bench.exe'),'frozen_plan_sha256':sha(base/'custom/frozen-production-plan.txt'),'environment':{a:b for a,b in env.items() if a.startswith(('H5_','P12_','EXPLORER_','QC_'))}}
   (target/'receipt.json').write_text(json.dumps(result,indent=2));pair[name]=result
  rec['matched_AB'][profile]={'control':pair['serving'],'candidate':pair['dict8'],'relative_tps':pair['dict8']['median_tps']/pair['serving']['median_tps'],'delta_percent':100*(pair['dict8']['median_tps']/pair['serving']['median_tps']-1)}
 rec['state']='matrix_completed'
except BaseException as e:
 rec['state']='failed_prefix_retained';rec['error']=repr(e);traceback.print_exc()
finally:
 rec['release_mutex_return']=k.ReleaseMutex(m);rec['close_mutex_return']=k.CloseHandle(m)
 rec['finish_epoch_s']=time.time();rec['children']='every invoked subprocess returned or was timeout-killed and waited; no successor spawned'
 rec['control_dll_final_sha256']=sha(base/'custom-wide256/build/quality_custom.dll');rec['control_plan_final_sha256']=sha(base/'custom/frozen-production-plan.txt')
 rec['serving_unchanged']=rec['control_dll_final_sha256']==plan['control_dll_sha256'] and rec['control_plan_final_sha256']==plan['control_plan_sha256']
 record.write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
raise SystemExit(0 if rec['state']=='matrix_completed' and rec['serving_unchanged'] and rec['release_mutex_return'] and rec['close_mutex_return'] else 7)
