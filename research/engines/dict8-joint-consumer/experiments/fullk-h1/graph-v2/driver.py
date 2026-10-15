from pathlib import Path
import json,time,traceback,hashlib,ctypes

root=Path('C:/work/models');base=root/'experiments/bonsai-quality-20261013';d=base/'fullk-h1-tail-v2'
record=d/'gpu-driver-v2-receipt.json'
if record.exists():raise RuntimeError('Original successor driver retained; no replay')
plan=json.loads((d/'h1-test-plan-v2.json').read_text())
names=['run-h1-v2-strict-'+x+'.py' for x in ['serving-reference','serving-repeat','h1-plain','h1-mixed','tail-plain','tail-mixed']]+['run-h1-v2-timing-'+p+'-'+x+'.py' for p in ['ordinary','deterministic'] for x in ['h1-plain','h1-mixed']]
for name in names:
 if hashlib.sha256((d/name).read_bytes()).hexdigest()!=plan['runner_sha256'][name]:raise RuntimeError('runner hash mismatch: '+name)
if hashlib.sha256((d/'h1_graph_contract.py').read_bytes()).hexdigest()!=plan['helper_sha256']:raise RuntimeError('helper hash mismatch')
rec={'state':'started','generation':plan['generation'],'compiled_dll_sha256':plan['candidate_dll_sha256'],'strict_warmup_steps':3,'steps':[],'native_bound_s':50,'window_bound_s':240}
record.write_text(json.dumps(rec,indent=2))
try:
 exec(compile((d/'begin-h1-window-v2.py').read_text(),'begin-h1-window-v2.py','exec'),{'__name__':'__main__'})
 window=json.loads((d/'gpu-window-v2.json').read_text());rec['window']=window
 for name in names:
  if time.time()>=window['deadline_epoch_s']-4:
   rec['state']='completed_prefix_window_exhausted';break
  print('DRIVER_START',name,flush=True);t=time.time()
  try:
   exec(compile((d/name).read_text(),name,'exec'),{'__name__':'__main__'})
  except BaseException as error:
   rec['steps'].append({'name':name,'state':'failed','elapsed_s':time.time()-t,'error':repr(error)})
   rec['state']='failed_prefix_retained';traceback.print_exc();break
  rec['steps'].append({'name':name,'state':'completed','elapsed_s':time.time()-t})
  record.write_text(json.dumps(rec,indent=2));print('DRIVER_FINISH',name,flush=True)
 else:rec['state']='matrix_completed'
except BaseException as error:
 rec['state']='setup_failed';rec['error']=repr(error);traceback.print_exc()
finally:
 k=ctypes.WinDLL('kernel32',use_last_error=True)
 k.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];k.CreateMutexW.restype=ctypes.c_void_p
 k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.ReleaseMutex.argtypes=[ctypes.c_void_p];k.CloseHandle.argtypes=[ctypes.c_void_p]
 m=k.CreateMutexW(None,False,'Local\\BonsaiGPU')
 if not m:rec['final_mutex_state']='create_failed'
 else:
  v=k.WaitForSingleObject(m,0);rec['final_mutex_wait']=v
  if v in (0,128):rec['final_mutex_release_return']=k.ReleaseMutex(m)
  rec['final_mutex_close_return']=k.CloseHandle(m)
 rec['finished_epoch_s']=time.time();rec['native_children']='all invoked subprocess.run calls returned or timeout-killed-and-waited before runner finally'
 rec['retained_receipts']={str(p.relative_to(d)):json.loads(p.read_text()) for p in d.rglob('*receipt.json') if p!=record and ('strict' in p.parts or p.parent.name.startswith('timing-'))}
 rec['final_serving_dll_sha256']=hashlib.sha256((base/'custom-wide256/build/quality_custom.dll').read_bytes()).hexdigest()
 rec['final_serving_plan_sha256']=hashlib.sha256((base/'custom/frozen-production-plan.txt').read_bytes()).hexdigest()
 record.write_text(json.dumps(rec,indent=2));print('DRIVER_FINAL',json.dumps(rec),flush=True)
