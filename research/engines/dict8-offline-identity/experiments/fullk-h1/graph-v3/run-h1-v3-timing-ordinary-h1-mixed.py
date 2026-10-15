from pathlib import Path
import ctypes,hashlib,json,os,subprocess,time,sys,statistics
r=Path('C:/work/models');base=r/'experiments/bonsai-quality-20261013';d=base/'fullk-h1-tail-v3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(d/'build/quality_custom.dll')=='398a9d2bb38ec8613af6663dd782a282727da04b5f255f43e4c10780f1c6d768'
assert sha(base/'custom-wide256/build/quality_custom.dll')=='a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c'
assert sha(base/'custom/frozen-production-plan.txt')=='713c060b627ef9955d68edfef17e12d75f07093c68b74c3fd8ecf4d19b3507bd'
for required in ['serving-reference','serving-repeat','h1-plain','h1-mixed','tail-plain','tail-mixed']:
 saved=json.loads((d/'strict'/required/'receipt.json').read_text())
 if not saved['accepted']:raise RuntimeError('strict prerequisite failed: '+required)
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];kernel.CreateMutexW.restype=ctypes.c_void_p
kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.ReleaseMutex.argtypes=[ctypes.c_void_p];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
mutex=kernel.CreateMutexW(None,False,'Local\\BonsaiGPU')
if not mutex:raise ctypes.WinError(ctypes.get_last_error())
acquired=kernel.WaitForSingleObject(mutex,0) in (0,128)
if not acquired:kernel.CloseHandle(mutex);raise RuntimeError('GPU mutex busy; no run launched')
print('MUTEX ACQUIRED',flush=True)
try:
 window=json.loads((d/'gpu-window-v3.json').read_text())
 if window['candidate_dll_sha256']!='398a9d2bb38ec8613af6663dd782a282727da04b5f255f43e4c10780f1c6d768':raise RuntimeError('window candidate mismatch')
 print('GPU_WINDOW',json.dumps(window),flush=True)
 def native_budget():
  remaining=window['deadline_epoch_s']-time.time()-2
  if remaining<=2:raise RuntimeError('window budget exhausted; no native work started')
  return min(50,remaining)
 for mode in []:
  dest=d/('boundary-'+mode);dest.mkdir(exist_ok=True)
  if (dest/'failure-receipt.json').exists():raise RuntimeError('failed original retained; name a successor')
  if (dest/'receipt.json').exists():
   prev=json.loads((dest/'receipt.json').read_text());assert prev['accepted'];print(json.dumps(prev),flush=True);continue
  env={k:v for k,v in os.environ.items() if not k.startswith(('H5_','P12_','EXPLORER_','SWEEP_','GGML_','QC_'))};env.update({'H5_AB_PARALLEL': '1', 'H5_ACT': 'fp4', 'H5_ATTN_DIRECT': '0', 'H5_CARRIER_WF0': '1', 'H5_COMPACT': 'dense5', 'H5_GDN_ARITH': 'fp32', 'H5_GDN_BITS': '6', 'H5_GDN_CG': '8', 'H5_GDN_OCC': '2', 'H5_GDN_PIPE': '0', 'H5_GDN_Q8_SCALE': 'ls', 'H5_GDN_SCALED': '0', 'H5_GDN_SINGLE': 'separate', 'H5_GEMM_PREWAIT': '5', 'H5_GRAPH_NODES': 'native', 'H5_HEAD_KS': '1', 'H5_KV': 'fp8', 'H5_LOAD_THREADS': '4', 'H5_NO_SPEC': '1', 'H5_PDL': '1', 'H5_PLAN_FILL': '1', 'H5_PLAN_READ': 'C:/work/models/experiments/bonsai-quality-20261013/custom/frozen-production-plan.txt', 'H5_PREP_CHUNK': '2', 'H5_PREP_CHUNK_WIDE': '1', 'H5_PREP_CLEAR': '0', 'H5_PREP_EARLY': '4', 'H5_PREP_G4': '0.015625', 'H5_PREP_WIDE': 'multi', 'H5_SCALE_CARRIER': '1', 'H5_STATE': 'q8', 'H5_STATE_FUSION': '0', 'H5_TERNARY_OPERANDS': 'packed', 'H5_TOPK_SPLIT': '1', 'H5_WEIGHTS': 'fp4', 'H5_WSCALE': 'f16', 'P12_FP4_CACHE': '0', 'P12_FP4_SLOTS': '64', 'QC_KV': 'fp4-direct', 'QC_Q6_LAYOUT': 'planes', 'EXPLORER_PASSES': '1', 'H5_GRAPH_MAX_ROWS': '64', 'H5_PLAN_WRITE': 'C:/work/models/experiments/bonsai-quality-20261013/fullk-h1-tail-v3/probe-plan.txt', 'H5_GU_COOP_CHECK': '1'});env['H5_GU_COOP']=mode;env['PATH']=';'.join(['C:/work/models/tools/cuda/bin', 'C:/work/models/tools/cuda/bin/x64', 'C:/work/models/tools/msvc/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64', 'C:/work/models/build/llama/bin'])+';'+env.get('PATH','')
  with (dest/'stdout.log').open('w') as out,(dest/'stderr.log').open('w') as err:
   budget=native_budget()
   try:
    a=subprocess.run([sys.executable,str(d/'boundary-worker.py'),mode,str(dest)],cwd=d,env=env,stdout=out,stderr=err,timeout=budget)
   except subprocess.TimeoutExpired:
    (dest/'failure-receipt.json').write_text(json.dumps({'state':'timeout','mode':mode,'accepted':False,'window':window,'native_timeout_s':budget,'native_process':'killed_and_waited_by_subprocess_run'},indent=2));raise
  print('BOUNDARY',mode,a.returncode,(dest/'stdout.log').read_text(),flush=True)
  if a.returncode:print((dest/'stderr.log').read_text(),flush=True);raise RuntimeError('Boundary failed '+mode)
 for mode in ['h1-mixed']:
  dest=d/('timing-ordinary-'+mode);dest.mkdir(exist_ok=True)
  if (dest/'failure-receipt.json').exists():raise RuntimeError('failed original retained; name a successor')
  if (dest/'receipt.json').exists():
   saved=json.loads((dest/'receipt.json').read_text());assert saved['state']=='measured','Failed generation retained; name a successor';print(json.dumps(saved),flush=True);continue
  env={k:v for k,v in os.environ.items() if not k.startswith(('H5_','P12_','EXPLORER_','SWEEP_','GGML_','QC_'))};env.update({'H5_AB_PARALLEL': '1', 'H5_ACT': 'fp4', 'H5_ATTN_DIRECT': '0', 'H5_CARRIER_WF0': '1', 'H5_COMPACT': 'dense5', 'H5_GDN_ARITH': 'fp32', 'H5_GDN_BITS': '6', 'H5_GDN_CG': '8', 'H5_GDN_OCC': '2', 'H5_GDN_PIPE': '0', 'H5_GDN_Q8_SCALE': 'ls', 'H5_GDN_SCALED': '0', 'H5_GDN_SINGLE': 'separate', 'H5_GEMM_PREWAIT': '5', 'H5_GRAPH_NODES': 'native', 'H5_HEAD_KS': '1', 'H5_KV': 'fp8', 'H5_LOAD_THREADS': '4', 'H5_NO_SPEC': '1', 'H5_PDL': '1', 'H5_PLAN_FILL': '1', 'H5_PLAN_READ': 'C:/work/models/experiments/bonsai-quality-20261013/custom/frozen-production-plan.txt', 'H5_PREP_CHUNK': '2', 'H5_PREP_CHUNK_WIDE': '1', 'H5_PREP_CLEAR': '0', 'H5_PREP_EARLY': '4', 'H5_PREP_G4': '0.015625', 'H5_PREP_WIDE': 'multi', 'H5_SCALE_CARRIER': '1', 'H5_STATE': 'q8', 'H5_STATE_FUSION': '0', 'H5_TERNARY_OPERANDS': 'packed', 'H5_TOPK_SPLIT': '1', 'H5_WEIGHTS': 'fp4', 'H5_WSCALE': 'f16', 'P12_FP4_CACHE': '0', 'P12_FP4_SLOTS': '64', 'QC_KV': 'fp4-direct', 'QC_Q6_LAYOUT': 'planes', 'EXPLORER_PASSES': '1', 'H5_GRAPH_MAX_ROWS': '64', 'H5_PLAN_WRITE': 'C:/work/models/experiments/bonsai-quality-20261013/fullk-h1-tail-v3/probe-plan.txt', 'H5_GU_COOP_CHECK': '1'});env.pop('H5_GU_COOP_CHECK',None);env['H5_GU_COOP']=mode;env['H5_PLAN_WRITE']=str(d/('plan-'+mode+'.txt'));env['EXPLORER_PASSES']='3';env['H5_DETERMINISTIC_GEMM']='0';env['H5_DOWN_CLEAR']='0';env['PATH']=';'.join(['C:/work/models/tools/cuda/bin', 'C:/work/models/tools/cuda/bin/x64', 'C:/work/models/tools/msvc/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64', 'C:/work/models/build/llama/bin'])+';'+env.get('PATH','')
  cmd=[str(d/'build/quality_length_bench.exe'),'C:/work/models/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf','C:/work/models/experiments/bonsai-quality-20261013/inputs/video-gsm-prompts256.bin','64','128'];before=time.time()
  with (dest/'stdout.jsonl').open('w') as out,(dest/'stderr.log').open('w') as err:
   budget=native_budget()
   try:
    a=subprocess.run(cmd,cwd=d/'build',env=env,stdout=out,stderr=err,timeout=budget)
   except subprocess.TimeoutExpired:
    (dest/'failure-receipt.json').write_text(json.dumps({'state':'timeout','mode':mode,'accepted':False,'window':window,'native_timeout_s':budget,'native_process':'killed_and_waited_by_subprocess_run'},indent=2));raise
  samples=[]
  for line in (dest/'stdout.jsonl').read_text().splitlines():
   try:row=json.loads(line)
   except json.JSONDecodeError:continue
   if 'generation_tps' in row:samples.append(row)
  rec={'state':'measured' if a.returncode==0 and len(samples)==3 else 'failed','mode':mode,'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('H5_','P12_','EXPLORER_','QC_'))},'samples':samples,'wall_s':time.time()-before,'exit':a.returncode,'dll_sha256':sha(d/'build/quality_custom.dll'),'binary_sha256':sha(d/'build/quality_length_bench.exe'),'frozen_plan_sha256':sha(base/'custom/frozen-production-plan.txt')}
  if samples:rec['median_tps']=statistics.median(x['generation_tps'] for x in samples)
  (dest/'receipt.json').write_text(json.dumps(rec,indent=2));print('TIMING',json.dumps(rec),flush=True)
  if rec['state']!='measured':raise RuntimeError('Timing failed '+mode)
 print('BASELINE_FINAL',sha(base/'custom-wide256/build/quality_custom.dll'),sha(base/'custom/frozen-production-plan.txt'),flush=True)
finally:
 kernel.ReleaseMutex(mutex);kernel.CloseHandle(mutex);print('MUTEX RELEASED',flush=True)
