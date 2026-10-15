from pathlib import Path
import subprocess,json,time


def execute(dest,model,cache,run_process):
 receipt=dest/'fault-receipt.json'
 if receipt.exists():raise RuntimeError('existing fault receipt')
 result={'state':'started','gpu_work':False,'native_engine_execution':False,'tests':[],'fault_storage_ceiling_bytes':3*1024**3,'headroom_bytes':20*1024**3}
 receipt.write_text(json.dumps(result,indent=2))
 def text(value):return value.decode(errors='replace') if isinstance(value,bytes) else value or ''
 def invoke(name,cmd,timeout):
  observed={'name':name,'command':cmd};result['tests'].append(observed);start=time.perf_counter()
  try:
   r=run_process(cmd,capture_output=True,text=True,timeout=timeout,cwd=dest)
   observed.update(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr);return r,observed
  except subprocess.TimeoutExpired as e:
   observed.update(state='timeout',stdout=text(e.stdout),stderr=text(e.stderr));raise
  finally:
   observed['seconds']=time.perf_counter()-start
   if 'stdout' in observed:
    (dest/(name+'.stdout.log')).write_text(observed['stdout'])
    (dest/(name+'.stderr.log')).write_text(observed['stderr'])
 def test(name,mode,modelpath,cachepath,budget,expected):
  cmd=[str(dest/'cache_tool.exe'),mode,str(modelpath),str(cachepath),'8',str(budget),str(20*1024**3)]
  r,observed=invoke(name,cmd,15);observed['expected_error']=expected
  parsed=json.loads(r.stdout);observed['result']=parsed
  if r.returncode==0 or parsed.get('error')!=expected:
   raise RuntimeError(name+': expected '+expected+', observed '+str(parsed))
 try:
  r,observed=invoke('fault-cpp',[str(dest/'fault_test.exe'),str(model),str(cache)],90)
  result.update(fault_cpp_exit=r.returncode,fault_cpp_stdout=r.stdout,fault_cpp_stderr=r.stderr,fault_cpp_s=observed['seconds'])
  if r.returncode!=0:raise RuntimeError('fault-cpp failed with exit '+str(r.returncode))
  test('existing-no-overwrite','prepare',model,cache,3*1024**3,'Exists')
  test('insufficient-budget','prepare',model,dest/'cache/not-published.d8',1,'Budget')
  staged=dest/'cache/stale-test.d8.preparing';staged.mkdir()
  try:test('staged-not-recomputed','prepare',model,dest/'cache/stale-test.d8',3*1024**3,'Staged')
  finally:staged.rmdir()
  malformed=dest/'cache/malformed-model.bin'
  with malformed.open('xb') as out:out.write(b'GGUF'+b'\xff'*124)
  try:test('model-rejected-before-unsafe-parse','verify',malformed,cache,3*1024**3,'ModelIdentity')
  finally:malformed.unlink()
  result['state']='passed'
 except BaseException as e:
  result.update(state='failed',failure_type=type(e).__name__,failure=str(e))
  raise
 finally:
  receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result))
 return result


if __name__=='__main__':
 owner=Path('C:/work/models');dest=owner/'experiments/bonsai-quality-20261013/gu-dict8-identity-build-v2'
 execute(dest,owner/'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf',dest/'cache/weights.d8',subprocess.run)
