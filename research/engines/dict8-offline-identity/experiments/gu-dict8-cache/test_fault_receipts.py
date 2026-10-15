"""CPU injected-process coverage of durable native fault-suite outcomes."""
from pathlib import Path
import contextlib,importlib.util,io,json,subprocess,tempfile
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fault_host',p/'fault_host.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tested=[]
with tempfile.TemporaryDirectory(prefix='dict8-fault-receipts-') as td:
 root=Path(td)
 for scenario in ['cpp-failure','cpp-timeout','launch-error','bad-cli-json','pass']:
  dest=root/scenario;dest.mkdir();(dest/'cache').mkdir();calls=[]
  def fake(cmd,**kwargs):
   calls.append(cmd)
   if len(calls)==1:
    if scenario=='cpp-failure':return subprocess.CompletedProcess(cmd,7,'failure stdout','expected Corrupt actual Truncated')
    if scenario=='cpp-timeout':raise subprocess.TimeoutExpired(cmd,90,output=b'partial stdout',stderr=b'partial stderr')
    if scenario=='launch-error':raise OSError('injected executable missing')
    return subprocess.CompletedProcess(cmd,0,'cpp passed','')
   if scenario=='bad-cli-json':return subprocess.CompletedProcess(cmd,1,'not json','retained CLI error')
   expected='ModelIdentity' if cmd[1]=='verify' else 'Budget' if cmd[5]=='1' else 'Staged' if 'stale-test' in cmd[3] else 'Exists'
   return subprocess.CompletedProcess(cmd,1,json.dumps({'ok':False,'error':expected}),'')
  raised=None
  try:
   with contextlib.redirect_stdout(io.StringIO()):module.execute(dest,root/'model',dest/'cache/weights.d8',fake)
  except Exception as e:raised=e
  receipt=json.loads((dest/'fault-receipt.json').read_text())
  if scenario=='pass':assert raised is None and receipt['state']=='passed' and len(receipt['tests'])==5
  else:assert raised is not None and receipt['state']=='failed' and receipt['failure_type']==type(raised).__name__
  if scenario=='cpp-failure':
   assert receipt['fault_cpp_exit']==7 and receipt['fault_cpp_stdout']=='failure stdout' and receipt['fault_cpp_stderr']=='expected Corrupt actual Truncated'
   assert (dest/'fault-cpp.stderr.log').read_text()=='expected Corrupt actual Truncated'
  if scenario=='cpp-timeout':assert receipt['tests'][0]['stdout']=='partial stdout' and receipt['tests'][0]['stderr']=='partial stderr'
  if scenario=='bad-cli-json':assert receipt['tests'][1]['stdout']=='not json' and receipt['tests'][1]['stderr']=='retained CLI error'
  assert not (dest/'cache/stale-test.d8.preparing').exists() and not (dest/'cache/malformed-model.bin').exists()
  try:module.execute(dest,root/'model',dest/'cache/weights.d8',fake);assert False
  except RuntimeError as e:assert str(e)=='existing fault receipt'
  tested.append(scenario)
result={'gpu_work':False,'native_execution':False,'injected_process_scenarios':tested,'failure_receipt_and_streams_retained':True,'existing_receipt_not_overwritten':True,'owned_fixture_cleanup':True}
(p/'fault-receipt-test-receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
