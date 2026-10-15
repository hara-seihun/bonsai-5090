from pathlib import Path
import json,hashlib,zipfile,io,base64
p=Path(__file__).parent;root=Path('~/work/five-wide-native')
r=json.loads((root/'dict8-engine-build-receipt-v1.json').read_text())
assert r['state']=='built' and r['exit_code']==0
names=['strict-worker.py','driver.py','environment.json','test_harness.py']
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name in names:z.write(p/name,name)
plan={'state':'CPU_staged_not_run','candidate_dll_sha256':r['candidate_dll_sha256'],'control_dll_sha256':r['control_dll_final_sha256'],'control_plan_sha256':r['control_plan_final_sha256'],'bench_sha256':r['bench_sha256'],'files':{name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in names},'immutable_inputs':{},'source_identity':json.loads((p/'engine-source-binding.json').read_text()),'window_bound_s':240,'child_bound_s':50,'strict':{'warmups':3,'cached_observations':2,'rows':64,'vocab':248320,'complete_logit_bytes':127139840,'teacher_forcing':True,'reference_repeat':True,'H5_DETERMINISTIC_GEMM':'1','H5_DOWN_CLEAR':'0','complete_endpoints':['rounded gate/up after activation scales/row means','SiLU','q4','s4','g4']},'matched_AB':{'profiles':['ordinary','deterministic'],'arms':['unchanged frozen serving','private dict8'],'passes':3,'streams':64,'tokens_per_stream':128,'warmup_tokens_per_stream':8},'GPU_acceptance':False,'GPU_request':False,'handoff':'Requires fresh8614 handback after0142faa9 reproduction release plus separate original-operation authorization; no handback file is fabricated.'}
owner='C:/work/models';b=owner+'/experiments/bonsai-quality-20261013'
plan['immutable_inputs']={b+'/custom-wide256/build/quality_custom.dll':plan['control_dll_sha256'],b+'/custom/frozen-production-plan.txt':plan['control_plan_sha256'],b+'/custom-wide256/build/quality_length_bench.exe':r['bench_sha256'],b+'/gu-dict8-engine-build-v1/build/quality_length_bench.exe':r['bench_sha256'],b+'/gu-dict8-engine-build-v1/build/quality_custom.dll':r['candidate_dll_sha256'],owner+'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf':'53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'}
plan['source_identity'].update({'dictionary_kernel_sha256':r['source']['src/dict8.cu'],'codec_sha256':r['source']['src/dict8_codec.hpp'],'source_parent':'03382927f534a9b8faefc1888157ba7ac38861f1'})
s='''from pathlib import Path
import hashlib,json,base64,io,zipfile,ast,subprocess,sys
owner=Path('C:/work/models');d=owner/'experiments/bonsai-quality-20261013/gu-dict8-engine-build-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
if (d/'harness-plan.json').exists() or (d/'gpu-driver-v1-receipt.json').exists():raise RuntimeError('existing stage/run generation: inspect; no overwrite')
assert not (d/'gpu-handback-v1.json').exists(),'unexpected handback before CPU staging'
plan=PLAN
for name,digest in plan['immutable_inputs'].items():assert sha(Path(name))==digest,name
payload=base64.b64decode(PAYLOAD);zipfile.ZipFile(io.BytesIO(payload)).extractall(d)
for name,digest in plan['files'].items():assert sha(d/name)==digest;compile((d/name).read_text(),name,'exec') if name.endswith('.py') else None
(d/'harness-plan.json').write_text(json.dumps(plan,indent=2))
# No native module import. -I uses only stdlib and pure contract-function extraction.
r=subprocess.run([sys.executable,'-I',str(d/'test_harness.py')],cwd=d,capture_output=True,text=True,timeout=8)
(d/'cpu-contract.log').write_text(r.stdout+r.stderr);assert r.returncode==0
cmd='@echo off\\ncall "'+str(owner/'tools/msvc/setup_x64.bat')+'"\\ndumpbin /exports "'+str(d/'build/quality_custom.dll')+'"\\nexit /b %ERRORLEVEL%\\n'
(d/'exports.cmd').write_text(cmd)
o=subprocess.run(['cmd.exe','/c',str(d/'exports.cmd')],capture_output=True,text=True,timeout=10,cwd=d)
(d/'exports.log').write_text(o.stdout+o.stderr);assert o.returncode==0
for name in ['qc_step','qc_logits','qc_graph_stats','h5_dict8_setup_stats','h5_dict8_check_stats','h5_dict8_reset_checks']:assert name in o.stdout,name
receipt={'state':'CPU_staged_not_run','gpu_work':False,'native_model_execution':False,'candidate_dll_sha256':sha(d/'build/quality_custom.dll'),'stage_payload_sha256':hashlib.sha256(payload).hexdigest(),'plan_sha256':sha(d/'harness-plan.json'),'isolated_contract_exit':r.returncode,'isolated_contract_output':r.stdout,'PE_exports_verified_without_loading':True,'handback_fabricated':False,'runtime_receipt_exists':(d/'gpu-driver-v1-receipt.json').exists(),'files':{name:sha(d/name) for name in plan['files']}}
(d/'cpu-stage-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'receipt':receipt,'plan':plan},indent=2))
'''
s=s.replace('PLAN',repr(plan)).replace('PAYLOAD',repr(base64.b64encode(buf.getvalue()).decode()))
(p/'stage-v1.py').write_text(s);(p/'harness-plan.json').write_text(json.dumps(plan,indent=2))
print('CPU-stage harness payload',len(buf.getvalue()))
