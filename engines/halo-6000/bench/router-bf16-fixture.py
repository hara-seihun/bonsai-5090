#!/usr/bin/env python3
"""Extract genuine per-layer router inputs; never reuse layer0 for other layers."""
import argparse,hashlib,json,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('output');a=p.parse_args()
manifest=Path(a.manifest);raw=manifest.read_bytes();m=json.loads(raw);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if m.get('model_bytes')!=8538502944:raise SystemExit('Expected unchanged v1 source identity')
records={}
for t in m['tensors']:
 k=(t['step'],t['name'])
 if k in records:raise SystemExit('Duplicate oracle frame')
 records[k]=t
fields={'inputs':('attn_post_norm',2048,'f32'),'oracle-logits':('ffn_moe_logits',256,'f32'),'oracle-probs':('ffn_moe_probs',256,'f32'),'oracle-ids':('ffn_moe_topk',8,'i32'),'oracle-weights':('ffn_moe_weights_norm',8,'f32'),'oracle-shared':('shared_expert_gate_sigmoid',1,'f32')}
files={}
for stem,(field,width,dtype) in fields.items():
 payload=bytearray()
 for step in range(16):
  for layer in range(40):
   t=records[(step,f'{field}-{layer}')]
   if t['batch']!=1 or t['position']!=step or t['type']!=dtype or t['bytes']!=width*4:raise SystemExit('Invalid actual frame geometry')
   data=(manifest.parent/t['file']).read_bytes()
   if len(data)!=width*4:raise SystemExit('Invalid actual frame extent')
   payload+=data
 target=out/(stem+('.i32' if dtype=='i32' else '.f32'));target.write_bytes(payload)
 files[stem]={'path':str(target.resolve()),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
result={'version':1,'actual_per_layer_inputs':True,'shape':[16,40,2048],'layout':'step,layer,K float32 little endian','oracle_manifest':str(manifest.resolve()),'oracle_manifest_sha256':hashlib.sha256(raw).hexdigest(),'model':m['model'],'model_bytes':m['model_bytes'],'files':files,'inference':False}
(out/'fixture.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
