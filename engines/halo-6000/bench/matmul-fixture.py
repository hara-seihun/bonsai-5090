#!/usr/bin/env python3
"""Real GGUF bytes -> immutable Halo cache + independent dense integer oracle.

PANE packing and PTQ decoding use their experiment owner; Q8_0 conversion keeps
all original FP16 scales and signed bytes. No quantizer is run here.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/workspace/src/prism-llama/gguf-py')
sys.path.insert(0,'/workspace/ternary-bench/experiments/packing')
import gguf
from repack import decode
# format.py collides with standard names only inside this explicit file load.
import importlib.util
spec=importlib.util.spec_from_file_location('pane_format','/workspace/ternary-bench/experiments/packing/format.py')
pane=importlib.util.module_from_spec(spec);spec.loader.exec_module(pane)
ROLES={
 'qkv':'blk.0.attn_qkv.weight','z':'blk.0.attn_gate.weight',
 'gdn_out':'blk.0.ssm_out.weight','q':'blk.3.attn_q.weight',
 'k':'blk.3.attn_k.weight','v':'blk.3.attn_v.weight',
 'o':'blk.3.attn_output.weight','shared_gate':'blk.0.ffn_gate_shexp.weight',
 'shared_down':'blk.0.ffn_down_shexp.weight','gate':'blk.0.ffn_gate_exps.weight',
 'up':'blk.0.ffn_up_exps.weight','down':'blk.0.ffn_down_exps.weight',
 'head':'output.weight'}
EXPERTS=[0,7,31,63,127,191,223,255]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('v1');ap.add_argument('hybrid');ap.add_argument('out');a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cases=[]
 for label,source in [('v1',a.v1),('hybrid',a.hybrid)]:
  r=gguf.GGUFReader(source);ts={t.name:t for t in r.tensors}
  for role,name in ROLES.items():
   t=ts[name];typ=int(t.tensor_type);shape=list(map(int,t.shape));k,n=shape[:2];experts=8 if len(shape)==3 else 1
   if label=='hybrid' and (typ!=8 or role=='head'):continue
   if typ not in (8,143):continue
   key=label+'-'+role;paths={x:out/(key+'.'+x) for x in ('cache','qs','scales')};meta=out/(key+'.json')
   if meta.exists() and all(p.exists() for p in paths.values()):
    rec=json.loads(meta.read_text());assert rec['shape']==shape and rec['type']==typ;cases.append(rec);continue
   block=128 if typ==143 else 32;size=28 if typ==143 else 34
   raw=np.asarray(t.data,np.uint8).reshape(-1,n*(k//block),size)
   sources=EXPERTS if experts>1 else [0]
   # Whole experts guarantee the16-block PANE tile boundary, independent oracle
   # has canonical signed coefficients plus one original-half scale perK32.
   for p in paths.values():p.unlink(missing_ok=True)
   with paths['cache'].open('wb') as fc,paths['qs'].open('wb') as fq,paths['scales'].open('wb') as fs:
    for e in sources:
     b=raw[e]
     if typ==143:
      pane.pack(b).tofile(fc)
      (decode(b).astype(np.int8)-1).tofile(fq)
      np.repeat(b[:,26:28].copy().view('<u2').reshape(-1),4).astype('<u2').tofile(fs)
     else:
      v=b.reshape(-1,4,34);cache=np.empty((len(v),136),np.uint8)
      cache[:,:128]=v[:,:,2:].reshape(-1,128);cache[:,128:]=v[:,:,:2].reshape(-1,8);cache.tofile(fc)
      b[:,2:].copy().view(np.int8).tofile(fq);b[:,:2].copy().tofile(fs)
   rec=dict(key=key,name=name,source=source,type=typ,shape=shape,n=n,k=k,experts=experts,format=1 if typ==143 else 2,
            expert_indices=sources,files={x:str(p) for x,p in paths.items()},sha256={x:sha(p) for x,p in paths.items()},bytes=paths['cache'].stat().st_size)
   meta.write_text(json.dumps(rec,indent=2)+'\n');cases.append(rec);print(json.dumps({'fixture':key,'bytes':rec['bytes']}),flush=True)
 (out/'cases.tsv').write_text(''.join('\t'.join(map(str,[c['key'],c['format'],c['n'],c['k'],c['experts'],c['bytes'],c['files']['cache'],c['files']['qs'],c['files']['scales']]))+'\n' for c in cases))
 (out/'manifest.json').write_text(json.dumps(cases,indent=2)+'\n')
if __name__=='__main__':main()
