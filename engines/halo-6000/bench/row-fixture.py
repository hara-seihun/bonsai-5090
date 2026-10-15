#!/usr/bin/env python3
"""Bind the eight requested actual cache matrices to their oracle inputs."""
import hashlib,json,pathlib,struct,sys
cache,oracle,out=map(pathlib.Path,sys.argv[1:4]);out.mkdir(parents=True,exist_ok=True)
selection={
 'qkv':('blk.0.attn_qkv.weight','attn_norm-0'),
 'z':('blk.0.attn_gate.weight','attn_norm-0'),
 'output':('blk.0.ssm_out.weight','final_output-0'),
 'router':('blk.0.ffn_gate_inp.weight','attn_post_norm-0'),
 'alpha':('blk.0.ssm_alpha.weight','attn_norm-0'),
 'beta':('blk.0.ssm_beta.weight','attn_norm-0'),
 'sharedrouter':('blk.0.ffn_gate_inp_shexp.weight','attn_post_norm-0'),
 'head':('output.weight','result_norm'),
}
with cache.open('rb') as f:
 def read(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
 def string():return f.read(read('I')).decode()
 assert f.read(8)==b'HALO6C01'
 version=read('I');geometry=read('Q');identity=[read('Q') for _ in range(5)];source=string();nt=read('I');records={}
 for _ in range(nt):
  name=string();shape=[read('Q') for _ in range(read('I'))]
  typ=read('I');fmt=read('I');size=read('Q');offset=read('Q');checksum=read('Q')
  records[name]=dict(name=name,shape=shape,type=typ,format=fmt,bytes=size,offset=offset,checksum=checksum)
 manifest=json.loads((oracle/'manifest.json').read_text());cases=[];tsv=[]
 for key,(name,callback) in selection.items():
  d=records[name];assert len(d['shape']) in (1,2) and d['format'] in (0,1,2),d
  k=d['shape'][0];n=d['shape'][1] if len(d['shape'])==2 else 1;assert k%128==0
  f.seek(d['offset']);data=f.read(d['bytes']);assert len(data)==d['bytes'];sha=hashlib.sha256(data).hexdigest()
  path=out/(key+'.weights')
  if key=='head':
   path=out.parent/'head-fixture/head.q8'
   if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  else:path.write_bytes(data)
  tensors=sorted([t for t in manifest['tensors'] if t['name']==callback],key=lambda t:t['position']);assert len(tensors)==16
  activation=out/(key+'.inputs.f32')
  with activation.open('wb') as w:
   for t in tensors:
    b=(oracle/t['file']).read_bytes();assert len(b)==k*4,(key,t);w.write(b)
  cases.append(dict(key=key,descriptor=d,sha256=sha,weights=str(path),inputs=str(activation),callbacks=tensors))
  tsv.append('\t'.join(map(str,[key,d['format'],n,k,d['bytes'],path,activation])))
 (out/'cases.tsv').write_text('\n'.join(tsv)+'\n')
 (out/'row-fixture.json').write_text(json.dumps(dict(cache=str(cache),version=version,geometry=geometry,identity=identity,source=source,cases=cases),indent=2)+'\n')
 print(json.dumps(dict(extracted=8,actual_inputs_each=16,cases=[(d['key'],d['descriptor']['shape'],d['descriptor']['format']) for d in cases])))
