#!/usr/bin/env python3
"""Extract immutable cached head bytes and all actual final-norm callbacks."""
import hashlib,json,pathlib,struct,sys
cache,oracle,out=map(pathlib.Path,sys.argv[1:4]); out.mkdir(parents=True,exist_ok=True)
with cache.open('rb') as f:
 def read(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
 def string():return f.read(read('I')).decode()
 assert f.read(8)==b'HALO6C01'
 version=read('I');geometry=read('Q');identity=[read('Q') for _ in range(5)];source=string();nt=read('I')
 desc=None
 for _ in range(nt):
  name=string();shape=[read('Q') for _ in range(read('I'))]
  typ=read('I');fmt=read('I');size=read('Q');offset=read('Q');checksum=read('Q')
  if name=='output.weight':desc=dict(name=name,shape=shape,type=typ,format=fmt,size=size,offset=offset,checksum=checksum)
 assert desc and desc['shape']==[2048,248320] and desc['format']==2 and desc['size']==248320*16*136,desc
 f.seek(desc['offset']);remaining=desc['size'];sha=hashlib.sha256()
 with (out/'head.q8').open('wb') as w:
  while remaining:
   b=f.read(min(remaining,8<<20));assert b;w.write(b);sha.update(b);remaining-=len(b)
manifest=json.loads((oracle/'manifest.json').read_text())
tensors=sorted([t for t in manifest['tensors'] if t['name']=='result_norm'],key=lambda t:t['position'])
assert len(tensors)==16
with (out/'head-input.f32').open('wb') as w:
 for t in tensors:
  b=(oracle/t['file']).read_bytes();assert len(b)==8192;w.write(b)
info=dict(cache=str(cache),cache_version=version,geometry=geometry,source=source,head=desc,head_sha256=sha.hexdigest(),activation_callbacks=tensors)
(out/'head-fixture.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(dict(head_bytes=desc['size'],head_sha256=sha.hexdigest(),actual_positions=len(tensors))))
