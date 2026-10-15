#!/usr/bin/env python3
"""Independent byte identity against the packing worker's established v1 image."""
import hashlib,json,struct
from pathlib import Path
E=Path('/workspace/halo-6000')
cache=Path(json.loads((E/'docs/results/loader-v1-repack.json').read_text())['cache'])
with cache.open('rb') as f:header=f.read(1024*1024)
o=8

def read(fmt):
 global o
 value=struct.unpack_from('<'+fmt,header,o);o+=struct.calcsize('<'+fmt)
 return value[0] if len(value)==1 else value

def string():
 global o
 n=read('I');value=header[o:o+n].decode();o+=n;return value

assert read('I')==1
read('Q');read('QQQQQ');string();count=read('I')
selected=None
for _ in range(count):
 name=string();nd=read('I');shape=[read('Q') for _ in range(nd)];type_=read('I');format_=read('I');bytes_=read('Q');offset=read('Q');read('Q')
 if name=='blk.0.ffn_gate_exps.weight':selected=(offset,bytes_,shape,type_,format_)
assert selected is not None

def digest(path,offset,n):
 h=hashlib.sha256()
 with path.open('rb') as f:
  f.seek(offset)
  while n:
   b=f.read(min(n,1024*1024));assert b;h.update(b);n-=len(b)
 return h.hexdigest()

offset,n,shape,type_,format_=selected
baseline=Path('/workspace/ternary-bench/experiments/packing/packing-pane.bin')
a=digest(cache,offset,n);b=digest(baseline,0,n);assert a==b
record={'tensor':'blk.0.ffn_gate_exps.weight','bytes':n,'shape':shape,'source_type':type_,'format':format_,'cache':str(cache),'packing_reference':str(baseline),'cache_offset':offset,'sha256':a,'byte_identical':True}
(E/'docs/results/loader-packing-reference.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
