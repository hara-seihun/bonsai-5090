"""Deterministic fixture/layout audit without GPU (sub-minute, no timing claim)."""
import json
import sys
from pathlib import Path
import numpy as np
from ffnlab.gguf import Layer
P=Path(sys.argv[1]);L=int(sys.argv[2]) if len(sys.argv)>2 else 10
w=Layer(L)
D,F=5120,17408
# Check all128 columns of16 deterministic row/group cases per projection/layout.
def read_packed(name,N,K,expanded=False):
    return np.memmap(P/(name+'.bin'),mode='r',dtype='<u4').reshape(K//128,N//32,4 if expanded else 2,32,4)
def decode(p,row,k,expanded=False):
    g,rr=divmod(row,32);m,rr=divmod(rr,16);rh,gid=divmod(rr,8)
    kb,ki=divmod(k,128);h,ki=divmod(ki,64);side,ki=divmod(ki,32);tq,j=divmod(ki,8);lane=gid*4+tq
    if expanded:
        c=int(p[kb,g,2*m+h,lane,rh+2*side])>>(4*j)&15
        return -int(c&7) if c&8 else int(c&7)
    word=int(p[kb,g,m,lane,2*h+rh]);c=word>>(4*j)&15
    mag=(c>>(1-side))&1;neg=(c>>(3-side))&1
    return -mag if neg else mag
checks=0
for name,permuted in [('gu',False),('gu_f1',True)]:
 p=read_packed(name,2*F,D)
 for row in [0,7,8,15,16,31,32,63,64,511,512,1023,17407,17408,34814,34815]:
  g,r=divmod(row,64);up=r>=32;rr=r%32
  if permuted:
    chunk,rem=divmod(g,32);cls,high=divmod(rem,8);src=chunk*32+high*4+cls
  else:src=g
  t=w.tu if up else w.tg
  for kb in [0,39]:
   for k in range(kb*128,kb*128+128):assert decode(p,row,k)==int(t[src*32+rr,k]);checks+=1
p=read_packed('down',D,F);q=read_packed('down_expanded',D,F,True)
for row in [0,7,8,15,16,31,32,63,64,511,512,1023,1740,4000,5118,5119]:
 for kb in [0,135]:
  for k in range(kb*128,kb*128+128):
   assert decode(p,row,k)==int(w.td[row,k]);checks+=1
   low=k%32;group=(k%128)//32
   v=[int(w.td[row,kb*128+j*32+low]) for j in range(4)]
   expected=sum((1 if (group&j).bit_count()%2==0 else -1)*v[j] for j in range(4))
   assert decode(q,row,k,True)==expected;checks+=1
r={'layout_checks':checks,'passed':True,'scope':'deterministic row/group endpoints,128 columns each; not GPU execution'}
(P/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
