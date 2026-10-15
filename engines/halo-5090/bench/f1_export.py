"""Export original layer10 fixture for our native discriminator; no model weights in Git.
Usage: PYTHONPATH=/opt/bench/work/ffn-label python3 bench/f1_export.py OUT [layer]
Keep OPENBLAS_NUM_THREADS=2 and run via eval-cap. Repeated rows are timing data only.
"""
import json
import sys
from pathlib import Path
import numpy as np
from ffnlab.gguf import Layer
from ffnlab.schemes import Dense, halo, fused, calib
from ffnlab.ffn import signs
from ffnlab.quant import hadamard, nvfp4, silu
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
L=int(sys.argv[2]) if len(sys.argv)>2 else 10
ROOT=Path('/opt/bench/work/ffn-label')
z=np.load(ROOT/'data'/'cap'/f'L{L:02d}.npz')
n=z['n'][z['seq']>0][:8].copy()
# 8 unique held rows, repeated to512 for timing; every1/8/512 test sees this prefix.
np.tile(n,(64,1)).astype('<f4').tofile(OUT/'input.bin')
s5,s17=signs();s5.astype('<f4').tofile(OUT/'sign_in.bin');s17.astype('<f4').tofile(OUT/'sign_hid.bin')
w=Layer(L)
def pack(t,expanded=False):
    N,K=t.shape;G=N//32;B=K//128
    v=t.reshape(G,32,B,128).transpose(2,0,1,3)
    out=np.zeros((B,G,4 if expanded else 2,32,4),dtype='<u4')
    lane=np.arange(32);gid=lane//4;tq=lane%4
    for m in range(2):
      for h in range(2):
       for r in range(2):
        rows=16*m+8*r+gid
        if not expanded:
          word=np.zeros((B,G,32),dtype=np.uint32)
          for j in range(8):
            x=v[:,:,rows,64*h+8*tq+j];y=v[:,:,rows,64*h+32+8*tq+j]
            cx=(x!=0).astype(np.uint32)*2+(x<0).astype(np.uint32)*8
            cy=(y!=0).astype(np.uint32)+(y<0).astype(np.uint32)*4
            word|=(cx|cy)<<(4*j)
          out[:,:,m,:,2*h+r]=word
        else:
          for side in range(2):
            word=np.zeros((B,G,32),dtype=np.uint32)
            for j in range(8):word|=v[:,:,rows,64*h+32*side+8*tq+j].astype(np.uint32)<<(4*j)
            out[:,:,2*m+h,:,r+side*2]=word
    return out

def scale(s):
    N,B=s.shape;v=s.astype('<f2').reshape(N//32,32,B).transpose(2,0,1)
    ix=np.array([16*m+8*r+g for g in range(8) for m in range(2) for r in range(2)])
    return v[:,:,ix].copy()

def gu(permuted):
    ix=np.arange(17408//32)
    if permuted:ix=np.array([c*32+h*4+b for c in range(17) for b in range(4) for h in range(8)])
    ti=np.empty((34816,5120),np.int8);si=np.empty((34816,40),np.float32)
    for dest,src in enumerate(ix):
      ti[dest*64:dest*64+32]=w.tg[src*32:src*32+32];ti[dest*64+32:dest*64+64]=w.tu[src*32:src*32+32]
      si[dest*64:dest*64+32]=w.sg[src*32:src*32+32];si[dest*64+32:dest*64+64]=w.su[src*32:src*32+32]
    key='gu_f1' if permuted else 'gu'
    pack(ti).tofile(OUT/f'{key}.bin');scale(si).tofile(OUT/f'{key}_scale.bin')
gu(False);gu(True)
pack(w.td).tofile(OUT/'down.bin');scale(w.sd).tofile(OUT/'down_scale.bin')
x=np.array(w.td).reshape(5120,136,4,32).astype(np.int16)
a,b,c,d=(x[:,:,j,:] for j in range(4))
y=np.stack((a+b+c+d,a-b+c-d,a+b-c-d,a-b-c+d),axis=2)
assert np.abs(y).max()<=4
codes=(np.abs(y)|(np.where(y<0,8,0))).astype(np.uint8).reshape(5120,17408)
pack(codes,True).tofile(OUT/'down_expanded.bin')
dense=Dense(L);cal=calib()[L]
h= silu(nvfp4(hadamard(n*s5),gscale=np.float32(cal['g_in']))@dense.g.T)*(nvfp4(hadamard(n*s5),gscale=np.float32(cal['g_in']))@dense.u.T)
refs={'baseline':halo(L,n,w=dense),'fixed':nvfp4(hadamard(h*s17),gscale=np.float32(cal['g_hidH']))@dense.d.T,'f1':fused(L,n,w=dense)}
for key,v in refs.items():v.astype('<f4').tofile(OUT/f'ref_{key}.bin')
np.array([cal['g_in'],cal['g_hidH'],cal['g_hid']],dtype='<f4').tofile(OUT/'fixed.bin')
(OUT/'manifest.json').write_text(json.dumps({'layer':L,'D':5120,'FF':17408,'timing_rows':512,'unique_held_rows':8,'fixed':cal,'files':{p.name:p.stat().st_size for p in OUT.glob('*.bin')}},indent=2)+'\n')
print((OUT/'manifest.json').read_text())
