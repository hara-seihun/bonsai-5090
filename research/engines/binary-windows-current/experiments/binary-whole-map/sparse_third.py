#!/usr/bin/env python3
"""Exact balanced E2M1 pair + sparse third plane on real captured inputs.

This is a carrier census, not native timing. All source bytes are retained;
no clipping or approximate quotient is introduced.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_capture import load,carrier

D=[0,1,2,3,4,6,8,12,-1,-2,-3,-4,-6,-8,-12]
pairs={}
for a in D:
 for b in D: pairs.setdefault(a+4*b,(a,b))
recode={}
for q in range(-128,128):
 choices=[]
 for c in D:
  if q-16*c in pairs:
   a,b=pairs[q-16*c]; choices.append((int(c!=0),abs(c),abs(a)+abs(b),a,b,c))
 if not choices: raise RuntimeError(f'no exact recode for{q}')
 _,_,_,a,b,c=min(choices)
 assert q==a+4*b+16*c
 recode[q]=(a,b,c)

def census(q):
 c=np.asarray([recode[int(v)][2] for v in q.ravel()],dtype=np.int8).reshape(q.shape)
 blocks=c.reshape(len(q),-1,128)
 counts=np.count_nonzero(blocks,axis=2)
 return {'rows':len(q),'k':q.shape[1],'nonzero_third_fraction':float(np.mean(c!=0)),
 'third_counts_per_row_K128':{'min':int(counts.min()),'mean':float(counts.mean()),'p50':float(np.percentile(counts,50)),'p95':float(np.percentile(counts,95)),'max':int(counts.max())},
 'one_row_third_zero_group_fraction':float(np.mean(counts==0)),
 'R32_union_third_support_mean':float(np.count_nonzero(np.any(blocks!=0,axis=0),axis=1).mean()),
 'zero_third_K64_tile_fractions':{str(tw):float(np.mean(~np.any(blocks.reshape(len(q)//tw,tw,-1,2,64)!=0,axis=(1,4)))) for tw in (8,16,32) if len(q)%tw==0},
 'source_scalar_codes':len(np.unique(q)),
 'scope':'actual32-row capture; regrouped for support union, not a model-produced simultaneous batch; hidden codes recomputed with original float32 producer formula'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 out={'identity':'q=a+4*b+16*c, a/b/c in2E2M1; source byte exact','full_signed_byte_coverage':256,'two_plane_representable_A8_codes':sum(q in pairs for q in range(-127,128)), 'two_plane_missing_positive_0to60':[q for q in range(61) if q not in pairs], 'recode':recode,'layers':{}}
 for l in (0,30,61):
  rows=load(a.capture,l); iq=np.stack([r[2] for r in rows]);h=np.stack([r[6] for r in rows]);qh,_,_=carrier(h,127)
  out['layers'][l]={'input':census(iq),'hidden':census(qh)}
 Path(a.out).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out['layers'],indent=2))
if __name__=='__main__':main()
