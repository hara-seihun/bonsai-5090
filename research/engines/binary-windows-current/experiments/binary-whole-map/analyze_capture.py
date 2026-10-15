#!/usr/bin/env python3
"""Captured real FFN fibres and complete trained down observation; one layer/call.

No approximate-distance classes are treated as exact equivalence classes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import argparse
import json
import struct
import time
from pathlib import Path
import numpy as np
from weight_census import GGUF, InputError

def load(path, layer):
    data=Path(path).read_bytes()
    magic,count,nlayer,d,ff=struct.unpack_from('<5i',data)
    if (magic,nlayer,d,ff)!=(0x574d4631,3,5120,17408):
        raise InputError('capture geometry/magic mismatch')
    off=20; records=[]
    while off<len(data):
        pos,l,r,rd,rf=struct.unpack_from('<5i',data,off); off+=20
        if (r,rd,rf)!=(1,d,ff): raise InputError('invalid capture row')
        norm=np.frombuffer(data,'<f4',d,off); off+=d*4
        iq=np.frombuffer(data,'i1',d,off); off+=d
        ix=np.frombuffer(data,'<f4',d//128,off); off+=d//128*4
        gu=np.frombuffer(data,'<f4',2*ff,off); off+=ff*8
        h=np.frombuffer(data,'<f4',ff,off); off+=ff*4
        if l==layer: records.append((pos,norm,iq,ix,gu[:ff],gu[ff:],h))
    if len(records)!=count: raise InputError('incomplete selected layer capture')
    return records

def carrier(h,half):
    shape=h.shape; blocks=h.reshape(len(h),-1,128)
    maximum=np.max(np.abs(blocks),axis=-1,keepdims=True)
    inv=np.divide(np.float32(half),maximum,out=np.zeros_like(maximum),where=maximum!=0)
    q=np.rint(blocks*inv).astype(np.int16)
    sx=maximum/np.float32(half)
    return q.reshape(shape),(q.astype(np.float32)*sx).reshape(shape),sx

def entropy_columns(q):
    ordered=np.sort(q,axis=0); boundaries=np.concatenate([np.ones((1,q.shape[1]),bool),np.diff(ordered,axis=0)!=0])
    unique=boundaries.sum(axis=0)
    entropy=np.zeros(q.shape[1],np.float64)
    # Exact empirical per-neuron counts, not population cardinalities.
    for col in range(q.shape[1]):
        _,counts=np.unique(q[:,col],return_counts=True); p=counts/len(q); entropy[col]=-(p*np.log2(p)).sum()
    return {'unique_min':int(unique.min()),'unique_mean':float(unique.mean()),'unique_max':int(unique.max()),'empirical_entropy_mean_bits':float(entropy.mean())}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--capture',required=True); p.add_argument('--model',required=True); p.add_argument('--layer',type=int,required=True); p.add_argument('--out',required=True); p.add_argument('--kv',choices=['f16','fp8'],required=True); a=p.parse_args(); start=time.monotonic()
    records=load(a.capture,a.layer); norm,iq,ix,g,u,h=[np.stack([r[i] for r in records]) for i in range(1,7)]
    if not all(np.isfinite(v).all() for v in (norm,ix,g,u,h)): raise InputError('nonfinite captured values')
    q8,h8,s8=carrier(h,127)
    model=GGUF(a.model); raw,scales=model.weight(a.layer,'down')
    signs=(2*np.unpackbits(raw[...,2:],axis=-1,bitorder='little').astype(np.int8)-1).reshape(5120,17408)
    w=signs.astype(np.float32); w.reshape(5120,-1,128)[:]*=scales.astype(np.float32)[...,None]
    observed=h8@w.T
    denom=float(np.sum(observed.astype(np.float64)**2))
    report={'layer':a.layer,'positions':[r[0] for r in records],'rows':len(records),'input_contract':f'actual postnorm A8 max/127 with original K128 scales, one-row pureA8 engine, F16 recurrent state/{a.kv.upper()} KV','observer':'complete trained 5120-output down projection, no residual/next-layer/logit substitution','gate_negative_fraction':float(np.mean(g<0)),'hidden_zero_fraction_exact':float(np.mean(h==0)),'next_a8_zero_fraction':float(np.mean(q8==0)),'next_a8_nonzero_negative_fraction':float(np.mean(q8<0)), 'negative_gate_hidden_energy_fraction':float(np.sum(h[g<0].astype(np.float64)**2)/np.sum(h.astype(np.float64)**2)),'input_a8_norm_relative_rms':float(np.sqrt(np.sum((iq.astype(np.float32)*np.repeat(ix,128,axis=1)-norm).astype(np.float64)**2)/np.sum(norm.astype(np.float64)**2))), 'h_f32_fibres':entropy_columns(h.view(np.int32)), 'next_a8_fibres':entropy_columns(q8), 'candidates':[]}
    for half in [63,31,15,7,3,1]:
        q,hq,sx=carrier(h,half); y=hq@w.T
        report['candidates'].append({'carrier':'signed scalar code plus dynamic K128 maximum','half_range':half,'state_count':2*half+1,'packed_bits':int(np.ceil(np.log2(2*half+1))),'max_code_abs':int(np.max(np.abs(q))),'zero_fraction':float(np.mean(q==0)),'down_nse_vs_a8':float(np.sum((y-observed).astype(np.float64)**2)/denom),'hidden_nse_vs_a8':float(np.sum((hq-h8).astype(np.float64)**2)/np.sum(h8.astype(np.float64)**2)),'fibres':entropy_columns(q)})
    for threshold in [.001,.003,.01,.03,.1]:
        maxh=np.repeat(np.max(np.abs(h.reshape(len(h),-1,128)),axis=-1),128,axis=1)
        keep=np.abs(h)>=threshold*maxh; candidate=h8*keep; y=candidate@w.T
        report['candidates'].append({'carrier':'a8 code with threshold erasure; original block scale retained','threshold_max_fraction':threshold,'discarded_fraction':float(np.mean(~keep)),'down_nse_vs_a8':float(np.sum((y-observed).astype(np.float64)**2)/denom)})
    # Same q at two real reachable observations need not carry the same value:
    # scale is part of the packet. This witnesses a scale-blind proposed name,
    # with the rest of the receiver coordinates fixed.
    witness=None
    for col in range(17408):
        for row in range(1,len(h)):
            same=np.flatnonzero(q8[:row,col]==q8[row,col])
            same=[int(s) for s in same if h8[s,col]!=h8[row,col]]
            if same:
                j=same[0]; delta=float(h8[row,col]-h8[j,col]); witness={'neuron':col,'rows':[j,row],'positions':[records[j][0],records[row][0]],'shared_q':int(q8[row,col]),'scales':[float(s8[j,col//128,0]),float(s8[row,col//128,0])],'effective_hidden_values':[float(h8[j,col]),float(h8[row,col])],'single_coordinate_down_delta_l2':float(abs(delta)*np.linalg.norm(w[:,col]))}; break
        if witness: break
    report['scale_blind_collision']=witness; report['seconds']=time.monotonic()-start
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'layer':a.layer,'seconds':report['seconds'],'a8_zero':report['next_a8_zero_fraction'],'sevenbit_down_nse':report['candidates'][0]['down_nse_vs_a8'],'scale_blind_collision':witness}))
if __name__=='__main__':
    main()
