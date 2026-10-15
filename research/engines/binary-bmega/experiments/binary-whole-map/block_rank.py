#!/usr/bin/env python3
"""Exact modular certificates for trained local down injectivity, one layer/call.

These are real/integer matrix rank witnesses, not an FP32-rounded whole-FFN
collision census. Original row/block half scales are nonzero diagonal factors.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from weight_census import GGUF, InputError

def certificate(signs):
    # 256 original output rows suffice in these checkpoints; record the chosen
    # 128 rows, not just a numerical rank estimate.
    a = (signs[:256].astype(np.int16) % 3).copy()
    labels = np.arange(len(a)); chosen=[]; determinant=1
    for c in range(128):
        indices=np.flatnonzero(a[c:,c])
        if not len(indices):
            raise InputError(f'first256 output rows fail to certify column{c}')
        p=c+int(indices[0])
        if p != c:
            a[[c,p]]=a[[p,c]]; labels[[c,p]]=labels[[p,c]]
            determinant = (-determinant) % 3
        chosen.append(int(labels[c]))
        pivot=int(a[c,c]); determinant=(determinant*pivot)%3
        a[c,c:] = (a[c,c:] * pivot) % 3 # inverse is itself in F3
        a[c+1:,c:] = (a[c+1:,c:] - a[c+1:,c:c+1]*a[c:c+1,c:]) % 3
    # Row swaps involving unused rows mean this determinant is for the pivot
    # order. Recompute the actual recorded square matrix independently.
    square=(signs[chosen].astype(np.int16)%3).copy(); det=1
    for c in range(128):
        p=c+int(np.flatnonzero(square[c:,c])[0])
        if p!=c: square[[c,p]]=square[[p,c]]; det=(-det)%3
        pivot=int(square[c,c]); det=(det*pivot)%3
        square[c,c:]=(square[c,c:]*pivot)%3
        square[c+1:,c:] = (square[c+1:,c:] - square[c+1:,c:c+1]*square[c:c+1,c:]) % 3
    if det == 0: raise InputError('recorded submatrix is singular')
    return {'output_rows':chosen,'determinant_mod3':det,
            'sign_submatrix_sha256':hashlib.sha256(signs[chosen].astype(np.int8).tobytes()).hexdigest()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--layer',required=True,type=int); p.add_argument('--out',required=True); a=p.parse_args(); start=time.monotonic()
    raw,scales=GGUF(a.model).weight(a.layer,'down')
    witnesses=[]
    for b in range(136):
        signs=(2*np.unpackbits(raw[:,b,2:],axis=-1,bitorder='little').astype(np.int8)-1)
        w=certificate(signs); w['block']=b; w['scale_words_sha256']=hashlib.sha256(scales[w['output_rows'],b].astype('<f2').tobytes()).hexdigest(); witnesses.append(w)
    result={'layer':a.layer,'modulus':3,'blocks':136,'rank_each':128,
        'scope':'exact integer sign matrix; original positive half row scales preserve real rank; fixed activation scales; not FP32 rounded full-block sufficiency',
        'witnesses':witnesses,'seconds':time.monotonic()-start}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='witnesses'}))
if __name__=='__main__': main()
