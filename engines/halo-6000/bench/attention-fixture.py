#!/usr/bin/env python3
"""Convert an oracle eval-callback NPZ capture to the independent phase guard.
No comparator library is linked into native inference or this converter.
"""
import argparse,json,struct
from pathlib import Path
import numpy as np

def write_fixture(source,output,context=None,layer=5,slot=1,slots=2):
    with np.load(source,allow_pickle=False) as cap:
        arrays=[]
        for name,shape in [('q_gate',(16,512)),('k',(2,256)),('v',(2,256)),('q_norm',(256,)),('k_norm',(256,))]:
            a=np.asarray(cap[name],dtype='<f4')
            if a.size!=int(np.prod(shape)) or not np.isfinite(a).all():raise ValueError(f'{name} shape/finite')
            arrays.append(a.reshape(shape))
        prefixes=[]
        for name in ['key_prefix','value_prefix']:
            a=np.asarray(cap[name])
            if a.ndim!=3 or a.shape[1:]!=(2,256):raise ValueError(f'{name} needs [position,2,256]')
            if a.dtype.kind=='u' and a.dtype.itemsize==2:a=a.astype('<u2',copy=False)
            else:
                a=a.astype('<f2')
                if not np.isfinite(a).all():raise ValueError(f'{name} finite')
                a=a.view('<u2')
            prefixes.append(a)
        position=len(prefixes[0])
        if len(prefixes[1])!=position:raise ValueError('key/value prefix lengths differ')
        context=context or position+1
        if not 0<=position<context or not 0<=layer<10 or not 0<=slot<slots:raise ValueError('state bounds')
        header=struct.pack('<8s5I',b'H6ATTN1\0',context,position,layer,slot,slots)
        Path(output).write_bytes(header+b''.join(a.tobytes() for a in arrays+prefixes))
        expected=None
        if 'attended' in cap:
            a=np.asarray(cap['attended'],dtype='<f4')
            if a.size!=4096 or not np.isfinite(a).all():raise ValueError('attended shape/finite')
            expected=str(output)+'.expected.f32';Path(expected).write_bytes(a.tobytes())
    return {'fixture':str(output),'expected_fork_output':expected,'context':context,'position':position,'layer_index':layer,'slot':slot,'slots':slots,'projection_included':False}

def read_output(path,position):
    buf=Path(path).read_bytes();offset=0;out={}
    for name,dtype,n,shape in [('q','<f4',4096,(16,256)),('k','<f4',512,(2,256)),('key_current','<u2',512,(2,256)),('value_current','<u2',512,(2,256)),('prob','<f4',16*(position+1),(16,position+1)),('attended','<f4',4096,(16,256))]:
        size=n*np.dtype(dtype).itemsize
        out[name]=np.frombuffer(buf[offset:offset+size],dtype=dtype).reshape(shape).copy();offset+=size
    if offset!=len(buf):raise ValueError('extra output bytes')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture_npz');p.add_argument('fixture');p.add_argument('--context',type=int);p.add_argument('--layer',type=int,default=5);p.add_argument('--slot',type=int,default=1);p.add_argument('--slots',type=int,default=2)
    a=p.parse_args();print(json.dumps(write_fixture(a.capture_npz,a.fixture,a.context,a.layer,a.slot,a.slots)))
