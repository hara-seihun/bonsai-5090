#!/usr/bin/env python3
"""Export actual fork layer0 projections plus current native F32 small weights.

Usage: gdn-prefill-fixture.py CACHE ORACLE OUT --start 4 --rows 12
No model inference. Validates both convolution history layout candidates against
preceding raw QKV; writes one H6GDNP1 image and provenance beside it.
"""
import argparse
import hashlib
import json
import pathlib
import struct
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('cache', type=pathlib.Path)
p.add_argument('oracle', type=pathlib.Path)
p.add_argument('out', type=pathlib.Path)
p.add_argument('--start', type=int, default=4)
p.add_argument('--rows', type=int, default=12)
a = p.parse_args()
assert a.start >= 3 and 1 <= a.rows <= 512
manifest = json.loads((a.oracle / 'manifest.json').read_text())
records = {(t['name'], t['position']): t for t in manifest['tensors']}
used = []
def tensor(name, position, count):
    t = records[name, position]
    assert t['type'] == 'f32' and t['bytes'] == count * 4, t
    b = (a.oracle / t['file']).read_bytes()
    assert len(b) == count * 4
    used.append(dict(name=name, position=position, file=t['file'], sha256=hashlib.sha256(b).hexdigest()))
    return np.frombuffer(b, dtype='<f4').copy()

qkv = np.stack([tensor('linear_attn_qkv_mixed-0', t, 8192) for t in range(a.start, a.start+a.rows)])
z = np.stack([tensor('z-0', t, 4096) for t in range(a.start, a.start+a.rows)])
alpha = np.stack([tensor('alpha-0', t, 32) for t in range(a.start, a.start+a.rows)])
beta = np.stack([tensor('beta-0', t, 32) for t in range(a.start, a.start+a.rows)])
previous = np.stack([tensor('linear_attn_qkv_mixed-0', t, 8192) for t in range(a.start-3,a.start)]).T.copy()
raw_history = tensor('conv_states-0', a.start, 8192*3)
candidates = {'channel-major': raw_history.reshape(8192,3),
              'time-major-to-channel-major': raw_history.reshape(3,8192).T.copy()}
matches = [name for name, x in candidates.items() if x.tobytes() == previous.tobytes()]
assert len(matches) == 1, ('initial history does not equal prior three raw projections', matches)
history = candidates[matches[0]]
# Fork ne=[key_row,value_col,value_head,1], key_row contiguous, native identical.
state = tensor('state_predelta-0', a.start, 32*128*128)
assert np.any(state != 0) and np.any(history != 0), 'fixture must exercise nonzero initial state/history'

weights = {}
with a.cache.open('rb') as f:
    def read(fmt):
        size = struct.calcsize('<'+fmt)
        return struct.unpack('<'+fmt, f.read(size))[0]
    def string(): return f.read(read('I')).decode()
    assert f.read(8) == b'HALO6C01'
    version, geometry = read('I'), read('Q')
    identity = [read('Q') for _ in range(5)]
    source = string()
    nt = read('I')
    for _ in range(nt):
        name = string()
        shape = [read('Q') for _ in range(read('I'))]
        typ, fmt, size, offset, checksum = read('I'), read('I'), read('Q'), read('Q'), read('Q')
        weights[name] = dict(shape=shape,type=typ,format=fmt,bytes=size,offset=offset,checksum=checksum)
    small = []
    for name, count in [('blk.0.ssm_conv1d.weight',8192*4),('blk.0.ssm_a',32),('blk.0.ssm_dt.bias',32),('blk.0.ssm_norm.weight',128)]:
        d = weights[name]
        assert d['format'] == 0 and d['bytes'] == count*4, (name,d)
        f.seek(d['offset']); b = f.read(d['bytes']); assert len(b) == count*4
        d['sha256'] = hashlib.sha256(b).hexdigest()
        small.append(np.frombuffer(b,dtype='<f4').copy())
        if name.endswith('ssm_conv1d.weight'): assert d['shape'] == [4,8192], d

assert np.all(small[1] < 0), 'ssm_a must already be negative exp(A_log)'
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('wb') as f:
    f.write(b'H6GDNP1\0'); f.write(struct.pack('<I',a.rows))
    for x in [qkv,z,alpha,beta,*small,state,history]: f.write(x.astype('<f4',copy=False).tobytes())
meta = dict(fixture=str(a.out),sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),start=a.start,rows=a.rows,
            oracle=str(a.oracle),cache=str(a.cache),source=source,version=version,geometry=geometry,identity=identity,
            history_axis_map=matches[0],state_axis_map='[value_head][value_col][key_row], key_row contiguous',
            tensors=used,weights={k:v for k,v in weights.items() if k in ['blk.0.ssm_conv1d.weight','blk.0.ssm_a','blk.0.ssm_dt.bias','blk.0.ssm_norm.weight']})
a.out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({k:meta[k] for k in ['fixture','sha256','start','rows','history_axis_map']}))
