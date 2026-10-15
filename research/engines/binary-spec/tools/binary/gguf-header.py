#!/usr/bin/env python3
"""Read only a complete GGUF header, also from a still-downloading payload."""
import argparse
import json
import struct
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('input', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
with a.input.open('rb') as f:
    def exact(n):
        b = f.read(n)
        if len(b) != n:
            raise ValueError('GGUF_HEADER_INCOMPLETE')
        return b
    def scalar(fmt):
        return struct.unpack('<' + fmt, exact(struct.calcsize('<' + fmt)))[0]
    def string():
        return exact(scalar('Q')).decode('utf-8')
    formats = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    def value(t):
        if t in formats:
            return scalar(formats[t])
        if t == 8:
            return string()
        if t == 9:
            element = scalar('I')
            n = scalar('Q')
            return [value(element) for _ in range(n)]
        raise ValueError(f'GGUF_TYPE_UNKNOWN:{t}')
    if exact(4) != b'GGUF' or scalar('I') != 3:
        raise ValueError('GGUF_V3_REQUIRED')
    nt, nk = scalar('Q'), scalar('Q')
    metadata = {}
    for _ in range(nk):
        name = string()
        v = value(scalar('I'))
        if not name.startswith('tokenizer.'):
            metadata[name] = v
    tensors = []
    for _ in range(nt):
        name = string()
        rank = scalar('I')
        shape = [scalar('Q') for _ in range(rank)]
        tensors.append({'name':name, 'shape_ggml':shape, 'type':scalar('I'), 'offset':scalar('Q')})
    result = {'input':str(a.input), 'header_end':f.tell(), 'tensor_count':nt,
              'metadata_count':nk, 'metadata':metadata, 'tensors':tensors}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(f'GGUF_HEADER_READY {nt} tensors, {nk} fields, {result["header_end"]} bytes')
