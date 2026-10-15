#!/usr/bin/env python3
"""Independently certify source router coefficient losslessness, not model quality.

Requires fork gguf-py on PYTHONPATH; only reads source weights and the guarded
compact image. Output is a new standalone audit, not a modified GPU receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model', type=Path)
p.add_argument('image', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
source = a.model.resolve(strict=True)
st = source.stat()
reader = GGUFReader(str(source))
tensors = {t.name: t for t in reader.tensors}
f32_hash = hashlib.sha256()
bf16_hash = hashlib.sha256()
per_tensor = []
changed = total = nonfinite = 0
maxabs = 0.0
for layer in range(40):
    for shared in (False, True):
        name = f'blk.{layer}.ffn_gate_inp{"_shexp" if shared else ""}.weight'
        t = tensors[name]
        shape = [2048] if shared else [2048, 256]
        if int(t.tensor_type) != 0 or list(t.shape) != shape:
            raise ValueError(f'Not original F32 router geometry: {name}')
        values = np.asarray(t.data).reshape(-1).astype('<f4', copy=False)
        finite = np.isfinite(values)
        nonfinite += int(np.count_nonzero(~finite))
        if not finite.all():
            raise ValueError(f'Nonfinite coefficient: {name}')
        bits = values.view('<u4')
        rne = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype('<u2')
        expanded_bits = rne.astype('<u4') << 16
        count = int(np.count_nonzero(bits != expanded_bits))
        error = float(np.max(np.abs(values.astype(np.float64) - expanded_bits.view('<f4'))))
        lowbits = int(np.count_nonzero(bits & np.uint32(65535)))
        changed += count
        total += len(values)
        maxabs = max(maxabs, error)
        f32_hash.update(values.tobytes())
        bf16_hash.update(rne.tobytes())
        per_tensor.append({'name': name, 'coefficients': len(values),
                           'changed_f32_bits_after_bf16_roundtrip': count,
                           'nonzero_low16_bits': lowbits, 'maxabs': error})
image = a.image.resolve(strict=True)
image_hash = hashlib.sha256()
with image.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
        image_hash.update(block)
if image.stat().st_size != total * 2 or image_hash.hexdigest() != bf16_hash.hexdigest():
    raise ValueError('Independent source packing differs from guarded compact image')
if source.stat() != st:
    raise ValueError('Source changed during audit')
result = {'source': str(source), 'source_stat': {'device': st.st_dev,
          'inode': st.st_ino, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
          'ctime_ns': st.st_ctime_ns}, 'source_tensor_count': len(per_tensor),
          'coefficient_count': total, 'nonfinite_coefficients': nonfinite,
          'source_coefficients_changed_by_bf16': changed,
          'source_nonzero_low16_bits': sum(t['nonzero_low16_bits'] for t in per_tensor),
          'coefficient_roundtrip_maxabs': maxabs,
          'source_router_f32_sha256': f32_hash.hexdigest(),
          'independent_source_rne_bf16_sha256': bf16_hash.hexdigest(),
          'guarded_image': str(image), 'guarded_image_sha256': image_hash.hexdigest(),
          'guarded_image_matches_independent_source_packing': True,
          'source_router_coefficients_losslessly_bf16_storable': changed == 0,
          'not_a_general_f32_to_bf16_losslessness_claim': True,
          'per_tensor': per_tensor}
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps({k: v for k, v in result.items() if k != 'per_tensor'}))
