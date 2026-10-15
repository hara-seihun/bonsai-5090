#!/usr/bin/env python3
"""Weight-stream roof, not an assertion that all work is bandwidth-only.

Run with PYTHONPATH pointing to the pinned Prism fork's gguf-py.
"""
import argparse
import json
from pathlib import Path
from gguf import GGUFReader

p = argparse.ArgumentParser()
p.add_argument('--model', type=Path, required=True)
p.add_argument('--bandwidth-bytes-second', type=float, required=True)
p.add_argument('--tokens-second', type=float, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if min(a.bandwidth_bytes_second, a.tokens_second) <= 0:
    p.error('POSITIVE_RATE_REQUIRED')
r = GGUFReader(str(a.model))
q1 = [t for t in r.tensors if int(t.tensor_type) == 41]
emb = [t for t in q1 if t.name == 'token_embd.weight']
if len(emb) != 1 or len(emb[0].shape) != 2:
    raise ValueError('Q1_EMBEDDING_GEOMETRY_REQUIRED')
e = emb[0]
row_bytes = int(e.shape[0]) // 128 * 18
q1_bytes = sum(int(t.n_bytes) for t in q1)
tensor_bytes = sum(int(t.n_bytes) for t in r.tensors)
def rate(b):
    floor = b / a.bandwidth_bytes_second
    roof = 1 / floor
    return {'bytes_per_token': b, 'weight_stream_floor_ms': floor * 1000,
            'roof_tokens_second': roof, 'measured_percent_of_roof': a.tokens_second / roof * 100}
result = {
    'model': str(a.model), 'tensor_count': len(r.tensors),
    'q1_block_bytes': 18, 'q1_block_values': 128,
    'q1_sign_bytes': q1_bytes * 16 // 18,
    'q1_fp16_scale_bytes': q1_bytes * 2 // 18,
    'embedding_matrix_bytes': int(e.n_bytes), 'one_embedding_row_bytes': row_bytes,
    'bandwidth_bytes_second': a.bandwidth_bytes_second, 'measured_tokens_second': a.tokens_second,
    'all_binary_weights_including_scales': rate(q1_bytes),
    'all_tensor_payload': rate(tensor_bytes),
    'decode_stream_tensor_payload': rate(tensor_bytes - int(e.n_bytes) + row_bytes),
    'scope': 'Batch1 one complete weight stream/token; 1.79TB/s is decimal advertised bandwidth. Embedding reads one selected row, not its entire matrix. Includes original scales; excludes activation/KV/state/partial/output traffic and dependency/compute latency. Those boundaries must be priced by each fusion receipt. Speculative/batched multi-token weight reuse has a different roof.'
}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
