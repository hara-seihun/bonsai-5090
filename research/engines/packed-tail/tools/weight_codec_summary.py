#!/usr/bin/env python3
"""Exact payload accounting from weight_codec_census JSONL; no GPU cost claim."""
import argparse
import json
import math
from pathlib import Path


def summarize(rows):
    blocks = sum(r['blocks'] for r in rows)
    hist = [sum(r['nnz_hist'][i] for r in rows) for i in range(129)]
    trits = [sum(r['trits'][i] for r in rows) for i in range(3)]
    if not blocks or sum(hist) != blocks or sum(trits) != blocks * 128:
        raise ValueError('invalid census counts')
    fast = hist[86]
    # Per-tensor bitmaps and uint64 exclusive payload-byte prefixes per512 blocks.
    # uint64 avoids an invalid >4GiB aggregate offset; local consumers can narrow explicitly.
    bitmap = sum((r['blocks'] + 7) // 8 for r in rows)
    index = sum(8 * ((r['blocks'] + 511) // 512 + 1) for r in rows)
    payload = fast * 25 + (blocks - fast) * 26
    dense = blocks * 26
    encoded = payload + bitmap + index
    rank_bits = sum(r['rank_mask_sign_bits'] for r in rows)
    return {
        'matrices': len(rows), 'blocks': blocks, 'trits': trits,
        'nonzero_fraction': (trits[0] + trits[2]) / (blocks * 128),
        'fast86_blocks': fast, 'fast86_fraction': fast / blocks,
        'rank_width_bits': (math.comb(128, 42) - 1).bit_length(),
        'dense5_label_bytes': dense,
        'fixed86_payload_bytes': payload, 'bitmap_bytes': bitmap,
        'prefix_index_u64_512_bytes': index,
        'fixed86_total_bytes': encoded,
        'label_bytes_saved': dense - encoded,
        'label_fraction_saved': 1 - encoded / dense,
        'original_fp16_scale_bytes_unchanged': blocks * 2,
        'rank_every_count_with_8bit_count_bytes': (rank_bits + 7) // 8,
        'scope': 'exact decoded trits; scales unchanged; no layout padding or execution bill',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('census', type=Path)
    args = ap.parse_args()
    all_rows = [json.loads(line) for line in args.census.read_text().splitlines()]
    if len(all_rows) < 2 or all_rows[-1]['tensor'] != 'TOTAL':
        raise ValueError('incomplete census')
    rows = all_rows[:-1]
    names = [r['tensor'] for r in rows]
    if len(set(names)) != len(names) or names.count('token_embd.weight') != 1 or names.count('output.weight') != 1:
        raise ValueError('model identity/duplicate tensor mismatch')
    aggregate = all_rows[-1]
    for row in rows + [aggregate]:
        if len(row['nnz_hist']) != 129 or len(row['trits']) != 3:
            raise ValueError('invalid histogram extent')
        values = row['nnz_hist'] + row['trits'] + [row['blocks']]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError('invalid count domain')
        if sum(row['nnz_hist']) != row['blocks'] or sum(row['trits']) != row['blocks'] * 128:
            raise ValueError('invalid row counts')
    for key, size in [('nnz_hist', 129), ('trits', 3)]:
        if aggregate[key] != [sum(r[key][i] for r in rows) for i in range(size)]:
            raise ValueError(f'aggregate {key} mismatch')
    for key in ['blocks', 'dense5_bits', 'rank_mask_sign_bits', 'raw_mask_sign_bits']:
        if aggregate[key] != sum(r[key] for r in rows):
            raise ValueError(f'aggregate {key} mismatch')
    selected = [r for r in rows if r['tensor'] != 'token_embd.weight']
    if any(not (r['tensor'].startswith('blk.') or r['tensor'] == 'output.weight') for r in selected):
        raise ValueError('unrecognized streamed projection')
    print(json.dumps({
        'all_ptq_matrices': summarize(rows),
        'selected_streamed_projections': summarize(selected),
        'selected_inventory': 'all blk.* PTQ matrices and output.weight; embedding excluded',
        'notes': ['FP16-scale traffic is absent in cached carrier; preserved scale bytes are storage only.',
                  'Selected engine auxiliary projections and row means are outside label budget.',
                  'Bitmap prefix popcount, rank decoding, memory layout and boundary conversions must be priced.'],
    }, indent=2))


if __name__ == '__main__':
    main()
