#!/usr/bin/env python3
"""Original Q1_0 weight census; each --layer run is an independent bounded stage.

python3 experiments/binary-whole-map/weight_census.py --model MODEL --layer 0 --report REPORT
Repeat for layers 30 and 61. No decoded/requantized weights or synthetic activations.
"""
import argparse
import json
import mmap
import os
import struct
import time
from pathlib import Path

import numpy as np


class InputError(Exception):
    pass


class GGUF:
    SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
               7: 'B', 10: 'Q', 11: 'q', 12: 'd'}

    def __init__(self, path):
        self.file = open(path, 'rb')
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        magic, version, nt, nk = self.take('IIQQ')
        if magic != 0x46554747 or version != 3:
            raise InputError('expected GGUF version 3')
        alignment = 32  # GGUF specification's established initial alignment.
        for _ in range(nk):
            key = self.string()
            typ, = self.take('I')
            if key == 'general.alignment':
                if typ != 4:
                    raise InputError('alignment metadata must be uint32')
                alignment, = self.take('I')
            else:
                self.skip(typ)
        self.tensors = {}
        for _ in range(nt):
            name = self.string()
            nd, = self.take('I')
            dims = self.take('Q' * nd)
            typ, offset = self.take('IQ')
            self.tensors[name] = (dims, typ, offset)
        self.start = (self.pos + alignment - 1) // alignment * alignment

    def take(self, fmt):
        values = struct.unpack_from('<' + fmt, self.data, self.pos)
        self.pos += struct.calcsize('<' + fmt)
        return values

    def string(self):
        n, = self.take('Q')
        value = self.data[self.pos:self.pos+n].decode('utf-8')
        self.pos += n
        return value

    def skip(self, typ):
        if typ in self.SCALARS:
            self.pos += struct.calcsize('<' + self.SCALARS[typ])
        elif typ == 8:
            n, = self.take('Q')
            self.pos += n
        elif typ == 9:
            subtype, n = self.take('IQ')
            if subtype in self.SCALARS:
                self.pos += n * struct.calcsize('<' + self.SCALARS[subtype])
            elif subtype == 8:
                for _ in range(n):
                    self.skip(8)
            else:
                raise InputError(f'unsupported array metadata type {subtype}')
        else:
            raise InputError(f'unsupported metadata type {typ}')

    def weight(self, layer, role):
        name = f'blk.{layer}.ffn_{role}.weight'
        if name not in self.tensors:
            raise InputError(f'missing {name}')
        dims, typ, offset = self.tensors[name]
        expected = (17408, 5120) if role == 'down' else (5120, 17408)
        if typ != 41 or dims != expected:
            raise InputError(f'{name}: expected Q1_0 {expected}, found {typ} {dims}')
        k, n = dims
        raw = np.ndarray((n, k // 128, 18), dtype=np.uint8,
                         buffer=self.data, offset=self.start + offset)
        scales = raw[..., :2].copy().view('<f2').reshape(n, k // 128)
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise InputError(f'{name}: exact sign-only equality assumes finite positive scales')
        return raw, scales


def records(array):
    a = np.ascontiguousarray(array)
    return a.reshape(a.shape[0], -1).view(np.dtype((np.void, a[0].nbytes))).ravel()


def duplicates(a, antipodal):
    if antipodal:
        a = a ^ ((a[:, :1] & 1) * np.uint8(255))
    _, counts = np.unique(records(a), return_counts=True)
    return {'items': len(a), 'classes': len(counts),
            'duplicate_items': int(np.sum(counts - 1)),
            'collision_pairs': int(np.sum(counts.astype(np.int64) * (counts - 1) // 2)),
            'maximum_multiplicity': int(counts.max())}


def summary(a):
    return {'min': int(a.min()), 'mean': float(a.mean()), 'max': int(a.max()),
            'p05': float(np.quantile(a, .05)), 'p50': float(np.quantile(a, .5)),
            'p95': float(np.quantile(a, .95))}


def summary_float(a):
    return {'min': float(a.min()), 'mean': float(a.mean()), 'max': float(a.max()),
            'p05': float(np.quantile(a, .05)), 'p50': float(np.quantile(a, .5)),
            'p95': float(np.quantile(a, .95))}


def matrix_census(raw, scales):
    signs = np.ascontiguousarray(raw[..., 2:])
    n, nb, _ = signs.shape
    rows = signs.reshape(n, -1)
    blocks = signs.reshape(-1, 16)
    result = {'rows': n, 'k': nb * 128, 'q1_bytes': int(raw.size),
              'scale_fp16_unique': int(np.unique(scales.view(np.uint16)).size),
              'scale_fp16_bits_or_hex': f'{int(np.bitwise_or.reduce(scales.view(np.uint16).ravel())):04x}',
              'scale_11bit_reconstruct_all_exact': bool(np.all(((scales.view(np.uint16) >> 3) & 2047) << 3 == scales.view(np.uint16))),
              'scale_storage_candidate': '11-bit label=FP16_bits>>3, exact original FP16_bits=label<<3; bit-reconstruction cost unpriced',
              'scale_min': float(scales.min()), 'scale_max': float(scales.max()),
              'sign_rows': duplicates(rows, False),
              'sign_rows_up_to_antipode': duplicates(rows, True),
              'sign_k128_blocks': duplicates(blocks, False),
              'sign_k128_blocks_up_to_antipode': duplicates(blocks, True)}
    words = {}
    for width in (8, 16, 32):
        w = signs.view(f'<u{width // 8}').reshape(-1)
        _, counts = np.unique(w, return_counts=True)
        p = counts.astype(np.float64) / len(w)
        local = signs.view(f'<u{width // 8}').reshape(-1, 128 // width)
        ordered = np.sort(local, axis=1)
        diversity = 1 + np.sum(ordered[:, 1:] != ordered[:, :-1], axis=1)
        words[str(width)] = {'aligned_words': len(w), 'distinct': len(counts),
                             'entropy_bits': float(-np.sum(p * np.log2(p))),
                             'maximum_frequency': int(counts.max()),
                             'distinct_per_original_k128': summary(diversity)}
    result['local_sign_words'] = words
    bits = np.unpackbits(signs, axis=2, bitorder='little').reshape(n, nb*128)
    # A 2x2 sign matrix has rank one precisely when its four bits have even parity.
    parity = bits[::2, ::2] ^ bits[1::2, ::2] ^ bits[::2, 1::2] ^ bits[1::2, 1::2]
    result['aligned_2x2'] = {'count': int(parity.size),
                             'rank_one': int(np.count_nonzero(parity == 0)),
                             'rank_two': int(np.count_nonzero(parity))}
    return result, bits


def down_census(bits, scales):
    n, k = bits.shape
    nb = k // 128
    packed_columns = np.packbits(bits.T, axis=1, bitorder='little')
    exact = []
    antipodal = []
    for block in range(nb):
        columns = packed_columns[block*128:(block+1)*128]
        exact.append(duplicates(columns, False)['duplicate_items'])
        antipodal.append(duplicates(columns, True)['duplicate_items'])
    result = {'original_k128_blocks': nb,
              'full_output_column_identical_duplicate_items': sum(exact),
              'full_output_column_antipodal_or_identical_duplicate_items': sum(antipodal),
              'scope': '128 columns compared only within each original K128; all 5120 outputs; each output scale shared by these 128 columns',
              'small_output_tile_carriers': {}}
    for r in (4, 8, 16):
        # Class is a column sign vector modulo antipode; orientation is its row-0 bit.
        tile = bits.reshape(n//r, r, nb, 128)
        code = np.zeros((n//r, nb, 128), dtype=np.uint16)
        for j in range(1, r):
            code |= ((tile[:, j] ^ tile[:, 0]).astype(np.uint16) << (j-1))
        ordered = np.sort(code, axis=2)
        distinct = 1 + np.sum(ordered[:, :, 1:] != ordered[:, :, :-1], axis=2)
        u = int(distinct.sum())
        tiles = distinct.size
        aggregate_adds = tiles*128-u
        output_adds = r * (u-tiles)
        direct_adds = tiles*r*127
        result['small_output_tile_carriers'][str(r)] = {
            'tile_k128_pairs': tiles, 'classes_per_128_columns': summary(distinct),
            'total_classes': u,
            'exact_recurrence': 'orient each input by row0 sign, sum inputs by relative column-sign code, form r signed class sums, then multiply each output by its ORIGINAL FP16 scale',
            'arithmetic_model': {'aggregation_additions': aggregate_adds,
                                 'output_reduction_additions': output_adds,
                                 'combined_additions': aggregate_adds+output_adds,
                                 'direct_signed_reduction_additions': direct_adds,
                                 'addition_ratio': (aggregate_adds+output_adds)/direct_adds,
                                 'scale_multiplications_unchanged': int(tiles*r)},
            'boundaries_not_priced': 'prepared class IDs, runtime indexed aggregation/scatter, class initialization, output signs, loads and kernel launch; this is NOT a measured speedup',
            'class_id_bits_upper_bound': r-1,
            'orientation_bits_per_column': 1}
        if r == 8:
            local = code[0, 0]
            pair = None
            for a in range(128):
                matches = np.flatnonzero(local[a+1:] == local[a])
                if matches.size:
                    pair = (a, a+1+int(matches[0]))
                    break
            if pair is not None:
                a, b = pair
                opposite = int(bits[0, a] ^ bits[0, b])
                differences = np.flatnonzero((bits[:, a] ^ bits[:, b]) != opposite)
                witness = {'k128': 0, 'input_columns': [a, b],
                           'observed_outputs': [0, 8], 'antipodal': bool(opposite),
                           'relative_sign_code': int(local[a]),
                           'original_output_scale_fp16_hex': [f'{x:04x}' for x in scales[:8, 0].view(np.uint16)],
                           'full_output_rejection': None}
                if differences.size:
                    j = int(differences[0])
                    witness['full_output_rejection'] = {'first_differing_output': j,
                                                        'column_sign_bits': [int(bits[j, a]), int(bits[j, b])],
                                                        'original_scale_fp16_hex': f'{scales[j, 0].view(np.uint16):04x}'}
                result['local_merge_and_global_rejection_witness'] = witness
    return result


def gu_census(g, u, sg, su):
    signs_g = np.ascontiguousarray(g[..., 2:])
    signs_u = np.ascontiguousarray(u[..., 2:])
    xor = signs_g ^ signs_u
    popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    disagreement = popcount[xor].sum(axis=2)
    same_scale = sg.view(np.uint16) == su.view(np.uint16)
    same_block = np.all(xor == 0, axis=2)
    opposite_block = np.all(xor == 255, axis=2)
    joined_sign_rows = np.concatenate([signs_g.reshape(g.shape[0], -1),
                                       signs_u.reshape(u.shape[0], -1)], axis=0)
    result = {'matched_sign_k128_blocks_identical': int(same_block.sum()),
              'matched_sign_k128_blocks_antipodal': int(opposite_block.sum()),
              'matched_scaled_k128_blocks_identical': int((same_block & same_scale).sum()),
              'matched_scaled_k128_blocks_antipodal': int((opposite_block & same_scale).sum()),
              'matched_original_fp16_scales_equal': int(same_scale.sum()),
              'matched_blocks': int(same_block.size),
              'matched_sign_disagreement_fraction': float(disagreement.sum()/xor.size/8),
              'disagreements_per_matched_k128': summary(disagreement),
              'disagreement_fraction_per_matched_whole_row': summary_float(disagreement.sum(axis=1) / (g.shape[1]*128)),
              'union_gate_up_sign_k128_blocks': duplicates(np.concatenate([signs_g.reshape(-1, 16), signs_u.reshape(-1, 16)]), False),
              'union_gate_up_sign_k128_blocks_up_to_antipode': duplicates(np.concatenate([signs_g.reshape(-1, 16), signs_u.reshape(-1, 16)]), True),
              'matched_sign_rows_identical': int(np.all(same_block, axis=1).sum()),
              'matched_sign_rows_antipodal': int(np.all(opposite_block, axis=1).sum()),
              'union_gate_up_sign_rows': duplicates(joined_sign_rows, False),
              'union_gate_up_sign_rows_up_to_antipode': duplicates(joined_sign_rows, True),
              'two_class_k128_carrier': {
                  'same_sign_columns': summary(128-disagreement),
                  'opposite_sign_columns': summary(disagreement),
                  'recurrence': 'P=sum(gsign*x where gsign=usign), Q=sum(gsign*x where gsign!=usign); G=original_dG*(P+Q), U=original_dU*(P-Q)',
                  'signed_addition_model_per_k128': {'carrier': 128, 'direct_two_reductions': 254},
                  'scope': 'exact real linear algebra, not floating-point accumulation order; no activation capture, quantization or timing',
                  'recurring_boundary': 'two sign bits/class+orientation per input, two accumulators, independent original G/U scales; no SiLU/product closure asserted'}}
    return result


def run(args):
    start = time.perf_counter()
    model = GGUF(args.model)
    g, sg = model.weight(args.layer, 'gate')
    u, su = model.weight(args.layer, 'up')
    d, sd = model.weight(args.layer, 'down')
    layer = {'GU': gu_census(g, u, sg, su), 'matrices': {}}
    for role, raw, scales in [('gate', g, sg), ('up', u, su), ('down', d, sd)]:
        census, bits = matrix_census(raw, scales)
        layer['matrices'][role] = census
        if role == 'down':
            layer['down_observer'] = down_census(bits, scales)
        del bits
    layer['elapsed_seconds'] = time.perf_counter()-start
    report_path = Path(args.report)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        stat = os.stat(args.model)
        if (report['model']['path'] != str(Path(args.model).resolve()) or
                report['model']['bytes'] != stat.st_size or
                report['model']['mtime_ns'] != stat.st_mtime_ns):
            raise InputError('report model identity mismatch')
    else:
        stat = os.stat(args.model)
        report = {'status': 'ok', 'model': {'path': str(Path(args.model).resolve()),
                  'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
                  'layout': 'GGUF type41 Q1_0: row-major K128, FP16 little-endian d then 16 sign bytes, bit1=+d bit0=-d, little-bit-order',
                  'scope': 'exhaustive signs and original positive finite FP16 scales on layers explicitly listed; statistical observations and priced arithmetic candidates, not runtime speedup or a whole-FFN carrier proof',
                  'layers': {}}
    report['layers'][str(args.layer)] = layer
    temporary = report_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(report_path)
    print(json.dumps({'status': 'ok', 'layer': args.layer,
                      'elapsed_seconds': layer['elapsed_seconds'], 'report': str(report_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--layer', type=int, required=True, choices=(0, 30, 61))
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (InputError, OSError, ValueError, KeyError, struct.error) as error:
        print(json.dumps({'status': 'error', 'kind': type(error).__name__, 'detail': str(error)}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
