#!/usr/bin/env python3
"""CPU checks for captured weight metadata and historical/stable tag identities."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import smlit_analyze as lit
import smlit_spec as spec


class MetadataTests(unittest.TestCase):
    def read_weights(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.csv'
            path.write_text(text)
            return spec.load_draft_weights(path)

    def test_captured_mixed_format_round_includes_output_row_scales(self):
        weights = self.read_weights('N,K,bytes_per_128,launches\n'
                                   '5120,25600,52,1\n10240,5120,52,1\n'
                                   '1280,5120,52,10\n6144,5120,52,5\n'
                                   '5120,4096,52,5\n34816,5120,36,5\n'
                                   '5120,17408,52,5\n256,5120,72,1\n'
                                   '248320,5120,26,1\n')
        counts = {1024000: 1, 409600: 1, 51200: 10, 245760: 5, 163840: 5,
                  1392640: 5, 696320: 5, 10240: 1, 9932800: 1}
        self.assertEqual(sum(weights[blocks].bytes * count for blocks, count in counts.items()), 899513344)
        self.assertEqual(weights[1392640].format, 'b2')
        self.assertEqual(weights[10240].bytes, 738304)
        self.assertEqual(weights[9932800].bytes, 258252800)

    def test_metadata_rejects_missing_unknown_and_indistinguishable_shapes(self):
        for text in ['N,K\n32,128\n', 'N,K,bytes_per_128\n32,127,36\n',
                     'N,K,bytes_per_128\n32,128,35\n',
                     'N,K,bytes_per_128\n32,256,36\n64,128,36\n',
                     'N,K,bytes_per_128\n32,128,36\n32,128,52\n']:
            with self.subTest(text=text):
                self.assertIsInstance(self.read_weights(text), str)
        duplicate = self.read_weights('N,K,bytes_per_128\n32,128,36\n32,128,36\n')
        self.assertEqual(len(duplicate), 1)

    def test_tern_format_includes_row_scales(self):
        self.assertEqual(spec.DraftWeight(32, 128, 34).bytes, 1216)

    def test_included_header_collisions_require_explicit_source_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'kernels.cu').write_text('#include "a.cuh"\n#include "b.cuh"\n')
            (root / 'a.cuh').write_text('__global__ void k_gdn_single() { H5_LIT(0); }\n')
            (root / 'b.cuh').write_text('__global__ void k_gdn_verify() { H5_LIT(0); }\n')
            names = lit.tag_names(root)
            self.assertEqual(names[100001], 'k_gdn_single|k_gdn_verify')
            records = np.zeros(1, dtype=lit.REC); records['tag'] = 100001
            with self.assertRaisesRegex(SystemExit, 'ambiguous'):
                lit.validate_tags(records, names)
            lit.validate_tags(records, lit.tag_names(root, ['100001=k_gdn_verify']))
            with self.assertRaisesRegex(SystemExit, 'not a source tag candidate'):
                lit.tag_names(root, ['100001=k_other'])
            with self.assertRaisesRegex(SystemExit, 'conflicting'):
                lit.tag_names(root, ['100001=k_gdn_verify', '100001=k_gdn_single'])

    def test_current_source_has_unique_stable_site_tags(self):
        names = lit.tag_names(Path(__file__).resolve().parents[1] / 'src')
        self.assertFalse({tag: name for tag, name in names.items() if '|' in name})
        expected = {0x80010001: 'k_gdn_single', 0x80010002: 'k_gdn_verify',
                    0x80010003: 'k_conv_gdn_verify', 0x80010004: 'k_prep2w_ordered',
                    0x80010005: 'k_attn_combine_tail'}
        self.assertEqual({tag: name for tag, name in names.items() if tag & 0x80000000}, expected)

    def test_named_sites_survive_header_line_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'smlit.cuh').write_text('enum class SiteTag : uint32_t { GdnVerify = 0x80010002u };\n')
            (root / 'kernels.cu').write_text('#include "verify.cuh"\n')
            header = root / 'verify.cuh'
            marker = '__global__ void k_gdn_verify() { H5_LIT_SITE(GdnVerify, 0); }\n'
            header.write_text(marker)
            before = lit.tag_names(root)
            header.write_text('\n' * 40 + marker)
            self.assertEqual(lit.tag_names(root), before)
            self.assertEqual(before[0x80010002], 'k_gdn_verify')


if __name__ == '__main__':
    unittest.main()
