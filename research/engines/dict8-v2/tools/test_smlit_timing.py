#!/usr/bin/env python3
"""CPU regressions for sub-256 ns landmarks at epoch-sized GPU globaltimer values.
Run: python3 tools/test_smlit_timing.py
"""
import contextlib
import csv
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import smlit_analyze as lit
import smlit_gemm_release as gemm
import smlit_spec as spec

ORIGIN = 1_791_480_000_000_000_000


def records(origin):
    r = np.zeros(2, dtype=lit.REC)
    r['tag'] = 400001; r['grid'] = 1; r['threads'] = 32
    r['t0'] = [origin, origin + 40032]
    r['t1'] = [origin + 30000, origin + 70032]
    r['cw'] = [33, 65]; r['cx'] = 30000
    return r


def dump(path, r):
    path.write_bytes(struct.pack('<4Q', 0x544c4d53, 10, 1, len(r)) + r.tobytes())


class RelativeTimingTests(unittest.TestCase):
    def test_analyzer_is_epoch_translation_invariant(self):
        summaries = []; launches = []
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i, origin in enumerate([0, ORIGIN]):
                raw = root / f'{i}.bin'; out = root / str(i)
                dump(raw, records(origin))
                with patch.object(lit, 'tag_names', return_value={400001: 'graph_copy'}), \
                     patch('sys.argv', ['smlit_analyze.py', str(raw), str(root), str(out), '--sms', '1']), \
                     contextlib.redirect_stdout(io.StringIO()):
                    lit.main()
                summaries.append(json.loads((out / 'summary.json').read_text()))
                with (out / 'launches.csv').open() as f:
                    launches.append(list(csv.DictReader(f)))
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(launches[0], launches[1])
        self.assertAlmostEqual(float(launches[1][0]['wait_release_first_us']), .033)
        self.assertAlmostEqual(float(launches[1][1]['wait_release_first_us']), 40.097)
        self.assertAlmostEqual(summaries[1]['rounds'][0]['time_zero_sms_active_us'], 10.130)

    def test_spec_uses_common_relative_origin_across_segments(self):
        low = spec.seg_stats(records(0), {400001: 'graph_copy'}, 1.0, 0)
        high = spec.seg_stats(records(ORIGIN), {400001: 'graph_copy'}, 1.0, ORIGIN)
        self.assertEqual(low, high)
        self.assertEqual(high[0]['release'], 33)
        self.assertEqual(high[1]['release'] - high[0]['exit'], 10097)
        second = records(ORIGIN + 80000)
        later = spec.seg_stats(second, {400001: 'graph_copy'}, 1.0, ORIGIN)
        self.assertEqual(later[0]['entry'] - high[-1]['exit'], 9968)

    def test_gemm_marks_subtract_uint64_origin_before_float_conversion(self):
        reports = []
        with tempfile.TemporaryDirectory() as d:
            for i, origin in enumerate([0, ORIGIN]):
                r = records(origin)
                r['tag'] = 200001; r['cta'] = 0; r['threads'] = 64
                r['warp'] = [0, 1]
                r['t0'] = [origin, origin + 32]
                r['t1'] = [origin + 30000, origin + 30032]
                r['mk'] = [[33, 65, 97, 129], [35, 67, 99, 131]]
                raw = Path(d) / f'{i}.bin'; dump(raw, r)
                with patch.object(gemm, 'tag_names', return_value={200001: 'k_gemm_fp4'}):
                    reports.append(gemm.analyze(raw, d))
        self.assertEqual(reports[0], reports[1])
        g = reports[1]['rounds'][0]['gemms'][0]
        self.assertAlmostEqual(g['release_first_us'], .129)
        self.assertAlmostEqual(g['release_last_us'], .163)
        self.assertAlmostEqual(g['entry_last_us'], .032)
        self.assertAlmostEqual(g['consumer_first_stage_ready_us']['p50'], .033)


if __name__ == '__main__':
    unittest.main()
