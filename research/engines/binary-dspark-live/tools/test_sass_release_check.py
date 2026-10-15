#!/usr/bin/env python3
"""CPU-only SASS audit contracts: python3 tools/test_sass_release_check.py."""
import contextlib
import io
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import sass_release_check as audit

NATIVE = Path(__file__).with_name('sass_release_check_native.sass')
LOAD = (0, 'LDS.128 R0, [R4]', 0, 0)
LATER_LOAD = (16, 'LDSM.16.M88.4 R8, [R12]', 1, 0)
RELEASE = (48, '@!P0 SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R0], RZ', 7, 0)


class ReleaseContract(unittest.TestCase):
    def test_recorded_native_hazard(self):
        # CUDA 12.8 sm_120a halo-fix-gemm/build-gemm-ring/fp4.o, first affected
        # function. Its two outstanding LDS loads precede the predicated release.
        with NATIVE.open() as source:
            [(name, body)] = audit.parse_functions(source)
        self.assertEqual(len(body), 15)
        self.assertEqual(body[0][2:], (0, 0))
        self.assertEqual(body[10][2:], (1, 0))
        [(address, _, pending)] = audit.check(body)
        self.assertEqual(address, 0x4fa0)
        self.assertEqual([entry[0] for entry in pending], [0x4ec0, 0x4f60])

    def test_later_shared_load_wait_completes_older_loads(self):
        wait = (32, 'NOP', 7, 2)
        self.assertEqual(audit.check([LOAD, LATER_LOAD, wait, RELEASE]), [])
        wait_first = (32, 'NOP', 7, 1)
        [(_, _, pending)] = audit.check([LOAD, LATER_LOAD, wait_first, RELEASE])
        self.assertEqual(pending, [(16, 'LDSM.16.M88.4 R8, [R12]', 1)])

    def test_no_scoreboard_is_not_waitable(self):
        load = (0, 'LDS R0, [R4]', 7, 0)
        [(_, _, pending)] = audit.check([load, (16, 'NOP', 7, 63), RELEASE])
        self.assertEqual(pending, [(0, 'LDS R0, [R4]', 7)])

    def test_fences_and_unconditional_control_reset_pending(self):
        for instruction in ['MEMBAR.CTA', 'FENCE.VIEW.ASYNC.S', 'BRA 0x80',
                            'EXIT', 'BAR.SYNC 0x0', 'RET']:
            with self.subTest(instruction=instruction):
                self.assertEqual(audit.check([LOAD, (16, instruction, 7, 0), RELEASE]), [])
        self.assertEqual(len(audit.check([LOAD, (16, '@P0 BRA 0x80', 7, 0), RELEASE])), 1)

    def test_padding_and_expect_tx_are_not_empty_releases(self):
        self.assertEqual(audit.check([(0, '@!PT LDS R0, [R4]', 0, 0), RELEASE]), [])
        self.assertEqual(audit.check([LOAD, (16, 'SYNCS.ARRIVE.TRANS64 RZ, [R4], R0', 7, 0)]), [])

    def test_whitespace_width_and_full_function_coverage(self):
        source = NATIVE.read_text()
        wide = source.replace(' &wr=', ' ' * 10000 + '&wr=')
        [(name, body)] = audit.parse_functions(io.StringIO(wide))
        self.assertEqual(len(body), 15)
        [(address, _, pending)] = audit.check(body)
        self.assertEqual(address, 0x4fa0)
        self.assertEqual([entry[0] for entry in pending], [0x4ec0, 0x4f60])
        other = ('Function : clean\n/*0000*/ EXIT; /* 0x0000000000000000 */\n'
                 '/* 0x000fc00000000000 */\n')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = audit.audit(audit.parse_functions(io.StringIO(source + other)), True)
        self.assertEqual(result, 1)
        self.assertIn('2 function(s) audited, 16 instruction(s)', output.getvalue())
        self.assertIn('pending 04ec0', output.getvalue())
        self.assertIn('1 function(s) affected', output.getvalue())

    def test_incomplete_dump_never_reports_clean(self):
        for source in ['', 'Function : empty\n',
                       'Function : truncated\n/*0000*/ LDS R0, [R4]; /* 0x0000 */\n',
                       'Function : bad\n/*0000*/ LDS R0, [R4]; /* 0x0000 */\nnot a control word\n']:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'input.sass'
                path.write_text(source)
                output, errors = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    result = audit.main(['--sass', str(path)])
                self.assertEqual(result, 2)
                self.assertNotIn('function(s) affected', output.getvalue())
                self.assertIn('SASS audit error:', errors.getvalue())

    def test_disassembler_failure_reports_stderr_without_clean_summary(self):
        popen = audit.subprocess.Popen
        program = 'import sys; sys.stderr.write("bad binary\\n" + "x" * 131072); sys.exit(42)'
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(audit.subprocess, 'Popen',
                          side_effect=lambda command, **kwargs: popen([sys.executable, '-c', program], **kwargs)), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(audit.main(['binary']), 2)
        self.assertIn('bad binary', errors.getvalue())
        self.assertNotIn('function(s) affected', output.getvalue())

    def test_missing_disassembler_is_an_input_error(self):
        with patch.object(audit.subprocess, 'Popen', side_effect=FileNotFoundError('cuobjdump')), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(audit.main(['binary']), 2)


if __name__ == '__main__':
    unittest.main()
