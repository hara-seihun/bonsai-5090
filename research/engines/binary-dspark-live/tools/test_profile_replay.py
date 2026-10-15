#!/usr/bin/env python3
"""CPU-only replay boundary tests; set H5_REPLAY_ORDINARY_SOURCE for the recorded S64 check."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import profile_replay_s64 as s64

ROOT = Path(__file__).resolve().parents[1]


class ReplayContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.binary = cls.root / "profile-replay-cpu"
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-DH5_REPLAY_CPU_ONLY", "-I" + str(ROOT / "src"),
                        str(ROOT / "tools/profile_replay.cpp"), "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def replay(self, events, *options, dry_run=True):
        path = self.root / "events.txt"
        path.write_text(events)
        return subprocess.run([str(self.binary), "nonexistent-model", str(path), str(self.root / "unused"),
                               *options, *(["--dry-run"] if dry_run else [])], capture_output=True, text=True)

    def test_retained_capacities_and_explicit_capacities(self):
        result = self.replay("RESET 0\nKEEP 0 0\nSTEP 1 0 1 4095\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        self.assertEqual((config["slots"], config["ctx"], config["rowcap"]), (1, 4096, 520))
        self.assertFalse((self.root / "unused.csv").exists())
        self.assertNotEqual(self.replay("RESET 1\n").returncode, 0)
        result = self.replay("RESET 63\nKEEP 63 1\nACCEPT 63 8\nHEADSTEP 1 63 1 174 1 0\n",
                             "--slots", "64", "--ctx", "175")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_slot_position_and_reset_checked_before_execution(self):
        invalid = [
            "RESET -1\n", "RESET 64\n",
            "RESET 0\nKEEP -1 0\n", "RESET 0\nKEEP 64 0\n",
            "RESET 0\nACCEPT -1 0\n", "RESET 0\nACCEPT 64 0\n",
            "RESET 0\nSTEP 1 -1 1 0\n", "RESET 0\nSTEP 1 64 1 0\n",
            "RESET 0\nSTEP 1 0 1 -1\n", "RESET 0\nSTEP 1 0 1 175\n",
            "RESET 0\nHEADSTEP 1 0 1 175 0\n",
            "RESET 0\nSTEP 1 1 1 0\n", "RESET 0\nKEEP 1 0\n", "RESET 0\nACCEPT 1 0\n",
            "RESET 0\nSTEP 2 0 1 0 1 1 0\n",
            "RESET 0\nSTEP 1 0 1 0\nRESET 64\n",
        ]
        for events in invalid:
            with self.subTest(events=events):
                result = self.replay(events, "--slots", "64", "--ctx", "175")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("event line", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_invalid_and_duplicate_cli(self):
        invalid = [
            ["--slots", "0"], ["--slots", "-1"], ["--slots", "64x"], ["--slots", "2147483648"],
            ["--slots"], ["--slots", "2", "--slots", "2"],
            ["--ctx", "0"], ["--ctx", "-1"], ["--ctx", "175x"], ["--ctx", "2147483647"],
            ["--ctx"], ["--ctx", "175", "--ctx", "175"],
            ["--passes", "0"], ["--passes", "1x"], ["--passes", "101"],
            ["--passes", "1", "--passes", "1"], ["--decode-only", "--decode-only"],
            ["--ordinary", "--ordinary"], ["--dry-run"], ["--unknown"],
            ["--graph-upload", "capture"], ["--graph-nodes", "native"],
            ["--graph-upload", "capture", "--graph-upload", "launch"],
            ["--graph-timing", "off", "--graph-timing", "off"],
            ["--graph-nodes", "native", "--graph-nodes", "native"],
            ["--graph-upload", "unknown"], ["--graph-timing", "unknown"], ["--graph-nodes", "unknown"],
            ["--head-rows", "all", "--head-rows", "all"], ["--head-rows", "unknown"],
        ]
        for options in invalid:
            with self.subTest(options=options):
                result = self.replay("RESET 0\n", *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_malformed_events(self):
        invalid = ["", "\n", "KEEP 0 0\n", "RESET\n", "RESET 0 0\n", "OTHER 0\n",
                   "RESET 0\nKEEP 0 2\n", "RESET 0\nACCEPT 0 9\n", "RESET 0\nSTEP 0\n",
                   "RESET 0\nSTEP 521\n", "RESET 0\nSTEP 1 0 1\n",
                   "RESET 0\nSTEP 1 0 1 999999999999999999999\n",
                   "RESET 0\nSTEP 1 0 1 0x\n", "RESET 0\nHEADSTEP 1 0 1 0 2 0 0\n",
                   "RESET 0\nHEADSTEP 2 0 1 0 0 1 1 2 1 0\n"]
        for events in invalid:
            with self.subTest(events=events):
                self.assertNotEqual(self.replay(events).returncode, 0)
        self.assertIn("requires --dry-run", self.replay("RESET 0\n", dry_run=False).stderr)

    def test_rowcap_and_all_slots_reset_before_use(self):
        rows = " ".join(f"0 1 {pos}" for pos in range(520))
        self.assertEqual(self.replay(f"RESET 0\nSTEP 520 {rows}\n").returncode, 0)
        resets = "".join(f"RESET {slot}\n" for slot in range(64))
        rows = " ".join(f"{slot} 1 0" for slot in range(64))
        result = self.replay(resets + f"STEP 64 {rows}\n", "--slots", "64")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_generator_rejects_guessed_source(self):
        source = self.root / "guessed-source.txt"
        source.write_text("RESET 0\nSTEP 1 0 1 0\n")
        with self.assertRaisesRegex(ValueError, "source SHA256"):
            s64.generate(source, self.root / "guessed-s64.txt", self.root / "guessed-manifest.json")
        self.assertFalse((self.root / "guessed-s64.txt").exists())

    @unittest.skipUnless(os.environ.get("H5_REPLAY_ORDINARY_SOURCE"), "recorded source not supplied")
    def test_actual_recording_s64_replication_and_measurement_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Path(os.environ["H5_REPLAY_ORDINARY_SOURCE"])
            output, manifest_path = root / "s64.txt", root / "manifest.json"
            manifest = s64.generate(source, output, manifest_path)
            actual = s64.recorded_steps(source.read_bytes())
            generated = [line.split() for line in output.read_text().splitlines()]
            prefill = [line for line in generated if line[:2] == ["STEP", "48"]]
            decodes = [line for line in generated if line[:2] == ["STEP", "64"]]
            self.assertEqual(len(prefill), 64)
            self.assertEqual(len(decodes), 127)
            for slot, line in enumerate(prefill):
                self.assertEqual(list(map(int, line[2:])), [v for _, token, pos in actual[0] for v in [slot, token, pos]])
            for original, line in zip(actual[1:], decodes):
                _, token, pos = original[0]
                self.assertEqual(list(map(int, line[2:])), [v for slot in range(64) for v in [slot, token, pos]])
            result = subprocess.run([str(self.binary), "nonexistent-model", str(output), "unused", "--slots", "64",
                                     "--ctx", "175", "--ordinary", "--decode-only", "--dry-run"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(result.stdout)
            self.assertEqual((config["steps"], config["prefills"], config["decode_steps"]), (191, 64, 127))
            self.assertEqual(manifest["fixture"]["events"], config["events"])
            synthetic = io.StringIO(newline="")
            writer = csv.writer(synthetic)
            for _ in range(4):
                for round_index in range(191):
                    rows = 48 if round_index < 64 else 64
                    writer.writerow([round_index, rows, 1.0] + [0] * rows)
            csv_path = root / "synthetic.csv"
            csv_path.write_text(synthetic.getvalue())
            decode_path, report_path = root / "decode.csv", root / "report.json"
            report = s64.summarize(csv_path, manifest_path, "ABBA", decode_path, report_path)
            self.assertEqual(len(decode_path.read_text().splitlines()), 4 * 127)
            self.assertEqual([p["arm"] for p in report["passes"]], list("ABBA"))
            self.assertEqual(report["passes"][0]["aggregate_tokens_per_second"], 64)
            csv_path.write_text("\n".join(csv_path.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "all 127"):
                s64.summarize(csv_path, manifest_path, "ABBA", root / "truncated.csv", root / "truncated.json")
            for old, new in [("0,48,1.0", "1,48,1.0"), ("64,64,1.0", "64,48,1.0"),
                             ("64,64,1.0", "64,64,nan"), ("64,64,1.0", "64,64,0")]:
                csv_path.write_text(synthetic.getvalue().replace(old, new, 1))
                with self.subTest(replacement=new), self.assertRaises(ValueError):
                    s64.summarize(csv_path, manifest_path, "ABBA", root / "invalid.csv", root / "invalid.json")
            output.write_text(output.read_text() + "RESET 0\n")
            with self.assertRaisesRegex(ValueError, "fixture SHA256"):
                s64.summarize(csv_path, manifest_path, "ABBA", root / "mutated.csv", root / "mutated.json")


if __name__ == "__main__":
    unittest.main()
