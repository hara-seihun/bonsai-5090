#!/usr/bin/env python3
"""Fast CPU contracts for report quality gates, pooling, and evidence custody."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import summarize_gpu_grammar as report


def fixture():
    config = {"depth": 12, "source": "abc123", "binary": "/tmp/engine",
              "binary_sha256": "a" * 64, "environment": {"H5_NO_SPEC": "0"}}
    arms = ["unconstrained_spec", "constrained_spec", "constrained_gpu_spec"]
    cases = [{"id": "a"}, {"id": "b"}, {"id": "c", "suite": "expanded"}]
    meta = {"cases": cases, "arms": arms, "draft_limits": [12], "args": {"repetitions": 1}}
    records = []
    for i, case in enumerate(cases):
        for arm in arms:
            gpu = "gpu" in arm
            grammar = {"mode": "ordinary", "gpu": gpu, "draft_token_waits": 0 if gpu else 4,
                       "gpu_grammar_states": 20 if gpu else None, "gpu_grammar_bytes": 20000 if gpu else None,
                       "forced_tokens": i, "observed_tokens": 1 if i == 0 else 9,
                       "legal_tokens_sum": (1 if i == 0 else 9) * 4,
                       "draft_body_s": .2, "draft_selection_s": .1, "draft_token_wait_s": 0 if gpu else .02}
            if arm == "unconstrained_spec":
                grammar = {}
            records.append({"phase": "measured", "case_id": case["id"], "arm": arm,
                            "repetition": 0, "draft_limit": 12, "status": "ok", "text": "true",
                            "parsed": True, "schema_valid": True, "semantic_correct": True,
                            "stopped": True, "capped": False, "grammar_complete": True,
                            "finish_reason": "stop", "stop_reason": "grammar_complete", "contract_errors": [],
                            "metrics": {"decode_tok_per_s": 200 if gpu else 100,
                                        "client_tok_per_s": 180 if gpu else 90,
                                        "accepted_draft_length": 1 if i == 0 else 3,
                                        "forced_token_fraction": 0 if i == 0 else i / 9},
                            "timings": {"grammar": grammar, "accepted_drafts": 1 if i == 0 else 9,
                                        "verify_rounds": 1 if i == 0 else 3, "proposed_drafts": 10}})
    return config, meta, {"arms": {}}, records


class Contracts(unittest.TestCase):
    def build(self, inputs):
        result = report.build_report(*inputs)
        self.assertNotIsInstance(result, report.Error)
        return result

    def comparison(self, result):
        return result["by_request_depth"]["12"]["full"]["comparisons"]["constrained_spec__vs__constrained_gpu_spec"]

    def test_quality_and_exact_output_are_distinct(self):
        inputs = fixture()
        for row in inputs[3]:
            if "gpu" in row["arm"]:
                row["text"] = " true \n"
        result = self.build(inputs)
        pair = self.comparison(result)
        self.assertEqual(pair["verdict"], "matched_quality_native_speed_win")
        self.assertEqual(pair["text_identical_pairs"], 0)
        self.assertEqual(pair["parsed_identical_pairs"], 3)
        self.assertFalse(pair["same_law_exact_output_all"])
        unconstrained = result["by_request_depth"]["12"]["full"]["comparisons"]["unconstrained_spec__vs__constrained_spec"]
        self.assertTrue(unconstrained["matched_quality_all"])
        self.assertIsNone(unconstrained["pairs"][0]["grammar_law_identical"])

    def test_no_speed_win_from_failed_quality_or_missing_requests(self):
        for failure in ("schema_valid", "semantic_correct", "grammar_complete"):
            with self.subTest(failure=failure):
                inputs = fixture()
                inputs[3][2][failure] = False
                pair = self.comparison(self.build(inputs))
                self.assertEqual(pair["verdict"], "not_a_matched_quality_speed_win")
                self.assertEqual(pair["raw_paired_rate_ratios"]["decode_tok_per_s"]["mean"], 2)
        inputs = fixture()
        inputs[3].pop()
        pair = self.comparison(self.build(inputs))
        self.assertFalse(pair["matched_quality_all"])
        self.assertIn("right:missing_request", pair["pairs"][2]["refusal_reasons"])

    def test_caps_contracts_and_request_errors_never_qualify(self):
        for kind in ("cap", "contract", "error"):
            with self.subTest(kind=kind):
                inputs = fixture()
                row = inputs[3][2]
                if kind == "cap":
                    row["capped"] = True
                elif kind == "contract":
                    row["contract_errors"] = ["speculation_not_observed"]
                else:
                    row["status"] = "error"
                    row["error"] = {"kind": "transport_error", "message": "test"}
                result = self.build(inputs)
                self.assertEqual(self.comparison(result)["verdict"], "not_a_matched_quality_speed_win")
                gpu = result["by_request_depth"]["12"]["full"]["arms"]["constrained_gpu_spec"]
                self.assertFalse(gpu["quality_pass"])
                self.assertEqual(gpu["failures"][0]["case_id"], "a")

    def test_unknown_wait_is_not_zero(self):
        inputs = fixture()
        del inputs[3][2]["timings"]["grammar"]["draft_token_waits"]
        result = self.build(inputs)
        gpu = result["by_request_depth"]["12"]["full"]["arms"]["constrained_gpu_spec"]
        self.assertIsNone(gpu["token_wait_count_all_zero"])
        self.assertIn("gpu_token_wait_count_unknown", gpu["failures"][0]["reasons"])
        self.assertFalse(self.comparison(result)["matched_quality_all"])

    def test_pooled_denominators_and_splits(self):
        inputs = fixture()
        warmup = copy.deepcopy(inputs[3][2])
        warmup["phase"] = "warmup"
        warmup["schema_valid"] = False
        inputs[3].append(warmup)
        result = self.build(inputs)
        base = result["by_request_depth"]["12"]["base24"]
        self.assertEqual(base["fixture_count"], 2)
        arm = base["arms"]["constrained_gpu_spec"]
        self.assertEqual(arm["metrics"]["accepted_draft_length"]["mean"], 2)
        self.assertEqual(arm["pooled"]["accepted_draft_length"]["value"], 2.5)
        self.assertEqual(arm["pooled"]["forced_token_fraction"]["value"], .1)
        self.assertTrue(arm["quality_pass"])
        del inputs[3][2]["timings"]["verify_rounds"]
        arm = self.build(inputs)["by_request_depth"]["12"]["base24"]["arms"]["constrained_gpu_spec"]
        self.assertIsNone(arm["pooled"]["accepted_draft_length"]["value"])

    def test_duplicate_record_is_typed_error(self):
        inputs = fixture()
        inputs[3].append(copy.deepcopy(inputs[3][0]))
        self.assertEqual(report.build_report(*inputs).kind, "duplicate_record")

    def test_evidence_hashes_reproducible_output_and_no_overwrite(self):
        config, metadata, summary, rows = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / "panel"
            panel.mkdir()
            for path, value in ((root / "config.json", config), (panel / "metadata.json", metadata), (panel / "summary.json", summary)):
                path.write_text(json.dumps(value))
            (panel / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            out = root / "report1"
            self.assertIsNone(report.run(root, out))
            data = (out / "report.json").read_bytes()
            self.assertIn("quality_rule", json.loads(data)["scope"])
            self.assertIn("ELF SHA256", (out / "report.md").read_text())
            self.assertEqual(report.run(root, out).kind, "output_exists")
            self.assertEqual((out / "report.json").read_bytes(), data)
            out2 = root / "report2"
            self.assertIsNone(report.run(root, out2))
            self.assertEqual((out2 / "report.json").read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
