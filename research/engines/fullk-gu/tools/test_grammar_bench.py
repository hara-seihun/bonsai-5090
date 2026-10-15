#!/usr/bin/env python3
"""CPU request/telemetry/pairing contracts; no server or GPU required."""
import copy
import unittest
from types import SimpleNamespace

import grammar_bench as bench


class Validator:
    def iter_errors(self, value):
        return []


class GrammarBenchContracts(unittest.TestCase):
    def setUp(self):
        self.arms = bench.ARMS
        bench.ARMS = ("unconstrained_plain", "unconstrained_spec", "constrained_plain", "constrained_spec",
                      "constrained_canonical_plain", "constrained_canonical_spec", "constrained_jump_plain", "constrained_jump_spec",
                      "constrained_sparse_plain", "constrained_sparse_spec", "constrained_canonical_sparse_plain",
                      "constrained_canonical_sparse_spec", "constrained_jump_sparse_plain", "constrained_jump_sparse_spec",
                      "constrained_bulk_spec", "constrained_canonical_bulk_spec", "constrained_jump_bulk_spec",
                      "constrained_gpu_plain", "constrained_gpu_spec")
        self.case = {"id": "test", "schema": {"const": "red"}, "prompt": "Emit red", "max_tokens": 32, "expected": "red"}
        self.args = SimpleNamespace(model="bonsai", max_tokens=None, seed=42, stream=True, draft_limit=7)

    def tearDown(self):
        bench.ARMS = self.arms

    def response(self, arm, text='"red"', legal_sum=10, observed=2):
        spec = arm.endswith("_spec")
        grammar = {"complete": True, "stop_reason": "grammar_complete", "sparse_head": "sparse" in arm or "gpu" in arm,
                   "gpu": "gpu" in arm, "gpu_grammar_states": 12, "gpu_grammar_bytes": 12300,
                   "bulk_select": "bulk" in arm,
                   "mode": "canonical_fixed" if "canonical" in arm or "jump" in arm else "ordinary",
                   "canonical_law": "ordinary-byte-grammar-forced-spans-only-v3" if "canonical" in arm or "jump" in arm else None,
                   "jump_forward": "jump" in arm,
                   "legal_tokens_sum": legal_sum, "mean_legal_set_size": legal_sum / observed,
                   "forced_tokens": 1, "observed_tokens": observed, "forced_token_fraction": 1 / observed}
        if "gpu" in arm:
            grammar["draft_token_waits"] = 0
        return bench.Reply({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                            "usage": {"completion_tokens": observed},
                            "timings": {"decode_tok_per_s": 100, "verify_rounds": int(spec),
                                        "accepted_drafts": int(spec), "proposed_drafts": 7 * int(spec), "grammar": grammar}}, 1, .1)

    def row(self, arm, limit=7, **kwargs):
        result = bench.evaluate(self.response(arm, **kwargs), self.case, arm, Validator(), True)
        self.assertNotIsInstance(result, bench.Error)
        return {"phase": "measured", "case_id": "test", "repetition": 0, "draft_limit": limit, "arm": arm, **result}

    def test_request_law_and_projection_are_independent(self):
        for arm in bench.ARMS:
            request = bench.request_for(self.case, arm, self.args, 24)
            self.assertEqual(request["draft_verify_limit"], 24 if arm.endswith("_spec") else 0)
            self.assertTrue(request["stream_options"]["include_usage"])
            if arm.startswith("unconstrained_"):
                self.assertNotIn("response_format", request)
                self.assertNotIn("grammar_sparse_head", request)
                self.assertNotIn("grammar_bulk_select", request)
                continue
            self.assertEqual(request["grammar_sparse_head"], "sparse" in arm or "gpu" in arm)
            self.assertEqual(request.get("grammar_gpu", False), "gpu" in arm)
            self.assertEqual(request["grammar_bulk_select"], "bulk" in arm)
            self.assertFalse(request["grammar_sparse_head"] and request["grammar_bulk_select"])
            self.assertEqual(request.get("grammar_jump_forward", False), "jump" in arm)
            self.assertEqual(request["response_format"]["json_schema"].get("grammar_mode", "ordinary"),
                             "canonical_fixed" if "canonical" in arm or "jump" in arm else "ordinary")

    def test_missing_sparse_evidence_is_not_success(self):
        reply = self.response("constrained_sparse_plain")
        bad = copy.deepcopy(reply.response)
        del bad["timings"]["grammar"]["sparse_head"]
        del bad["timings"]["grammar"]["mean_legal_set_size"]
        result = bench.evaluate(bench.Reply(bad, 1, None), self.case, "constrained_sparse_plain", Validator())
        self.assertEqual(set(result["contract_errors"]),
                         {"missing_or_mismatched_sparse_head_telemetry", "missing_mean_legal_set_size"})
        self.assertIsNone(result["metrics"]["mean_legal_set_size"])
        dense = bench.evaluate(bench.Reply(bad, 1, None), self.case, "constrained_plain", Validator())
        self.assertEqual(dense["contract_errors"], [])
        strict_dense = bench.evaluate(bench.Reply(bad, 1, None), self.case, "constrained_plain", Validator(), True)
        self.assertIn("missing_or_mismatched_sparse_head_telemetry", strict_dense["contract_errors"])

    def test_component_timing_is_observed_not_synthesized(self):
        reply = self.response("constrained_bulk_spec")
        fields = {"draft_body_s": .5, "draft_selection_s": .3, "draft_token_wait_s": 0, "draft_token_waits": 0}
        reply.response["timings"]["grammar"].update(fields)
        result = bench.evaluate(reply, self.case, "constrained_bulk_spec", Validator())
        for key, value in fields.items():
            self.assertEqual(result["metrics"][key], value)
        missing = bench.evaluate(self.response("constrained_plain"), self.case, "constrained_plain", Validator())
        for key in fields:
            self.assertIsNone(missing["metrics"][key])

    def test_bulk_route_requires_its_own_evidence(self):
        reply = self.response("constrained_bulk_spec")
        reply.response["timings"]["grammar"]["bulk_select"] = False
        result = bench.evaluate(reply, self.case, "constrained_bulk_spec", Validator())
        self.assertIn("missing_or_mismatched_bulk_select_telemetry", result["contract_errors"])
        reply = self.response("constrained_spec")
        reply.response["timings"]["grammar"]["bulk_select"] = True
        result = bench.evaluate(reply, self.case, "constrained_spec", Validator(), False, True)
        self.assertIn("missing_or_mismatched_bulk_select_telemetry", result["contract_errors"])

    def test_gpu_route_requires_zero_token_waits_and_geometry(self):
        reply = self.response("constrained_gpu_spec")
        reply.response["timings"]["grammar"]["draft_token_waits"] = 7
        del reply.response["timings"]["grammar"]["gpu_grammar_bytes"]
        result = bench.evaluate(reply, self.case, "constrained_gpu_spec", Validator(), True)
        self.assertEqual(set(result["contract_errors"]), {"gpu_grammar_per_token_host_wait", "missing_gpu_grammar_geometry"})

    def test_pairing_preserves_draft_limit_and_grammar_law(self):
        rows = [self.row(arm, limit) for limit in (7, 24) for arm in bench.ARMS]
        for row in rows:
            if row["arm"] == "constrained_sparse_plain" and row["draft_limit"] == 24:
                row["text"], row["parsed"] = '"blue"', "blue"
        summary = bench.summarize(rows, [self.case])
        pair = summary["matched_law_comparisons"]["constrained_plain_vs_sparse"]
        self.assertEqual(pair["comparable_pairs"], 2)
        self.assertEqual(pair["text_identity_rate"], .5)
        self.assertEqual(summary["by_draft_limit"]["7"]["matched_law_comparisons"]["constrained_plain_vs_sparse"]["text_identity_rate"], 1)
        self.assertEqual(summary["by_draft_limit"]["24"]["matched_law_comparisons"]["constrained_plain_vs_sparse"]["text_identity_rate"], 0)
        self.assertIn("constrained_canonical_sparse_plain_vs_jump", summary["matched_law_comparisons"])
        self.assertIn("constrained_spec_vs_bulk", summary["matched_law_comparisons"])
        self.assertIn("constrained_spec_vs_gpu", summary["matched_law_comparisons"])
        self.assertIn("constrained_plain_vs_gpu", summary["matched_law_comparisons"])
        self.assertIn("constrained_sparse_spec_vs_gpu", summary["matched_law_comparisons"])
        self.assertIn("constrained_bulk_spec_vs_gpu", summary["matched_law_comparisons"])
        self.assertIn("constrained_sparse_spec_vs_bulk", summary["matched_law_comparisons"])
        self.assertIn("constrained_canonical_bulk_spec_vs_jump", summary["matched_law_comparisons"])
        self.assertNotIn("constrained_plain_vs_jump", summary["matched_law_comparisons"])
        self.assertEqual(summary["constrained_plain_vs_spec"]["text_identity_rate"], 1)

    def test_pooled_legal_size_uses_emitted_token_weights(self):
        rows = [self.row("constrained_plain", legal_sum=10, observed=2),
                self.row("constrained_plain", legal_sum=90, observed=10)]
        rows[1]["repetition"] = 1
        warmup = self.row("constrained_plain", legal_sum=10000, observed=1)
        warmup["phase"] = "warmup"
        summary = bench.summarize([*rows, warmup], [self.case])["arms"]["constrained_plain"]
        self.assertAlmostEqual(summary["pooled_metrics"]["mean_legal_set_size"], 100 / 12)
        self.assertEqual(summary["metrics"]["mean_legal_set_size"]["mean"], 7)
        self.assertEqual(summary["requests"], 2)
        rows[0]["timings"]["grammar"].pop("legal_tokens_sum")
        summary = bench.summarize(rows, [self.case])["arms"]["constrained_plain"]
        self.assertIsNone(summary["pooled_metrics"]["mean_legal_set_size"])


if __name__ == "__main__":
    unittest.main()
