#!/usr/bin/env python3
"""Paired schema decoding benchmark; run against a strict native Bonsai server.

GPU box: python3 -m pip install jsonschema
  python3 tools/grammar_bench.py --server http://127.0.0.1:8000 \
    --out results/grammar-smoke --smoke --repetitions 2 --draft-limit 7
  python3 tools/grammar_bench.py --server http://127.0.0.1:8000 \
    --out results/grammar-full --repetitions 4 --draft-limit 7 --stream

Use H5_DRAFT_MARGIN=0, no relaxed policy/tree/lookup/penalties/thinking and a
loaded drafter. Requests run serially. Odd repetitions reverse the selected arms;
all arms see identical messages and caps. Warmups are recorded but not scored.
TTFT is native server timing, plus actual first content receipt for SSE. Client
throughput uses reported completion tokens, never a character/token estimate.
Absent native telemetry stays null with per-metric observation counts. A stop
is recorded separately from native grammar completion (not inferred from it).
Tool fixtures generate representations; they do not execute tool calls.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any
import urllib.error
import urllib.request

ARMS = ("unconstrained_plain", "unconstrained_spec", "constrained_plain", "constrained_spec")
METRICS = ("decode_tok_per_s", "client_tok_per_s", "client_decode_tok_per_s", "ttft_s", "client_ttft_s",
           "wall_s", "accepted_draft_length", "draft_acceptance_rate", "grammar_compile_s",
           "grammar_target_mask_s", "grammar_draft_mask_s", "grammar_mask_s", "forced_token_fraction",
           "mean_legal_set_size", "draft_body_s", "draft_selection_s", "draft_token_wait_s", "draft_token_waits",
           "jumped_tokens", "jump_spans", "jump_prefill_s", "gpu_grammar_states", "gpu_grammar_bytes")


@dataclass(frozen=True)
class Error:
    kind: str
    message: str
    http_status: int | None = None


@dataclass(frozen=True)
class Reply:
    response: dict[str, Any]
    wall_s: float
    first_content_s: float | None


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate object key: " + key)
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError("non-JSON numeric constant: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def read_json(path):
    try:
        return strict_json(path.read_text())
    except (OSError, ValueError) as exc:
        return Error("invalid_input", str(exc))


def http(url, payload, timeout, raw_path):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, raw_path.open("wb") as raw:
            if payload is None or not payload["stream"]:
                data = response.read()
                raw.write(data)
                parsed = strict_json(data.decode())
            else:
                texts, finish, usage, timings = [], None, None, None
                done = False
                for line in response:
                    raw.write(line)
                    raw.flush()
                    if not line.startswith(b"data: "):
                        continue
                    body = line[6:].strip()
                    if body == b"[DONE]":
                        done = True
                        break
                    event = strict_json(body.decode())
                    if "error" in event:
                        return Error("server_error", json.dumps(event["error"]))
                    for choice in event.get("choices", []):
                        piece = choice.get("delta", {}).get("content")
                        if piece:
                            if first is None:
                                first = time.perf_counter() - start
                            texts.append(piece)
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
                    if "usage" in event:
                        usage = event["usage"]
                    if "timings" in event:
                        timings = event["timings"]
                if not done or finish is None or usage is None:
                    return Error("incomplete_stream", "SSE requires DONE, finish_reason, and terminal usage")
                parsed = {"choices": [{"message": {"content": "".join(texts)}, "finish_reason": finish}],
                          "usage": usage, "timings": timings}
        if not isinstance(parsed, dict):
            return Error("invalid_response", "Response must be an object")
        return Reply(parsed, time.perf_counter() - start, first)
    except urllib.error.HTTPError as exc:
        data = exc.read()
        raw_path.write_bytes(data)
        return Error("http_error", data.decode(errors="replace"), exc.code)
    except (OSError, urllib.error.URLError) as exc:
        return Error("transport_error", str(exc))
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return Error("invalid_response", str(exc))


def request_for(case, arm, args, draft_limit=None):
    schema = json.dumps(case["schema"], ensure_ascii=False, separators=(",", ":"))
    body = {"model": args.model, "messages": [
        {"role": "system", "content": "Return only one JSON value matching this schema. No markdown or explanation. " + schema},
        {"role": "user", "content": case["prompt"]}],
        "max_tokens": case["max_tokens"] if args.max_tokens is None else args.max_tokens,
        "temperature": 0, "seed": args.seed, "thinking": False, "stream": args.stream,
        "draft_verify_limit": (args.draft_limit if draft_limit is None else draft_limit) if arm.endswith("_spec") else 0}
    if args.stream:
        body["stream_options"] = {"include_usage": True}
    if arm.startswith("constrained_"):
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": case["id"], "strict": True, "schema": case["schema"]}}
        if "canonical" in arm or "jump" in arm:
            body["response_format"]["json_schema"]["grammar_mode"] = "canonical_fixed"
        if "jump" in arm:
            body["grammar_jump_forward"] = True
        body["grammar_sparse_head"] = "sparse" in arm or "gpu" in arm
        body["grammar_bulk_select"] = "bulk" in arm
        if "gpu" in arm:
            body["grammar_gpu"] = True
    return body


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def ratio(a, b):
    return a / b if number(a) is not None and number(b) is not None and b > 0 else None


def evaluate(reply, case, arm, validator, require_sparse_telemetry=False, require_bulk_telemetry=False):
    response = reply.response
    try:
        choice = response["choices"][0]
        text = choice["message"]["content"]
        finish = choice["finish_reason"]
        tokens = response["usage"]["completion_tokens"]
        if not isinstance(text, str) or not isinstance(finish, str) or type(tokens) is not int or tokens < 0:
            return Error("invalid_response", "content/finish_reason/completion_tokens have invalid types")
        if "error" in response:
            return Error("server_error", json.dumps(response["error"]))
        native = response.get("timings")
        if native is None:
            native = {}
        if not isinstance(native, dict):
            return Error("invalid_response", "timings must be an object or unset")
        grammar = native.get("grammar", {})
        if not isinstance(grammar, dict):
            return Error("invalid_response", "timings.grammar must be an object")
    except (KeyError, IndexError, TypeError) as exc:
        return Error("invalid_response", str(exc))
    try:
        value = strict_json(text)
        errors = [error.message for error in validator.iter_errors(value)]
        valid = not errors
        parse_error = None
    except ValueError as exc:
        value, errors, valid, parse_error = None, [], False, str(exc)
    accepted, rounds, proposed = (number(native.get(key)) for key in ("accepted_drafts", "verify_rounds", "proposed_drafts"))
    target, draft = (number(grammar.get(key)) for key in ("target_mask_s", "draft_mask_s"))
    contract_errors = []
    if arm.startswith("constrained_") and not grammar:
        contract_errors.append("missing_grammar_telemetry")
    if arm.startswith("constrained_") and grammar:
        expected_mode = "canonical_fixed" if "canonical" in arm or "jump" in arm else "ordinary"
        if grammar.get("mode") != expected_mode:
            contract_errors.append("missing_or_mismatched_grammar_mode")
        if grammar.get("jump_forward") is not ("jump" in arm):
            contract_errors.append("missing_or_mismatched_jump_forward_telemetry")
        if expected_mode == "canonical_fixed" and not isinstance(grammar.get("canonical_law"), str):
            contract_errors.append("missing_canonical_law")
    if arm.startswith("constrained_") and grammar and (require_sparse_telemetry or "sparse" in arm):
        expected_sparse = "sparse" in arm or "gpu" in arm
        if grammar.get("sparse_head") is not expected_sparse:
            contract_errors.append("missing_or_mismatched_sparse_head_telemetry")
        if number(grammar.get("mean_legal_set_size")) is None:
            contract_errors.append("missing_mean_legal_set_size")
    if arm.startswith("constrained_") and grammar and (require_bulk_telemetry or "bulk" in arm):
        if grammar.get("bulk_select") is not ("bulk" in arm):
            contract_errors.append("missing_or_mismatched_bulk_select_telemetry")
    if "gpu" in arm and grammar:
        if grammar.get("gpu") is not True:
            contract_errors.append("missing_gpu_grammar_telemetry")
        if number(grammar.get("draft_token_waits")) != 0:
            contract_errors.append("gpu_grammar_per_token_host_wait")
        if number(grammar.get("gpu_grammar_states")) is None or number(grammar.get("gpu_grammar_bytes")) is None:
            contract_errors.append("missing_gpu_grammar_geometry")
    if arm.startswith("constrained_") and grammar and type(grammar.get("complete")) is not bool:
        contract_errors.append("missing_or_invalid_grammar_completion")
    stop_reason = grammar.get("stop_reason")
    if arm.startswith("constrained_") and grammar and stop_reason not in ("grammar_complete", "eos", "max_tokens", "context_limit", "error", "cancelled"):
        contract_errors.append("missing_or_invalid_grammar_stop_reason")
    # A jump-enabled request can finish through forced spans and branching
    # anchors without ever needing a speculative verification round.
    jumped = number(grammar.get("jumped_tokens"))
    jump_only = grammar.get("jump_forward") is True and jumped is not None and jumped > 0
    if arm.endswith("_spec") and tokens > 1 and rounds == 0 and not jump_only:
        contract_errors.append("speculation_not_observed")
    if arm.endswith("_plain") and rounds is not None and rounds != 0:
        contract_errors.append("plain_arm_speculated")
    if number(native.get("relax_offered")) not in (None, 0):
        contract_errors.append("relaxed_verification_observed")
    metrics = {"decode_tok_per_s": number(native.get("decode_tok_per_s")),
               "client_tok_per_s": ratio(tokens, reply.wall_s),
               "client_decode_tok_per_s": ratio(tokens - 1, reply.wall_s - reply.first_content_s) if reply.first_content_s is not None and tokens > 1 else None,
               "ttft_s": number(native.get("ttft_s")), "client_ttft_s": reply.first_content_s,
               "wall_s": reply.wall_s, "accepted_draft_length": ratio(accepted, rounds),
               "draft_acceptance_rate": ratio(accepted, proposed),
               "grammar_compile_s": number(grammar.get("compile_s")),
               "grammar_target_mask_s": target, "grammar_draft_mask_s": draft,
               "grammar_mask_s": target + draft if target is not None and draft is not None else None,
               "forced_token_fraction": number(grammar.get("forced_token_fraction")),
               "mean_legal_set_size": number(grammar.get("mean_legal_set_size")),
               **{key: number(grammar.get(key)) for key in ("draft_body_s", "draft_selection_s", "draft_token_wait_s", "draft_token_waits",
                                                           "jumped_tokens", "jump_spans", "jump_prefill_s", "gpu_grammar_states", "gpu_grammar_bytes")}}
    return {"status": "ok", "text": text, "parsed": value, "schema_valid": valid,
            "semantic_correct": valid and value == case["expected"] if "expected" in case else None,
            "parse_error": parse_error, "schema_errors": errors,
            "finish_reason": finish, "stopped": finish == "stop", "capped": finish == "length",
            "grammar_complete": grammar.get("complete"), "stop_reason": stop_reason,
            "eos_observed": stop_reason == "eos" if stop_reason is not None else None,
            "completion_tokens": tokens,
            "timings": native, "metrics": metrics, "contract_errors": contract_errors}


def stats(values):
    values = sorted(value for value in values if number(value) is not None)
    if not values:
        return {"observations": 0, "mean": None, "median": None, "p95": None}
    return {"observations": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": values[math.ceil(len(values) * .95) - 1]}


def summarize(records, cases, include_limit_groups=True):
    scored = [row for row in records if row["phase"] == "measured"]
    groups = {}
    def counter_total(rows, name, grammar=False):
        values = [number((row["timings"].get("grammar", {}) if grammar else row["timings"]).get(name)) for row in rows]
        observed = [value for value in values if value is not None]
        return {"observations": len(observed), "sum": sum(observed) if observed else None}

    for arm in ARMS:
        rows = [row for row in scored if row["arm"] == arm]
        success = [row for row in rows if row["status"] == "ok"]
        semantic = [row for row in success if row["semantic_correct"] is not None]
        counters = {key: counter_total(success, key) for key in ("accepted_drafts", "proposed_drafts", "verify_rounds")}
        counters.update({key: counter_total(success, key, grammar=True) for key in ("target_mask_rows", "draft_mask_rows", "forced_tokens", "observed_tokens", "legal_tokens_sum", "draft_token_waits", "jumped_tokens", "jump_spans")})
        pooled = {}
        for key, numerator, denominator in (("accepted_draft_length", "accepted_drafts", "verify_rounds"),
                                            ("draft_acceptance_rate", "accepted_drafts", "proposed_drafts"),
                                            ("forced_token_fraction", "forced_tokens", "observed_tokens"),
                                            ("mean_legal_set_size", "legal_tokens_sum", "observed_tokens")):
            complete = counters[numerator]["observations"] == counters[denominator]["observations"] == len(success)
            pooled[key] = ratio(counters[numerator]["sum"], counters[denominator]["sum"]) if complete else None
        groups[arm] = {"requests": len(rows), "successful_requests": len(success),
            "error_count": len(rows) - len(success),
            "schema_valid_count": sum(row["schema_valid"] for row in success),
            "schema_valid_output_rate": ratio(sum(row["schema_valid"] for row in success), len(rows)),
            "semantic_correct_count": sum(row["semantic_correct"] for row in semantic),
            "semantic_observations": len(semantic),
            "semantic_correct_rate": ratio(sum(row["semantic_correct"] for row in semantic), len(rows)) if len(semantic) == len(success) else None,
            "finish_reasons": dict(Counter(row["finish_reason"] for row in success)),
            "caps": sum(row["capped"] for row in success), "stops": sum(row["stopped"] for row in success),
            "grammar_complete_count": sum(row["grammar_complete"] is True for row in success),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in success if row["stop_reason"] is not None)),
            "eos_observations": sum(row["eos_observed"] is not None for row in success),
            "eos_count": sum(row["eos_observed"] is True for row in success),
            "contract_errors": dict(Counter(err for row in success for err in row["contract_errors"])),
            "counter_totals": counters, "pooled_metrics": pooled,
            "metrics": {key: stats(row["metrics"][key] for row in success) for key in METRICS}}
    indexed = {(row["case_id"], row["repetition"], row.get("draft_limit"), row["arm"]): row for row in scored}
    identities = sorted({(row["case_id"], row["repetition"], row.get("draft_limit")) for row in scored})
    def compare(left_arm, right_arm):
        pairs = []
        for case_id, repetition, draft_limit in identities:
            left = indexed.get((case_id, repetition, draft_limit, left_arm))
            right = indexed.get((case_id, repetition, draft_limit, right_arm))
            comparable = left is not None and right is not None and left["status"] == right["status"] == "ok"
            pair = {"case_id": case_id, "repetition": repetition, "draft_limit": draft_limit, "comparable": comparable,
                    "text_identical": left["text"] == right["text"] if comparable else None,
                    "parsed_identical": left["parsed"] == right["parsed"] if comparable and left["schema_valid"] and right["schema_valid"] else None,
                    "finish_identical": left["finish_reason"] == right["finish_reason"] if comparable else None,
                    "contract_clean": not (left["contract_errors"] or right["contract_errors"]) if comparable else None,
                    "grammar_law_identical": tuple(left["timings"].get("grammar", {}).get(key) for key in ("mode", "canonical_law")) ==
                        tuple(right["timings"].get("grammar", {}).get(key) for key in ("mode", "canonical_law")) if comparable else None,
                    "decode_speedup": ratio(right["metrics"]["decode_tok_per_s"], left["metrics"]["decode_tok_per_s"]) if comparable else None,
                    "client_speedup": ratio(right["metrics"]["client_tok_per_s"], left["metrics"]["client_tok_per_s"]) if comparable else None}
            if comparable and not pair["text_identical"]:
                pair["first_different_character"] = next((i for i, (a, b) in enumerate(zip(left["text"], right["text"])) if a != b), min(len(left["text"]), len(right["text"])))
            pairs.append(pair)
        count = sum(pair["comparable"] for pair in pairs)
        return {"left_arm": left_arm, "right_arm": right_arm, "pairs": pairs, "comparable_pairs": count,
                "text_identity_rate": ratio(sum(pair["text_identical"] is True for pair in pairs), count),
                "parsed_identity_rate": ratio(sum(pair["parsed_identical"] is True for pair in pairs), count),
                "decode_speedup": stats(pair["decode_speedup"] for pair in pairs),
                "client_speedup": stats(pair["client_speedup"] for pair in pairs)}
    comparisons = {}
    for arm in ARMS:
        if arm.endswith("_plain") and arm[:-6] + "_spec" in ARMS:
            comparisons[arm[:-6] + "_plain_vs_spec"] = compare(arm, arm[:-6] + "_spec")
        if not any(execution in arm for execution in ("sparse", "bulk", "gpu")) and arm.startswith("constrained_"):
            for execution in ("sparse", "bulk", "gpu"):
                alternate = arm.rsplit("_", 1)[0] + "_" + execution + "_" + arm.rsplit("_", 1)[1]
                if alternate in ARMS:
                    comparisons[arm + "_vs_" + execution] = compare(arm, alternate)
        if "sparse" in arm:
            bulk = arm.replace("_sparse_", "_bulk_")
            if bulk in ARMS:
                comparisons[arm + "_vs_bulk"] = compare(arm, bulk)
        if arm.startswith(("constrained_sparse_", "constrained_bulk_")):
            gpu = "constrained_gpu_" + arm.rsplit("_", 1)[1]
            if gpu in ARMS:
                comparisons[arm + "_vs_gpu"] = compare(arm, gpu)
    for suffix in ("plain", "spec", "sparse_plain", "sparse_spec", "bulk_spec"):
        left, right = "constrained_canonical_" + suffix, "constrained_jump_" + suffix
        if left in ARMS and right in ARMS:
            comparisons[left + "_vs_jump"] = compare(left, right)
    result = {"version": 1, "arms": groups, "cases": [case["id"] for case in cases],
              "constrained_plain_vs_spec": comparisons.get("constrained_plain_vs_spec"),
              "matched_law_comparisons": comparisons}
    limits = sorted({row["draft_limit"] for row in scored if "draft_limit" in row})
    if include_limit_groups and len(limits) > 1:
        result["by_draft_limit"] = {str(limit): summarize([row for row in records if row["draft_limit"] == limit], cases, False) for limit in limits}
    return result


def main():
    global ARMS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True)
    parser.add_argument("--out", required=True, type=Path, help="New output directory; existing paths are rejected")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("grammar_cases.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fixture-set", choices=("all", "base", "expanded"), default="all", help="Original 24 fixtures, 10 new union/ref fixtures, or all; unmarked fixtures belong to base")
    parser.add_argument("--case", action="append", help="Select exact fixture ID (repeatable)")
    parser.add_argument("--arm", action="append", help="Select generated execution arm (repeatable)")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--draft-limit", required=True, type=int, action="append", help="Speculative proposal cap; repeat to sweep e.g. 7,12,16,24")
    parser.add_argument("--warmups", type=int, default=1, help="Number of complete selected-arm warmup passes per draft limit on the first fixture")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--model", default="bonsai")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write exact request plan without HTTP or jsonschema dependency")
    parser.add_argument("--canonical", action="store_true", help="Add minified canonical fixed-span plain/spec arms")
    parser.add_argument("--jump-forward", action="store_true", help="Add canonical jump plain/spec arms; requires --canonical")
    parser.add_argument("--sparse-head", action="store_true", help="Add sparse target-head plain/spec controls under each selected constrained grammar law")
    parser.add_argument("--bulk-select", action="store_true", help="Add full-head bulk-transfer/CPU-selector spec controls; each request uses either sparse or bulk, never both")
    parser.add_argument("--gpu-grammar", action="store_true", help="Add GPU-resident ordinary-law sparse plain/spec arms with zero per-token host grammar synchronization")
    args = parser.parse_args()
    if args.jump_forward and not args.canonical:
        parser.error('--jump-forward requires --canonical')
    ARMS = ("unconstrained_plain", "unconstrained_spec", "constrained_plain", "constrained_spec")
    if args.canonical:
        ARMS += ("constrained_canonical_plain", "constrained_canonical_spec")
    if args.jump_forward:
        ARMS += ("constrained_jump_plain", "constrained_jump_spec")
    dense_arms = ARMS
    if args.sparse_head:
        ARMS += tuple(arm.rsplit("_", 1)[0] + "_sparse_" + arm.rsplit("_", 1)[1] for arm in dense_arms if arm.startswith("constrained_"))
    if args.bulk_select:
        ARMS += tuple(arm[:-5] + "_bulk_spec" for arm in dense_arms if arm.startswith("constrained_") and arm.endswith("_spec"))
    if args.gpu_grammar:
        ARMS += ("constrained_gpu_plain", "constrained_gpu_spec")
    if args.arm:
        if set(args.arm) - set(ARMS):
            parser.error("Unknown or unenabled arms: " + ",".join(sorted(set(args.arm) - set(ARMS))))
        ARMS = tuple(arm for arm in ARMS if arm in args.arm)
    args.draft_limits = list(dict.fromkeys(args.draft_limit))
    args.draft_limit = args.draft_limits[0]
    if args.repetitions < 1 or any(limit < 1 for limit in args.draft_limits) or args.warmups < 0 or args.timeout <= 0 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("repetitions/draft-limit/timeout/max-tokens must be positive; warmups nonnegative")
    fixtures = read_json(args.cases)
    if isinstance(fixtures, Error):
        parser.error(fixtures.message)
    if not isinstance(fixtures, dict) or fixtures.get("version") != 1 or not isinstance(fixtures.get("cases"), list):
        parser.error("Expected version 1 fixture document with cases array")
    cases = fixtures["cases"]
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not all(key in case for key in ("id", "category", "schema", "prompt", "max_tokens")):
            parser.error("Fixture missing required fields")
        if case["id"] in ids or not isinstance(case["id"], str) or not case["id"].replace("_", "").isalnum():
            parser.error("Fixture IDs must be unique alphanumeric/underscore names")
        ids.add(case["id"])
        if not isinstance(case["schema"], dict) or not isinstance(case["prompt"], str) or type(case["max_tokens"]) is not int or case["max_tokens"] < 1:
            parser.error("Invalid fixture schema/prompt/cap")
        if case.get("suite", "base") not in ("base", "expanded"):
            parser.error("Fixture suite must be base or expanded; unmarked fixtures are base")
    if args.case and set(args.case) - ids:
        parser.error("Unknown case IDs: " + ",".join(sorted(set(args.case) - ids)))
    cases = [case for case in cases if (args.fixture_set == "all" or case.get("suite", "base") == args.fixture_set)
             and (not args.smoke or case.get("smoke") is True) and (not args.case or case["id"] in args.case)]
    if not cases:
        parser.error("No fixtures selected")
    validators = {}
    if not args.dry_run:
        try:
            import jsonschema
        except ImportError:
            parser.error("Install jsonschema on the GPU box: python3 -m pip install jsonschema")
        for case in cases:
            try:
                validator_type = jsonschema.validators.validator_for(case["schema"])
                validator_type.check_schema(case["schema"])
                validators[case["id"]] = validator_type(case["schema"])
                if "expected" in case:
                    validators[case["id"]].validate(case["expected"])
            except jsonschema.exceptions.SchemaError as exc:
                parser.error("Invalid schema " + case["id"] + ": " + exc.message)
            except jsonschema.exceptions.ValidationError as exc:
                parser.error("Invalid expected value " + case["id"] + ": " + exc.message)
    if args.out.exists():
        parser.error("Output already exists; choose a new --out to preserve raw evidence")
    args.out.mkdir(parents=True)
    raw = args.out / "raw"
    raw.mkdir()
    metadata = {"version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "fixtures_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                "cases": cases, "arms": ARMS, "draft_limits": args.draft_limits, "server_geometry": None,
                "draft_limit_scope": "request proposal cap within loaded geometry; true draft-depth costs require a separate engine run per geometry",
                "order": "listed arms on even repetition, reversed on odd repetition; draft limits alternate order by case/repetition",
                "metric_definitions": {"client_tok_per_s": "reported completion_tokens / full client wall time",
                    "client_decode_tok_per_s": "(completion_tokens - 1) / (wall_s - first content receipt); SSE only, chunk-level observation",
                    "accepted_draft_length": "accepted_drafts / verify_rounds; excludes bonus token",
                    "mean_legal_set_size": "sum of legal token counts before emitted non-EOS target tokens / observed_tokens; pooled counter ratio, not unweighted request mean",
                    "matched_law_comparisons": "plain/spec and dense/sparse/bulk pairs matched by fixture, repetition and draft limit; ordinary and canonical laws never pooled as an identity claim",
                    "draft_body_s": "native grammar drafter body time, disjoint from draft_selection_s",
                    "draft_selection_s": "native grammar drafter head/selection time; includes draft_token_wait_s",
                    "draft_token_wait_s": "GPU token synchronization wall time; subset of selection, not an extra additive bucket",
                    "draft_token_waits": "native grammar count of GPU token synchronizations; zero in the bulk selector route",
                    "grammar_mask_s": "CPU target_mask_s + draft_mask_s; excludes compile; native grammar.mask_timing_scope records coverage. GPU mask/selection remains inside verify_s/draft_s",
                    "eos_observed": "explicit timings.grammar.stop_reason == eos; null when native stop reason is absent (unconstrained arms)",
                    "schema_valid_output_rate": "valid entire JSON values / attempted measured requests; errors count as not valid",
                    "semantic_correct": "schema-valid parsed value equals independent fixture expected value; array order significant"}}
    dump(args.out / "metadata.json", metadata)
    if not args.dry_run:
        for endpoint in ("health", "metrics", "v1/models"):
            result = http(args.server.rstrip("/") + "/" + endpoint, None, args.timeout, raw / (endpoint.replace("/", "-") + ".json"))
            if isinstance(result, Error):
                dump(args.out / "startup_error.json", vars(result))
                print(json.dumps(vars(result)), file=sys.stderr)
                return 1
            if endpoint == "metrics":
                metrics = result.response
                invalid = []
                if metrics.get("speculation") is not True:
                    invalid.append("server must have speculation enabled and drafter loaded")
                if metrics.get("relax_policy") not in (None, "none") or metrics.get("tree2_enabled") is True or metrics.get("lookup_extra_max", 0) != 0:
                    invalid.append("server must disable relaxed policy/tree/lookup")
                proposals = metrics.get("draft_proposals")
                if type(proposals) is not int or proposals < 1:
                    invalid.append("server must report positive draft_proposals to establish supported geometry")
                elif max(args.draft_limits) > proposals:
                    invalid.append(f"draft-limit {max(args.draft_limits)} exceeds server proposals {proposals}; start engine with larger proposal geometry")
                if invalid:
                    dump(args.out / "startup_error.json", {"kind": "invalid_server_configuration", "errors": invalid})
                    print("; ".join(invalid), file=sys.stderr)
                    return 1
                metadata["server_geometry"] = {key: metrics.get(key) for key in ("draft_proposals", "draft_query_rows", "verifier_rows_per_full_round")}
        dump(args.out / "metadata.json", metadata)
    records, plan = [], []
    work = [("warmup", repetition, cases[0], limit) for repetition in range(args.warmups) for limit in args.draft_limits]
    work += [("measured", repetition, case, limit) for index, case in enumerate(cases) for repetition in range(args.repetitions)
             for limit in (args.draft_limits if (index + repetition) % 2 == 0 else list(reversed(args.draft_limits)))]
    with (args.out / "records.jsonl").open("w") as journal:
        for phase, repetition, case, draft_limit in work:
            arms = ARMS if repetition % 2 == 0 else tuple(reversed(ARMS))
            for order, arm in enumerate(arms):
                stem = f"{phase}-{case['id']}-r{repetition:03d}-d{draft_limit}-{order}-{arm}"
                body = request_for(case, arm, args, draft_limit)
                request_path, response_path = raw / (stem + ".request.json"), raw / (stem + ".response.body")
                dump(request_path, body)
                identity = {"phase": phase, "case_id": case["id"], "category": case["category"],
                            "repetition": repetition, "draft_limit": draft_limit, "arm": arm, "order": order,
                            "request_path": str(request_path.relative_to(args.out)), "response_path": str(response_path.relative_to(args.out))}
                plan.append(identity)
                if args.dry_run:
                    continue
                result = http(args.server.rstrip("/") + "/v1/chat/completions", body, args.timeout, response_path)
                if isinstance(result, Reply):
                    result = evaluate(result, case, arm, validators[case["id"]], args.sparse_head, args.bulk_select)
                row = {**identity, **({"status": "error", "error": vars(result)} if isinstance(result, Error) else result)}
                records.append(row)
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                dump(args.out / "summary.json", summarize(records, cases))
                print(f"{phase} {case['id']} r{repetition} d{draft_limit} {arm}: " + (row["error"]["kind"] if row["status"] == "error" else f"valid={row['schema_valid']} semantic={row['semantic_correct']} finish={row['finish_reason']} tok/s={row['metrics']['decode_tok_per_s']}"), flush=True)
                if isinstance(result, Error) and result.kind in ("transport_error", "incomplete_stream"):
                    dump(args.out / "interrupted.json", {"kind": result.kind, "message": "Stopped without retry: provider effect may still be in flight"})
                    return 1
    dump(args.out / "plan.json", plan)
    if args.dry_run:
        print(f"Planned {len(plan)} requests across {len(cases)} fixtures in {args.out}")
        return 0
    summary = summarize(records, cases)
    dump(args.out / "summary.json", summary)
    dump(args.out / "completed.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "requests": len(records)})
    print(json.dumps({arm: {key: value for key, value in group.items() if key in ("requests", "schema_valid_output_rate", "semantic_correct_rate", "caps", "contract_errors")} for arm, group in summary["arms"].items()}, indent=2))
    return int(any(row["status"] == "error" or row.get("contract_errors") for row in records))


if __name__ == "__main__":
    try:
        exit_code = main()
    except OSError as exc:
        print(json.dumps({"kind": "filesystem_error", "message": str(exc)}), file=sys.stderr)
        exit_code = 1
    sys.exit(exit_code)
