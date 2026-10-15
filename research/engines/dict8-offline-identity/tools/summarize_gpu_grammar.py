#!/usr/bin/env python3
"""CPU-only report for one run_sparse_panel.py run.

Usage: python3 tools/summarize_gpu_grammar.py RUN_DIR --out NEW_REPORT_DIR
Writes report.json and report.md without overwriting evidence. The input must
include config.json and panel/{metadata.json,records.jsonl,summary.json}.
Null means unobserved, never zero. Warmups are excluded from every comparison.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import grammar_bench as bench

ARMS = (
    "unconstrained_spec", "constrained_plain", "constrained_spec",
    "constrained_sparse_plain", "constrained_sparse_spec",
    "constrained_bulk_spec", "constrained_gpu_plain", "constrained_gpu_spec",
)
COMPARISONS = (
    ("unconstrained_spec", "constrained_spec"),
    ("unconstrained_spec", "constrained_gpu_spec"),
    ("constrained_plain", "constrained_spec"),
    ("constrained_sparse_plain", "constrained_sparse_spec"),
    ("constrained_gpu_plain", "constrained_gpu_spec"),
    ("constrained_plain", "constrained_sparse_plain"),
    ("constrained_spec", "constrained_sparse_spec"),
    ("constrained_spec", "constrained_bulk_spec"),
    ("constrained_sparse_spec", "constrained_bulk_spec"),
    ("constrained_plain", "constrained_gpu_plain"),
    ("constrained_spec", "constrained_gpu_spec"),
    ("constrained_sparse_plain", "constrained_gpu_plain"),
    ("constrained_sparse_spec", "constrained_gpu_spec"),
    ("constrained_bulk_spec", "constrained_gpu_spec"),
)
RATES = ("decode_tok_per_s", "client_tok_per_s", "client_decode_tok_per_s")
POOLED = {
    "accepted_draft_length": ("native", "accepted_drafts", "verify_rounds"),
    "draft_acceptance_rate": ("native", "accepted_drafts", "proposed_drafts"),
    "forced_token_fraction": ("grammar", "forced_tokens", "observed_tokens"),
    "mean_legal_set_size": ("grammar", "legal_tokens_sum", "observed_tokens"),
}


@dataclass(frozen=True)
class Error:
    kind: str
    message: str


@dataclass(frozen=True)
class Document:
    value: Any
    sha256: str


def load(path: Path, lines: bool = False) -> Document | Error:
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
        value = [bench.strict_json(line) for line in text.splitlines() if line.strip()] if lines else bench.strict_json(text)
        return Document(value, hashlib.sha256(data).hexdigest())
    except OSError as exc:
        return Error("filesystem_error", f"{path}: {exc}")
    except (ValueError, UnicodeError) as exc:
        return Error("invalid_json", f"{path}: {exc}")


def numeric(value):
    return bench.number(value)


def aggregate(values):
    values = [value for value in values if numeric(value) is not None]
    return {"observations": len(values), "sum": sum(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def identity(row):
    return row["case_id"], row["repetition"], row["draft_limit"]


def native(row):
    return row.get("timings", {})


def grammar(row):
    return native(row).get("grammar", {})


def failure_reasons(row):
    if row["status"] == "error":
        return ["request_error"]
    failures = []
    for key in ("schema_valid", "semantic_correct", "stopped"):
        if row.get(key) is not True:
            failures.append(key + ("_unknown" if row.get(key) is None else "_failed"))
    if row.get("capped") is not False:
        failures.append("capped" if row.get("capped") is True else "cap_unknown")
    if row["arm"].startswith("constrained_") and row.get("grammar_complete") is not True:
        failures.append("grammar_incomplete_or_unknown")
    failures.extend(row["contract_errors"])
    if row["arm"].startswith("constrained_") and grammar(row).get("mode") != "ordinary":
        failures.append("ordinary_law_not_observed")
    if "gpu" in row["arm"]:
        if grammar(row).get("gpu") is not True:
            failures.append("gpu_not_observed")
        waits = numeric(grammar(row).get("draft_token_waits"))
        if waits is None:
            failures.append("gpu_token_wait_count_unknown")
        elif waits != 0:
            failures.append("gpu_token_wait_count_nonzero")
        for key in ("gpu_grammar_bytes", "gpu_grammar_states"):
            if numeric(grammar(row).get(key)) is None:
                failures.append(key + "_unknown")
    return sorted(set(failures))


def validate(config, metadata, summary, rows):
    for name, value in (("config", config), ("metadata", metadata), ("summary", summary)):
        if not isinstance(value, dict):
            return Error("invalid_input", name + " must be an object")
    if type(config.get("depth")) is not int or config["depth"] < 1:
        return Error("invalid_input", "config.depth must report actual positive loaded depth")
    for key in ("source", "binary_sha256", "binary"):
        if not isinstance(config.get(key), str) or not config[key]:
            return Error("invalid_input", "config." + key + " required for provenance")
    if not isinstance(config.get("environment"), dict):
        return Error("invalid_input", "config.environment must be an object")
    if not isinstance(metadata.get("cases"), list) or not metadata["cases"]:
        return Error("invalid_input", "metadata.cases required to identify full/base/expanded fixture splits")
    cases = metadata["cases"]
    if any(not isinstance(case, dict) or not isinstance(case.get("id"), str)
           or case.get("suite", "base") not in ("base", "expanded") for case in cases):
        return Error("invalid_input", "invalid fixture ID/suite in metadata")
    ids = {case["id"] for case in cases}
    if len(ids) != len(cases):
        return Error("invalid_input", "duplicate metadata fixture ID")
    if not isinstance(metadata.get("arms"), list) or not all(isinstance(arm, str) for arm in metadata["arms"]):
        return Error("invalid_input", "metadata.arms must be an array of names")
    if len(set(metadata["arms"])) != len(metadata["arms"]) or not set(metadata["arms"]) & set(ARMS):
        return Error("invalid_input", "metadata.arms must be unique and include a requested report arm")
    if not isinstance(summary.get("arms"), dict) or any(not isinstance(group, dict) for group in summary["arms"].values()):
        return Error("invalid_input", "summary.arms must contain arm summary objects")
    limits = metadata.get("draft_limits")
    if not isinstance(limits, list) or not limits or any(type(x) is not int or x < 1 for x in limits):
        return Error("invalid_input", "metadata.draft_limits must report positive request caps")
    args = metadata.get("args")
    if not isinstance(args, dict) or type(args.get("repetitions")) is not int or args["repetitions"] < 1:
        return Error("invalid_input", "metadata.args.repetitions required to detect missing requests")
    if not isinstance(rows, list):
        return Error("invalid_input", "records must be JSON objects, one per line")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("phase") not in ("warmup", "measured") or row.get("status") not in ("ok", "error"):
            return Error("invalid_record", f"record {index}: unknown phase/status")
        if (not isinstance(row.get("case_id"), str) or row["case_id"] not in ids
                or not isinstance(row.get("arm"), str) or row["arm"] not in metadata["arms"]
                or type(row.get("repetition")) is not int or row["repetition"] < 0
                or type(row.get("draft_limit")) is not int or row["draft_limit"] not in limits):
            return Error("invalid_record", f"record {index}: invalid case/arm/repetition/depth identity")
        if row["phase"] == "measured":
            if row["repetition"] >= args["repetitions"]:
                return Error("invalid_record", f"record {index}: repetition outside metadata plan")
            key = (*identity(row), row["arm"])
            if key in seen:
                return Error("duplicate_record", f"record {index}: {key}")
            seen.add(key)
        if row["status"] == "error":
            if not isinstance(row.get("error"), dict):
                return Error("invalid_record", f"record {index}: error detail missing")
            continue
        if (not isinstance(row.get("text"), str) or "parsed" not in row
                or not isinstance(row.get("metrics"), dict) or not isinstance(row.get("timings"), dict)
                or not isinstance(grammar(row), dict)
                or not isinstance(row.get("contract_errors"), list)
                or any(not isinstance(e, str) for e in row["contract_errors"])):
            return Error("invalid_record", f"record {index}: invalid successful record")
        for key in ("schema_valid", "semantic_correct", "stopped", "capped", "grammar_complete"):
            if row.get(key) is not None and type(row[key]) is not bool:
                return Error("invalid_record", f"record {index}: {key} must be boolean or null")
    return None


def bucket_totals(rows):
    cdfs = [grammar(row).get("legal_set_cdf") for row in rows]
    observed = [cdf for cdf in cdfs if isinstance(cdf, list)]
    if not observed:
        return {"observations": 0, "cumulative": None}
    cuts = sorted({item["max_legal_tokens"] for cdf in observed for item in cdf
                   if isinstance(item, dict) and numeric(item.get("max_legal_tokens")) is not None})
    totals = []
    for cut in cuts:
        values = [numeric(item.get("emitted_tokens")) for cdf in observed for item in cdf
                  if isinstance(item, dict) and item.get("max_legal_tokens") == cut]
        totals.append({"max_legal_tokens": cut, **aggregate(values)})
    return {"observations": len(observed), "cumulative": totals,
            "scope": "cumulative emitted target-token legal-set counts, not disjoint GPU timing buckets"}


def arm_report(rows, expected):
    good = [row for row in rows if row["status"] == "ok"]
    failures = []
    for row in rows:
        reasons = failure_reasons(row)
        if reasons:
            failures.append({"case_id": row["case_id"], "repetition": row["repetition"],
                             "draft_limit": row["draft_limit"], "reasons": reasons,
                             "error": row.get("error"), "parse_error": row.get("parse_error"),
                             "schema_errors": row.get("schema_errors")})
    metrics = {key: aggregate(row["metrics"].get(key) for row in good)
               for key in sorted(set(bench.METRICS) | {key for row in good for key in row["metrics"]})}
    telemetry = {}
    for scope, getter in (("native", native), ("grammar", grammar)):
        keys = sorted({key for row in good for key, value in getter(row).items() if numeric(value) is not None})
        telemetry[scope] = {key: aggregate(getter(row).get(key) for row in good) for key in keys}
    pooled = {}
    for name, (scope, numerator, denominator) in POOLED.items():
        getter = native if scope == "native" else grammar
        observed = [row for row in good if numeric(getter(row).get(numerator)) is not None
                    and numeric(getter(row).get(denominator)) is not None]
        num = sum(getter(row)[numerator] for row in observed) if observed else None
        den = sum(getter(row)[denominator] for row in observed) if observed else None
        pooled[name] = {"observations": len(observed), "numerator": num, "denominator": den,
                        "value": bench.ratio(num, den) if len(observed) == len(good) and good else None}
    waits = [numeric(grammar(row).get("draft_token_waits")) for row in good]
    all_zero = all(value == 0 for value in waits) if good and all(value is not None for value in waits) else None
    semantic = [row for row in good if row.get("semantic_correct") is not None]
    return {"expected_requests": expected, "requests": len(rows), "missing_requests": expected - len(rows),
            "errors": len(rows) - len(good), "schema_valid": sum(row.get("schema_valid") is True for row in good),
            "semantic_correct": sum(row.get("semantic_correct") is True for row in good),
            "semantic_observations": len(semantic), "caps": sum(row.get("capped") is True for row in good),
            "stops": sum(row.get("stopped") is True for row in good),
            "grammar_complete": sum(row.get("grammar_complete") is True for row in good),
            "finish_reasons": dict(Counter(row.get("finish_reason") for row in good)),
            "stop_reasons": dict(Counter(row.get("stop_reason") for row in good)),
            "contract_errors": dict(Counter(e for row in good for e in row["contract_errors"])),
            "quality_pass": bool(expected) and len(rows) == expected and not failures,
            "failures": failures, "metrics": metrics, "telemetry": telemetry, "pooled": pooled,
            "token_wait_count_all_zero": all_zero, "legal_set_cdf": bucket_totals(good),
            "telemetry_scopes": {key: sorted({grammar(row)[key] for row in good if isinstance(grammar(row).get(key), str)})
                                 for key in ("mask_timing_scope", "draft_breakdown_scope", "mask_scope")},
            "gpu_plans": sorted({(grammar(row).get("gpu_grammar_states"), grammar(row).get("gpu_grammar_bytes"))
                                 for row in good if numeric(grammar(row).get("gpu_grammar_states")) is not None
                                 and numeric(grammar(row).get("gpu_grammar_bytes")) is not None})}


def paired_report(rows, cases, repetitions, limit, left_arm, right_arm):
    indexed = {(*identity(row), row["arm"]): row for row in rows}
    pairs = []
    for case in cases:
        for repetition in range(repetitions):
            left = indexed.get((case["id"], repetition, limit, left_arm))
            right = indexed.get((case["id"], repetition, limit, right_arm))
            reasons = []
            for label, row in (("left", left), ("right", right)):
                reasons.extend([label + ":missing_request"] if row is None else [label + ":" + r for r in failure_reasons(row)])
            comparable = left is not None and right is not None and left["status"] == right["status"] == "ok"
            text_equal = left["text"] == right["text"] if comparable else None
            parsed_equal = left["parsed"] == right["parsed"] if comparable and left.get("schema_valid") is True and right.get("schema_valid") is True else None
            law_equal = (grammar(left).get("mode"), grammar(left).get("canonical_law")) == (grammar(right).get("mode"), grammar(right).get("canonical_law")) if comparable else None
            same_law = law_equal if left_arm.startswith("constrained_") and right_arm.startswith("constrained_") else None
            exact = not reasons and same_law is True and text_equal is True and parsed_equal is True
            rates = {key: bench.ratio(right["metrics"].get(key), left["metrics"].get(key)) if comparable else None for key in RATES}
            pairs.append({"case_id": case["id"], "repetition": repetition, "draft_limit": limit,
                          "comparable": comparable, "text_identical": text_equal, "parsed_identical": parsed_equal,
                          "grammar_law_identical": same_law, "matched_quality": not reasons,
                          "same_law_exact_output": exact,
                          "refusal_reasons": sorted(set(reasons)), "right_over_left_rates": rates})
    eligible = bool(pairs) and all(pair["matched_quality"] for pair in pairs)
    rates = {key: aggregate(pair["right_over_left_rates"][key] for pair in pairs) for key in RATES}
    matched = {key: aggregate(pair["right_over_left_rates"][key] for pair in pairs if pair["matched_quality"]) for key in RATES}
    native_rate = rates["decode_tok_per_s"]
    verdict = "not_a_matched_quality_speed_win"
    if eligible:
        if native_rate["observations"] != len(pairs):
            verdict = "matched_quality_native_rate_unobserved"
        elif native_rate["mean"] > 1:
            verdict = "matched_quality_native_speed_win"
        else:
            verdict = "matched_quality_no_native_speed_win"
    rate_verdicts = {}
    for key, data in rates.items():
        rate_verdicts[key] = ("not_quality_qualified" if not eligible else "unobserved" if data["observations"] != len(pairs)
                              else "matched_quality_speed_win" if data["mean"] > 1 else "matched_quality_no_speed_win")
    return {"left_arm": left_arm, "right_arm": right_arm, "expected_pairs": len(pairs),
            "comparable_pairs": sum(p["comparable"] for p in pairs),
            "text_identical_pairs": sum(p["text_identical"] is True for p in pairs),
            "parsed_identical_pairs": sum(p["parsed_identical"] is True for p in pairs),
            "matched_quality_pairs": sum(p["matched_quality"] for p in pairs),
            "same_law_exact_output_pairs": sum(p["same_law_exact_output"] for p in pairs),
            "same_law_exact_output_all": bool(pairs) and all(p["same_law_exact_output"] for p in pairs),
            "matched_quality_all": eligible, "verdict": verdict, "rate_verdicts": rate_verdicts,
            "raw_paired_rate_ratios": rates, "matched_quality_paired_rate_ratios": matched, "pairs": pairs}


def build_report(config, metadata, summary, records):
    invalid = validate(config, metadata, summary, records)
    if invalid is not None:
        return invalid
    measured = [r for r in records if r["phase"] == "measured"]
    selected = [arm for arm in ARMS if arm in metadata["arms"]]
    repetitions = metadata["args"]["repetitions"]
    splits = {}
    for limit in sorted(metadata["draft_limits"]):
        groups = {}
        for suite in ("full", "base24", "expanded10"):
            cases = [case for case in metadata["cases"] if suite == "full" or case.get("suite", "base") == ("base" if suite == "base24" else "expanded")]
            ids = {case["id"] for case in cases}
            rows = [r for r in measured if r["case_id"] in ids and r["draft_limit"] == limit]
            groups[suite] = {"fixture_count": len(cases), "fixture_ids": sorted(ids),
                             "arms": {arm: arm_report([r for r in rows if r["arm"] == arm], len(cases) * repetitions) for arm in selected},
                             "comparisons": {left + "__vs__" + right: paired_report(rows, cases, repetitions, limit, left, right)
                                             for left, right in COMPARISONS if left in selected and right in selected}}
        splits[str(limit)] = groups
    compact_metadata = {key: value for key, value in metadata.items() if key != "cases"}
    compact_metadata["cases"] = [{key: case[key] for key in ("id", "category", "suite", "max_tokens") if key in case}
                                 for case in metadata["cases"]]
    return {"version": 1, "config": config, "panel_metadata": compact_metadata,
            "input_summary_quality": {arm: {key: value for key, value in group.items() if key not in ("metrics", "counter_totals", "pooled_metrics")}
                                      for arm, group in summary.get("arms", {}).items()},
            "scope": {"actual_loaded_depth": config["depth"], "request_draft_limits": metadata["draft_limits"],
                      "server_geometry": metadata.get("server_geometry"), "draft_limit_scope": metadata.get("draft_limit_scope"),
                      "warmup_records_excluded": sum(r["phase"] == "warmup" for r in records),
                      "excluded_arms": sorted(set(metadata["arms"]) - set(selected)),
                      "split_names": "base24/expanded10 denote declared suites; fixture_count is actual selected count",
                      "timing": "CPU body and selection disjoint; token wait is a subset of selection. GPU body/selection event telemetry is not CPU mask time. Native verify_s/draft_s contain GPU grammar work; do not add them to body/selection or mask time.",
                      "quality_rule": "No matched-quality speed win unless every planned pair is present and both arms pass schema/semantic/stop/completion/contract quality. Text/parsed identity is reported separately, not a quality gate. Same-law exact-output claims additionally require ordinary laws and exact text/parsed identity. Raw rates include failed-quality outputs and are not quality-qualified wins.",
                      "unknown_rule": "Absent telemetry stays null/unobserved. Zero CPU mask time does not establish zero GPU grammar cost or zero token waits. Exact text/parsed identity does not establish token-ID identity.",
                      "ratio": "right/left rate per fixture/repetition/request-depth, then unweighted mean; pooled counters use jointly observed numerator/denominator and require complete telemetry over successful requests"},
            "by_request_depth": splits}


def fmt(value):
    return "?" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)


def markdown(report):
    cfg = report["config"]
    lines = ["# GPU grammar panel report", "", f"Loaded depth: **{cfg['depth']}**. Request caps: {report['scope']['request_draft_limits']}.",
             f"Source: `{cfg['source']}`. ELF SHA256: `{cfg['binary_sha256']}`.", "",
             report["scope"]["quality_rule"], "", report["scope"]["timing"], "", report["scope"]["unknown_rule"], "",
             "Rates are request means in tok/s; paired ratios are mean right/left. `?` is unobserved. `mean/pooled` distinguishes unweighted request ratios from pooled counters.", ""]
    for depth, splits in report["by_request_depth"].items():
        for name, split in splits.items():
            lines.extend([f"## Request depth {depth} / {name} ({split['fixture_count']} fixtures)", "",
                          "| Arm | attempts/planned | valid | semantic/observed | caps/errors | stopped/complete | quality | native/client/SSE-decode |",
                          "|---|---:|---:|---:|---:|---:|---|---|"])
            for arm, group in split["arms"].items():
                rates = "/".join(fmt(group["metrics"][key]["mean"]) for key in RATES)
                lines.append(f"| {arm} | {group['requests']}/{group['expected_requests']} | {group['schema_valid']} | {group['semantic_correct']}/{group['semantic_observations']} | {group['caps']}/{group['errors']} | {group['stops']}/{group['grammar_complete']} | {group['quality_pass']} | {rates} |")
            lines.extend(["", "| Arm | accepted/round mean/pooled | forced mean/pooled | legal mean/pooled | body/selection/wait (s) | native verify/draft/sampling (s) | waits=0 | GPU states/bytes |", "|---|---|---|---|---|---|---|---|"])
            for arm, group in split["arms"].items():
                ratios = [fmt(group["metrics"][key]["mean"]) + "/" + fmt(group["pooled"][key]["value"]) for key in ("accepted_draft_length", "forced_token_fraction", "mean_legal_set_size")]
                times = "/".join(fmt(group["metrics"][key]["mean"]) for key in ("draft_body_s", "draft_selection_s", "draft_token_wait_s"))
                native_times = "/".join(fmt(group["telemetry"]["native"].get(key, {}).get("mean")) for key in ("verify_s", "draft_s", "sampling_s"))
                lines.append(f"| {arm} | {' | '.join(ratios)} | {times} | {native_times} | {fmt(group['token_wait_count_all_zero'])} | {group['gpu_plans']} |")
            for arm, group in split["arms"].items():
                for scope, values in group["telemetry_scopes"].items():
                    if values:
                        lines.append(f"\n{arm} `{scope}`: {', '.join(values)}")
                extra = {key: data for key, data in group["telemetry"]["grammar"].items() if key.startswith("gpu_") and key not in ("gpu_grammar_states", "gpu_grammar_bytes")}
                if extra:
                    lines.append("\nGPU buckets: `" + json.dumps(extra, sort_keys=True) + "`")
                if "gpu" in arm:
                    lines.append("\nLegal-set CDF: `" + json.dumps(group["legal_set_cdf"], sort_keys=True) + "`")
            lines.extend(["", "| Paired right / left | identical text/parsed/planned | native/client/SSE ratio | verdict |", "|---|---|---|---|"])
            for pair in split["comparisons"].values():
                rates = "/".join(fmt(pair["raw_paired_rate_ratios"][key]["mean"]) for key in RATES)
                lines.append(f"| {pair['right_arm']} / {pair['left_arm']} | {pair['text_identical_pairs']}/{pair['parsed_identical_pairs']}/{pair['expected_pairs']} | {rates} | {pair['verdict']} |")
            failures = [(arm, failure) for arm, group in split["arms"].items() for failure in group["failures"]]
            missing = [(arm, group["missing_requests"]) for arm, group in split["arms"].items() if group["missing_requests"]]
            if failures or missing:
                lines.extend(["", "### Failures by fixture and arm", ""])
                for arm, count in missing:
                    lines.append(f"- {arm}: {count} planned requests missing.")
                for arm, fail in failures:
                    lines.append(f"- {fail['case_id']} r{fail['repetition']} / {arm}: {', '.join(fail['reasons'])}; details: `{json.dumps({k: fail[k] for k in ('error', 'parse_error', 'schema_errors')}, ensure_ascii=False)}`")
            mismatch = [(pair, row) for pair in split["comparisons"].values() for row in pair["pairs"] if not row["matched_quality"]]
            if mismatch:
                lines.extend(["", "### Pair qualification failures", ""])
                for pair, row in mismatch:
                    lines.append(f"- {row['case_id']} r{row['repetition']} / {pair['left_arm']} → {pair['right_arm']}: {', '.join(row['refusal_reasons'])}")
            lines.append("")
    lines.extend(["## Reproduction and environment", "", "Input SHA256 manifests and exact command/environment are retained in report.json.", "", "```json", json.dumps(cfg, indent=2, ensure_ascii=False), "```", ""])
    return "\n".join(lines)


def run(input_dir, output_dir):
    paths = {"config": input_dir / "config.json", "metadata": input_dir / "panel/metadata.json",
             "records": input_dir / "panel/records.jsonl", "summary": input_dir / "panel/summary.json"}
    loaded, hashes = {}, {}
    for name, path in paths.items():
        result = load(path, name == "records")
        if isinstance(result, Error):
            return result
        loaded[name] = result.value
        hashes[name] = result.sha256
    report = build_report(loaded["config"], loaded["metadata"], loaded["summary"], loaded["records"])
    if isinstance(report, Error):
        return report
    try:
        report["inputs"] = {name: {"path": str(path.resolve()), "sha256": hashes[name]} for name, path in paths.items()}
        report["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (output_dir / "report.md").write_text(markdown(report))
    except FileExistsError:
        return Error("output_exists", f"Choose a new output directory: {output_dir}")
    except (OSError, ValueError) as exc:
        return Error("report_write_error", str(exc))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing paths rejected")
    args = parser.parse_args()
    result = run(args.input, args.out)
    if isinstance(result, Error):
        print(json.dumps({"kind": result.kind, "message": result.message}), file=sys.stderr)
        return 1
    print(str(args.out / "report.json"))
    print(str(args.out / "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
