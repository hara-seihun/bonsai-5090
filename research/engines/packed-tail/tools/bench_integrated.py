#!/usr/bin/env python3
"""Bounded native ABBA integration/ablation replay; no GPU admission or provisioning."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from typing import Generic, Literal, TypeVar

T = TypeVar("T")
ErrorKind = Literal["configuration", "io", "timeout", "process_exit", "output_schema",
                    "interrupted", "budget_exhausted"]


@dataclass(frozen=True)
class Error:
    kind: ErrorKind
    message: str


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Arm:
    name: str
    binary: Path
    interface: Literal["probe", "fixed-single-pass"]
    plan: Path
    nodes: Literal["native", "kernel-memory"]
    prep: Literal["0", "2"]
    fusion: Literal["0", "1"]
    cg: Literal["4", "8"]
    pdl: Literal["0", "1"]
    build: str

    def environment(self) -> dict[str, str]:
        env = {"H5_COMPACT": "dense5", "H5_PLAN_READ": str(self.plan),
               "H5_PDL": self.pdl, "H5_STATE": "f16", "H5_KV": "fp8",
               "H5_WEIGHTS": "fp4", "H5_ACT": "fp4", "H5_PREP_CHUNK": self.prep,
               "H5_STATE_FUSION": self.fusion, "H5_GDN_CG": self.cg,
               "H5_ATTN_DIRECT": "0"}
        if self.prep == "2":
            env["H5_PREP_G4"] = ".015625"
        return env

    def command(self, model: Path, events: Path, prefix: Path,
                workload: str, passes: int) -> list[str]:
        cmd = [str(self.binary), str(model), str(events), str(prefix),
               "--ordinary" if workload == "ordinary" else "--decode-only"]
        if self.interface == "probe":
            cmd += ["--graph-upload", "launch", "--graph-timing", "off",
                    "--graph-nodes", self.nodes, "--passes", str(passes)]
        return cmd


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fixture_steps(path: Path, workload: str) -> Ok[list[int]] | Error:
    try:
        lines = path.read_text().splitlines()
        steps = []
        if not lines or lines[0] != "RESET 0":
            return Error("configuration", f"{path}: must begin with RESET 0")
        for line in lines:
            parts = line.split()
            if not parts:
                return Error("configuration", f"{path}: empty event")
            op, values = parts[0], list(map(int, parts[1:]))
            if op == "STEP":
                if not values or not 1 <= values[0] <= 520 or len(values) != 1 + 3 * values[0]:
                    return Error("configuration", f"{path}: invalid STEP")
                if any(values[i] != 0 or values[i + 1] < 0 or not 0 <= values[i + 2] < 4096
                       for i in range(1, len(values), 3)):
                    return Error("configuration", f"{path}: invalid slot/token/position")
                steps.append(values[0])
            elif op == "RESET":
                if values != [0]:
                    return Error("configuration", f"{path}: invalid RESET")
            elif op in ("KEEP", "ACCEPT"):
                limit = 1 if op == "KEEP" else 8
                if len(values) != 2 or values[0] != 0 or not 0 <= values[1] <= limit:
                    return Error("configuration", f"{path}: invalid {op}")
            else:
                return Error("configuration", f"{path}: unknown event {op}")
        expected_rows = 1 if workload == "ordinary" else 8
        if len(steps) < 2 or any(n != expected_rows for n in steps[1:]):
            return Error("configuration", f"{path}: not the expected {workload} workload")
        return Ok(steps)
    except OSError as e:
        return Error("io", str(e))
    except ValueError as e:
        return Error("configuration", f"{path}: {e}")


def finite_json(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_json(x) for x in value)
    return True


def read_output(prefix: Path, steps: list[int], passes: int, arm: Arm) -> Ok[dict] | Error:
    try:
        lines = prefix.with_suffix(".stdout.log").read_text().splitlines()
        metrics = json.loads(lines[-1]) if lines else None
        if not isinstance(metrics, dict) or not finite_json(metrics):
            return Error("output_schema", "missing/nonfinite replay metrics")
        for key in ("graph_replays", "graph_captures"):
            if type(metrics.get(key)) is not int or metrics[key] < 0:
                return Error("output_schema", f"invalid {key}")
        if metrics["graph_replays"] == 0:
            return Error("output_schema", "no native graph replays")
        if type(metrics.get("total_s")) not in (float, int) or metrics["total_s"] <= 0:
            return Error("output_schema", "invalid total_s")
        if arm.interface == "probe":
            required = {"passes": passes, "graph_upload": 0, "graph_timing": 0,
                        "graph_nodes": 0 if arm.nodes == "native" else 1}
            if any(type(metrics.get(k)) is not int or metrics[k] != v for k, v in required.items()):
                return Error("output_schema", "replayer did not apply requested graph/pass configuration")
        with prefix.with_suffix(".csv").open(newline="") as f:
            raw = list(csv.reader(f))
        if len(raw) != len(steps) * passes:
            return Error("output_schema", "missing or additional measured STEP rows")
        rows = []
        for i, row in enumerate(raw):
            round_id, n = map(int, row[:2])
            elapsed = float(row[2])
            predictions = list(map(int, row[3:]))
            if (round_id, n) != (i % len(steps), steps[i % len(steps)]):
                return Error("output_schema", f"changed recorded work at CSV row {i}")
            if len(row) != 3 + n or any(t < 0 for t in predictions):
                return Error("output_schema", f"incomplete/invalid predictions at CSV row {i}")
            if not math.isfinite(elapsed) or not 1e-12 <= elapsed <= 1800:
                return Error("output_schema", f"timing outside finite replay bounds at CSV row {i}")
            rows.append({"pass": i // len(steps), "round": round_id, "rows": n,
                         "elapsed_s": elapsed, "predictions": predictions})
        decode = [r["elapsed_s"] * 1000 for r in rows if r["round"] > 0]
        return Ok({"metrics": metrics, "rows": rows, "decode_mean_ms": statistics.mean(decode),
                   "decode_median_ms": statistics.median(decode), "decode_min_ms": min(decode),
                   "decode_max_ms": max(decode), "decode_sum_s": sum(decode) / 1000,
                   "logits": {"state": "unavailable", "reason": "profile_replay API emits argmax only"}})
    except OSError as e:
        return Error("io", str(e))
    except (ValueError, IndexError, TypeError) as e:
        return Error("output_schema", str(e))


def execute(command: list[str], env: dict[str, str], prefix: Path, timeout: float) -> Ok[float] | Error:
    proc = None
    start = time.monotonic()
    try:
        with prefix.with_suffix(".stdout.log").open("w") as stdout, \
                prefix.with_suffix(".stderr.log").open("w") as stderr:
            proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            code = proc.wait(timeout=timeout)
        if code:
            return Error("process_exit", f"replayer exit {code}")
        return Ok(time.monotonic() - start)
    except subprocess.TimeoutExpired:
        return Error("timeout", f"process exceeded {timeout:.3f}s")
    except KeyboardInterrupt:
        return Error("interrupted", "SIGINT/SIGTERM; foreground child reaped")
    except OSError as e:
        return Error("io", str(e))
    finally:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def prediction_changes(a: dict, b: dict, decode_only: bool) -> int:
    return sum(x != y for ra, rb in zip(a["rows"], b["rows"])
               if not decode_only or ra["round"] > 0
               for x, y in zip(ra["predictions"], rb["predictions"]))


def panel_summary(records: list[dict]) -> dict:
    a0, b0, b1, a1 = [r["result"] for r in records]
    a = statistics.mean([a0["decode_mean_ms"], a1["decode_mean_ms"]])
    b = statistics.mean([b0["decode_mean_ms"], b1["decode_mean_ms"]])
    return {"state": "complete", "A_mean_ms": a, "B_mean_ms": b,
            "B_latency_reduction_pct": 100 * (1 - b / a), "A_minus_B_ms": a - b,
            "run_mean_ms_ABBA": [r["result"]["decode_mean_ms"] for r in records],
            "predictions_decode": {"count": sum(r["rows"] for r in a0["rows"] if r["round"] > 0),
                                   "A_self_changes": prediction_changes(a0, a1, True),
                                   "B_self_changes": prediction_changes(b0, b1, True),
                                   "A0_to_each_run_changes": [prediction_changes(a0, r["result"], True)
                                                               for r in records]},
            "predictions_all_A0_to_each_run_changes": [prediction_changes(a0, r["result"], False)
                                                       for r in records]}


def contrast_selection(args: argparse.Namespace) -> Ok[dict] | Error:
    available = []
    if args.phase in ("pair", "all"):
        available.append("integrated")
    if args.phase in ("ablations", "all"):
        available += ["without_graph", "without_gemm", "without_prep", "without_fusion", "cg4",
                      "without_statepreload", "standalone_statepreload", "without_PDL_macros", "PDL_off"]
        if args.pdl_statepreload_binary is not None:
            available.append("without_early")
    if args.contrasts is None:
        return Ok({"state": "all_available", "selected": available, "excluded": []})
    if len(set(args.contrasts)) != len(args.contrasts):
        return Error("configuration", "--contrasts must contain unique names")
    unknown = [name for name in args.contrasts if name not in available]
    if unknown:
        return Error("configuration", f"contrasts unavailable for this phase/binary selection: {unknown}; available: {available}")
    return Ok({"state": "explicit", "requested": args.contrasts,
               "selected": [name for name in available if name in args.contrasts],
               "excluded": [name for name in available if name not in args.contrasts]})


def configuration(args: argparse.Namespace) -> Ok[dict] | Error:
    try:
        if not 1 <= args.passes <= 100 or not 1 <= args.blocks <= 10:
            return Error("configuration", "passes must be 1..100 and blocks 1..10")
        if not all(math.isfinite(x) and 0 < x <= 1800 for x in
                   (args.process_timeout, args.total_timeout)):
            return Error("configuration", "process/total timeout must be finite, positive and <=1800s")
        if args.baseline_interface == "fixed-single-pass" and args.passes != 1:
            return Error("configuration", "fixed-single-pass baseline requires --passes 1")
        if args.phase != "pair" and (args.pdl_control_binary is None or args.pdl_early_binary is None):
            return Error("configuration", "ablations require --pdl-control-binary and --pdl-early-binary")
        if os.environ.get("LD_PRELOAD"):
            return Error("configuration", "unset LD_PRELOAD: native uninstrumented execution required")
        binaries = {}
        for key in ("baseline_binary", "bundle_binary", "pdl_control_binary", "pdl_early_binary",
                    "pdl_statepreload_binary"):
            value = getattr(args, key)
            if value is not None:
                value = value.resolve(strict=True)
                if not value.is_file() or not os.access(value, os.X_OK):
                    return Error("configuration", f"{key}: not an executable file")
                binaries[key] = value
        inputs = {}
        for key in ("model", "ordinary_events", "verify_events", "baseline_plan", "bundle_plan"):
            path = getattr(args, key).resolve(strict=True)
            if not path.is_file():
                return Error("configuration", f"{key}: not a file")
            inputs[key] = path
        steps = {}
        for workload, key in (("ordinary", "ordinary_events"), ("verify", "verify_events")):
            result = fixture_steps(inputs[key], workload)
            if isinstance(result, Error):
                return result
            steps[workload] = result.value
        return Ok({"binaries": binaries, "inputs": inputs, "steps": steps})
    except OSError as e:
        return Error("io", str(e))


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    report = {"schema": "halo-integrated-v1", "state": "preparing", "runs": [], "panels": [],
              "scope": "native teacher-forced target latency; prefill excluded from decode summary; no independent gain summing",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    save(output / "results.json", report)
    configured = configuration(args)
    if isinstance(configured, Error):
        report.update(state="error", error=asdict(configured))
        save(output / "results.json", report)
        save(output / "summary.json", {"state": "error", "error": asdict(configured), "panels": []})
        print(json.dumps(report["error"]), file=sys.stderr)
        return 1
    config = configured.value
    selection = contrast_selection(args)
    assert isinstance(selection, Ok)
    report["contrast_selection"] = selection.value
    inputs, binaries = config["inputs"], config["binaries"]
    (output / "inputs").mkdir()
    for key in ("ordinary_events", "verify_events", "baseline_plan", "bundle_plan"):
        frozen = output / "inputs" / f"{key}.txt"
        shutil.copyfile(inputs[key], frozen)
        inputs[key] = frozen
    model_stat = inputs["model"].stat()
    report["custody"] = {"binaries": {k: identity(v) for k, v in binaries.items()},
                         "fixtures_and_plans": {k: identity(v) for k, v in inputs.items() if k != "model"},
                         "model": {"state": "stat_identity", "path": str(inputs["model"]),
                                   "bytes": model_stat.st_size, "mtime_ns": model_stat.st_mtime_ns,
                                   "device": model_stat.st_dev, "inode": model_stat.st_ino}}
    bundle = Arm("bundle", binaries["bundle_binary"], "probe", inputs["bundle_plan"],
                 "kernel-memory", "2", "1", "8", "1", "early+standalone-statepreload")
    baseline = Arm("baseline", binaries["baseline_binary"], args.baseline_interface,
                   inputs["baseline_plan"], "native", "0", "0", "4", "1", "original/no-PDL-macros")
    contrasts = []
    if args.phase in ("pair", "all"):
        contrasts.append(("integrated", baseline, bundle))
    if args.phase in ("ablations", "all"):
        contrasts += [("without_graph", replace(bundle, name="without_graph", nodes="native"), bundle),
                      ("without_gemm", replace(bundle, name="without_gemm", plan=inputs["baseline_plan"]), bundle),
                      ("without_prep", replace(bundle, name="without_prep", prep="0"), bundle),
                      ("without_fusion", replace(bundle, name="without_fusion", fusion="0"), bundle),
                      ("cg4", replace(bundle, name="cg4", cg="4"), bundle),
                      ("without_statepreload", replace(bundle, name="early_only",
                         binary=binaries["pdl_early_binary"], build="early-only"), bundle),
                      ("standalone_statepreload", replace(bundle, name="standalone_early_only", fusion="0",
                         binary=binaries["pdl_early_binary"], build="early-only"),
                         replace(bundle, name="standalone_both", fusion="0")),
                      ("without_PDL_macros", replace(bundle, name="no_PDL_macros",
                         binary=binaries["pdl_control_binary"], build="no-PDL-macros"), bundle),
                      ("PDL_off", replace(bundle, name="PDL_off", pdl="0"), bundle)]
        if "pdl_statepreload_binary" in binaries:
            contrasts.append(("without_early", replace(bundle, name="statepreload_only",
                              binary=binaries["pdl_statepreload_binary"], build="standalone-statepreload-only"), bundle))
            report["early_isolation"] = {"state": "scheduled"}
        else:
            report["early_isolation"] = {"state": "not_requested", "reason": "--pdl-statepreload-binary unset"}
    contrasts = [contrast for contrast in contrasts if contrast[0] in selection.value["selected"]]
    if "early_isolation" in report and "without_early" not in selection.value["selected"]:
        report["early_isolation"] = {"state": "not_requested", "reason": "without_early not selected"}
    report["arms"] = {arm.name: dict(asdict(arm), binary=str(arm.binary), plan=str(arm.plan),
                                     environment=arm.environment()) for _, a, b in contrasts for arm in (a, b)}
    for contrast, a, b in contrasts:
        for workload in ("ordinary", "verify"):
            for block in range(args.blocks):
                panel_id = f"{contrast}-{workload}-{block}"
                panel = {"id": panel_id, "contrast": contrast, "workload": workload,
                         "block": block, "A": a.name, "B": b.name, "state": "planned", "run_ids": []}
                report["panels"].append(panel)
                for slot, arm in enumerate((a, b, b, a)):
                    run_id = f"{panel_id}-{slot}-{arm.name}"
                    prefix = output / run_id
                    command = arm.command(inputs["model"], inputs[f"{workload}_events"], prefix,
                                          workload, args.passes)
                    report["runs"].append({"id": run_id, "panel_id": panel_id, "arm": arm.name,
                                           "state": "planned", "command": command,
                                           "environment": arm.environment(), "prefix": str(prefix)})
                    panel["run_ids"].append(run_id)
    report["state"] = "running"
    save(output / "results.json", report)
    deadline = time.monotonic() + args.total_timeout
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("H5_")}
    for panel in report["panels"]:
        records = [r for r in report["runs"] if r["panel_id"] == panel["id"]]
        panel["state"] = "running"
        for record in records:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result = Error("budget_exhausted", "total foreground execution budget exhausted")
            else:
                record["state"] = "running"
                save(output / "results.json", report)
                prefix = Path(record["prefix"])
                result = execute(record["command"], dict(base_env, **record["environment"]), prefix,
                                 min(args.process_timeout, remaining))
                record["artifacts"] = sorted(str(p) for p in output.glob(prefix.name + "*"))
                if isinstance(result, Ok):
                    record["process_wall_s"] = result.value
                    arm = next(arm for _, a, b in contrasts for arm in (a, b) if arm.name == record["arm"])
                    result = read_output(prefix, config["steps"][panel["workload"]], args.passes, arm)
            if isinstance(result, Error):
                record.update(state="error", error=asdict(result))
                panel.update(state="error", error=asdict(result))
                report.update(state="error", error=asdict(result))
                for pending in report["runs"]:
                    if pending["state"] == "planned":
                        pending.update(state="not_run", reason="earlier error; no automatic continuation")
                for pending in report["panels"]:
                    if pending["state"] == "planned":
                        pending.update(state="not_run", reason="earlier error")
                save(output / "results.json", report)
                save(output / "summary.json", {"state": "error", "error": asdict(result), "panels": report["panels"]})
                print(json.dumps({"panel": panel["id"], "error": asdict(result)}), file=sys.stderr)
                return 1
            record.update(state="complete", result=result.value)
            save(output / "results.json", report)
        panel.update(panel_summary(records))
        save(output / "results.json", report)
        save(output / "summary.json", {"state": "running", "panels": report["panels"]})
        print(json.dumps(panel, allow_nan=False), flush=True)
    report["state"] = "complete"
    save(output / "results.json", report)
    save(output / "summary.json", {"state": "complete", "panels": report["panels"]})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-binary", "bundle-binary", "model", "ordinary-events", "verify-events",
                 "baseline-plan", "bundle-plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("pdl-control-binary", "pdl-early-binary", "pdl-statepreload-binary"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--baseline-interface", choices=("probe", "fixed-single-pass"), required=True)
    parser.add_argument("--phase", choices=("pair", "ablations", "all"), required=True)
    parser.add_argument("--contrasts", nargs="+", help="Unique contrast names; omitted selects all available in --phase. Canonical panel order is retained.")
    parser.add_argument("--passes", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--process-timeout", type=float, required=True)
    parser.add_argument("--total-timeout", type=float, required=True)
    args = parser.parse_args()
    selection = contrast_selection(args)
    if isinstance(selection, Error):
        print(json.dumps(asdict(selection)), file=sys.stderr)
        return 1
    try:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        print(json.dumps(asdict(Error("configuration", f"fresh output directory required: {e}"))), file=sys.stderr)
        return 1
    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        return run(args)
    except (OSError, KeyboardInterrupt) as e:
        error = Error("interrupted" if isinstance(e, KeyboardInterrupt) else "io", str(e))
        if (args.output / "results.json").is_file():
            report = json.loads((args.output / "results.json").read_text())
            report.update(state="error", error=asdict(error))
            for collection in (report["runs"], report["panels"]):
                for item in collection:
                    if item["state"] in ("planned", "running"):
                        item.update(state="not_run" if item["state"] == "planned" else "error",
                                    error=asdict(error))
            save(args.output / "results.json", report)
            save(args.output / "summary.json", {"state": "error", "error": asdict(error), "panels": report["panels"]})
        print(json.dumps(asdict(error)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
