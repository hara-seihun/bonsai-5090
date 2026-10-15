#!/usr/bin/env python3
"""CPU-only replay and proposal comparisons for coupled sampling JSONL traces."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

MASK64 = (1 << 64) - 1
SCALES = (0.7, 1.0, 1.3, 1.6, 2.0)


@dataclass(frozen=True)
class TraceError:
    message: str


@dataclass(frozen=True)
class Filter:
    temperature: float
    top_p: float
    min_p: float
    top_k: int


@dataclass(frozen=True)
class Row:
    target: tuple[tuple[int, float], ...]
    draft: tuple[tuple[int, float], ...]
    target_token: int
    draft_token: int


@dataclass(frozen=True)
class Block:
    seed: int
    first_index: int
    target: Filter
    draft: Filter
    rows: tuple[Row, ...]
    accepted_prefix: int


@dataclass(frozen=True)
class Distribution:
    # Keep pre-normalization log weights: these are the CPU sampler's race scores.
    log_weights: dict[int, float]
    probabilities: dict[int, float]

    def entropy(self) -> float:
        return -sum(p * math.log(p) for p in self.probabilities.values() if p > 0)


def noise_mix(x: int) -> int:
    x &= MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def noise_gumbel(seed: int, index: int, token: int) -> float:
    key = (noise_mix(seed + 0x9E3779B97F4A7C15)
           ^ noise_mix(index + 0xD1B54A32D192ED03)
           ^ noise_mix(token + 0x8CB92BAA3F3D8DD7))
    uniform = ((noise_mix(key) >> 12) + 0.5) * (2.0 ** -52)
    return -math.log(-math.log(uniform))


def distribution(candidates: tuple[tuple[int, float], ...], config: Filter) -> Distribution:
    ordered = sorted(candidates, key=lambda item: (-item[1], item[0]))
    if config.top_k:
        ordered = ordered[:config.top_k]
    best = ordered[0][1]
    weighted = [(token, (score - best) / config.temperature,
                 math.exp((score - best) / config.temperature))
                for token, score in ordered]
    weighted = [item for item in weighted if item[2] >= config.min_p]
    total = sum(item[2] for item in weighted)
    retained = []
    cumulative = 0.0
    for item in weighted:
        retained.append(item)
        cumulative += item[2]
        if cumulative >= total * config.top_p:
            break
    norm = sum(item[2] for item in retained)
    return Distribution({token: logw for token, logw, _ in retained},
                        {token: weight / norm for token, _, weight in retained})


def select(dist: Distribution, noise: dict[int, float]) -> int:
    return max(dist.log_weights, key=lambda token: (dist.log_weights[token] + noise[token], -token))


def integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def number(value: Any, low: float, high: float, exclusive_low: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (low < value if exclusive_low else low <= value) and value <= high
    except OverflowError:
        return False


def parse_candidates(row: dict[str, Any], name: str, where: str) -> tuple[tuple[int, float], ...] | TraceError:
    ids, scores = row.get(name + "_ids"), row.get(name + "_scores")
    if not isinstance(ids, list) or not isinstance(scores, list) or not ids or len(ids) != len(scores):
        return TraceError(f"{where}: {name}_ids/{name}_scores must be nonempty equal-length arrays")
    if not all(integer(token, 0, (1 << 31) - 1) for token in ids) or len(set(ids)) != len(ids):
        return TraceError(f"{where}: {name}_ids must contain distinct nonnegative int32 tokens")
    if not all(number(score, -sys.float_info.max, sys.float_info.max) for score in scores):
        return TraceError(f"{where}: {name}_scores must contain finite numbers")
    return tuple(zip(ids, map(float, scores)))


def parse_block(value: Any, where: str) -> Block | TraceError:
    if not isinstance(value, dict):
        return TraceError(f"{where}: expected a JSON block object")
    for field in ("seed", "first_index"):
        if not integer(value.get(field), 0, MASK64):
            return TraceError(f"{where}: {field} must be uint64")
    configs = []
    for prefix in ("", "draft_"):
        for field, low, high, exclusive in (("temperature", 0, 2, True),
                                            ("top_p", 0, 1, True), ("min_p", 0, 1, False)):
            if not number(value.get(prefix + field), low, high, exclusive):
                detail = "; temperature 0 is unsupported" if field == "temperature" else ""
                return TraceError(f"{where}: {prefix + field} must be finite in "
                                  f"{'(' if exclusive else '['}{low},{high}]{detail}")
        if not prefix and not integer(value.get("top_k"), 1, 128):
            return TraceError(f"{where}: top_k must be in [1,128]; a full-vocabulary trace is unsupported")
        top_k = value.get("draft_top_k", value["top_k"]) if prefix else value["top_k"]
        if prefix and not integer(top_k, 0, 128):
            return TraceError(f"{where}: draft_top_k must be in [0,128]")
        configs.append(Filter(float(value[prefix + "temperature"]), float(value[prefix + "top_p"]),
                              float(value[prefix + "min_p"]), top_k))
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        return TraceError(f"{where}: rows must be a nonempty array")
    accepted = value.get("accepted_prefix")
    if not integer(accepted, 0, len(raw_rows)):
        return TraceError(f"{where}: accepted_prefix must be in [0,len(rows)]")
    rows = []
    for i, raw in enumerate(raw_rows):
        row_where = f"{where}, row {i}"
        if not isinstance(raw, dict):
            return TraceError(f"{row_where}: expected a row object")
        target = parse_candidates(raw, "target", row_where)
        draft = parse_candidates(raw, "draft", row_where)
        if isinstance(target, TraceError):
            return target
        if isinstance(draft, TraceError):
            return draft
        if len(target) != value["top_k"] or len(draft) > 16:
            return TraceError(f"{row_where}: require exactly top_k target candidates and 1..16 draft candidates")
        for name in ("target", "draft"):
            if not integer(raw.get(name + "_token"), 0, (1 << 31) - 1):
                return TraceError(f"{row_where}: {name}_token must be a nonnegative int32 token")
        rows.append(Row(target, draft, raw["target_token"], raw["draft_token"]))
    for i, row in enumerate(rows[:min(accepted + 1, len(rows))]):
        if (row.target_token == row.draft_token) != (i < accepted):
            return TraceError(f"{where}, row {i}: token agreement contradicts accepted_prefix={accepted}")
    return Block(value["seed"], value["first_index"], configs[0], configs[1], tuple(rows), accepted)


def proposal_configs(base: Filter) -> list[tuple[str, float, Filter]]:
    return [(f"{mode}/temperature_scale={scale:g}", scale,
             Filter(min(2.0, base.temperature * scale),
                    base.top_p if mode == "configured" else 1.0,
                    base.min_p if mode == "configured" else 0.0, base.top_k))
            for mode in ("configured", "untruncated") for scale in SCALES]


class Analyzer:
    def __init__(self, seeds: int):
        self.seeds = seeds
        self.blocks = self.realized_rows = self.ignored_rows = self.accepted = 0
        self.target_entropy = self.raw_mass = 0.0
        self.configs: dict[str, dict[str, Any]] = {}

    def add(self, block: Block, where: str) -> TraceError | None:
        for row_index, row in enumerate(block.rows[:block.accepted_prefix + 1]):
            index = (block.first_index + row_index) & MASK64
            p = distribution(row.target, block.target)
            q = distribution(row.draft, block.draft)
            tokens = set(p.log_weights) | {token for token, _ in row.draft}
            actual_noise = {token: noise_gumbel(block.seed, index, token) for token in tokens}
            for name, dist, expected in (("target", p, row.target_token), ("draft", q, row.draft_token)):
                replay = select(dist, actual_noise)
                if replay != expected:
                    return TraceError(f"{where}, row {row_index}: {name} replay mismatch: "
                                      f"logged {expected}, replay {replay}; check score/filter/index schema")
            proposals = [(name, scale, config, distribution(row.draft, config))
                         for name, scale, config in proposal_configs(block.draft)]
            agreements = [0] * len(proposals)
            # Successive uint64 keys are reproducible Monte Carlo seeds, shared across all configs.
            for trial in range(self.seeds):
                seed = (block.seed + trial + 1) & MASK64
                noise = {token: noise_gumbel(seed, index, token) for token in tokens}
                chosen = select(p, noise)
                for i, (_, _, _, candidate) in enumerate(proposals):
                    agreements[i] += select(candidate, noise) == chosen
            self.realized_rows += 1
            self.target_entropy += p.entropy()
            self.raw_mass += sum(p.probabilities.get(token, 0.0) for token, _ in row.draft)
            for i, (name, scale, config, candidate) in enumerate(proposals):
                if name not in self.configs:
                    self.configs[name] = dict(name=name, temperature_scale=scale,
                                             filter_mode=name.split("/")[0], agreement_count=0,
                                             entropy_sum=0.0, mass_sum=0.0, support_sum=0,
                                             temperature_min=config.temperature, temperature_max=config.temperature,
                                             clamped_rows=0)
                acc = self.configs[name]
                acc["agreement_count"] += agreements[i]
                acc["entropy_sum"] += candidate.entropy()
                acc["mass_sum"] += sum(p.probabilities.get(token, 0.0) for token in candidate.probabilities)
                acc["support_sum"] += len(candidate.probabilities)
                acc["temperature_min"] = min(acc["temperature_min"], config.temperature)
                acc["temperature_max"] = max(acc["temperature_max"], config.temperature)
                acc["clamped_rows"] += block.draft.temperature * scale > 2
        self.blocks += 1
        self.accepted += block.accepted_prefix
        self.ignored_rows += max(0, len(block.rows) - block.accepted_prefix - 1)
        return None

    def report(self) -> dict[str, Any]:
        count = self.realized_rows
        ranking = []
        for acc in self.configs.values():
            ranking.append(dict(name=acc["name"], temperature_scale=acc["temperature_scale"],
                                filter_mode=acc["filter_mode"],
                                agreement=acc["agreement_count"] / (count * self.seeds),
                                agreement_count=acc["agreement_count"],
                                mean_q_entropy_nats=acc["entropy_sum"] / count,
                                mean_target_mass_retained_q=acc["mass_sum"] / count,
                                mean_q_support=acc["support_sum"] / count,
                                temperature_range=[acc["temperature_min"], acc["temperature_max"]],
                                temperature_clamped_rows=acc["clamped_rows"]))
        ranking.sort(key=lambda item: (-item["agreement"], -item["mean_target_mass_retained_q"], item["name"]))
        for rank, item in enumerate(ranking, 1):
            item["rank"] = rank
        baseline = next(item for item in ranking if item["name"] == "configured/temperature_scale=1")
        return dict(blocks=self.blocks, realized_rows=count, ignored_rows=self.ignored_rows,
                    accepted_prefix_tokens=self.accepted, observed_agreement=self.accepted / count,
                    independent_seeds_per_row=self.seeds, draws_per_config=count * self.seeds,
                    mean_target_entropy_nats=self.target_entropy / count,
                    mean_target_mass_raw_draft=self.raw_mass / count,
                    baseline=baseline, ranking=ranking)


def analyze(path: Path, seeds: int) -> dict[str, Any] | TraceError:
    analyzer = Analyzer(seeds)
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                where = f"{path}:{line_number}"
                if not line.strip():
                    return TraceError(f"{where}: empty JSONL record")
                try:
                    value = json.loads(line)
                except (ValueError, RecursionError) as error:
                    return TraceError(f"{where}: invalid JSON: {error}")
                block = parse_block(value, where)
                if isinstance(block, TraceError):
                    return block
                error = analyzer.add(block, where)
                if error is not None:
                    return error
    except (OSError, UnicodeError) as error:
        return TraceError(f"{path}: {error}")
    if not analyzer.blocks:
        return TraceError(f"{path}: no trace blocks")
    return analyzer.report()


def self_test() -> None:
    candidates = ((7, 0.0), (2, 0.0), (11, -2.0))
    config = Filter(1.0, 1.0, 0.0, 3)
    p = distribution(candidates, config)
    token = select(p, {t: noise_gumbel(42, 9, t) for t, _ in candidates})
    block = Block(42, 9, config, config, (Row(candidates, candidates, token, token),), 1)
    analyzer = Analyzer(256)
    assert analyzer.add(block, "synthetic") is None
    report = analyzer.report()
    assert report["baseline"]["agreement"] == 1.0
    assert report["mean_target_mass_raw_draft"] == 1.0
    assert list(distribution(candidates, Filter(1, 0.5, 1, 2)).probabilities) == [2]
    assert list(distribution(((0, 0.0), (1, 0.0)), Filter(1, 0.5, 0, 2)).probabilities) == [0]
    assert list(distribution(((0, 0.0), (1, 0.0)), Filter(1, 0.51, 0, 2)).probabilities) == [0, 1]
    assert noise_mix(0) == 0
    assert noise_mix(1) == 0x5692161D100B05E5
    # Golden values evaluated directly by the canonical C++ noise header.
    assert math.isclose(noise_gumbel(42, 0, 7), -0.91069390048136623, rel_tol=1e-15)
    assert math.isclose(noise_gumbel(0, MASK64, 7), 2.2580910192363137, rel_tol=1e-15)
    print("self-test passed: identical p=q agreement 256/256; filtering, tie order, canonical keyed noise")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, help="one block per JSONL line")
    parser.add_argument("--seeds", type=int, default=256, help="Monte Carlo seeds per realized row (default: 256)")
    parser.add_argument("--self-test", action="store_true", help="run a tiny CPU-only synthetic contract check")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.self_test:
        if args.trace is not None:
            parser.error("--self-test does not take a trace")
        self_test()
        return 0
    if args.trace is None:
        parser.error("trace is required unless --self-test is used")
    result = analyze(args.trace, args.seeds)
    if isinstance(result, TraceError):
        print(f"trace error: {result.message}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
