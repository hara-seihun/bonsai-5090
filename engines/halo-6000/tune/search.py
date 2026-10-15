#!/usr/bin/env python3
"""halo-autotune whole-token search (runs on pod B under the job supervisor).

Loop: propose compile-time configurations -> build variant .so files in parallel
(CPU) -> one resident-driver batch under /workspace/halo-6000/gpu.lock (guard
every candidate bit-for-bit, time warm s1 decode) -> update population.
Runtime grids are tuned per elite by coordinate descent without recompiling.

Every measurement is appended to RUN/trace.jsonl; RUN/best.json holds the best
exact configuration so far.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import space  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf"
CACHE = "/workspace/models/halo6000-cache"
TEACHER = "/workspace/ternary-bench/oracle/teacher-v1/tokens.i32"
LOCK = "/workspace/halo-6000/gpu.lock"

# Which kinds each gene touches; used to weight mutations by measured slice cost.
GENE_KINDS = {
    "qkv_lanes": ["gdn_proj", "attn_proj"], "z_lanes": ["gdn_proj", "attn_proj"], "ab_lanes": ["gdn_proj"],
    "out_lanes": ["out_proj"], "router_rows": ["router"], "select_body": ["select"],
    "head_lanes": ["head"], "head_contiguous": ["head"],
}


class Run:
    def __init__(self, path: Path, args):
        self.path = path
        self.args = args
        path.mkdir(parents=True, exist_ok=True)
        # Frozen source snapshot: engine headers can change under a long search; every
        # variant and the driver of one run are built from the same tree.
        self.src = path / "src"
        if not (self.src / "build/tune/driver").exists():
            subprocess.run(["rm", "-rf", str(self.src)], check=True)
            self.src.mkdir()
            for d in ("include", "src", "tune"):
                subprocess.run(["cp", "-a", str(ROOT / d), str(self.src / d)], check=True)
            t0 = time.time()
            subprocess.run([str(self.src / "tune/build.sh"), "engine"], check=True, capture_output=True)
            print(f"snapshot engine+driver built in {time.time() - t0:.0f}s", flush=True)
        self.driver = self.src / "build/tune/driver"
        (path / "cfg").mkdir(parents=True, exist_ok=True)
        (path / "so").mkdir(parents=True, exist_ok=True)
        (path / "batches").mkdir(parents=True, exist_ok=True)
        self.trace = path / "trace.jsonl"
        self.rng = random.Random(args.seed)
        self.results = {}       # (compile_key, grid tuple) -> record
        self.configs = {}       # compile_key -> Config
        self.failed_builds = set()
        self.kind_cost = {}     # kind -> us per token (reference profile)
        self.ref_rate = None
        if self.trace.exists():
            for line in self.trace.read_text().splitlines():
                r = json.loads(line)
                if r.get("kind") == "measure":
                    c = space.Config.from_json(r["config"])
                    self.configs[c.compile_key()] = c
                    self.results[(c.compile_key(), tuple(r["grid_vector"]))] = r
                elif r.get("kind") == "build_fail":
                    self.failed_builds.add(r["compile_key"])
                elif r.get("kind") == "kind_cost":
                    self.kind_cost = r["kind_cost"]

    def log(self, rec):
        rec["t"] = time.time()
        with self.trace.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- build -----------------------------------------------------------
    def build(self, configs):
        todo = []
        for c in configs:
            key = c.compile_key()
            so = self.path / "so" / f"{key}.so"
            self.configs[key] = c
            if key in self.failed_builds or so.exists():
                continue
            hdr = self.path / "cfg" / f"{key}.hpp"
            hdr.write_text(c.header())
            todo.append((key, hdr, so))

        def one(item):
            key, hdr, so = item
            tu = "schedule_batch" if self.args.rows else "schedule"
            p = subprocess.run([str(self.src / "tune/build.sh"), "variant", str(hdr), str(so), tu], capture_output=True, text=True)
            return key, p.returncode, (p.stderr[-2000:] if p.returncode else "")

        t0 = time.time()
        with cf.ThreadPoolExecutor(self.args.jobs) as ex:
            for key, rc, err in ex.map(one, todo):
                if rc:
                    self.failed_builds.add(key)
                    ptx = (self.path / "so" / f"{key}.ptxas")
                    self.log({"kind": "build_fail", "compile_key": key, "stderr": err,
                              "ptxas": ptx.read_text()[-2000:] if ptx.exists() else ""})
        if todo:
            print(f"built {len(todo)} variants in {time.time() - t0:.1f}s", flush=True)

    # ---- measure ---------------------------------------------------------
    def measure(self, items, profile=False, reps=None, tag="batch", time=True):
        """items: list of (Config, grid_vector). Chunks keep each GPU-lock hold short."""
        out = []
        for i in range(0, len(items), self.args.chunk):
            out += self.measure_chunk(items[i:i + self.args.chunk], profile, reps, f"{tag}-{i // self.args.chunk}", time)
        return out

    def measure_chunk(self, items, profile=False, reps=None, tag="batch", time_it=True):
        jobs = []
        for c, gv in items:
            key = c.compile_key()
            so = self.path / "so" / f"{key}.so"
            if key in self.failed_builds or not so.exists():
                continue
            if (key, tuple(gv)) in self.results and not profile:
                continue
            jobs.append({"id": f"{key}:{','.join(map(str, gv))}", "so": str(so), "grids": gv, "profile": profile, "time": time_it, "cfg": c, "gv": gv})
        if not jobs:
            return []
        stamp = time.strftime("%Y%m%dT%H%M%S")
        batch = self.path / "batches" / f"{stamp}-{tag}"
        batch.mkdir(parents=True, exist_ok=True)
        jf = batch / "jobs.jsonl"
        with jf.open("w") as f:
            for j in jobs:
                f.write(json.dumps({k: j[k] for k in ("id", "so", "grids", "profile", "time")}) + "\n")
        out = batch / "results.jsonl"
        snap = "date -u +%FT%TZ; nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv"
        cmd = (f"{snap} > {batch}/context-before.txt; "
               f"{self.driver} --model {MODEL} --cache-dir {CACHE} --teacher {TEACHER} "
               f"--jobs {jf} --out {out} --reps {reps or self.args.reps} --ref-time {1 if self.args.ref_time else 0} --rows {self.args.rows}; rc=$?; "
               f"{snap} > {batch}/context-after.txt; exit $rc")
        t0 = time.time()
        p = subprocess.run(["flock", LOCK, "bash", "-c", cmd], capture_output=True, text=True, timeout=self.args.batch_timeout)
        (batch / "driver.stderr").write_text(p.stderr)
        print(f"{tag}: {len(jobs)} jobs in {time.time() - t0:.1f}s rc={p.returncode}", flush=True)
        recs = []
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                r = json.loads(line)
                if r.get("kind") == "reference":
                    if "tok_s" in r:
                        self.ref_rate = r["tok_s"]
                    self.log({"kind": "reference", **r, "batch": batch.name})
                    continue
                j = next((j for j in jobs if j["id"] == r["id"]), None)
                if j is None:
                    continue
                done.add(j["id"])
                rec = {"kind": "measure", "batch": batch.name, "config": j["cfg"].to_json(), "compile_key": j["cfg"].compile_key(),
                       "grid_vector": j["gv"], "exact": r.get("guard", {}).get("exact", False), "tok_s": r.get("tok_s"),
                       "tok_s_reps": r.get("tok_s_reps"), "guard": r.get("guard"), "error": r.get("error"),
                       "describe": r.get("describe"), "epoch_us": r.get("epoch_us"), "profile_token_us": r.get("profile_token_us"),
                       "ref_tok_s": self.ref_rate}
                self.log(rec)
                if not profile:
                    self.results[(rec["compile_key"], tuple(j["gv"]))] = rec
                recs.append(rec)
        if p.returncode and len(done) < len(jobs):
            # A crashed candidate (sticky CUDA error) kills the driver; record it and continue with the rest.
            first = next(j for j in jobs if j["id"] not in done)
            rec = {"kind": "measure", "batch": batch.name, "config": first["cfg"].to_json(), "compile_key": first["cfg"].compile_key(),
                   "grid_vector": first["gv"], "exact": False, "tok_s": None, "error": "driver crashed: " + p.stderr[-500:]}
            self.log(rec)
            self.results[(rec["compile_key"], tuple(first["gv"]))] = rec
            rest = [(j["cfg"], j["gv"]) for j in jobs if j["id"] not in done and j is not first]
            if rest:
                recs += self.measure_chunk(rest, profile, reps, tag + "-resume", time_it)
        return recs

    # ---- population ------------------------------------------------------
    def exact_ranked(self):
        rows = [r for r in self.results.values() if r.get("exact") and r.get("tok_s")]
        return sorted(rows, key=lambda r: -r["tok_s"])

    def best(self):
        e = self.exact_ranked()
        return e[0] if e else None

    def save_best(self):
        b = self.best()
        if b:
            (self.path / "best.json").write_text(json.dumps(b, indent=1))

    def gene_weights(self):
        w = {}
        for g in space.BODY_GENES:
            if len(space.BODY_GENES[g]) < 2:
                continue
            w[g] = 1.0 + sum(self.kind_cost.get(k, 0) for k in GENE_KINDS.get(g, [])) / 100.0
        return w

    def propose(self, parent: space.Config, n):
        out, seen = [], set(self.configs) | self.failed_builds
        tries = 0
        weights = self.gene_weights()
        while len(out) < n and tries < 50 * n:
            tries += 1
            c = parent
            for _ in range(self.rng.choice([1, 1, 1, 2, 2, 3])):
                c = space.mutate(c, self.rng)
                # Re-roll body gene by slice-cost weight half the time.
                if self.rng.random() < 0.5:
                    g = self.rng.choices(list(weights), weights=list(weights.values()))[0]
                    c.body[g] = self.rng.choice(space.BODY_GENES[g])
            k = c.compile_key()
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        return out

    def grid_neighbours(self, rec):
        c = space.Config.from_json(rec["config"])
        desc = rec.get("describe") or {}
        classes = desc.get("classes", [])
        gv = list(rec["grid_vector"])
        out = []
        for i, m in enumerate(c.class_list()):
            info = classes[i] if i < len(classes) else {}
            cur = gv[i] or info.get("grid", 188)
            coop = c.classes()[m][0]
            cand = space.legal_grids(coop, c.min_blocks.get(m, 1), info.get("active_per_sm"))
            for g in cand:
                if g == cur:
                    continue
                if abs(g - cur) > 2 * max(cur, 188):
                    continue
                v = list(gv)
                v[i] = g
                out.append((c, v))
        self.rng.shuffle(out)
        return out


EVERY = [k for k in space.CUTTABLE if k not in ("rot_writer", "rot_down")]
GRID_SWEEP = [-94, -188, -376, -564, -752, -1128, -1504]   # negative = clamp to cooperative residency


def clone(c):
    return space.Config.from_json(json.loads(json.dumps(c.to_json())))


def rate(run, c, gv=None):
    r = run.results.get((c.compile_key(), tuple(gv if gv is not None else c.grid_vector())))
    return r["tok_s"] if r and r.get("exact") and r.get("tok_s") else None


def measure_configs(run, configs, tag):
    run.build(configs)
    run.measure([(c, c.grid_vector()) for c in configs], tag=tag)
    return [(rate(run, c), c) for c in configs]


def coordinate_bodies(run, base, tag, rounds=2):
    """Coordinate descent over body genes, whole-token tok/s, all values of one gene per batch row."""
    best_rate = rate(run, base) or measure_configs(run, [base], tag + "-base")[0][0]
    for rnd in range(rounds):
        improved = False
        cands = []
        for g, vals in space.BODY_GENES.items():
            for v in vals:
                if v == base.body.get(g):
                    continue
                c = clone(base)
                c.body[g] = v
                c.min_blocks = {m: b for m, b in base.min_blocks.items() if m in c.classes()}
                c.grids = {m: x for m, x in base.grids.items() if m in c.classes()}
                cands.append((g, c))
        res = measure_configs(run, [c for _, c in cands], f"{tag}-r{rnd}")
        # Apply the best value per gene greedily, then confirm the combination.
        per_gene = {}
        for (g, c), (r, _) in zip(cands, res):
            if r and r > best_rate * 1.001 and (g not in per_gene or r > per_gene[g][0]):
                per_gene[g] = (r, c.body[g])
        if not per_gene:
            break
        combo = clone(base)
        for g, (_, v) in per_gene.items():
            combo.body[g] = v
        combo.min_blocks = {m: b for m, b in base.min_blocks.items() if m in combo.classes()}
        combo.grids = {m: x for m, x in base.grids.items() if m in combo.classes()}
        single_best = max(per_gene.items(), key=lambda kv: kv[1][0])
        single = clone(base)
        single.body[single_best[0]] = single_best[1][1]
        single.min_blocks = dict(combo.min_blocks)
        single.grids = {m: x for m, x in base.grids.items() if m in single.classes()}
        res2 = measure_configs(run, [combo, single], f"{tag}-r{rnd}-combo")
        for r, c in res2:
            if r and r > best_rate:
                best_rate, base, improved = r, c, True
        print(f"{tag} round {rnd}: {best_rate:.2f} tok/s body={base.body}", flush=True)
        if not improved:
            break
    return base, best_rate


def separable_grids(run, base, tag, min_blocks_options=(1, 2, 3, 4)):
    """Per-class (min_blocks, grid) from per-epoch profiles. Valid when classes are
    separate kernels (time is additive); confirmed by a whole-token measurement."""
    variants = []
    for mb in min_blocks_options:
        c = clone(base)
        c.min_blocks = {m: mb for m in c.class_list()}
        c.grids = {}
        variants.append(c)
    run.build(variants)
    table = {}   # mask -> list of (us, mb, grid)
    for c in variants:
        mb = next(iter(c.min_blocks.values()))
        items = [(c, [g] * len(c.class_list())) for g in GRID_SWEEP]
        recs = run.measure(items, profile=3, tag=f"{tag}-mb{mb}", time=False)
        for rec in recs:
            if not rec.get("exact") or not rec.get("epoch_us"):
                continue
            d = rec["describe"]
            first = d["segment_first"] + [d["epochs"]]
            kinds = d["epoch_kinds"]
            cls_of = {}
            for si in range(len(first) - 1):
                mask = 0
                for e in range(first[si], first[si + 1]):
                    mask |= 1 << kinds[e]
                cls_of[si] = f"{mask:#x}"
            per = {}
            for si in range(len(first) - 1):
                per.setdefault(cls_of[si], 0.0)
                per[cls_of[si]] += sum(rec["epoch_us"][first[si]:first[si + 1]])
            clist = c.class_list()
            for m, us in per.items():
                i = clist.index(m) if m in clist else None
                if i is None:
                    continue
                g = d["classes"][i]["grid"]
                table.setdefault(m, []).append((us, mb, g))
    best = clone(base)
    best.min_blocks, best.grids = {}, {}
    for m, rows in table.items():
        us, mb, g = min(rows)
        best.min_blocks[m] = mb
        best.grids[m] = g
    run.log({"kind": "grid_table", "tag": tag, "table": table})
    r = measure_configs(run, [best], f"{tag}-confirm")[0][0]
    print(f"{tag}: separable grids -> {r} tok/s", flush=True)
    return best, r


def merged(base, removed, mb=1):
    c = clone(base)
    for k in removed:
        if k in c.boundaries:
            c.boundaries.remove(k)
    cls = c.classes()
    c.min_blocks = {m: b for m, b in base.min_blocks.items() if m in cls}
    c.grids = {m: g for m, g in base.grids.items() if m in cls}
    for m in cls:
        if m not in base.classes():
            c.min_blocks[m] = mb
            c.grids[m] = -188 * mb
    return c


def greedy_merges(run, base, base_rate, tag, max_rounds=8):
    """Try removing each kernel boundary (merging into a cooperative segment) at min_blocks 1 and 2;
    then try all improving merges together; keep the best."""
    for rnd in range(max_rounds):
        singles = []
        for k in base.live_boundaries():
            for mb in (1, 2):
                singles.append((k, mb, merged(base, [k], mb)))
        if not singles:
            break
        res = measure_configs(run, [c for _, _, c in singles], f"{tag}-r{rnd}")
        good = sorted([(r, k, mb, c) for (k, mb, c), (r, _) in zip(singles, res) if r and r > base_rate * 1.0005], key=lambda t: -t[0])
        if not good:
            break
        best_r, _, _, best_c = good[0]
        ks = []
        for r, k, mb, c in good:
            if k not in ks:
                ks.append(k)
        if len(ks) > 1:
            combos = [merged(base, ks, 1), merged(base, ks, 2), merged(base, ks[: max(2, len(ks) // 2)], 1)]
            for r, c in measure_configs(run, combos, f"{tag}-r{rnd}-combo"):
                if r and r > best_r:
                    best_r, best_c = r, c
        base, base_rate = best_c, best_r
        print(f"{tag} round {rnd}: {base_rate:.2f} tok/s boundaries={base.live_boundaries()}", flush=True)
    return base, base_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="build/tune/runs/s1")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=18)
    ap.add_argument("--ref-time", type=int, default=1)
    ap.add_argument("--batch-timeout", type=int, default=900)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--elites", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--rows", type=int, default=0, help="batch rows (0 = single-stream)")
    ap.add_argument("--seed-config", default="", help="best.json of an earlier run: skip stages 1-2 and refine from it")
    args = ap.parse_args()
    run = Run(ROOT / args.run if not args.run.startswith("/") else Path(args.run), args)
    deadline = time.time() + args.max_hours * 3600

    if args.rows:
        batch_search(run, args, deadline)
        return
    ref = space.reference()
    measure_configs(run, [ref], "baseline")
    if args.seed_config:
        seed = json.loads(Path(args.seed_config).read_text())
        base = space.Config.from_json(seed["config"])
        base.grids = {m: g for m, g in zip(base.class_list(), seed["grid_vector"]) if g}
        r = measure_configs(run, [base], "seed")[0][0]
        print(f"seed {r} tok/s", flush=True)
        base, r = coordinate_bodies(run, base, "seed-bodies", rounds=2)
        run.save_best()
        args.skip_to_refine = True
    else:
        args.skip_to_refine = False
    # Stage 1: every phase its own kernel -> bodies, then separable (min_blocks, grid) per class.
    if not args.skip_to_refine:
        stage12(run)
    refine(run, args, deadline)


def stage12(run):
    ref = space.reference()
    every = clone(ref)
    every.boundaries = list(EVERY)
    every, r = coordinate_bodies(run, every, "s1-bodies")
    every, r2 = separable_grids(run, every, "s1-grids")
    if r2 and r2 > r:
        r = r2
    else:
        every.min_blocks, every.grids = {}, {}
    every, r = coordinate_bodies(run, every, "s1-bodies2", rounds=1)
    run.save_best()
    # Stage 2: greedy merges of kernel boundaries into cooperative segments.
    merged, rm = greedy_merges(run, every, r, "s2-merge")
    run.save_best()
    merged, rm2 = separable_grids(run, merged, "s2-grids", min_blocks_options=(1, 2, 3))
    run.save_best()


BATCH_X = [-1, -2, -3, -4, -6, -8, -12, -16, -23, -32, -47]


def batch_search(run, args, deadline):
    """Batch tokens: bodies are fixed (include/batch.cuh); search segmentation,
    __launch_bounds__ and per-class x grid. Reference = run_batch_token default."""
    global GRID_SWEEP
    GRID_SWEEP = BATCH_X
    one = space.Config(rows=args.rows)                       # one cooperative kernel, like run_batch_token
    every = space.Config(rows=args.rows, boundaries=[k for k in space.BATCH_CUTTABLE if k in set(one.sequence())])
    res = measure_configs(run, [one, every], "b-seeds")
    print("batch seeds:", [r for r, _ in res], flush=True)
    every, r = separable_grids(run, every, "b-grids")
    run.save_best()
    merged, rm = greedy_merges(run, every, r or 0, "b-merge")
    run.save_best()
    separable_grids(run, merged, "b-grids2", min_blocks_options=(1, 2, 3))
    run.save_best()
    refine(run, args, deadline)


def refine(run, args, deadline):
    # Stage 3: evolutionary refinement around the elites.
    gen = 0
    while gen < args.generations and time.time() < deadline:
        gen += 1
        elites = run.exact_ranked()[: args.elites]
        props = []
        for i, rec in enumerate(elites):
            share = max(1, args.population * (len(elites) - i) // sum(range(1, len(elites) + 1)))
            parent = space.Config.from_json(rec["config"])
            parent.grids = {m: g for m, g in zip(parent.class_list(), rec["grid_vector"]) if g}
            props += run.propose(parent, share)
        measure_configs(run, props, f"s3-g{gen}")
        for rec in run.exact_ranked()[:2]:
            run.measure(run.grid_neighbours(rec)[:12], tag=f"s3-g{gen}-grids")
        run.save_best()
        b = run.best()
        print(f"gen {gen}: best {b['tok_s']:.2f} tok/s ({b['compile_key']} grids {b['grid_vector']}), reference {run.ref_rate}", flush=True)
    b = run.best()
    c = space.Config.from_json(b["config"])
    run.measure([(c, b["grid_vector"])], profile=5, reps=5, tag="best-profile")
    run.save_best()
    print("best:", json.dumps({"tok_s": b["tok_s"], "config": c.describe(), "grid_vector": b["grid_vector"]}), flush=True)


if __name__ == "__main__":
    main()
