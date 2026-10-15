"""halo-autotune configuration space, as data.

A configuration has two layers:
  * compile-time genes, which become a generated C++ header (tune/cfg/<id>.hpp)
    consumed by src/schedule.cu through include/tune.hpp;
  * runtime genes (one grid per compiled segment class), which the resident
    driver applies without recompiling.

Every gene selects among bodies that are bit-identical to the reference engine.
The driver's 16-step logits guard is still run on every candidate.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict

KINDS = ["embed", "in_norm", "gdn_proj", "conv_prep", "recur", "finish_q", "rot_writer", "out_proj",
         "res_norm", "router", "select", "gate_up", "rot_down", "down", "attn_proj", "norm_rope",
         "scores", "softmax", "value_gate", "final_norm", "head", "arg_part", "arg_finish",
         "pack_hidden", "pack_inner"]   # batch-only kinds (include/batch.hpp BatchEpochKind)
K = {k: i for i, k in enumerate(KINDS)}
SMS = 188

# Compile-time body genes and their legal values (first value = reference default).
BODY_GENES = {
    "qkv_lanes": [16, 8, 4, 32],
    "z_lanes": [16, 8, 4, 32],
    "ab_lanes": [16, 32, 8],
    "out_lanes": [16, 8, 4, 32],
    "router_rows": [2, 1, 4, 8, 0],
    "select_body": [1, 0],
    "head_lanes": [16, 8, 4, 32],
    "head_contiguous": [0, 1],
    # L2 prefetch bits (header emits prefetch = sum): proj epoch prefetches out_proj+router (1),
    # gate_up prefetches the 8 selected experts' down (2), down prefetches next layer's
    # projections (4), router prefetches the shared expert (8).
    "pf_proj": [0, 1],
    "pf_down": [0, 1],
    "pf_next": [0, 1],
    "pf_shared": [0, 1],
    "mma_proj": [0, 1],
    "mma_out": [0, 1],
    "mma_head": [0, 1],
    "proj_rows": [1, 2, 4],
    "out_rows": [1, 2, 4],
    "router_mrows": [1, 2, 4],
    "down_rows": [1, 2, 4],
    "load_nc": [0, 1],
    "router_bf16": [0, 1],
    "l2_router": [0, 1],
    "l2_state": [0, 1],
    "l2_ratio": [100, 80, 60],
    "fuse_norm": [0, 1],
    "fuse_select": [0, 1],
}
# Kinds whose epoch may start a new kernel (any kind can; listed for mutation).
CUTTABLE = [k for k in KINDS if k not in ("embed", "pack_hidden", "pack_inner")]
# Batch ArgPart/ArgFinish partitions depend on gridDim.x, so they share a launch.
BATCH_CUTTABLE = [k for k in KINDS if k not in ("embed", "arg_finish", "rot_writer", "rot_down")]


def batch_sequence(rotation_writer=False, rotation_down=False, layers=40):
    """Mirror of BatchWorkspace::create's epoch list (src/batch.cu)."""
    seq = ["embed"]
    for i in range(layers):
        attention = (i + 1) % 4 == 0
        seq += ["in_norm", "pack_hidden"]
        seq += (["attn_proj", "norm_rope", "scores", "softmax", "value_gate"] if attention
                else ["gdn_proj", "conv_prep", "recur", "finish_q"])
        if rotation_writer:
            seq.append("rot_writer")
        seq += ["pack_inner", "out_proj", "res_norm", "router", "select", "gate_up"]
        if rotation_down:
            seq.append("rot_down")
        seq.append("down")
    seq += ["final_norm", "pack_hidden", "head", "arg_part", "arg_finish"]
    return seq


def epoch_sequence(rotation_writer=False, rotation_down=False, layers=40, fuse_norm=0, fuse_select=0):
    seq = ["embed"]
    for i in range(layers):
        attention = (i + 1) % 4 == 0
        if not fuse_norm:
            seq.append("in_norm")
        seq += (["attn_proj", "norm_rope", "scores", "softmax", "value_gate"] if attention
                else ["gdn_proj", "conv_prep", "recur", "finish_q"])
        if rotation_writer:
            seq.append("rot_writer")
        seq += ["out_proj"] + ([] if fuse_norm else ["res_norm"]) + ["router"] + ([] if fuse_select else ["select"]) + ["gate_up"]
        if rotation_down:
            seq.append("rot_down")
        seq.append("down")
    seq += ["final_norm", "head", "arg_part", "arg_finish"]
    return seq


@dataclass
class Config:
    body: dict = field(default_factory=lambda: {g: v[0] for g, v in BODY_GENES.items()})
    boundaries: list = field(default_factory=list)        # kinds that start a new kernel
    min_blocks: dict = field(default_factory=dict)        # segment mask (hex str) -> __launch_bounds__ min blocks
    grids: dict = field(default_factory=dict)             # segment mask (hex str) -> grid (runtime)
    rotation: tuple = (False, False)
    rows: int = 0                                         # 0: s1 schedule.cu; >0: batch schedule_batch.cu

    # ---- derived ---------------------------------------------------------
    def sequence(self):
        if self.rows:
            return batch_sequence(*self.rotation)
        return epoch_sequence(*self.rotation, fuse_norm=self.body.get("fuse_norm", 0), fuse_select=self.body.get("fuse_select", 0))

    def live_boundaries(self):
        """Boundaries on kinds that occur; others are inert and must not split compile keys."""
        present = set(self.sequence())
        return sorted((k for k in set(self.boundaries) if k in present), key=K.get)

    def segments(self):
        """List of (mask, n_epochs) for every segment in token order."""
        b = set(self.boundaries)
        segs = []
        for i, kind in enumerate(self.sequence()):
            if i == 0 or kind in b:
                segs.append([0, 0])
            segs[-1][0] |= 1 << K[kind]
            segs[-1][1] += 1
        return segs

    def classes(self):
        """Distinct segment masks -> (cooperative?, count)."""
        out = {}
        for mask, n in self.segments():
            key = f"{mask:#x}"
            coop, count = out.get(key, (False, 0))
            out[key] = (coop or n > 1, count + 1)
        return dict(sorted(out.items(), key=lambda kv: int(kv[0], 16)))

    def compile_key(self):
        """Everything that changes the binary (grids excluded)."""
        cls = self.classes()
        d = {"body": self.body, "classes": {m: self.min_blocks.get(m, 1) for m in cls},
             "boundaries": self.live_boundaries(), "rotation": list(self.rotation)}
        if self.rows:
            d = {"batch": True, "classes": d["classes"], "boundaries": d["boundaries"], "rotation": d["rotation"]}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def class_list(self):
        return list(self.classes().keys())

    def grid_vector(self):
        """Runtime grid per class, in header order (0 = driver default)."""
        return [int(self.grids.get(m, 0)) for m in self.class_list()]

    def header(self, with_grids=False):
        cid = self.compile_key()
        cls = self.class_list()
        boundary_mask = 0
        for k in self.live_boundaries():
            boundary_mask |= 1 << K[k]
        lines = ["#pragma once", "// generated by tune/space.py; do not edit", "#include <cstdint>",
                 "namespace halo6000::tune {", f'static constexpr const char*config_id="{cid}";']
        pf = 0
        for g, v in self.body.items():
            if g.startswith("pf_"):
                pf |= {"pf_proj": 1, "pf_down": 2, "pf_next": 4, "pf_shared": 8}[g] * int(bool(v))
                continue
            if g == "prefetch":
                continue
            lines.append(f"static constexpr int {g}={int(v)};")
        lines.append(f"static constexpr int prefetch={pf};")
        lines.append(f"static constexpr int segment_count={len(cls)};")
        lines.append("static constexpr uint32_t segment_kinds[]={" + ",".join(f"{m}u" for m in cls) + "};")
        lines.append("static constexpr int segment_min_blocks[]={" + ",".join(str(self.min_blocks.get(m, 1)) for m in cls) + "};")
        lines.append(f"static constexpr uint32_t boundary_kinds={boundary_mask:#x}u;")
        if with_grids and any(self.grids.get(m) for m in cls):
            lines.append("static constexpr int segment_grids[]={" + ",".join(str(int(self.grids.get(m, 0))) for m in cls) + "};")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_json(self):
        d = asdict(self)
        d["rotation"] = list(self.rotation)
        return d

    @staticmethod
    def from_json(d):
        body = {g: v[0] for g, v in BODY_GENES.items()}
        body.update({g: v for g, v in d["body"].items() if g in BODY_GENES})
        c = Config(body=body, boundaries=list(d["boundaries"]), min_blocks=dict(d.get("min_blocks", {})),
                   grids=dict(d.get("grids", {})), rotation=tuple(d.get("rotation", (False, False))), rows=int(d.get("rows", 0)))
        return c

    def describe(self):
        segs = self.segments()
        return {"compile_key": self.compile_key(), "body": self.body, "boundaries": self.boundaries,
                "segments": len(segs), "barriers": sum(n - 1 for _, n in segs),
                "classes": {m: {"coop": c, "count": n, "min_blocks": self.min_blocks.get(m, 1), "grid": self.grids.get(m, 0)}
                            for m, (c, n) in self.classes().items()}}


def kinds_of(mask_hex):
    m = int(mask_hex, 16)
    return [k for k in KINDS if m >> K[k] & 1]


def reference():
    return Config()


def legal_grids(coop, min_blocks, active=None):
    """Candidate grids for a class. Cooperative classes are capped at residency."""
    cap = (active or min_blocks) * SMS
    base = [SMS // 4, SMS // 2, SMS, 2 * SMS, 3 * SMS, 4 * SMS, 6 * SMS, 8 * SMS]
    if coop:
        return sorted({g for g in base if g <= cap} | {cap})
    return sorted(set(base) | {1, 8, 32, 64})


def mutate(c: Config, rng: random.Random, p_body=0.35, p_cut=0.4, p_mb=0.25):
    if c.rows:
        p_body, p_cut = 0.0, 0.6
    n = Config.from_json(json.loads(json.dumps(c.to_json())))
    moved = False
    while not moved:
        r = rng.random()
        if r < p_body:
            g = rng.choice([g for g, v in BODY_GENES.items() if len(v) > 1])
            n.body[g] = rng.choice(BODY_GENES[g])
        elif r < p_body + p_cut:
            k = rng.choice(BATCH_CUTTABLE if n.rows else CUTTABLE)
            if k in n.boundaries:
                n.boundaries.remove(k)
            else:
                n.boundaries.append(k)
        else:
            cls = n.class_list()
            m = rng.choice(cls)
            n.min_blocks[m] = rng.choice([1, 2, 3, 4])
        moved = n.compile_key() != c.compile_key()
    n.grids = {m: g for m, g in c.grids.items() if m in n.classes()}
    n.min_blocks = {m: b for m, b in n.min_blocks.items() if m in n.classes()}
    return n
