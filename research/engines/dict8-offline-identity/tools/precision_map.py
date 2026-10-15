#!/usr/bin/env python3
"""Precision map of every tensor in the Ternary Bonsai 27B hot loop, as measured in halo-5090.

Each row is one input or output of an operation. Marks, in bits per element (scales amortised):
  green dot   tested and passes the quality gate (GSM8K / KLD vs fork / drafter acceptance)
  red x       tested and fails
  grey square what the engine uses where nothing narrower was tried (the narrowest safe width is unknown)
  black bar   information floor where one is known
Writes results/precision-map.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OK, FAIL, UNT = "ok", "fail", "untested"
# (group, tensor, [(bits, status, label)], floor or None)
rows = [
    ("weights", "Projection weights (in/out/gate-up/down)", [(2.0, OK, "2-bit trits + FP16/128 (stored)"), (1.625, OK, "trits + E4M3/128 scale: GSM8K 94.0"), (1.24, FAIL, "post-hoc pair-4:8 sparse: GSM8K 0%")], 1.56),
    ("weights", "Embedding and LM head weights", [(1.75, OK, "PTQ1_0 ternary")], 1.56),
    ("weights", "Drafter (DFlash2) weights", [(4.5, OK, "NVFP4: acceptance unchanged")], None),
    ("weights", "alpha/beta projection", [(4.5, OK, "rotated NVFP4"), (16, OK, "bf16 (exact control)")], None),
    ("activations", "GEMM input activations", [(4.5, OK, "NVFP4: KLD .005-.01, GSM8K = fork"), (8, OK, "INT8: KLD 1e-4 (control)")], None),
    ("activations", "Residual stream x", [(32, UNT, "fp32")], None),
    ("activations", "Projection outputs", [(32, UNT, "fp32")], None),
    ("activations", "MMA accumulator (per 16-K block)", [(32, UNT, "fp32"), (12, OK, "exact integer sum needs ~12 bits")], None),
    ("activations", "Norm / silu / softmax / RoPE math", [(32, UNT, "fp32")], None),
    ("memory", "Attention KV cache", [(8.125, OK, "FP8 E4M3 + scale per (pos, head)"), (16, OK, "fp16 (control)")], None),
    ("memory", "Attention q and probabilities", [(16, UNT, "fp16 MMA operands")], None),
    ("memory", "GDN state S (128x128 per head)", [(16, OK, "bf16: GSM8K = fork"), (32, OK, "fp32 (exact control)")], None),
    ("memory", "GDN pending (verify replay) cache", [(16, OK, "bf16")], None),
    ("memory", "Conv state", [(32, UNT, "fp32")], None),
    ("memory", "Drafter KV ring", [(16, UNT, "fp16")], None),
    ("output", "Logits (argmax only)", [(32, UNT, "fp32 in GEMM epilogue")], None),
]

OFFS = {"post-hoc pair-4:8 sparse: GSM8K 0%": (-4, -15), "trits + E4M3/128 scale: GSM8K 94.0": (0, 10), "2-bit trits + FP16/128 (stored)": (8, -15),
        "PTQ1_0 ternary": (10, -3)}
colors = {"weights": "#7a1f2b", "activations": "#1f4e7a", "memory": "#3c6b3c", "output": "#555555"}
fig, ax = plt.subplots(figsize=(13, 9.5), dpi=150)
ax.set_xscale("log", base=2)
ticks = [1, 1.585, 2, 4, 4.5, 8, 12, 16, 32]
ax.set_xticks(ticks); ax.set_xticklabels(["1", "1.58\n(log2 3)", "2", "4", "4.5\n(NVFP4)", "8", "12", "16", "32"])
ax.set_xlim(0.9, 48)
n = len(rows)
for i, (grp, name, marks, floor) in enumerate(rows):
    y = n - 1 - i
    ax.axhspan(y - 0.5, y + 0.5, color=colors[grp], alpha=0.06, lw=0)
    if floor:
        ax.plot([floor, floor], [y - 0.38, y + 0.38], color="black", lw=2.5)
    # line from narrowest passing mark to widest mark, to show the span tried
    bs = sorted(b for b, _, _ in marks)
    ax.plot([bs[0], bs[-1]], [y, y], color="#999999", lw=1, zorder=1)
    for j, (b, st, lab) in enumerate(sorted(marks, key=lambda m: m[0])):
        if st == OK:
            ax.scatter(b, y, s=90, color="#2e8b3e", zorder=3, edgecolor="white")
        elif st == FAIL:
            ax.scatter(b, y, s=110, color="#c0392b", marker="x", zorder=3, linewidths=2.5)
        else:
            ax.scatter(b, y, s=80, color="#9a9a9a", marker="s", zorder=3, edgecolor="white")
        off = OFFS.get(lab, (0, 9 if j % 2 == 0 else -15))
        ax.annotate(lab, (b, y), xytext=off, textcoords="offset points", ha="center" if off[0] == 0 else ("left" if off[0] > 0 else "right"), fontsize=7.2, color="#333333")
ax.set_yticks(range(n)); ax.set_yticklabels([r[1] for r in rows][::-1], fontsize=9)
for tl, r in zip(ax.get_yticklabels(), rows[::-1]):
    tl.set_color(colors[r[0]])
ax.set_ylim(-0.6, n - 0.4)
ax.grid(axis="x", color="#dddddd", lw=0.6)
ax.set_xlabel("bits per element (block scales amortised)")
ax.set_title("Ternary Bonsai 2 27B on halo-5090: what precision each tensor needs, as measured", fontsize=12, loc="left")
from matplotlib.lines import Line2D
leg = [Line2D([], [], marker="o", color="w", markerfacecolor="#2e8b3e", markersize=9, label="tested, passes quality gate"),
       Line2D([], [], marker="x", color="#c0392b", lw=0, markersize=9, markeredgewidth=2.5, label="tested, fails"),
       Line2D([], [], marker="s", color="w", markerfacecolor="#9a9a9a", markersize=8, label="what we use; nothing narrower tried"),
       Line2D([], [], color="black", lw=2.5, label="information floor (86/128 fixed-count trits)")]
ax.legend(handles=leg, loc="lower left", fontsize=8.5, framealpha=0.95)
for grp, c in colors.items():
    pass
fig.text(0.01, 0.005, "Row colours: weights (red), activations (blue), per-stream memory (green), output (grey). Grey rows are open territory: the true minimum is unknown, not 32 bits.", fontsize=8, color="#444444")
plt.tight_layout(rect=(0, 0.02, 1, 1))
plt.savefig(__file__.rsplit("/", 2)[0] + "/results/precision-map.png")
print("ok")
