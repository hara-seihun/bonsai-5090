#!/usr/bin/env python3
"""Speed vs cost when trading per-stream speed for batch size: Ternary Bonsai 2 27B, "everything works" model.

Step time for S streams = max(memory, compute) + latency floor, where
  memory  = weights (5.3 GB, information floor) + drafter + S x per-stream state/KV bytes
  compute = S x rows_per_stream x 27e9 MAC x 1.1 (drafter) / (effective FP4 MAC rate)
Each step yields S x accepted tokens. Assumptions are printed on the chart. Modelled, not measured,
except the "measured today" markers. Writes results/speed-cost-curve.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W, DRAFT = 5.3e9, 0.5e9
MAC = 27e9 * 1.1
ROWS, ACC = 7, 6                       # verify rows per stream per step, tokens accepted
CTX = 4096
STATE = 2 * 38e6                       # GDN state at ~4 bits (optimal store), read + write
KV = 16 * CTX * 2 * 1024 * 0.5         # 16 attention layers, K+V, 4 KV heads x 256, ~4-bit KV
PER = STATE + KV
cards = {
    "RTX 5090 ($0.69/h)": dict(bw=1.79e12, mac=1.04e15 * 2, lat=0.5e-3, price=0.69, vram=32e9, c="#c0392b"),
    "B200 ($5.00/h)": dict(bw=8.0e12, mac=4.5e15 * 2, lat=0.3e-3, price=5.00, vram=180e9, c="#1f4e9a"),
}
EFF_BW, EFF_MAC = 0.85, 0.75           # achieved fraction of peak
fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
for name, c in cards.items():
    smax = int((c["vram"] * 0.9 - W - DRAFT) / (PER / 2 * 1.2))   # resident bytes ~ half of traffic (state read+write), 20% slack
    xs, ys, lab = [], [], []
    S = 1
    while S <= smax:
        mem = (W + DRAFT + S * PER) / (c["bw"] * EFF_BW)
        cmp = S * ROWS * MAC / (c["mac"] * EFF_MAC)
        t = max(mem, cmp) + c["lat"]
        per_stream = ACC / t
        agg = S * per_stream
        cents = c["price"] * 1e6 / 3600 / agg * 100
        xs.append(per_stream); ys.append(cents); lab.append(S)
        S = S * 2 if S < 8 else int(S * 1.5)
    ax.plot(xs, ys, "-o", color=c["c"], label=f"{name}, up to {smax} streams at 4k", ms=4)
    for x, y, s in zip(xs, ys, lab):
        if s in (1, 4, 16, 64, 256) or s == lab[-1]:
            ax.annotate(f"{s}", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, color=c["c"])
# measured today, 5090 ($0.69/h basis)
meas = [(381, 0.69e6 / 3600 / 361 * 100, "1"), (233, 0.69e6 / 3600 / 1580 * 100, "8"), (126, 0.69e6 / 3600 / 3357 * 100, "32"), (78, 0.69e6 / 3600 / 4259 * 100, "64")]
ax.plot([m[0] for m in meas], [m[1] for m in meas], "x--", color="#555555", label="halo-5090 measured today (server, streaming)")
for x, y, s in meas:
    ax.annotate(s, (x, y), xytext=(5, -10), textcoords="offset points", fontsize=8, color="#555555")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("tokens per second, per stream (what one user sees)")
ax.set_ylabel("cost, ¢ per million output tokens (card rental only)")
ax.set_title("Ternary Bonsai 2 27B: per-stream speed vs cost as batch size grows (labels = concurrent streams)", loc="left", fontsize=11)
ax.grid(True, which="both", color="#e3e3e3", lw=0.6)
ax.legend(fontsize=9, loc="center right")
ax.text(0.01, 0.955, "Modelled 'everything works': weights 5.3 GB (1.56 b), GDN state ~4 b, KV ~4 b at 4k context, speculation 7 rows -> 6 tokens,\n"
        "2x effective FP4 rate (fibres or trained 4:8 sparsity), 85% bandwidth / 75% MAC efficiency, 0.5 / 0.3 ms per-step latency floor.",
        transform=ax.transAxes, fontsize=7.5, color="#444444")
plt.tight_layout()
plt.savefig(__file__.rsplit("/", 2)[0] + "/results/speed-cost-curve.png")
for name, c in cards.items():
    print(name)
    for S in (1, 4, 16, 64, 128, 256, 512, 1024):
        mem = (W + DRAFT + S * PER) / (c["bw"] * EFF_BW); cmp = S * ROWS * MAC / (c["mac"] * EFF_MAC); t = max(mem, cmp) + c["lat"]
        agg = S * ACC / t
        print(f"  S={S:5d} step {t*1e3:6.2f} ms ({'mem' if mem > cmp else 'mac'}) per-stream {ACC/t:7.0f} agg {agg:8.0f} tok/s  {c['price']*1e6/3600/agg*100:6.3f} c/M")
