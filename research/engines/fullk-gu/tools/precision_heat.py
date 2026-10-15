#!/usr/bin/env python3
"""Heat flowchart of one halo-5090 step: 64 streams x 4k context, 512 verify rows (64 x 8), Ternary Bonsai 2 27B.

  node fill   = GPU time per step (white -> yellow -> red), label gives ms and share
  edge colour = precision of the data on it (same palette as precision_flow.dot)
  edge width  = bytes moved per step on that edge, summed over all layers (computed from shapes)

Times: the 62.9 ms per-step breakdown measured at 64 x 4k (README, H5_PROF). Phases the profiler reports
as one total are split by the 64 x 500 profile's ratios (GEMMs) or by element count (prep); those carry "~".
Writes results/precision-heat.png.
"""
import math, subprocess

R, S, CTX = 512, 64, 4000             # verify rows, streams, context
D, FF, VOC = 5120, 17408, 248320
NG, NA = 48, 16                        # GDN and attention layers
TW = 2.125 / 8                         # ternary bytes per weight (2-bit trit + fp16 per 128)
NV = 0.5625                            # NVFP4 activation bytes per element (+ fp32 per row, negligible)
GB = 1e9

# ---- time per step (ms) at 64 x 4k
gemm = 24.5; r = {"in": 2.82, "out": 1.33, "gu": 6.22, "down": 2.87, "head": 0.77}; rs = sum(r.values())
T = {k: gemm * v / rs for k, v in r.items()}
prep = 7.0; el = {"p1": 64 * D, "pg": NG * 6144, "pa": NA * 6144, "p2": 64 * D, "p3": 64 * FF, "pf": D}; es = sum(el.values())
T.update({k: prep * v / es for k, v in el.items()})
T.update({"gdn": 10.1, "attn": 9.3, "attnpre": 0.35, "comb": 0.35, "conv": 2.3, "ab": 0.5, "embed": 0.05, "draft": 7.7})
TOT = sum(T.values())
approx = {"in", "out", "gu", "down", "head", "p1", "pg", "pa", "p2", "p3", "pf", "attn", "attnpre", "comb", "ab"}

def heat(ms):
    f = min(1.0, math.sqrt(ms / 11.0))
    # white -> yellow -> red
    if f < 0.5:
        t = f / 0.5; c = (255, 255, int(255 - t * 155))
    else:
        t = (f - 0.5) / 0.5; c = (255, int(255 - t * 175), int(100 - t * 60))
    return "#%02x%02x%02x" % c

def tlab(k):
    ms = T[k]; return ("~" if k in approx else "") + f"{ms:.1f} ms · {100 * ms / TOT:.0f}%"

# ---- bytes per step on each edge
B = {
    "x_p1": R * D * 4 * 64, "p1_in": R * D * NV * 64,
    "W_in": (16384 * NG + 14336 * NA) * D * TW,
    "in_qkv": R * 10240 * 4 * NG, "in_z": R * 6144 * 4 * NG, "in_att": R * 14336 * 4 * NA,
    "p1_ab": R * D * 4 * NG, "W_ab": 96 * D * 0.5625 * NG, "ab_conv": R * 96 * 4 * NG,
    "cst": S * 3 * 10240 * 4 * NG * 2, "conv_gdn": R * 10240 * 4 * NG,
    "S": S * NG * 128 * 128 * 2 * NG * 2 / NG * 1,   # 48 heads x 128x128 bf16 per stream per layer, read + write
    "pend": R * (2048 + 6144 + 96) * 2 * NG * 2, "gdn_o": R * 6144 * 4 * NG,
    "kvw": R * 2 * 1024 * 1 * NA, "kvr": S * CTX * 2 * 1024 * 1 * NA, "q": R * 6144 * 2 * NA,
    "part": R * 6144 * 4 * 4 * NA, "comb": R * 6144 * 4 * NA,
    "py_out": R * 6144 * NV * 64, "W_out": D * 6144 * TW * 64, "out_add": R * D * 4 * 64,
    "add_p2": R * D * 4 * 64, "p2_gu": R * D * NV * 64, "W_gu": 2 * FF * D * TW * 64, "gu_h": R * FF * 4 * 64,
    "p3_d": R * FF * NV * 64, "W_d": D * FF * TW * 64, "d_add": R * D * 4 * 64, "res": R * D * 4 * 64,
    "pf_h": R * D * NV, "W_h": VOC * D * TW, "ids": R * 4,
    "cap": R * 5 * D * 4, "W_df": 0.9e9, "ring": S * 1056 * 5 * 2 * 1024 * 2,
}
B["S"] = S * NG * 128 * 128 * 2 * NG * 2   # per stream per layer: 48 heads x 128 x 128 bf16; read and write
def w(k):
    b = B[k]; return max(1.0, min(14.0, 1.6 * math.log2(1 + b / 5e7)))
def bl(k):
    b = B[k]; return f"{b / GB:.2f} GB" if b >= 1e8 else (f"{b / 1e6:.0f} MB" if b >= 1e6 else f"{b / 1e3:.0f} KB")

C = {"fp32": "#1f4e9a", "bf16": "#2e9e8a", "fp16": "#8e44ad", "fp8": "#e67e22", "nvfp4": "#d62728", "tern": "#111111", "int": "#7f7f7f"}
def e(a, b, k, prec, extra=""):
    st = ', style=dashed' if prec == "tern" or k in ("W_ab", "W_df") else ""
    return f'  {a} -> {b} [color="{C[prec]}", penwidth={w(k):.1f}, label=" {bl(k)}"{st}{extra}];\n'
def n(name, title, k=None, shape="box", extra=""):
    fill = heat(T[k]) if k else "#f4f4f4"
    lab = title + (f"\\n<{tlab(k)}>" if k else "")
    lab = lab.replace("<", "").replace(">", "")
    return f'  {name} [label="{lab}", fillcolor="{fill}", shape={shape}{extra}];\n'
def store(name, title, fill):
    return f'  {name} [label="{title}", shape=cylinder, fillcolor="{fill}"];\n'
def wnode(name, title):
    return f'  {name} [label="{title}", shape=note, fillcolor="#e6e6e6"];\n'

g = []
g.append('digraph G {\n  graph [rankdir=TB, fontname="Helvetica", nodesep=0.35, ranksep=0.45, labelloc=t, fontsize=16,\n'
         f'    label="halo-5090, one step at 64 streams x 4k context, 512 verify rows — fill = GPU time ({TOT:.0f} ms/step), line width = bytes moved per step, line colour = precision"];\n'
         '  node [style="rounded,filled", color="#888888", fontname="Helvetica", fontsize=11];\n  edge [arrowsize=0.7, fontname="Helvetica", fontsize=9, fontcolor="#333333"];\n')
legend = ''.join(f'<TR><TD BGCOLOR="{C[k]}" WIDTH="40" HEIGHT="6"></TD><TD ALIGN="LEFT">{t}</TD></TR>' for k, t in [
    ("fp32", "fp32"), ("bf16", "bf16"), ("fp16", "fp16"), ("fp8", "fp8 E4M3"), ("nvfp4", "NVFP4 (dashed: NVFP4 weights)"), ("tern", "dashed: ternary weights ≈ 2.1 b"), ("int", "int32 token ids")])
heatrow = ''.join(f'<TD BGCOLOR="{heat(v)}" WIDTH="34">{v:g}</TD>' for v in [0.1, 0.5, 1, 2, 4, 7, 11])
g.append(f'  legend [shape=plaintext, style="", label=<<TABLE BORDER="1" COLOR="#bbbbbb" CELLBORDER="0" CELLSPACING="3"><TR><TD COLSPAN="2"><B>line colour = precision</B></TD></TR>{legend}'
         f'<TR><TD COLSPAN="2"><B>fill = ms per step</B></TD></TR><TR><TD COLSPAN="2"><TABLE BORDER="0" CELLSPACING="1"><TR>{heatrow}</TR></TABLE></TD></TR>'
         '<TR><TD COLSPAN="2">~ = measured phase total split by ratio</TD></TR></TABLE>>];\n')
g.append(n("tok", "token ids", shape="oval") + n("emb", "embedding lookup", "embed") + store("x", "residual stream x\\n512 rows x 5120", "#dfe8f6"))
g.append('  legend -> tok [style=invis];\n')
g.append(e("tok", "emb", "ids", "int") + f'  emb -> x [color="{C["fp32"]}", penwidth=1.5];\n')
g.append('  subgraph cluster_layer {\n    label="x 64 layers (48 Gated DeltaNet, 16 attention)"; fontsize=13; color="#999999"; style="rounded,dashed";\n')
g.append('  subgraph cluster_in { label="mixer input"; color="#bbbbbb"; style=rounded;\n' + n("p1", "RMSNorm · Hadamard → NVFP4", "p1") +
         n("win", "input projection GEMM\\n512 rows x 16k outputs, all independent", "in") + wnode("Win", "W_in") + '  }\n')
g.append(e("x", "p1", "x_p1", "fp32") + e("p1", "win", "p1_in", "nvfp4") + e("Win", "win", "W_in", "tern"))
g.append('  subgraph cluster_gdn { label="Gated DeltaNet (48 layers)"; color="#2e9e8a"; style=rounded;\n' +
         n("ab", "α/β projection", "ab") + wnode("Wab", "W_αβ") + n("conv", "conv · silu · q/k norm · gates", "conv") + store("cst", "conv state", "#dfe8f6") +
         n("gdn", "delta-rule recurrence\\n64 streams x 48 heads in parallel,\\n8 rows of a stream in sequence", "gdn") +
         store("S", "state S, bf16\\n151 MB per stream", "#dff2ee") + store("pend", "pending replay cache", "#dff2ee") +
         n("pg", "gated RMSNorm · Hadamard → NVFP4", "pg") + '  }\n')
g.append(e("p1", "ab", "p1_ab", "fp32") + e("Wab", "ab", "W_ab", "nvfp4") + e("ab", "conv", "ab_conv", "fp32") + e("win", "conv", "in_qkv", "fp32", ', xlabel="qkv"') +
         e("cst", "conv", "cst", "fp32", ", dir=both") + e("conv", "gdn", "conv_gdn", "fp32") + e("S", "gdn", "S", "bf16", ", dir=both") +
         e("gdn", "pend", "pend", "bf16", ", dir=both") + e("gdn", "pg", "gdn_o", "fp32") + e("win", "pg", "in_z", "fp32", ', xlabel="z"'))
g.append('  subgraph cluster_attn { label="attention (16 layers)"; color="#e67e22"; style=rounded;\n' +
         n("qk", "q/k norm · RoPE · FP8 k, v", "attnpre") + store("kv", "KV cache, fp8\\n133 MB per stream at 4k", "#fdebd8") +
         n("fa", "split-KV flash attention\\n64 streams x 4 KV heads x splits", "attn") + n("comb", "combine splits", "comb") +
         n("pa", "Hadamard → NVFP4", "pa") + '  }\n')
g.append(e("win", "qk", "in_att", "fp32") + e("qk", "kv", "kvw", "fp8") + e("kv", "fa", "kvr", "fp8") + e("qk", "fa", "q", "fp16") +
         e("fa", "comb", "part", "fp32") + e("comb", "pa", "comb", "fp32"))
g.append('  subgraph cluster_out { label="mixer output"; color="#bbbbbb"; style=rounded;\n' + n("wout", "output projection GEMM", "out") + wnode("Wout", "W_out") +
         '  add1 [label="+", shape=circle, fillcolor="#dfe8f6", width=0.35];\n  }\n')
g.append(f'  pg -> wout [color="{C["nvfp4"]}", penwidth={w("py_out"):.1f}, label=" {bl("py_out")} (both kinds)"];\n' + f'  pa -> wout [color="{C["nvfp4"]}", penwidth=1.5];\n' +
         e("Wout", "wout", "W_out", "tern") + e("wout", "add1", "out_add", "fp32"))
g.append('  subgraph cluster_mlp { label="MLP"; color="#bbbbbb"; style=rounded;\n' + n("p2", "RMSNorm · Hadamard → NVFP4", "p2") +
         n("gu", "gate/up GEMM  (5120 → 2 x 17408)\\nepilogue silu(gate)·up", "gu") + wnode("Wgu", "W_gate,up") + n("p3", "Hadamard (17408) → NVFP4", "p3") +
         n("dn", "down GEMM  (17408 → 5120)", "down") + wnode("Wd", "W_down") + '  add2 [label="+", shape=circle, fillcolor="#dfe8f6", width=0.35];\n  }\n')
g.append(e("add1", "p2", "add_p2", "fp32") + e("p2", "gu", "p2_gu", "nvfp4") + e("Wgu", "gu", "W_gu", "tern") + e("gu", "p3", "gu_h", "fp32") +
         e("p3", "dn", "p3_d", "nvfp4") + e("Wd", "dn", "W_d", "tern") + e("dn", "add2", "d_add", "fp32") + e("add1", "add2", "res", "fp32", ", constraint=false"))
g.append('  }\n')
g.append(e("x", "add1", "res", "fp32", ", constraint=false") + f'  add2 -> x [color="{C["fp32"]}", penwidth=2, constraint=false, label=" next layer"];\n')
g.append(n("pf", "final RMSNorm → NVFP4", "pf") + n("head", "LM head GEMM  (5120 → 248,320)\\nargmax epilogue", "head") + wnode("Wh", "W_head") +
         '  out [label="target tokens (verify / accept)", shape=oval, fillcolor="#eeeeee"];\n')
g.append(f'  add2 -> pf [color="{C["fp32"]}", penwidth=2];\n' + e("pf", "head", "pf_h", "nvfp4") + e("Wh", "head", "W_h", "tern") + e("head", "out", "ids", "int"))
g.append('  subgraph cluster_df { label="DFlash2 drafter"; color="#8e44ad"; style=rounded;\n' + n("df", "5 drafter layers · head\\ntop-16 · selector", "draft") +
         store("ring", "drafter KV ring, fp16", "#efe3f6") + wnode("Wdf", "drafter W (NVFP4)") + '  }\n')
g.append(e("add2", "df", "cap", "fp32", ', constraint=false, xlabel="captures"') + e("ring", "df", "ring", "fp16", ", dir=both") + e("Wdf", "df", "W_df", "nvfp4") +
         f'  df -> tok [color="{C["int"]}", penwidth=1.5, constraint=false, label=" 7 drafts / stream"];\n  out -> df [color="{C["int"]}", penwidth=1.5, constraint=false];\n')
g.append('}\n')
root = __file__.rsplit("/", 2)[0]
open(root + "/results/precision-heat.dot", "w").write("".join(g))
subprocess.run(["dot", "-Tpng", "-Gdpi=130", root + "/results/precision-heat.dot", "-o", root + "/results/precision-heat.png"], check=True)
print(f"total {TOT:.1f} ms; biggest edges:", sorted(((v / GB, k) for k, v in B.items()), reverse=True)[:6])
