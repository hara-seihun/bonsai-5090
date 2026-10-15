#!/usr/bin/env python3
"""Modelled decode rooflines for two Flash architectures; see flash_heat.md.

Run: python3 tools/flash_heat.py [deepseek|glm|all]
No online access or Python packages required; Graphviz `dot` renders the charts.
Edges are decimal bytes per step, aggregated across the labelled layers. Node
traffic is the sum of its incident edges (read plus write), not parameter count.
"""
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
S, CTX, R = 64, 4000, 256
NV, TRIT, FP8 = 0.5625, 0.265625, 1.015625
COL = {'fp32': '#1f4e9a', 'bf16': '#2e9e8a', 'fp16': '#8e44ad',
       'fp8': '#e67e22', 'nvfp4': '#d62728', 'tern': '#111111', 'int': '#7f7f7f'}


def coverage(rows, topk, experts):
    return experts * (1 - (1 - topk / experts) ** rows)


def size(b):
    return f'{b / 1e9:.2f} GB' if b >= 1e8 else f'{b / 1e6:.1f} MB' if b >= 1e6 else f'{b / 1e3:.1f} KB'


def heat(ms):
    f = min(1, math.sqrt(ms / 11))
    if f < .5:
        t = f * 2
        rgb = (255, 255, int(255 - t * 155))
    else:
        t = 2 * f - 1
        rgb = (255, int(255 - t * 175), int(100 - t * 60))
    return '#%02x%02x%02x' % rgb


class Chart:
    def __init__(self, slug, title, gpus, d, layers):
        self.slug, self.title, self.gpus, self.d, self.layers = slug, title, gpus, d, layers
        self.nodes, self.edges, self.sections = {}, [], []
        self.resident_weights = self.resident_cache = 0
        self.notes = []

    def node(self, key, label, section='', kind='op', flops=0, rate='vector'):
        assert key not in self.nodes, key
        self.nodes[key] = dict(label=label, section=section, kind=kind, flops=flops, rate=rate, bytes=0)
        if section and section not in self.sections:
            self.sections.append(section)
        return key

    def edge(self, a, b, count, prec='fp32', label='', both=False, back=False):
        assert count >= 0
        self.edges.append(dict(source=a, target=b, bytes=count, precision=prec,
                               label=label, both=both, back=back))
        # A bidirectional store edge already carries read+write, counted once.
        for key in (a, b):
            if self.nodes[key]['kind'] == 'op':
                self.nodes[key]['bytes'] += count

    def store(self, key, label, resident, section='', prec='fp8'):
        self.resident_cache += resident
        return self.node(key, label + '\nresident ' + size(resident), section, 'store')

    def gemm(self, key, label, source, rows, kin, kout, layers=1, groups=1,
             prec='nvfp4', touched=None, active=None, section=''):
        """groups independent equal matrices, rows per group for headwise GEMMs.

        MoE: groups=E, touched=expected E touched, active=top-k. Dispatch rows
        replicate top-k times, but weights are read once per touched expert.
        FP32 between GEMMs, explicit norm/quant packs before EVERY GEMM.
        """
        p = kin * kout * layers * groups
        wbytes = {'nvfp4': NV, 'tern': TRIT, 'fp8': FP8}[prec]
        reads = p * wbytes if touched is None else kin * kout * layers * touched * wbytes
        copies = groups if active is None else active
        in_elements = rows * kin * layers * copies
        out_elements = rows * kout * layers * copies
        prep = self.node(key + '_pack', 'norm / pack -> NVFP4', section, flops=in_elements * 12)
        self.edge(source, prep, in_elements * 4, label='dispatch top-k' if active else '')
        self.node(key, label, section, flops=2 * rows * kin * kout * layers * copies, rate='fp4')
        self.edge(prep, key, in_elements * NV + rows * layers * copies * 4, 'nvfp4')
        self.node(key + '_W', 'W: ' + ('2-bit trits + fp16 / 128' if prec == 'tern' else prec.upper()), section, 'weight')
        self.edge(key + '_W', key, reads, prec, label='expected touched weights' if touched else '')
        self.resident_weights += p * wbytes
        return key, out_elements * 4

    def output(self, result, key, label, section='', flops=0):
        source, count = result
        self.node(key, label, section, flops=flops)
        self.edge(source, key, count)
        return key

    def mlp(self, prefix, source, rows, ff, layers, experts=1, topk=1, tern=False, section=''):
        touched = coverage(rows, topk, experts) if experts > 1 else None
        gu = self.gemm(prefix + '_gu', 'gate/up GEMM\n' + f'{layers} layers; D -> 2 x {ff}',
                       source, rows, self.d, 2 * ff, layers, experts,
                       'tern' if tern else 'nvfp4', touched, topk if experts > 1 else None, section)
        act = self.output(gu, prefix + '_act', 'clamp / SiLU(gate) * up', section,
                          rows * ff * layers * topk * 12)
        # Output of gate/up comprises two FF arrays; activation produces one.
        down = self.gemm(prefix + '_down', 'down GEMM\n' + f'{ff} -> {self.d}',
                         act, rows, ff, self.d, layers, experts,
                         'tern' if tern else 'nvfp4', touched, topk if experts > 1 else None, section)
        return down

    def finish(self):
        # Peak dense (not 2:4 sparse) rates. This is an optimistic hardware floor.
        rates = {'fp4': 9e15, 'bf16': 2.25e15, 'vector': 80e12, 'network': 80e12}
        for n in self.nodes.values():
            bandwidth = self.gpus * (900e9 if n['rate'] == 'network' else 8e12)
            n['memory_ms'] = n['bytes'] / bandwidth * 1000
            n['compute_ms'] = n['flops'] / (self.gpus * rates[n['rate']]) * 1000
            n['ms'] = max(n['memory_ms'], n['compute_ms']) if n['kind'] == 'op' else 0
        total = sum(n['ms'] for n in self.nodes.values())
        topnodes = sorted(((n['ms'], k, n['label']) for k, n in self.nodes.items()), reverse=True)[:3]
        topedges = sorted(self.edges, key=lambda e: e['bytes'], reverse=True)[:3]
        report = dict(model=self.title, serving=dict(streams=S, context=CTX, verify_rows=R,
                      accepted_tokens_assumed=4), gpus=self.gpus, hardware='B200 SXM, 180 GB each',
                      aggregate_hbm_bytes_per_s=self.gpus * 8e12, dense_fp4_flops_per_s=self.gpus * 9e15,
                      resident_weights_GB=self.resident_weights / 1e9,
                      resident_cache_GB=self.resident_cache / 1e9, sequential_roofline_ms=total,
                      notes=self.notes, hottest_nodes=topnodes, fattest_edges=topedges,
                      nodes=self.nodes, edges=self.edges)
        out = ROOT / 'results'
        out.mkdir(exist_ok=True)
        (out / (self.slug + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        def quote(text):
            return json.dumps(text).replace('\\n', '\\n')
        target = 'NVLink-connected' if self.gpus > 1 else 'single GPU'
        title = (f'{self.title} | MODELLED on {self.gpus} x B200 (180 GB/GPU, {target})\n'
                 f'64 streams x 4k context, 256 verify rows; forced ~4 accepted/stream/step\n'
                 f'fill = roofline ms ({total:.1f} ms sequential floor, NOT measured); width = bytes/step; colour = precision\n'
                 f'weights {self.resident_weights/1e9:.1f} GB + caches {self.resident_cache/1e9:.1f} GB; '
                 'NVFP4 GEMM inputs / FP32 outputs; FP8 KV\n'
                 '8 TB/s, 9 PFLOP/s dense FP4, 2.25 PFLOP/s BF16, 80 TFLOP/s FP32 per GPU\n'
                 + '\n'.join(self.notes))
        g = ['digraph G {', 'graph [rankdir=TB, nodesep=.3, ranksep=.4, fontname="Helvetica", fontsize=15, labelloc=t, label=' + quote(title) + '];',
             'node [style="rounded,filled", fontname="Helvetica", fontsize=10, color="#888888"];',
             'edge [fontname="Helvetica", fontsize=9, arrowsize=.65];']
        legend = ''.join(f'<TR><TD BGCOLOR="{c}" WIDTH="35"></TD><TD ALIGN="LEFT">{p}</TD></TR>' for p, c in COL.items())
        hr = ''.join(f'<TD BGCOLOR="{heat(v)}">{v:g}</TD>' for v in [.1, .5, 1, 2, 4, 7, 11])
        g.append('legend [shape=plaintext, style="", label=<<TABLE BORDER="1" CELLBORDER="0">'
                 '<TR><TD COLSPAN="2"><B>edge precision</B></TD></TR>' + legend +
                 '<TR><TD COLSPAN="2">dashed = weight reads</TD></TR><TR><TD COLSPAN="2"><B>fill = modelled ms</B></TD></TR>'
                 '<TR><TD COLSPAN="2"><TABLE BORDER="0"><TR>' + hr + '</TR></TABLE></TD></TR>'
                 '<TR><TD COLSPAN="2">width = 1.6 log2(1 + bytes/50 MB), clipped 1..14</TD></TR>'
                 '<TR><TD COLSPAN="2">ASSUMED = execution choice / unpublished shape</TD></TR></TABLE>>];')
        def render(key):
            n = self.nodes[key]
            label = n['label']
            if n['kind'] == 'op':
                label += f'\nMODELLED {n["ms"]:.2f} ms | {100*n["ms"]/total:.0f}%'
            shape = {'op': 'box', 'store': 'cylinder', 'weight': 'note', 'meta': 'oval'}[n['kind']]
            fill = heat(n['ms']) if n['kind'] == 'op' else '#eeeeee'
            return f'{key} [shape={shape}, label={quote(label)}, fillcolor="{fill}"];'
        for key, n in self.nodes.items():
            if not n['section']:
                g.append(render(key))
        for i, section in enumerate(self.sections):
            g.append(f'subgraph cluster_{i} {{ label={quote(section)}; style="rounded,dashed"; color="#aaaaaa";')
            g.extend(render(k) for k, n in self.nodes.items() if n['section'] == section)
            g.append('}')
        g.append('legend -> tok [style=invis];')
        for e in self.edges:
            width = max(1, min(14, 1.6 * math.log2(1 + e['bytes'] / 5e7)))
            p = e['precision']
            attrs = [f'color="{COL[p]}"', f'penwidth={width:.2f}', 'label=' + quote(size(e['bytes']) + (' | ' + e['label'] if e['label'] else ''))]
            if self.nodes[e['source']]['kind'] == 'weight':
                attrs.append('style=dashed')
            if e['both']:
                attrs.append('dir=both')
            if e['back']:
                attrs.append('constraint=false')
            g.append(f'{e["source"]} -> {e["target"]} [' + ','.join(attrs) + '];')
        g.append('}')
        dot = out / (self.slug + '.dot')
        dot.write_text('\n'.join(g) + '\n')
        subprocess.run(['dot', '-Tpng', '-Gdpi=115', str(dot), '-o', str(out / (self.slug + '.png'))], check=True)
        print(json.dumps({k: report[k] for k in ['model', 'resident_weights_GB', 'resident_cache_GB', 'sequential_roofline_ms', 'hottest_nodes', 'fattest_edges']}, indent=2))


def start(c):
    c.node('tok', 'verify block token IDs (int32)', kind='meta')
    c.node('emb_W', 'token embedding table, FP8', kind='weight')
    c.resident_weights += c.vocab * c.d * FP8
    c.node('emb', 'embedding lookup -> 4 residual streams\nvision tower: prefill only, excluded')
    c.edge('tok', 'emb', R * 4, 'int')
    c.edge('emb_W', 'emb', R * c.d * FP8, 'fp8')
    c.node('x', f'mHC residual stream\n{R} x 4 x {c.d}', kind='meta')
    c.edge('emb', 'x', R * 4 * c.d * 4)
    c.node('mhc', f'mHC mix / residual update / Sinkhorn\n4 streams; {c.layers} layers x 2 sublayers',
           'residual mixing', flops=R * c.layers * c.d * 4 * 2 * 40)
    c.edge('x', 'mhc', R * 4 * c.d * 4 * c.layers * 2 * 2, label='read + write residual')
    return 'mhc'


def moe(c, source, layers, experts, topk, ff):
    section = f'MoE x {layers}: {experts} routed / top-{topk} + 1 shared'
    touched = coverage(R, topk, experts)
    c.notes.append(f'Uniform independent routing ASSUMED: {touched:.1f}/{experts} experts touched ({touched/experts:.1%}); correlated routes may differ')
    router = c.gemm('router', f'router GEMM D -> {experts}\nNVFP4 weights; FP32 top-k / probabilities', source, R, c.d, experts, layers, section=section)
    select = c.output(router, 'select', f'top-{topk} / dispatch / weighted reduce', section, R * experts * layers * 20)
    routed = c.mlp('routed', source, R, ff, layers, experts, topk, True, section)
    c.edge(select, 'routed_gu_pack', R * topk * layers * 4, 'int', label='expert IDs')
    c.edge(select, 'routed_down', R * topk * layers * 4, 'fp32', label='routing probabilities', back=True)
    shared = c.mlp('shared', source, R, ff, layers, section=section)
    c.node('moe_add', 'sum shared + weighted routed + residual', section, flops=R * c.d * layers * (topk + 2))
    c.edge(routed[0], 'moe_add', routed[1])
    c.edge(shared[0], 'moe_add', shared[1])
    c.edge(source, 'moe_add', R * c.d * layers * 4, label='residual')
    return 'moe_add'


def head(c, source):
    h = c.gemm('head', f'LM head GEMM D -> {c.vocab}\nNVFP4 weights; FP32 logits', source, R, c.d, c.vocab)
    c.node('accept', 'greedy sample / verify / accept\n~4 accepted per stream ASSUMED')
    c.edge(h[0], 'accept', h[1])
    c.node('accepted', 'accepted tokens', kind='meta')
    c.edge('accept', 'accepted', R * 4, 'int')


def deepseek():
    c = Chart('heat-deepseek-v4.1-flash', 'DeepSeek V4.1 Flash: 20 encoder + 20 decoder layers', 4, 5120, 40)
    c.vocab = 129280
    c.notes += ['CSA2/MQA: 64 x 512 Q heads, one 512-channel KV latent (K=V), not V3 MLA; SWA 128',
                '3 Full encoder KV sources (ratio 2), 1 Full decoder source (ratio 1 from encoder layer 19); 8 indexers',
                '196B Engram FP8 tables resident on GPU; 3-block DSpark (NOT conventional MTP); draft shapes marked ASSUMED',
                'Ideal four-way sharding; NVLink payload floor included, launch/dequant/poor MoE occupancy NOT included']
    source = start(c)
    sec = 'CSA2 attention x 40 (2 SWA-only, 18 encoder CSA2, 20 decoder CSA2)'
    low = c.gemm('q_low', 'Q low-rank GEMM\n5120 -> 1280', source, R, c.d, 1280, 40, section=sec)
    qlat = c.output(low, 'qlat', 'Q latent RMSNorm', sec, R * 1280 * 40 * 8)
    qup = c.gemm('q_up', 'Q expansion GEMM\n1280 -> 64 x 512', qlat, R, 1280, 64 * 512, 40, section=sec)
    q = c.output(qup, 'q_rope', 'Q norm / RoPE (64 rotary channels)', sec, R * 64 * 512 * 40 * 8)
    # Local SWA KV is layer-specific, global compressed KV is shared by 4 source layers.
    local = c.gemm('local', 'layer-local SWA KV GEMM\n5120 -> 512; K=V latent', source, R, c.d, 512, 40, section=sec)
    c.node('local_kpack', 'SWA KV norm / RoPE / FP8', sec)
    c.edge(local[0], 'local_kpack', local[1])
    local_bytes = S * 128 * 512 * FP8 * 40
    c.store('swa', 'SWA ring: per-layer, FP8', local_bytes, sec)
    c.edge('local_kpack', 'swa', R * 512 * FP8 * 40, 'fp8')
    c.node('enc_final', 'encoder layer 19 output\nsource of decoder global KV', sec, 'meta')
    c.edge(source, 'enc_final', R * c.d * 4, label='encoder boundary', back=True)
    compressor = c.gemm('global', 'global KV + compression-gate GEMMs\nASSUMED two D -> 512 maps / source', source, R, c.d, 2 * 512, 4, section=sec)
    c.edge('enc_final', 'global_pack', R * c.d * 4, label='decoder source', back=True)
    c.node('compress', 'compress + norm / FP8\n3 sources ratio 2, decoder ratio 1', sec)
    c.edge(compressor[0], 'compress', compressor[1])
    global_bytes = S * (3 * CTX / 2 + CTX) * 512 * FP8
    c.store('global_kv', 'shared global KV, FP8\n4 sources, not 38 independent caches', global_bytes, sec)
    c.edge('compress', 'global_kv', R * (3 / 2 + 1) * 512 * FP8, 'fp8')
    idxq = c.gemm('idxq', 'indexer Q + head-weight GEMMs\n1280 -> 32 x 128; 8 indexers', qlat, R, 1280, 32 * 128, 8, section=sec)
    idx = c.output(idxq, 'index', 'sparse index scores / top-512\nFull: 2,8,14,20; Reindex: 24,28,32,36\n4k < 16,384 candidate cap', sec,
                   R * (3 * CTX / 2 + 5 * CTX) * 32 * 128 * 2)
    c.nodes[idx]['rate'] = 'bf16'
    idxk = c.gemm('idxk', 'indexer K + compression gate GEMMs\nASSUMED 2 x D -> 128, 4 sources', source, R, c.d, 2 * 128, 4, section=sec)
    idxkp = c.output(idxk, 'idxkp', 'compress index K / FP8', sec)
    index_res = S * (3 * CTX / 2 + CTX) * 128 * FP8
    c.store('index_k', 'shared index K cache, FP8', index_res, sec)
    c.edge('idxkp', 'index_k', R * 2.5 * 128 * FP8, 'fp8')
    c.edge('index_k', idx, S * (3 * CTX / 2 + 5 * CTX) * 128 * FP8, 'fp8', label='scan once per stream/indexer')
    att = c.node('attention', 'CSA2 sparse + local attention\nFP8 KV -> BF16 arithmetic -> FP32\n512 selected + 128 local / row', sec,
                 flops=4 * R * 64 * 512 * (38 * 512 + 40 * 128), rate='bf16')
    c.edge(q, att, R * 64 * 512 * 40 * 4)
    c.edge(idx, att, R * 512 * 38 * 4, 'int', label='top-k reuse')
    c.edge('swa', att, local_bytes, 'fp8', label='4 queries share ring pass')
    union_global = S * (18 * min(CTX / 2, 4 * 512) + 20 * min(CTX, 4 * 512)) * 512 * FP8
    c.edge('global_kv', att, union_global, 'fp8', label='ASSUMED union of 4 selections; reread each layer')
    outlow = c.gemm('o_low', 'grouped output GEMM\n8 groups: 4096 -> 1024', att, R, 4096, 1024, 40, 8, section=sec)
    mid = c.output(outlow, 'o_mid', 'grouped output latent FP32', sec)
    out = c.gemm('o_up', 'output expansion GEMM\n8192 -> 5120', mid, R, 8192, c.d, 40, section=sec)
    mixed = c.output(out, 'mix_add', 'attention residual / mHC update', sec)
    # Engram: 3 orders x 8 heads x 256 features in each of two modules.
    es = 'Engram at layers 1 and 14 (sparse conditional memory)'
    c.node('engram_tables', '196B Engram embedding tables\nFP8; full tables resident, sparse reads only', es, 'weight')
    c.resident_weights += (384006168 + 384016682) * 256 * FP8
    c.node('engram_lookup', 'hash n-grams {2,3,4}\n8 heads x 256 / order -> 6144 features', es)
    c.edge('tok', 'engram_lookup', R * 4 * 4 * 2, 'int', label='4-token history', back=True)
    c.edge('engram_tables', 'engram_lookup', R * 6144 * 2 * FP8, 'fp8')
    ep = c.gemm('engram_proj', 'Engram key/value GEMMs\nASSUMED 6144 -> 2 x 5120; NVFP4 weights', 'engram_lookup', R, 6144, 2 * c.d, 2, section=es)
    eg = c.output(ep, 'engram_gate', 'context-aware gating / integration\nno short convolution', es, R * c.d * 2 * 20)
    c.edge(mixed, eg, R * c.d * 4 * 2, label='context', back=True)
    end = moe(c, mixed, 40, 384, 6, 2304)
    c.edge(eg, 'moe_add', R * c.d * 4 * 2, label='Engram branches', back=True)
    head(c, end)
    ds = 'native DSpark: 3 SWA Transformer blocks, 5 parallel positions'
    dr = S * 5
    cap = c.gemm('draft_join', 'DSpark feature projection\nASSUMED concat layers 37/38/39: 3D -> D', end, dr, 3 * c.d, c.d, section=ds)
    din = c.output(cap, 'draft_in', 'draft feature + token / noise embeddings', ds)
    dp = c.gemm('draft_q', 'DSpark attention projections\nASSUMED inherited Q low/up + grouped O', din, dr, c.d, 1280, 3, section=ds)
    dlat = c.output(dp, 'draft_lat', 'draft Q latent', ds)
    dq = c.gemm('draft_qup', 'draft Q expansion\n1280 -> 64 x 512', dlat, dr, 1280, 64 * 512, 3, section=ds)
    dqq = c.output(dq, 'draft_qq', 'draft Q / RoPE', ds)
    dkv = c.gemm('draft_kvproj', 'draft SWA KV projection\nD -> 512', din, dr, c.d, 512, 3, section=ds)
    dk = c.output(dkv, 'draft_kpack', 'FP8 draft KV pack', ds)
    c.store('draft_kv', 'DSpark SWA KV, FP8', S * 128 * 512 * 3 * FP8, ds)
    c.edge(dk, 'draft_kv', dr * 512 * 3 * FP8, 'fp8')
    c.node('draft_att', 'DSpark SWA attention / RoPE', ds,
           flops=4 * dr * 64 * 512 * 128 * 3, rate='bf16')
    c.edge(dqq, 'draft_att', dq[1])
    c.edge('draft_kv', 'draft_att', S * 128 * 512 * 3 * FP8, 'fp8')
    do = c.gemm('draft_olow', 'draft grouped output\n8 x (4096 -> 1024)', 'draft_att', dr, 4096, 1024, 3, 8, section=ds)
    dm = c.output(do, 'draft_omid', 'draft output latent', ds)
    duo = c.gemm('draft_oup', 'draft output GEMM\n8192 -> D', dm, dr, 8192, c.d, 3, section=ds)
    dx = c.output(duo, 'draft_x', 'draft residual / mHC', ds)
    dff = c.mlp('draft_moe', dx, dr, 2304, 3, 128, 3, True, ds)
    c.notes.append(f'DSpark ASSUMED inherited FFN/attention dims: {coverage(dr,3,128):.1f}/128 experts touched; 3 routed + 1 shared')
    dshared = c.mlp('draft_shared', dx, dr, 2304, 3, section=ds)
    drouter = c.gemm('draft_router', 'draft router / top-3\nD -> 128', dx, dr, c.d, 128, 3, section=ds)
    c.node('draft_reduce', 'draft expert reduce + shared / residual', ds)
    c.edge(dff[0], 'draft_reduce', dff[1])
    c.edge(dshared[0], 'draft_reduce', dshared[1])
    c.edge(drouter[0], 'draft_reduce', drouter[1])
    dh = c.gemm('draft_head', 'DSpark base LM head\nshared target W, read again', 'draft_reduce', dr, c.d, c.vocab, section=ds)
    c.resident_weights -= c.d * c.vocab * NV
    mark = c.gemm('markov', 'Markov + confidence heads\nASSUMED D -> rank 256', 'draft_reduce', dr, c.d, 256, section=ds)
    ml = c.output(mark, 'mark_lat', 'Markov latent', ds)
    mv = c.gemm('mark_vocab', 'Markov rank -> vocabulary\nASSUMED full-vocab evaluation', ml, dr, 256, c.vocab, section=ds)
    c.node('draft_schedule', 'Markov sampling / confidence schedule\n5-position block -> 3 drafts + anchor ASSUMED', ds)
    c.edge(dh[0], 'draft_schedule', dh[1])
    c.edge(mv[0], 'draft_schedule', mv[1])
    c.edge('accept', 'draft_schedule', S * 4, 'int', back=True)
    c.edge('draft_schedule', 'tok', R * 4, 'int', back=True)
    # Four-way TP payload: ring allreduce 2*(p-1)/p bytes per rank,
    # two residual-width reductions / layer plus aggregate expert dispatch.
    c.node('nvlink', '4-way NVLink TP / expert exchange\n900 GB/s/GPU one-way payload ASSUMED\nno startup latency', 'inter-GPU', flops=0, rate='network')
    payload_per_gpu = 2 * 3 / 4 * R * c.d * 4 * 40 * 2 + R * 6 * c.d * 4 * 40 * 2 / 4
    c.edge(end, 'nvlink', payload_per_gpu * 4, label='payload sum across GPUs', back=True)
    c.finish()


def glm():
    c = Chart('heat-glm-5.3-flash', 'GLM 5.3 Flash: 34 KDA + 11 DSA/MLA, 45 layers', 1, 4096, 45)
    c.vocab = 154880
    c.notes += ['First 3 FFNs dense (12288); other 42: 288 routed / top-8 + 1 shared (2048); mHC x4',
                'MLA: q-rank 1536, KV-rank 512, 64 heads x qk/v 256, no RoPE; index 32 x 128, k-pool 4, top-2048',
                'KDA: 64 x 128, conv kernel 4; FP32 recurrent state retained; rejected drafts use pending replay ASSUMED',
                'One native MTP layer: assumed DSA/MoE dims inherited; 3 serial calls at 64 rows to propose 3 drafts',
                'All times ideal roofline floors: no launch/dequant/MoE-occupancy penalty; acceptance not measured']
    source = start(c)
    ks = 'Kimi Delta Attention x 34 (linear / recurrent)'
    kp = c.gemm('kda_qkv', 'Q/K/V GEMMs\n4096 -> 3 x 8192', source, R, c.d, 3 * 8192, 34, section=ks)
    conv = c.output(kp, 'kda_conv', 'depthwise conv4 / SiLU / QK norm', ks, R * 3 * 8192 * 34 * 16)
    c.store('conv_state', 'conv state, FP32\n3 previous positions', S * 3 * 3 * 8192 * 34 * 4, ks)
    c.edge('conv_state', conv, S * 3 * 3 * 8192 * 34 * 4 * 2, both=True)
    ga = c.gemm('kda_ga', 'decay + output-gate low-rank GEMMs\n2 x (D -> 128), beta D -> 64', source, R, c.d, 2 * 128 + 64, 34, section=ks)
    gaout = c.output(ga, 'kda_gate_lat', 'two 128-wide gate latents + beta', ks)
    gb = c.gemm('kda_gb', 'two gate expansion GEMMs\n2 x (128 -> 8192)', gaout, R, 128, 8192, 34, 2, section=ks)
    gates = c.output(gb, 'kda_gates', 'decay / beta / gated RMSNorm', ks)
    state_bytes = S * 34 * 64 * 128 * 128 * 4
    c.store('kda_state', 'KDA recurrent matrix, FP32\n64 x 128 x 128 per layer/stream', state_bytes, ks)
    c.node('kda', 'delta recurrence + gated norm\n4 rows sequential per stream\nverify + accepted-block replay ASSUMED', ks,
           flops=R * 34 * 64 * 128 * 128 * 10 * 2)
    c.edge(conv, 'kda', R * 3 * 8192 * 34 * 4)
    c.edge(gates, 'kda', R * (2 * 8192 + 64) * 34 * 4)
    c.edge('kda_state', 'kda', state_bytes * 2, both=True, label='read once + commit once')
    # Deferred commit lets the verifier discard rejected rows without matrix snapshots.
    pending = R * 34 * (3 * 8192 + 8192 + 64) * 4
    c.store('pending', 'Q/K/V/decay/beta pending replay, FP32', pending, ks)
    c.edge('pending', 'kda', pending * 2, both=True, label='save verify + read accepted replay')
    ko = c.gemm('kda_out', 'KDA output GEMM\n8192 -> 4096', 'kda', R, 8192, c.d, 34, section=ks)
    kout = c.output(ko, 'kda_add', 'KDA residual output', ks)
    ms = 'DSA / MLA x 11 (latent attention, no expanded KV cache)'
    lp = c.gemm('mla_low', 'Q + KV low-rank GEMMs\n4096 -> 1536 + 512', source, R, c.d, 1536 + 512, 11, section=ms)
    latent = c.output(lp, 'mla_latent', 'Q / KV latent norm', ms)
    qp = c.gemm('mla_q', 'Q expansion GEMM\n1536 -> 64 x 256', latent, R, 1536, 64 * 256, 11, section=ms)
    qout = c.output(qp, 'mla_qout', 'Q FP32', ms)
    qa = c.gemm('mla_absorb', 'headwise Q * W_K GEMMs\n64 x (256 -> 512)', qout, R, 256, 512, 11, 64, section=ms)
    aq = c.output(qa, 'mla_aq', 'absorbed Q, 64 x 512', ms)
    c.node('mla_kpack', 'KV latent -> FP8', ms)
    c.edge(latent, 'mla_kpack', R * 512 * 11 * 4)
    kvbytes = S * CTX * 512 * FP8 * 11
    c.store('mla_kv', 'MLA KV cache, FP8\n512-channel latent / token / DSA layer', kvbytes, ms)
    c.edge('mla_kpack', 'mla_kv', R * 512 * 11 * FP8, 'fp8')
    iq = c.gemm('index_q', 'index Q GEMM\n1536 -> 32 x 128', latent, R, 1536, 32 * 128, 11, section=ms)
    iqq = c.output(iq, 'index_qq', 'index Q / head weights', ms)
    ik = c.gemm('index_k', 'index K + k-pool gate + head weights\nD -> 2 x 128 + 32', source, R, c.d, 2 * 128 + 32, 11, section=ms)
    ikk = c.output(ik, 'index_kk', 'k-pool compress4 / index K FP8', ms)
    ikbytes = S * CTX / 4 * 128 * FP8 * 11
    c.store('index_cache', 'k-pool index cache, FP8', ikbytes, ms)
    c.edge(ikk, 'index_cache', R / 4 * 128 * FP8 * 11, 'fp8')
    c.node('index_score', 'index score / top-512 pools\nexpand -> 2048 raw positions + tail', ms,
           flops=2 * R * CTX / 4 * 32 * 128 * 11, rate='bf16')
    c.edge(iqq, 'index_score', R * 32 * 128 * 11 * 4)
    c.edge('index_cache', 'index_score', ikbytes, 'fp8')
    c.node('mla_att', 'sparse latent attention\nFP8 KV -> BF16 arithmetic -> FP32', ms,
           flops=4 * R * 2048 * 64 * 512 * 11, rate='bf16')
    c.edge(aq, 'mla_att', R * 64 * 512 * 11 * 4)
    c.edge('index_score', 'mla_att', R * 2051 * 11 * 4, 'int')
    c.edge('mla_kv', 'mla_att', S * min(CTX, 4 * 2048) * 512 * FP8 * 11, 'fp8', label='ASSUMED union of 4 selections = full 4k')
    vp = c.gemm('mla_v', 'headwise latent * W_V GEMMs\n64 x (512 -> 256)', 'mla_att', R, 512, 256, 11, 64, section=ms)
    vout = c.output(vp, 'mla_vout', 'expanded attention values FP32', ms)
    op = c.gemm('mla_out', 'attention output GEMM\n64 x 256 -> 4096', vout, R, 64 * 256, c.d, 11, section=ms)
    aout = c.output(op, 'mla_add', 'DSA residual output', ms)
    c.node('mixer_done', 'mixer residual / mHC update', 'residual mixing')
    c.edge(kout, 'mixer_done', R * c.d * 34 * 4)
    c.edge(aout, 'mixer_done', R * c.d * 11 * 4)
    dense = c.mlp('dense', 'mixer_done', R, 12288, 3, section='dense FFN: first 3 layers')
    end = moe(c, 'mixer_done', 42, 288, 8, 2048)
    c.edge(dense[0], end, dense[1], label='dense layer outputs', back=True)
    head(c, end)
    # One MTP block is reused three times, not three different MTP layers.
    ds = 'native MTP: one block reused for 3 serial draft iterations'
    c.node('mtp_input', 'target hidden + next-token embedding\nASSUMED two norms + concat', ds)
    c.edge(end, 'mtp_input', S * c.d * 4 * 3, label='3 iterations', back=True)
    c.edge('emb_W', 'mtp_input', S * c.d * FP8 * 3, 'fp8', label='shared embedding', back=True)
    mi = c.gemm('mtp_join', 'MTP concat projection\nASSUMED 2D -> D, 3 calls x 64 rows', 'mtp_input', S, 2 * c.d, c.d, 3, section=ds)
    mx = c.output(mi, 'mtp_x', 'MTP hidden', ds)
    # Combine the inherited MLA projection chain into a roofline sum in a single
    # node. Input/output activations below include intermediate materialization.
    mla_params = c.d * (1536 + 512) + 1536 * 64 * 256 + 2 * 64 * 256 * 512 + 64 * 256 * c.d
    iparams = 1536 * 32 * 128 + c.d * (2 * 128 + 32)
    c.node('mtp_mla_W', 'MTP MLA / index W, NVFP4\nASSUMED inherited dims, one resident copy', ds, 'weight')
    c.resident_weights += (mla_params + iparams) * NV
    c.node('mtp_att_pack', 'pack EACH internal GEMM input -> NVFP4\nMTP MLA/index chain grouped', ds,
           flops=S * 3 * (c.d + 1536 + 64 * 256 + 64 * 512) * 12)
    internal = S * 3 * (2 * c.d + 2 * 1536 + 64 * (256 + 512 + 512 + 256))
    c.edge(mx, 'mtp_att_pack', internal * 4, label='includes internal FP32 intermediates')
    c.node('mtp_att', 'MTP MLA/index GEMMs + sparse attention\n3 serial calls, 2048 keys/call\nASSUMED aggregated roofline', ds,
           flops=2 * S * 3 * (mla_params + iparams), rate='fp4')
    c.edge('mtp_att_pack', 'mtp_att', internal * NV, 'nvfp4')
    c.edge('mtp_mla_W', 'mtp_att', (mla_params + iparams) * NV * 3, 'nvfp4', label='one copy read on each serial call')
    c.store('mtp_kv', 'MTP MLA latent + k-pool index, FP8', S * CTX * (512 + 128 / 4) * FP8, ds)
    c.edge('mtp_kv', 'mtp_att', S * (2048 * 512 + CTX / 4 * 128) * FP8 * 3, 'fp8', label='3 serial reads')
    c.edge('mtp_att', 'mtp_kv', S * 3 * (512 + 128 / 4) * FP8, 'fp8')
    # Account BF16 tensor arithmetic separately rather than at FP4 rate.
    c.node('mtp_sparse', 'MTP attention tensor arithmetic\n3 x 64 x 2048 x 64 x 512', ds,
           flops=4 * S * 3 * 2048 * 64 * 512, rate='bf16')
    c.edge('mtp_att', 'mtp_sparse', S * 3 * 64 * 512 * 4)
    mout = c.node('mtp_out', 'MTP residual / norm', ds)
    c.edge('mtp_sparse', mout, S * c.d * 3 * 4)
    # Each of three serial calls touches fewer experts than one 256-row pass.
    dg = c.mlp('mtp_moe', mout, S, 2048, 3, 288, 8, True, ds)
    # layers=3 here means 3 reads/calls of one block; undo excess residency.
    c.resident_weights -= 2 * 3 * c.d * 2048 * 288 * TRIT
    dsh = c.mlp('mtp_shared', mout, S, 2048, 3, section=ds)
    c.resident_weights -= 2 * 3 * c.d * 2048 * NV
    rt = c.gemm('mtp_router', 'MTP router / top-8\n3 calls x 64 rows', mout, S, c.d, 288, 3, section=ds)
    c.resident_weights -= 2 * c.d * 288 * NV
    c.node('mtp_reduce', 'MTP routed/shared reduce / residual', ds)
    c.edge(dg[0], 'mtp_reduce', dg[1])
    c.edge(dsh[0], 'mtp_reduce', dsh[1])
    c.edge(rt[0], 'mtp_reduce', rt[1])
    dh = c.gemm('mtp_head', 'MTP shared LM head\n3 serial reads of target W', 'mtp_reduce', S, c.d, c.vocab, 3, section=ds)
    c.resident_weights -= 3 * c.d * c.vocab * NV
    c.resident_weights -= 2 * 2 * c.d * c.d * NV  # shared concat projection across 3 calls
    c.node('draft', 'MTP greedy proposals\n3 drafts + anchor -> verify 256 rows', ds)
    c.edge(dh[0], 'draft', dh[1])
    c.edge('draft', 'tok', S * 3 * 4, 'int', back=True)
    c.edge('accept', 'mtp_input', S * 4, 'int', back=True)
    c.notes.append(f'MTP ASSUMED serial weight rereads: {coverage(S,8,288):.1f}/288 experts per call; shared LM head/embedding not duplicated')
    c.finish()


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if arg not in ('all', 'deepseek', 'glm'):
        raise SystemExit('usage: flash_heat.py [all|deepseek|glm]')
    if arg in ('all', 'deepseek'):
        deepseek()
    if arg in ('all', 'glm'):
        glm()
