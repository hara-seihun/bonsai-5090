#!/usr/bin/env python3
"""Collate completed fresh scoped task panels with exact pooled rates."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
rows = []
for f in sorted(a.root.rglob('summary.json')):
    d = json.loads(f.read_text())
    if d.get('schema') != 'halo-quality-gate-v1':
        continue
    receipt_path = f.parent / 'run-receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    timing = d['generation_timing']['session']
    if timing['status'] != 'measured' or timing['reused_n']:
        raise ValueError(f'FRESH_MEASURED_PANEL_REQUIRED:{f}')
    metrics = {}
    for bench, b in d['benchmarks'].items():
        metrics[bench] = {'metrics':b['metrics'], 'truncated':b['truncated'],
                          'completion_tokens':b['completion_tokens'], 'generation_s':b['generation_s'],
                          'http_tokens_second':b['completion_tokens']/b['generation_s']}
    row = {'path':str(f.parent), 'receipt':receipt, 'decoding':d['decoding'],
           'manifest_sha256':d['data_manifest_sha256'], 'tokens':timing['completion_tokens'],
           'generation_seconds':timing['generation_s'],
           'http_tokens_second':timing['completion_tokens']/timing['generation_s'],
           'benchmarks':metrics}
    ma, mb = f.parent/'metrics-after.json', f.parent/'metrics-before.json'
    if ma.exists() and mb.exists():
        after, before = json.loads(ma.read_text()), json.loads(mb.read_text())
        counters = ('accepted_drafts','proposed_drafts','verify_rounds','generated_tokens','lookup_proposed',
                    'lookup_accepted','lookup_rounds','verify_s','draft_s','busy_s')
        delta = {k:after[k]-before[k] for k in counters}
        delta['acceptance_fraction'] = delta['accepted_drafts']/delta['proposed_drafts'] if delta['proposed_drafts'] else None
        delta['emitted_per_verify_round'] = delta['generated_tokens']/delta['verify_rounds'] if delta['verify_rounds'] else None
        row['native_counters'] = delta
    rows.append(row)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({'scope':'fresh64request quick32+32 ladder; not full gate', 'rows':rows}, indent=2)+'\n')
for row in rows:
    print(f'{row["path"]}: {row["http_tokens_second"]:.6f}tok/s')
