#!/usr/bin/env python3
"""Compress Nsight raw CSV into a per-launch pipe ledger (no invented missing counters)."""
import argparse
import csv
import io
import json
from pathlib import Path

METRICS = {
    'duration_ns': 'gpu__time_duration.sum',
    'issue_pct': 'sm__issue_active.avg.pct_of_peak_sustained_elapsed',
    'tensor_pct': 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'fma_pct': 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'alu_pct': 'sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed',
    'alu_active_pct': 'sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active',
    'lsu_active_pct': 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active',
    'lsu_pct': 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed',
    'xu_pct': 'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed',
    'uniform_pct': 'sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_elapsed',
    'tma_pct': 'sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'dram_pct': 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    'dram_read_bytes': 'dram__bytes_read.sum',
    'dram_write_bytes': 'dram__bytes_write.sum',
    'l2_sectors': 'lts__t_sectors.sum',
    'l2_hit_pct': 'lts__t_sector_hit_rate.pct',
    'l2_pct': 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'issued': 'smsp__inst_issued.sum',
    'executed': 'smsp__inst_executed.sum',
    **{f'{pipe}_instructions': f'smsp__inst_executed_pipe_{pipe}.sum'
       for pipe in ['adu', 'alu', 'cbu', 'fma', 'fp64', 'ipa', 'lsu', 'tensor', 'tex', 'tma', 'uniform', 'workid', 'xu', 'fmaheavy', 'fmalite', 'fma_type_fp16']},
    'active_warps_per_scheduler': 'smsp__warps_active.avg.per_cycle_active',
    'eligible_warps_per_scheduler': 'smsp__warps_eligible.avg.per_cycle_active',
}


def parse(path):
    lines = Path(path).read_text().splitlines()
    start = next((i for i, s in enumerate(lines) if s.startswith('"ID"')), None)
    if start is None:
        raise ValueError(f'{path}: no Nsight raw CSV header')
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    units = next(reader)
    rows = []
    for row in reader:
        if not row.get('ID', '').isdigit():
            continue
        out = {'id': int(row['ID']), 'kernel': row.get('Kernel Name'), 'grid': row.get('Grid Size'), 'block': row.get('Block Size')}
        for label, metric in METRICS.items():
            value = row.get(metric)
            out[label] = float(value.replace(',', '')) if value and value not in ('n/a', 'N/A') else None
        out['alu_basis'] = 'direct elapsed counter'
        if out['alu_pct'] is None:
            if out['alu_active_pct'] is not None and out['lsu_pct'] is not None and out['lsu_active_pct'] is not None and out['lsu_active_pct'] > 0:
                out['alu_pct'] = out['alu_active_pct'] * out['lsu_pct'] / out['lsu_active_pct']
                out['alu_basis'] = 'active ALU counter times elapsed/active normalization from identical LSU counter'
            else:
                out['alu_basis'] = 'unavailable'
        # --print-units base yields ns; imported older reports may use other duration units.
        unit = units.get(METRICS['duration_ns'])
        if out['duration_ns'] is not None:
            if unit not in ('nsecond', 'ns', 'usecond', 'us', 'msecond', 'ms', 'second', 's'):
                raise ValueError(f'{path}: unknown duration unit {unit!r}')
            out['duration_ns'] *= {'nsecond': 1, 'ns': 1, 'usecond': 1e3, 'us': 1e3, 'msecond': 1e6, 'ms': 1e6, 'second': 1e9, 's': 1e9}[unit]
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = [dict(source=p, **r) for p in a.csv for r in parse(p)]
    Path(a.out + '.json').write_text(json.dumps({'metrics': METRICS, 'launches': rows}, indent=2) + '\n')
    cols = ['issue_pct','tensor_pct','fma_pct','alu_pct','lsu_pct','xu_pct','uniform_pct','tma_pct','dram_pct','l2_pct']
    text = ['# Hardware pipe ledger', '', 'Percentages use elapsed device capacity, not only active SM cycles. ALU is converted from its active counter using the elapsed/active ratio of the identical LSU counter when Nsight omits its elapsed form (basis saved in JSON). Pipes overlap; columns are not additive. `—` is an unavailable counter, not zero. This is a launch sample unless the input is a complete range replay.', '', '| kernel / grid | µs | ' + ' | '.join(c.removesuffix('_pct') + ' %' for c in cols) + ' |', '|---|---:' + '|---:'*len(cols) + '|']
    for r in rows:
        fmt = lambda n: '—' if n is None else f'{n:.2f}'
        name = (r['kernel'] or '').replace('|', '\\|')
        text.append(f'| `{name}` / {r["grid"]} | {fmt(r["duration_ns"] / 1e3 if r["duration_ns"] is not None else None)} | ' + ' | '.join(fmt(r[k]) for k in cols) + ' |')
    Path(a.out + '.md').write_text('\n'.join(text) + '\n')


if __name__ == '__main__':
    main()
