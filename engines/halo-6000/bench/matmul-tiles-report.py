#!/usr/bin/env python3
"""Summarize guarded complete-consumer tile reuse, not MMA peak rates."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'matmul-tiles-results'


def main():
    summary = {'scope': 'exact Q8-input projection only; prep and routing not included',
               'baseline': 'best measured original splitK1 block128/256 wide/resident188',
               'direct_output_comparisons': 0, 'graph_output_comparisons': 0,
               'independent_K32_integer_comparisons': 0,
               'new_tiles_direct_output_comparisons': 0,
               'new_tiles_graph_output_comparisons': 0, 'chunks': [], 'resources': {}}
    receipts = sorted(OUT.glob('*-receipt.json'))
    assert len(receipts) == 47, len(receipts)
    for path in receipts:
        receipt = json.loads(path.read_text())
        assert receipt['state'] == 'succeeded', receipt
        assert receipt['wall_seconds'] < 90, receipt
        name = path.name.removesuffix('-receipt.json')
        rows = [json.loads(line) for line in (OUT / (name + '.jsonl')).read_text().splitlines()]
        assert rows[-1].get('passed')
        guards = [r for r in rows if r.get('guard') == 'all_K32_integer_partials']
        assert len(guards) == 2 and all(r['different'] == 0 for r in guards)
        summary['independent_K32_integer_comparisons'] += sum(r['partials'] for r in guards)
        prices = [r for r in rows if 'cold_us' in r]
        assert len(prices) == 12
        assert all(r['direct_bit_differences'] == r['graph_bit_differences'] == 0 for r in prices)
        summary['direct_output_comparisons'] += sum(r['outputs'] for r in prices)
        summary['graph_output_comparisons'] += sum(r['outputs'] for r in prices)
        new_comparisons = sum(r['outputs'] for r in prices if r['tiles'] != 1)
        summary['new_tiles_direct_output_comparisons'] += new_comparisons
        summary['new_tiles_graph_output_comparisons'] += new_comparisons
        best = {t: min((r for r in prices if r['tiles'] == t), key=lambda r: r['cold_us']) for t in (1, 2, 4)}
        summary['chunks'].append({'case': receipt['case'], 'columns': receipt['columns'],
            'subset': receipt['subset'], 'wall_seconds': receipt['wall_seconds'],
            'best': best, 'T2_speedup': best[1]['cold_us']/best[2]['cold_us'],
            'T4_speedup': best[1]['cold_us']/best[4]['cold_us']})
        for r in prices:
            key = f"F{r['format']}/K{r['K']}/T{r['tiles']}/block{r['block']}"
            resource = {field: r[field] for field in ('registers', 'local_bytes', 'shared_bytes', 'active_CTAs_per_SM')}
            if key in summary['resources']:
                assert summary['resources'][key] == resource
            summary['resources'][key] = resource
        if not receipt['subset']:
            print(receipt['case'], receipt['columns'],
                  '/'.join(f"{best[t]['cold_us']:.2f}" for t in (1, 2, 4)),
                  f"{summary['chunks'][-1]['T2_speedup']:.2f}/{summary['chunks'][-1]['T4_speedup']:.2f}x")
    (ROOT / 'matmul-tiles-summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
