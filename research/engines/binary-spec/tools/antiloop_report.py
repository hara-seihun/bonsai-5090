#!/usr/bin/env python3
"""Summarise anti-loop accuracy runs (tools/pod/antiloop_matrix.py OUTDIR/acc/<arm>-s<rep>/items.jsonl).

  antiloop_report.py ACCDIR [--ref T0-off] [--json OUT]

Per arm (pooled over replicates): correct per run, accuracy, cut at the cap (unclosed think), exact loops,
degenerate, circling, mean thinking tokens, reconsideration phrases per 1000 words. Paired against the reference
arm at the same temperature: per-item mean correctness difference over replicates (mean ± SE over the 60 items),
and how many reference cut/loop items the arm finished correctly.
"""
import argparse, json, math, re, statistics
from collections import defaultdict
from pathlib import Path


def load(accdir):
    runs = defaultdict(dict)   # arm -> rep -> {id: item}
    for d in sorted(Path(accdir).iterdir()):
        m = re.fullmatch(r'(.+)-s(\d+)', d.name)
        if not m or not (d / 'items.jsonl').exists():
            continue
        items = {json.loads(l)['id']: json.loads(l) for l in (d / 'items.jsonl').read_text().splitlines()}
        runs[m.group(1)][int(m.group(2))] = items
    return runs


def summary(reps):
    rows = [it for items in reps.values() for it in items.values()]
    n = len(reps)
    return {'runs': n, 'items': len(rows), 'correct_per_run': [sum(it['correct'] for it in items.values()) for items in reps.values()],
            'accuracy': sum(it['correct'] for it in rows) / len(rows),
            'cut': sum(not it['think_closed'] for it in rows) / n, 'loops': sum(bool(it['loop_detected']) for it in rows) / n,
            'degenerate': sum(bool(it['degenerate']) for it in rows) / n, 'circling': sum(bool(it['circling']) for it in rows) / n,
            'think_tokens_mean': statistics.mean(it['think_tokens_est'] for it in rows),
            'reconsider_per_kword': statistics.mean(it['reconsider_per_kword'] for it in rows)}


def paired(arm, ref):
    ids = sorted(set.intersection(*[set(x) for x in list(arm.values()) + list(ref.values())]))
    mean = lambda reps, i: statistics.mean(r[i]['correct'] for r in reps.values())
    diffs = [mean(arm, i) - mean(ref, i) for i in ids]
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float('nan')
    bad = [i for i in ids if any(not r[i]['think_closed'] for r in ref.values())]
    rescued = sum(mean(arm, i) for i in bad)
    return {'items': len(ids), 'delta_problems': sum(diffs), 'delta_pct': 100 * statistics.mean(diffs), 'se_pct': 100 * se,
            'ref_cut_items': len(bad), 'ref_cut_items_correct_in_arm': rescued}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('accdir'); ap.add_argument('--json')
    a = ap.parse_args()
    runs = load(a.accdir)
    out = {}
    print(f"{'arm':14} {'runs':>4} {'correct/run':>14} {'acc%':>6} {'cut':>5} {'loops':>5} {'degen':>5} {'circ':>5} {'think':>6} {'rc/kw':>5}   paired vs off (problems, pct ± SE; ref-cut items solved)")
    for arm in sorted(runs, key=lambda s: (float(s.split('-')[0][1:]), s)):
        s = summary(runs[arm]); out[arm] = s
        ref = arm.split('-')[0] + '-off'
        p = paired(runs[arm], runs[ref]) if ref in runs and ref != arm else None
        if p: s['paired_vs_off'] = p
        ptxt = f"{p['delta_problems']:+5.1f}  {p['delta_pct']:+5.1f} ± {p['se_pct']:.1f}  {p['ref_cut_items_correct_in_arm']:.1f}/{p['ref_cut_items']}" if p else ''
        print(f"{arm:14} {s['runs']:4d} {str(s['correct_per_run']):>14} {100*s['accuracy']:6.1f} {s['cut']:5.1f} {s['loops']:5.1f} {s['degenerate']:5.1f} {s['circling']:5.1f} {s['think_tokens_mean']:6.0f} {s['reconsider_per_kword']:5.1f}   {ptxt}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
