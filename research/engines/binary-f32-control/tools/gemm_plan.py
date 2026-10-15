#!/usr/bin/env python3
"""Change explicit row-eight plan entries; never retune other families/splits."""
import argparse
from pathlib import Path

GEOMETRY = {18: (2, 2, 8, 8), 19: (4, 2, 8, 8), 20: (4, 1, 8, 8),
            21: (8, 1, 8, 8), 22: (2, 1, 8, 8), 23: (4, 2, 8, 4),
            24: (3, 2, 8, 4), 25: (3, 2, 8, 8), 26: (8, 1, 8, 4),
            27: (6, 1, 8, 4), 28: (4, 1, 8, 4), 29: (8, 2, 8, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--family', choices=['gu', 'input', 'output', 'head', 'linear'], required=True)
    p.add_argument('--cfg', type=int, choices=sorted(GEOMETRY), required=True)
    p.add_argument('--splits', type=int, help='Explicit split count; otherwise retain recorded splits')
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; choose a fresh path')
    if args.splits is not None and args.splits < 1:
        p.error('splits must be positive')
    if args.family == 'gu' and (GEOMETRY[args.cfg][1] != 2 or args.splits not in (None, 1)):
        p.error('fused gate/up requires WG2 and one split')
    lines, changed = [], []
    for at, line in enumerate(args.source.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 7:
            p.error(f'line {at}: expected seven integers')
        try:
            n, k, rows, add, wf, cfg, splits = map(int, parts)
        except ValueError:
            p.error(f'line {at}: noninteger field')
        family = ('gu' if wf == 100 else 'head' if n == 248320 and wf == 3
                  else 'input' if wf == 3 and n in (14336, 16384)
                  else 'output' if wf == 3 and n == 5120 else None)
        if rows == 8 and (family == args.family or args.family == 'linear' and family in ('input', 'output', 'head')):
            original = (cfg, splits)
            cfg = args.cfg
            if args.splits is not None:
                splits = args.splits
            if k // 128 < splits:
                p.error(f'line {at}: more splits than K blocks')
            changed.append({'N': n, 'K': k, 'add': add, 'from': original, 'to': (cfg, splits)})
        lines.append(f'{n} {k} {rows} {add} {wf} {cfg} {splits}')
    if not changed:
        p.error('no matching entries; source is not the expected dense5 plan')
    args.output.write_text('\n'.join(lines) + '\n')
    print(changed)


if __name__ == '__main__':
    main()
