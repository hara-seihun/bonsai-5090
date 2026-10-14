#!/usr/bin/env python3
"""Download PrismML's Ternary-Bonsai-2-27B PTQ1_0 GGUF at a pinned revision and verify its SHA256.

The weights are published by PrismML under Apache-2.0 at
https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf . This repository does not redistribute them.

  python scripts/fetch_model.py                 # -> models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
  python scripts/fetch_model.py --out D:/models
"""
import argparse, hashlib, os, sys, urllib.request
from pathlib import Path

REPO = 'prism-ml/Ternary-Bonsai-2-27B-gguf'
REVISION = 'b072e1d3b35a0a630cece372c2127528e0994386'
FILE = 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
SIZE = 5946648928
SHA256 = '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'
URL = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 23), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'models')
    ap.add_argument('--print-url', action='store_true')
    args = ap.parse_args()
    if args.print_url:
        print(URL)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / FILE
    if not (dest.exists() and dest.stat().st_size == SIZE):
        part = dest.with_suffix('.part')
        start = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-'} if start else {})
        with urllib.request.urlopen(req) as r, open(part, 'ab' if start else 'wb') as f:
            if start and r.status != 206:
                f.truncate(0)
            done = start
            while block := r.read(1 << 22):
                f.write(block)
                done += len(block)
                sys.stderr.write(f'\r{done / 1e9:6.2f} / {SIZE / 1e9:.2f} GB')
        sys.stderr.write('\n')
        os.replace(part, dest)
    got = sha256(dest)
    if got != SHA256:
        raise SystemExit(f'{dest}: sha256 {got} != {SHA256}')
    print(dest, 'sha256 OK')


if __name__ == '__main__':
    main()
