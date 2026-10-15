#!/usr/bin/env python3
"""Fetch only selected public, revision/hash-pinned datasets. No model inference."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmarks',nargs='+',required=True)
    p.add_argument('--max-bytes',type=int,required=True)
    a=p.parse_args()
    if a.max_bytes<=0:p.error('--max-bytes must be positive')
    pins=json.loads((ROOT/'datasets-original15.lock.json').read_text());used=0
    for name in a.benchmarks:
        pin=pins[name];dest=(ROOT/pin['snapshot_path']).resolve()
        for rel,expected in pin['files_sha256'].items():
            target=(dest/rel).resolve()
            if not target.is_relative_to(dest):raise ValueError('Unsafe dataset path')
            if target.exists():body=target.read_bytes()
            else:
                prefix=(f"https://raw.githubusercontent.com/{pin['repo_id']}/{pin['revision']}/data/"
                        if pin.get('hub')=='github' else
                        f"https://huggingface.co/datasets/{pin['repo_id']}/resolve/{pin['revision']}/")
                remaining=a.max_bytes-used
                if remaining<=0:raise ValueError('Declared download budget exhausted')
                with urllib.request.urlopen(prefix+rel,timeout=30) as response:
                    body=response.read(remaining+1)
                if len(body)>remaining:raise ValueError('Declared download budget exceeded')
                used+=len(body)
            if hashlib.sha256(body).hexdigest()!=expected:
                raise ValueError('Dataset source differs: '+name+'/'+rel)
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
        print(name,pin['revision'],'hashes match',flush=True)
if __name__=='__main__':main()
