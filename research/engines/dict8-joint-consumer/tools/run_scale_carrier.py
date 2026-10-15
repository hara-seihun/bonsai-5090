#!/usr/bin/env python3
"""Run bounded cached/refresh scale-carrier component panels; no provisioning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--pdl', choices=('0','1'), required=True)
    p.add_argument('--rounds', type=int, required=True)
    a = p.parse_args()
    if not 2 <= a.rounds <= 100:
        p.error('rounds must be2..100')
    binary = a.binary.resolve(strict=True)
    # Real matrices spanning GU, down, attention input, recurrent input/output, head.
    cases = [('constant',34816,5120,19,1,1), ('varying',34816,5120,19,1,1),
             ('gu0',34816,5120,19,1,1), ('gu32',34816,5120,19,1,1),
             ('down0',5120,17408,12,4,0), ('attnq3',12288,5120,26,1,0),
             ('ssmqkv0',10240,5120,26,1,0), ('ssmout0',5120,6144,12,4,0),
             ('head',248320,5120,12,1,0)]
    a.output.mkdir(parents=True, exist_ok=False)
    env = {k:v for k,v in os.environ.items() if not k.startswith('H5_')}
    env['H5_PDL'] = a.pdl
    manifest = {'source':a.source,'binary':str(binary),
                'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                'pdl':a.pdl,'runs':[]}
    for case,N,K,cfg,split,fused in cases:
        fixture = case if case in ('constant','varying') else str((a.fixtures/case).resolve(strict=True))
        for R in (1,2,8):
            name=f'{case}-r{R}'
            output=a.output/name
            cmd=[str(binary),str(R),str(N),str(K),str(cfg),str(split),str(fused),fixture,str(a.rounds),str(output)]
            with (a.output/(name+'.log')).open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
            results=[json.loads(line) for line in (a.output/(name+'.log')).read_text().splitlines()]
            summary={'case':case,'rows':R,**results[-1]}
            manifest['runs'].append(summary)
            (a.output/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
