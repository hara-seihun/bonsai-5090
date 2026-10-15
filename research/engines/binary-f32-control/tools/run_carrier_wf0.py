#!/usr/bin/env python3
"""Cached WF3 vs WF0 native ABBA; includes actual prep and all model families."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('binary','fixtures','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--source',required=True);p.add_argument('--pdl',choices=('0','1'),required=True)
 p.add_argument('--delayed',choices=('0','1'),required=True);p.add_argument('--rounds',type=int,required=True)
 a=p.parse_args()
 if not 2<=a.rounds<=100:p.error('rounds must be2..100')
 binary=a.binary.resolve(strict=True);a.output.mkdir(parents=True,exist_ok=False)
 env={k:v for k,v in os.environ.items() if not k.startswith('H5_')};env.update(H5_PDL=a.pdl,H5_PREP_CHUNK='0')
 cases=[('gu0',34816,5120,19,1,3),('gu32',34816,5120,19,1,3),
        ('down0',5120,17408,26,8,2),('attnq3',12288,5120,26,1,0),
        ('ssmqkv0',10240,5120,26,1,0),('ssmout0',5120,6144,12,4,2),
        ('head',248320,5120,12,1,0),('varying',1056,2048,26,1,1)]
 manifest={'source':a.source,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'pdl':a.pdl,'delayed':a.delayed,'runs':[]}
 for case,N,K,cfg,split,mode in cases:
  fixture=case if case=='varying' else str((a.fixtures/case).resolve(strict=True))
  for R in (1,2,8):
   name=f'{case}-c{cfg}-r{R}'
   cmd=[str(binary),str(R),str(N),str(K),str(cfg),str(split),str(mode),fixture,str(a.rounds),str(a.output/name),a.delayed]
   with (a.output/(name+'.log')).open('w')as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
   raw=[json.loads(l)for l in (a.output/(name+'.log')).read_text().splitlines()]
   summary={'case':case,'rows':R,'cfg':cfg,**raw[-1]};manifest['runs'].append(summary)
   (a.output/'results.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
