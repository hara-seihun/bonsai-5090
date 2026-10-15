#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4]
PREREQUISITE='9c4ed70089114706200215bf60910dbed2c35c95'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def git(path,args):return subprocess.check_output(['git','-C',str(path),*args],timeout=10)
def main():
 new=[]
 for folder in ['method/packing-swarm/tools/round22','results/packing-swarm/round22','archive/packing-swarm/round22']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.name not in ['custody.json'] :new.append(p)
 for name in ['README.md','FINDINGS.md','OBLIGATIONS.md','ROUND23.md','method/packing-swarm/BOARD.md','method/packing-swarm/ROUNDS.md','results/packing-swarm/ROUND-22.md']:new.append(ROOT/name)
 compiler=ROOT/'archive/packing-swarm/round22/compiler1/sources.json';closure=json.loads(compiler.read_text())['sources']
 borrowed={}
 for name,digest in closure.items():
  p=pathlib.Path(name)
  if sha(p)!=digest:raise ValueError('changed compiler source:'+name)
  if not p.is_relative_to(ROOT):borrowed[name]=digest
 cpu=json.loads((ROOT/'results/packing-swarm/round22/cpu/receipt.json').read_text());inputs={}
 for case in cpu['cases'].values():
  for kind in ('ordinary','ubsan'):
   r=case[kind];name=r['inherited_archive'];digest=r['inherited_archive_sha256'];p=pathlib.Path(name)
   if sha(p)!=digest:raise ValueError('changed inherited tape:'+name)
   relative='archive/packing-swarm/round20/cpu/'+p.name
   if subprocess.run(['git','cat-file','-e',PREREQUISITE+':'+relative],cwd=ROOT,timeout=10).returncode:raise ValueError('private input custody unset:'+relative)
   inputs[name]={'SHA256':digest,'private_commit':PREREQUISITE,'private_tree_path':relative,'exact_cwd_is_not_custody':True}
 canonical=pathlib.Path('~/projects/research/program-12-packed-compute');engine=pathlib.Path('~/projects/halo-5090')
 preservation={'canonical_HEAD':git(canonical,['rev-parse','HEAD']).decode().strip(),'canonical_index_entries_SHA256':hashlib.sha256(git(canonical,['ls-files','-s'])).hexdigest(),'engine_HEAD':git(engine,['rev-parse','HEAD']).decode().strip(),'engine_refs':len(git(engine,['show-ref']).splitlines()),'engine_refs_SHA256':hashlib.sha256(git(engine,['show-ref'])).hexdigest(),'engine_index_entries_SHA256':hashlib.sha256(git(engine,['ls-files','-s'])).hexdigest()}
 expected={'canonical_HEAD':'846c92ba79be94db8d843e17e2c488aacdaf4ad9','canonical_index_entries_SHA256':'4c636ad0fb4829f3b202a65d0bbb354c965131e0f77394d227d40408cce736ff','engine_HEAD':'772d9060ec5b5eba5b7429833331843b93eca93e','engine_refs':144,'engine_refs_SHA256':'04936ae6b1221a2c72572cc85171c0e8721b49c0a8e53b49346470536f3fd9d9','engine_index_entries_SHA256':'2022a51f5796c3f91d017857ddc4c59d1be7f337bcef96e40512a757abbe28cc'}
 if preservation!=expected:raise ValueError('opening canonical/engine capture changed:'+json.dumps(preservation))
 state={'state':'sealed_CPU_source_only_round22','prerequisite':PREREQUISITE,'branch':'research/mma-round22-20261011','hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(new))},'borrowed_source_closure':borrowed,'borrowed_CPU_tapes':inputs,'preservation':preservation,'workspace_admission':{'id':'00000000-0000-0000-0000-000000000000','growth_MiB':512,'headroom_GiB':8,'intent':'budgeted','sparse':True},'durable_custody':{'private_remote':'~/data/research-remotes/program-12-packed-compute.git','incremental_bundle':'~/work/packing-mma-round22.bundle','external_delivery':'~/work/packing-mma-round22-delivery.json','stable_evidence':'~/work/packing-mma-round22-evidence/snapshot','released_clone_retention':'not promised; exact clone cwd is not custody'},'GPU_execution':False,'spending':False,'new_packing_bound':False,'nominee':'none','sidecars':'all three completed and explicitly closed','next':'ROUND23.md'}
 (ROOT/'results/packing-swarm/round22/custody.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({'state':state['state'],'new_hash_files':len(state['hashes']),'borrowed_sources':len(borrowed),'borrowed_tapes':len(inputs),'canonical_and_144_engine_refs':'unchanged'}))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:raise SystemExit('typed_custody_error:'+str(e))
