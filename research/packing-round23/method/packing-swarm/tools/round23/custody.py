#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[4]
BASE='031d82a54b3a61a40ffabcd28cc8f526f5bb0109'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],timeout=20)
def main():
 opening=json.loads((ROOT/'results/packing-swarm/round23/preservation-open.json').read_text());now={}
 for kind,path in [('canonical','~/projects/research/program-12-packed-compute'),('engine','~/projects/halo-5090')]:
  refs=git(path,'for-each-ref','--format=%(refname) %(objectname)')
  now[kind]={'HEAD':git(path,'rev-parse','HEAD').decode().strip(),'index_SHA256':hashlib.sha256(git(path,'ls-files','-s')).hexdigest(),'refs_SHA256':hashlib.sha256(refs).hexdigest(),'refs':len(refs.splitlines())}
 if now!=opening:raise ValueError('changed canonical/engine opening capture')
 compiler=json.loads((ROOT/'results/packing-swarm/round23/compiler.json').read_text())
 if compiler['GPU_execution'] is not False:raise ValueError('source scope changed')
 closure=json.loads((ROOT/'archive/packing-swarm/round23/compiler2/sources.json').read_text())['sources'];borrowed={}
 for name,digest in closure.items():
  p=pathlib.Path(name)
  if sha(p)!=digest:raise ValueError('changed final compiler source:'+name)
  if not p.is_relative_to(ROOT):borrowed[name]=digest
 snapshot=ROOT/'archive/packing-swarm/round23/compiler1'
 for name,s in json.loads((snapshot/'source-snapshot.json').read_text()).items():
  if sha(snapshot/s['snapshot_relative'])!=s['SHA256']:raise ValueError('changed first compiler source:'+name)
 inherited=json.loads(pathlib.Path('~/work/packing-mma-round22-evidence/artifact-custody.json').read_text())['inherited_CPU_tapes']
 for s in inherited.values():
  actual=git(ROOT,'show',s['private_commit']+':'+s['private_tree_path'])
  if hashlib.sha256(actual).hexdigest()!=s['SHA256']:raise ValueError('changed private input custody')
 paths=[]
 for folder in ['method/packing-swarm/tools/round23','results/packing-swarm/round23','archive/packing-swarm/round23']:
  paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='custody.json')
 paths.extend(ROOT/p for p in ['README.md','FINDINGS.md','OBLIGATIONS.md','ROUND24.md','method/packing-swarm/BOARD.md','method/packing-swarm/ROUNDS.md','results/packing-swarm/ROUND-23.md'])
 state={'state':'sealed_PRIVATE_CPU_source_only_round23','prerequisite':BASE,'branch':'research/mma-round23-20261011','hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))},'borrowed_source_closure':borrowed,'inherited_CPU_tapes':inherited,'preservation':now,'workspace_admission':{'id':'00000000-0000-0000-0000-000000000000','growth_MiB':512,'headroom_GiB':8,'intent':'budgeted','sparse':True},'private_remote':'~/data/research-remotes/program-12-packed-compute.git','stable_evidence':'~/work/packing-mma-round23-evidence/snapshot','external_delivery':'~/work/packing-mma-round23-delivery.json','incremental_bundle':'~/work/packing-mma-round23.bundle','released_cwd_retention':False,'GPU_execution':False,'spending':False,'new_packing_bound':False,'next':'ROUND24.md'}
 (ROOT/'results/packing-swarm/round23/custody.json').write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps({'state':state['state'],'source_evidence_files':len(state['hashes']),'borrowed_sources':len(borrowed),'private_tapes':len(inherited),'canonical_engine':'unchanged'}))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:sys.exit('typed_custody_error:'+str(e))
