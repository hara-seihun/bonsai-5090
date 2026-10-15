#!/usr/bin/env python3
import hashlib,json,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[4]
DEST=pathlib.Path('~/work/packing-mma-round23-evidence')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],timeout=20)
def main():
 if len(sys.argv)!=2:raise ValueError('required exact published commit')
 commit=sys.argv[1]
 if git('rev-parse','HEAD').decode().strip()!=commit:raise ValueError('wrong source commit')
 if git('status','--porcelain'):raise ValueError('source not committed clean')
 remote=pathlib.Path('~/data/research-remotes/program-12-packed-compute.git')
 ack=subprocess.check_output(['git','--git-dir',str(remote),'rev-parse','refs/heads/research/mma-round23-20261011'],timeout=10).decode().strip()
 if ack!=commit:raise ValueError('private remote custody not acknowledged')
 seal=json.loads((ROOT/'results/packing-swarm/round23/custody.json').read_text())
 files=dict(seal['hashes']);files['results/packing-swarm/round23/custody.json']=sha(ROOT/'results/packing-swarm/round23/custody.json')
 if (DEST/'snapshot').exists():raise ValueError('fresh snapshot required')
 snap=DEST/'snapshot';snap.mkdir(parents=True)
 source={};borrowed={}
 for relative,digest in files.items():
  p=ROOT/relative
  if sha(p)!=digest:raise ValueError('changed sealed file:'+relative)
  dst=snap/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
  if sha(dst)!=digest:raise ValueError('artifact copy mismatch:'+relative)
  source[relative]={'SHA256':digest,'bytes':dst.stat().st_size,'private_tree_path':relative}
 for name,digest in seal['borrowed_source_closure'].items():
  p=pathlib.Path(name);relative=pathlib.Path('borrowed')/p.relative_to('~');dst=snap/relative;dst.parent.mkdir(parents=True,exist_ok=True)
  if sha(p)!=digest:raise ValueError('changed borrowed source:'+name)
  shutil.copyfile(p,dst)
  if sha(dst)!=digest:raise ValueError('borrowed copy mismatch:'+name)
  borrowed[name]={'SHA256':digest,'bytes':dst.stat().st_size,'snapshot_relative':str(relative)}
 inherited=pathlib.Path('~/work/packing-mma-round22-evidence/artifact-custody.json')
 if sha(inherited)!='e8785ec999459b747ecbc9b3e33f178013967572525f18931c3156d36d02403b':raise ValueError('changed inherited custody')
 manifest={'state':'immutable_private_artifact_custody_verified','commit':commit,'private_remote':str(remote),'source_and_evidence_files':source,'borrowed_sources':borrowed,'inherited_CPU_tapes':seal['inherited_CPU_tapes'],'inherited_round22':{'commit':'031d82a54b3a61a40ffabcd28cc8f526f5bb0109','snapshot':'~/work/packing-mma-round22-evidence/snapshot','manifest':str(inherited),'manifest_SHA256':sha(inherited)},'snapshot_bytes':sum(s['bytes'] for s in source.values())+sum(s['bytes'] for s in borrowed.values()),'exact_released_cwd_retention':False}
 (DEST/'artifact-custody.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'state':manifest['state'],'files':len(source),'borrowed_sources':len(borrowed),'bytes':manifest['snapshot_bytes'],'manifest_SHA256':sha(DEST/'artifact-custody.json')}))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:sys.exit('typed_artifact_error:'+str(e))
