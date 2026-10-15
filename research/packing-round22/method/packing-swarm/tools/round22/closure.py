#!/usr/bin/env python3
import sys,json,hashlib,pathlib
root=pathlib.Path(__file__).resolve().parents[4]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if len(sys.argv)!=2: raise SystemExit('required: exact dependencies.mk')
p=pathlib.Path(sys.argv[1]);p=p if p.is_absolute() else root/p
raw=p.read_text().replace('\\\n',' ')
paths=[]
for token in raw.split(':',1)[1].split():
 q=pathlib.Path(token);q=q if q.is_absolute() else root/q
 if str(q).startswith('~/'):
  if not q.is_file(): raise SystemExit(f'missing source: {q}')
  paths.append(q.resolve())
if not paths:raise SystemExit('empty source closure')
output=dict(state='complete_home_source_dependency_closure',dependencies=str(p),sources={str(q):sha(q) for q in sorted(set(paths))})
(p.parent/'sources.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(dict(source_files=len(output['sources']),home_source_bytes=sum(q.stat().st_size for q in set(paths)))))
