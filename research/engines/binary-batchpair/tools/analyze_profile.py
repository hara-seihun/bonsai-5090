#!/usr/bin/env python3
"""Analyze captured kernel intervals without adding overlapped durations as walltime."""
import sqlite3, pathlib, json, collections, statistics, argparse
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=pathlib.Path(a.directory)
def union(intervals):
 total=0;end=None
 for s,e in sorted(intervals):
  if end is None or s>end:total+=e-s;end=e
  elif e>end:total+=e-end;end=e
 return total
for label, relative in [('pdl1','replay/pdl1-node.sqlite'), ('pdl0','replay/pdl0-node.sqlite'), ('ordinary','ordinary/node.sqlite')]:
 db=sqlite3.connect(f'file:{root/relative}?mode=ro', uri=True);db.row_factory=sqlite3.Row
 ids={r['id']:r['value'] for r in db.execute('select * from StringIds')}
 ranges=list(db.execute("select start,end,text from NVTX_EVENTS where text='replay-target' order by start"))
 kernels=list(db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
 grouped=collections.defaultdict(list);families=collections.defaultdict(float);rounds=[]
 for rr in ranges:
  kk=[r for r in kernels if r['start']>=rr['start'] and r['end']<=rr['end']]
  rounds.append({'wall_ms':(rr['end']-rr['start'])/1e6,'kernel_union_ms':union((r['start'],r['end']) for r in kk)/1e6,'kernel_sum_ms':sum(r['end']-r['start'] for r in kk)/1e6,'launches':len(kk)})
  for r in kk:
   name=ids[r['demangledName']];grouped[name].append(r)
   family='gemm' if 'gemm_fp4' in name else 'prep' if 'prep2' in name else 'gdn' if 'k_gdn' in name else 'conv' if 'k_conv' in name else 'attention' if 'attn' in name else 'head-select' if 'argmax' in name else 'other'
   families[family]+=(r['end']-r['start'])/1e6
 details=[]
 for name,rs in grouped.items():
  shapes=collections.Counter((r['gridX'],r['gridY'],r['gridZ'],r['blockX'],r['blockY'],r['blockZ'],r['registersPerThread'],r['dynamicSharedMemory']) for r in rs)
  details.append({'name':name,'mangled_name':ids[rs[0]['mangledName']],'count':len(rs),'sum_ms':sum(r['end']-r['start'] for r in rs)/1e6,'median_us':statistics.median(r['end']-r['start'] for r in rs)/1e3,'shapes':[{'geometry':shape,'count':n} for shape,n in shapes.items()]})
 result={'rounds':rounds,'families_sum_ms':dict(families),'kernels':sorted(details,key=lambda r:-r['sum_ms']),'note':'Kernel sums overlap when PDL is on. PDL0 is a changed scheduling intervention, not the baseline critical path.'}
 (root/f'{label}-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 print(label,'rounds',len(rounds),'wall',sum(r['wall_ms'] for r in rounds),'busy',sum(r['kernel_union_ms'] for r in rounds),'sum',sum(r['kernel_sum_ms'] for r in rounds),'families',dict(families))
 for r in result['kernels'][:12]:print(r['name'],r['sum_ms'],r['median_us'],r['shapes'][:3])
