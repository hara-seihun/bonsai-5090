#!/usr/bin/env python3
"""Price CUDA interval overlap inside explicit replay-target NVTX ranges."""
import argparse, collections, json, sqlite3
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('sqlite',type=Path);a=p.parse_args()
if not a.sqlite.is_file():p.error('trace SQLite file does not exist')
c=sqlite3.connect(a.sqlite.resolve().as_uri()+'?mode=ro',uri=True)
ranges=c.execute("select n.start,n.end from NVTX_EVENTS n left join StringIds s on s.id=n.textId where coalesce(n.text,s.value)='replay-target' order by n.start").fetchall()
if not ranges or any(y is None or y<=x for x,y in ranges):raise ValueError('missing/invalid replay-target ranges')
rows=c.execute('select k.start,k.end,s.value,k.demangledName from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start').fetchall()
def family(n):
 if n=='k_gemm_fp4':return 'gemm'
 if n.startswith('k_prep'):return 'prep'
 if n in ('k_conv','k_gdn'):return 'conv_gdn'
 if n.startswith('k_attn'):return 'attention'
 if n.startswith('graph_'):return 'memory'
 return 'other'
res=collections.defaultdict(lambda:collections.Counter());inside=0;span=0
for lo,hi in ranges:
 active=collections.Counter();ev=[];ends=[]
 for st,en,n,_ in rows:
  if en<=lo:continue
  if st>=hi:break
  if st<lo or en>hi:raise ValueError('kernel crosses target boundary')
  f=family(n);res[f]['residency_ns']+=en-st;res[f]['count']+=1;inside+=1;ev.extend([(st,f,1),(en,f,-1)]);ends.extend([st,en])
 if not ev:raise ValueError('empty target range')
 span+=max(ends)-min(ends);last=min(ends)
 for t,f,d in sorted(ev):
  delta=t-last;live=[k for k,v in active.items() if v>0]
  if live:
   res['all']['union_ns']+=delta
   if len(live)>1:res['all']['cross_family_overlap_ns']+=delta
   else:res[live[0]]['exclusive_ns']+=delta
  active[f]+=d;last=t
 if any(active.values()):raise ValueError('unclosed interval')
N=len(ranges)
print(json.dumps({'scope':'CUDA kernel intervals fully inside replay-target NVTX; prefill excluded; traced residency includes waits/observation perturbation', 'steps':N,'kernels':inside,'gpu_span_ms_per_step':span/N/1e6,'families':{k:dict(v)|{'residency_ms_per_step':v['residency_ns']/N/1e6,'exclusive_ms_per_step':v['exclusive_ns']/N/1e6,'count_per_step':v['count']/N} for k,v in res.items()},'union_ms_per_step':res['all']['union_ns']/N/1e6,'cross_family_overlap_ms_per_step':res['all']['cross_family_overlap_ns']/N/1e6},indent=2))
