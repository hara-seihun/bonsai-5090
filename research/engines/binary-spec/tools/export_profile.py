#!/usr/bin/env python3
"""Export every observed launch/API/copy and a first-round overlap timeline."""
import argparse, collections, csv, html, json, pathlib, sqlite3
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=pathlib.Path(a.directory)
for label,rel in [('request','node/trace.sqlite'),('verify','replay/pdl1-node.sqlite'),('serial','replay/pdl0-node.sqlite'),('ordinary','ordinary/node.sqlite')]:
 db=sqlite3.connect(f'file:{root/rel}?mode=ro',uri=True);db.row_factory=sqlite3.Row
 ids=dict(db.execute('select id,value from StringIds'))
 out=root/'inventory'/label;out.mkdir(parents=True,exist_ok=True)
 for table,short in [('CUPTI_ACTIVITY_KIND_KERNEL','kernels'),('CUPTI_ACTIVITY_KIND_MEMCPY','copies'),('CUPTI_ACTIVITY_KIND_MEMSET','memsets'),('CUPTI_ACTIVITY_KIND_RUNTIME','api'),('NVTX_EVENTS','ranges')]:
  rows=[]
  for r in db.execute(f'select * from {table} order by start'):
   r=dict(r)
   for col in ['nameId','demangledName','mangledName','shortName','textId']:
    if col in r and r[col] is not None:r[col+'_text']=ids[r[col]]
   rows.append(r)
  if not rows:continue
  with (out/f'{short}.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 if label=='request':continue
 rr=db.execute("select start,end from NVTX_EVENTS where text='replay-target' order by start limit 1").fetchone()
 if rr is None:raise RuntimeError('no replay-target range')
 start,end=rr
 rows=list(db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=? order by start',(start,end)))
 def family(name):
  return 'GEMM' if 'gemm_fp4' in name else 'Preparation' if 'prep2' in name else 'GDN' if 'k_gdn' in name else 'Convolution' if 'k_conv' in name else 'Attention' if 'attn' in name else 'Other'
 families=['GEMM','Preparation','GDN','Convolution','Attention','Other'];colors=['#a23cec','#f44f4f','#28bd90','#e9b232','#43a4db','#bbbbbb']
 svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1500 350">','<rect width="1500" height="350" fill="#151515"/>',f'<text x="25" y="30" fill="white" font-family="sans-serif" font-size="20">{label}: first fixed-work round, {(end-start)/1e6:.3f} ms — overlapping intervals</text>']
 for i,cat in enumerate(families):
  y=70+i*38;svg.append(f'<text x="20" y="{y+17}" fill="white" font-family="sans-serif">{cat}</text>')
  for r in rows:
   name=ids[r['demangledName']]
   if family(name)!=cat:continue
   x=160+(r['start']-start)/(end-start)*1300;width=max(.2,(r['end']-r['start'])/(end-start)*1300)
   title=html.escape(f"{name}; {(r['end']-r['start'])/1000:.3f} us; grid {r['gridX']}x{r['gridY']}x{r['gridZ']}; {r['registersPerThread']} registers/thread")
   svg.append(f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="25" fill="{colors[i]}" opacity="0.75"><title>{title}</title></rect>')
 for i in range(8):
  x=160+1300*i/7;svg.append(f'<text x="{x:.1f}" y="325" fill="#ccc" font-family="sans-serif">{(end-start)/1e6*i/7:.2f} ms</text>')
 svg.append('</svg>');(root/f'{label}-timeline.svg').write_text('\n'.join(svg)+'\n')
 print(label,len(rows),'kernels in first round')
