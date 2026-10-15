#!/usr/bin/env python3
import json
from pathlib import Path
root=Path(__file__).resolve().parent
records=[json.loads(line) for p in sorted((root/'results').glob('*-r*.jsonl')) for line in p.read_text().splitlines()]
summary=[]
for key in sorted({(r['kind'],r['r']) for r in records}):
    subset=[r for r in records if (r['kind'],r['r'])==key]
    routes={}
    for route in ('ordered_two_pass','joint_pair','dp4a','native_sf1','native_sf2'):
        rows=[r for r in subset if r['route']==route]
        if not rows: raise RuntimeError(f'missing route {key} {route}')
        routes[route]={'warm':min(rows,key=lambda r:r['inclusive_warm_us']),'cold':min(rows,key=lambda r:r['inclusive_cold_us'])}
    for route in ('native_sf1','native_sf2'):
        for cache in ('warm','cold'):
            for control in ('ordered_two_pass','joint_pair','dp4a'):
                routes[route][cache]['speedup_vs_'+control]=routes[control][cache][f'inclusive_{cache}_us']/routes[route][cache][f'inclusive_{cache}_us']
    summary.append({'kind':key[0],'r':key[1],'routes':routes})
(root/'summary.json').write_text(json.dumps({'records':len(records),'projections':summary},indent=2)+'\n')
for s in summary:
    print(s['kind'],s['r'])
    for route,rs in s['routes'].items():
        c=rs['cold'];print(route,'warm',rs['warm']['inclusive_warm_us'],'cold',c['inclusive_cold_us'],'speedup vs ordered/joint/dp4a',[c.get('speedup_vs_'+s) for s in ('ordered_two_pass','joint_pair','dp4a')],'cfg/split',c['cfg'],c['splits'],'projection rms',c['relative_rms_ordered'],'scale rms',c['scale_relative_rms'],'bytes',c['weight_resident_bytes'])
