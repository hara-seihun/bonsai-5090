#!/usr/bin/env python3
import hashlib,json,re,gzip
from pathlib import Path
E=Path(__file__).resolve().parent
D=E/'sparse-digit-results'
summary={'result':'bounded native loss; retain dense exact3; no full-model promotion', 'scope':'actual layer0 hidden F32 exports row-stacked for trained layer0 DOWN, not simultaneous model batch','rows':{},'witness':{},'source_pins':{}}
for r in (16,32):
 data=[json.loads(l) for l in (D/f'down-r{r}.jsonl').read_text().splitlines()]
 rows=[x for x in data if x['kind']=='down']
 result={'packet':data[0],'timed_arms':len(rows),'same_plan_exact3_bitexact_arms':sum(x['same_plan_exact3_bitexact'] for x in rows),'max_independent_raw_error':max(x['max_error'] for x in rows),'max_independent_raw_relative_rms':max(x['relative_rms'] for x in rows),'best':{}}
 for boundary in dict.fromkeys(x['boundary'] for x in rows):
  result['best'][boundary]={}
  for route in dict.fromkeys(x['route'] for x in rows):
   route_rows=[x for x in rows if x['route']==route and x['boundary']==boundary]
   result['best'][boundary][route]={key:min(route_rows,key=lambda x:x[key+'_us']) for key in ('cold','warm')}
 summary['rows'][r]=result
if (D/'witness-r16.jsonl').exists():
 witness=[json.loads(l) for l in (D/'witness-r16.jsonl').read_text().splitlines()]
 summary['witness']={'plans':len(witness),'max_independent_raw_error':max(x['max_error'] for x in witness),'max_independent_raw_relative_rms':max(x['relative_rms'] for x in witness),'same_plan_bitexact':all(x.get('same_plan_exact3_bitexact',False) for x in witness),'modes':sorted({x['mode'] for x in witness if 'mode' in x})}
for p in (E/'sparse_digits.cu',E/'sparse_digits.cuh',E/'sparse_digits_bench.cu',D/'sparse_digits.o',D/'sparse_digits.sass.gz',D/'timing-bench.cu',D/'captured-hidden.f32',D/'sparse_a8.lean'):
 summary['source_pins'][str(p.relative_to(E))]=hashlib.sha256(p.read_bytes()).hexdigest()
sass=gzip.decompress((D/'sparse_digits.sass.gz').read_bytes()).decode()
functions=[]
for body in sass.split('Function : ')[1:]:
 name=body.splitlines()[0].strip();ops=re.findall(r'\*/\s+([A-Z][A-Z0-9_.]+)\s',body)
 functions.append({'symbol':name,'static_instruction_count':len(ops),'native_fp4_OMMA_instructions':sum(o.startswith('OMMA.') for o in ops),'global_gather_U16_instructions':sum(o.startswith('LDG.E.U16') for o in ops),'raw_q_S8_loads':sum(o.startswith('LDG.E.S8') for o in ops),'note':'static compiled instruction census, not dynamic issue count'})
summary['sass']=functions
summary['timing_object_pins']={parts[1]:parts[0] for line in (D/'timing-provenance.sha256').read_text().splitlines() if len(parts:=line.split())==2}
summary['exported_object_scope']='Final stronger-witness rebuild; timing pins separately retained, same candidate source and flags'
summary['lean_receipt']=(D/'sparse_a8.lean.log').read_text().strip()
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Native sparse-third loss')
for r,x in summary['rows'].items():
 best=x['best']['producer_encode_project_residual_next_a8']
 print('R',r,[(route,y['cold']['cold_us'],y['cold']['cfg'],y['cold']['splits']) for route,y in best.items()])
print(summary['witness'])
