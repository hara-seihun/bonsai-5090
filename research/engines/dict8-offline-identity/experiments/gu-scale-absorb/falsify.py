"""CPU exact-coefficient and binary32 witness; no CUDA/module/model execution."""
from pathlib import Path
from fractions import Fraction
import json,struct,hashlib
root=Path(__file__).with_name('evidence')
b=json.loads((root/'bindings.json').read_text())
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
def scale512(c):
 e,m=c>>3,c&7
 return m if e==0 else (8+m)*(1<<(e-1))
q2=[0,1,2,3,4,6,8,12]
products={Fraction(q*scale512(c),1024):(q,c) for q in q2 for c in range(127)}
def decode(block):
 trit=[]
 for start,count in [(0,16),(16,8)]:
  for n in range(5):
   for m in range(count):
    q=block[start+m]
    for _ in range(n):q=(q*3)&255
    trit.append(((q*3)>>8)-1)
 for n in range(4):
  for h in range(2):
   q=block[24+h]
   for _ in range(n):q=(q*3)&255
   trit.append(((q*3)>>8)-1)
 assert len(trit)==128
 return trit
witnesses=[]
for row in b['rows']:
 raw=(root/(row['tensor']+'.rows32.bin')).read_bytes()[row['row']*1120:(row['row']+1)*1120]
 trits=sum([decode(raw[k*28:(k+1)*28]) for k in range(40)],[])
 nz=next(i for i,t in enumerate(trits) if t!=0)
 # A maximum activation at a zero-weight coordinate makes the global-scale=1
 # raw NVFP4 operand row coherent; it contributes nothing to this output.
 guard=next(i for i,t in enumerate(trits) if t==0 and i//16!=nz//16)
 mean=f32(row['mean']);target=f32(trits[nz]*f32(.5*mean))
 nearest=min(products,key=lambda p:abs(float(p)-mean))
 actual=f32(trits[nz]*f32(.5*float(nearest)))
 assert bits(target)!=bits(actual)
 witness={'tensor':row['tensor'],'row':row['row'],'weight_nz_k':nz,'weight_trit':trits[nz],'zero_weight_amax_guard_k':guard,'activation':{'nz_q4_positive_code':1,'nz_s4':56,'guard_q4_positive_code':7,'guard_s4':126,'g4':1.,'other_q4':0},'original_contraction':trits[nz]*.5,'mean_bits':row['mean_bits'],'target_bits':bits(target),'nearest_one_plane_coefficient':float(nearest),'nearest_q2_s4':products[nearest],'absorbed_bits':bits(actual),'different':True}
 witnesses.append(witness)
# An exactly representable 1.5 mean still cannot commute through binary32 reduction.
# Valid K5120 interface values: 4096 terms of 2048, then zeros, then 0.5.
# First 4096 and their absorbed counterparts sum exactly under any grouping;
# only the last OMMA stage has the half-ulp term.
a=f32(0);a_abs=f32(0)
for i in range(5120):
 x=2048. if i<4096 else .5 if i==5119 else 0.
 a=f32(a+x);a_abs=f32(a_abs+f32(1.5*x))
original=f32(a*1.5)
assert bits(original)!=bits(a_abs)
receipt={'state':'route-specific-falsifier','gpu_work':False,'native_execution':False,'input_custody_sha256':hashlib.sha256((root/'bindings.json').read_bytes()).hexdigest(),'source_engine_sha256':b['engine_sha256'],'source_fp4_sha256':b['fp4_sha256'],'model_sha256':b['model_sha256'],'rows_examined':len(witnesses),'one_plane_exact_rows':0,'power_of_two_residual_odd_bound_failed_rows':b['odd_bound45_failures'],'witnesses':witnesses,'rounding_noncommutation':{'K':5120,'mean':1.5,'first_terms':4096,'first_value':2048.,'last_value':.5,'before_scale_bits':bits(a),'original_bits':bits(original),'absorbed_bits':bits(a_abs),'grouping_scope':'4096 equal exact terms; one half-ulp term alone in final 128-K stage'},'scope':'Actual fixed model rows plus constructed legal q4/s4/g4 interface operands. Not saved model activations, native GPU acceptance or a final-observer/TPS claim.'}
Path(__file__).with_name('falsifier.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k not in ['witnesses']}));print(json.dumps(witnesses[0]))
