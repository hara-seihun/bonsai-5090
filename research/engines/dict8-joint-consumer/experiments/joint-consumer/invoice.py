from pathlib import Path
import collections,hashlib,json,re,struct
p=Path(__file__).resolve().parent
rows=[]
for name in ['native0','baseline','split53_shared','split53_shuffle','shared8','shared8_compact','shared8_dual','literal3']:
 kernel='0' if name=='native0' else '10'
 text=(p/'build'/f'{name}.sass').read_text()
 token=f'Function : _ZN2h513dict8_backend6kernelILi{kernel}ELb0E'
 part=text.split(token)[1].split('Function : ')[0]
 ins=[(int(a,16),x) for a,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?);',part)]
 candidates=[]
 for a,x in ins:
  if 'BRA ' not in x:continue
  t=re.search(r'0x([0-9a-f]+)',x)
  if not t or int(t[1],16)>=a:continue
  lo=int(t[1],16);body=[(ad,i) for ad,i in ins if lo<=ad<=a]
  if sum('OMMA.' in i for _,i in body)==32:
   candidates.append((len(body),lo,a,body))
 if not candidates:raise RuntimeError('no main-K loop: '+name)
 _,lo,hi,body=min(candidates)
 ops=collections.Counter(re.sub(r'^@!?P\d+ +','',x).split()[0] for _,x in body)
 log=(p/'build'/f'{name}.log').read_text()
 entry=log.split(f"Compiling entry function '_ZN2h513dict8_backend6kernelILi{kernel}ELb0E")[1].split('Compiling entry function')[0]
 regs=int(re.search(r'Used (\d+) registers',entry)[1]); stack=int(re.search(r'(\d+) bytes stack frame',entry)[1])
 resource=(p/'build'/f'{name}.resources').read_text().split(f'Function _ZN2h513dict8_backend6kernelILi{kernel}ELb0E')[1].split('Function ')[0]
 static_smem=int(re.search(r'SHARED:(\d+)',resource)[1]); local=int(re.search(r'LOCAL:(\d+)',resource)[1])
 # Dynamic resource requested by the exact generated wrapper.
 fw=4 if name=='shared8_dual' else 8; stages=2 if fw==4 else 4
 cw=1024 if name=='native0' else 864 if name=='literal3' else 832;extra=26368 if name.startswith('shared8') else 1152 if name=='split53_shared' else 1024 if name=='split53_shuffle' else 0
 sws=0 if name in ['shared8_compact','shared8_dual','literal3'] else fw*2*64
 dyn=stages*(fw*2*cw+sws+32*72)+stages*16+extra
 rows.append(dict(route=name,whole_kernel_static_instructions=len(ins),kloop_static_instructions=len(body),kloop_start=hex(lo),kloop_backedge=hex(hi),kloop_opcodes=dict(ops),registers=regs,stack=stack,static_shared_bytes=static_smem,local_bytes=local,zero_spills='0 bytes spill stores, 0 bytes spill loads' in entry,threads=(fw+1)*32,dynamic_shared_bytes=dyn,source_sha256=hashlib.sha256((p/f'{name}.cu').read_bytes()).hexdigest(),cubin_sha256=hashlib.sha256((p/'build'/f'{name}.cubin').read_bytes()).hexdigest()))
 (p/'build'/f'{name}-kloop.txt').write_text('\n'.join(hex(a)+' '+x for a,x in body)+'\n')
# Price book-bank address collisions from existing real-row payload; no CUDA load.
f=p/'fixtures/dict8.bin'
b=f.read_bytes();assert len(b)==40*1664
banks={k:[] for k in ['full8','low5','high3']}
for kb in range(40):
 for w in range(16):
  for arm in range(2):
   keys=[]
   for lane in range(32):
    plane=[struct.unpack_from('<I',b,kb*1664+j*128+lane*4)[0] for j in range(13)]
    key=(sum(x<<(32*i) for i,x in enumerate(plane))>>(26*w+13*arm))&8191
    assert key<6561;keys.append(key)
   for route,addresses in [('full8',keys),('low5',[k%243 for k in keys]),('high3',[k//243 for k in keys])]:
    buckets=[set() for _ in range(32)]
    for a in addresses:buckets[a%32].add(a)
    banks[route].append(max(map(len,buckets)))
book_init=6561*4*136
result={'gpu_execution':False,'compiler':'CUDA13.1 V13.1.115 sm_120a Linux, same flags across routes','kernel_scope':'R64 K5120 N34816, ordinary capture=false; whole kernel includes producer, setup, nonlinear suffix and outlined code; Kloop excludes out-of-line wait/collective paths','rows':rows,'logical_bytes':{'native_weights_layer':44564480,'compact_weights_layer':36208640,'compact_weights_64layers':2317352960,'native_weights_TMA_layer_decode':89128960,'compact_weights_TMA_layer_decode':72417280,'literal_weights_layer':37601280,'literal_weights_TMA_layer_decode':75202560,'literal_weights_64layers':2406481920,'full8_book_retained':26244,'full8_book_global_lane_read_bytes_per_layer_decode':book_init,'dual_book_global_lane_read_bytes_per_layer_decode':2*book_init,'full8_book_shared_lane_read_bytes_per_layer_decode':178257920,'padding':{'pair_weight_payload':0,'shared_book_reserved_minus_values':124,'unused_scale_region_removed_per_CTA':4096}},'book_bank_address_model':{r:{'warp_loads':len(v),'mean_max_distinct_words_per_bank':sum(v)/len(v),'worst_max_distinct_words_per_bank':max(v),'histogram':dict(collections.Counter(v))} for r,v in banks.items()},'fixture':{'path':str(f),'sha256':hashlib.sha256(b).hexdigest()},'limits':['logical load bytes and bank-address multiplicities are not DRAM/cache/latency measurements','register/shared-memory capacities do not establish actual occupancy or speed','no new observer, native startup, whole model, HTTP, or task acceptance','cache owner5bdb08fa remains independent; current shared8 payload is compatible, split53 payload requires a different semantic binding']}
(p/'invoice.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['route','kloop_static_instructions','registers','dynamic_shared_bytes','threads']} for r in rows],indent=2));print(json.dumps(result['book_bank_address_model'],indent=2))
