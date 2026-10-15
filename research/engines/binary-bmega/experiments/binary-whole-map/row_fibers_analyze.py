import hashlib,json,struct
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
root=Path('/workspace/binary/row-fibers');data=(root/'packets.bin').read_bytes();off=0;groups=defaultdict(list);calls=[];fingerprints=[];raw=bytearray();samples=[]
names={0:'wiki_chunk_prefill',1:'heterogeneous_panel_teacher_forcing',2:'identical_prompt_positive_control'}
token_data=(root/'packets.bin.tokens').read_bytes();toff=0;tokens={}
while toff<len(token_data):
 i,n=struct.unpack_from('<2i',token_data,toff);toff+=8;tokens[i]=struct.unpack_from('<'+str(n)+'i',token_data,toff);toff+=4*n
prefix_by_packet=defaultdict(set)
def sha(x):return hashlib.sha256(x).hexdigest()
def census(ps):
 # Python byte-key equality resolves hash collisions by comparing the complete packet.
 u=Counter(ps);q=Counter(x[:5120] for x in ps);sxq=Counter(x[:5280] for x in ps)
 antipodes=[]
 for x in ps:
  iq=np.frombuffer(x[:5120],np.int8).astype(np.int16)
  if np.any(iq==-128):raise ValueError('A8 negation outside supported domain')
  ndc=-np.frombuffer(x[5280:5440],'<i4')
  anti=(-iq).astype(np.int8).tobytes()+x[5120:5280]+ndc.tobytes()+x[5440:5600]
  antipodes.append(anti)
 orbits=Counter(min(x,a) for x,a in zip(ps,antipodes));anti_by_packet=dict(zip(ps,antipodes))
 return {'rows':len(ps),'distinct_complete_packets':len(u),'exact_duplicate_rows':len(ps)-len(u),'row_compression_ratio':len(ps)/len(u),'distinct_q_only':len(q),'distinct_q_sx':len(sxq),'antipodal_orbits':len(orbits),'antipodal_pair_classes':sum(1 for x in u if anti_by_packet[x] in u and anti_by_packet[x]!=x)//2,'zero_input_rows':sum(not any(x[:5120]) for x in ps),'multiplicity_histogram':dict(sorted(Counter(u.values()).items()))}
while off<len(data):
 if off+28>len(data):raise ValueError('truncated header')
 magic,mode,r,call,l,k,pos=struct.unpack_from('<7i',data,off);off+=28
 if magic!=0x52464231 or mode not in names or r not in (16,32) or l not in (0,30,61) or k!=5120:raise ValueError('invalid header')
 ps=[]
 for row in range(r):
  x=data[off:off+5600];off+=5600
  if len(x)!=5600 or any(x[5440:]):raise ValueError('truncated or nonvalid carrier')
  iq=np.frombuffer(x[:5120],np.int8).astype(np.int32).reshape(40,128)
  if not np.array_equal(iq.sum(axis=1),np.frombuffer(x[5280:5440],'<i4')):raise ValueError('DC does not match q')
  prefix=tokens[-1][:pos+row+1] if mode==0 else tokens[0 if mode==2 else row][:pos+1]
  prefix_by_packet[(mode,r,l,x)].add(prefix)
  ps.append(x);fingerprints.append({'token_prefix_sha256':sha(struct.pack('<'+str(len(prefix))+'i',*prefix)),'regime':names[mode],'R':r,'call':call,'position_base':pos,'layer':l,'row':row,'q_sha256':sha(x[:5120]),'complete_packet_sha256':sha(x)})
 groups[(mode,r,l)].extend(ps)
 prefixes=[tokens[-1][:pos+row+1] if mode==0 else tokens[0 if mode==2 else row][:pos+1] for row in range(r)]
 c={'regime':names[mode],'R':r,'call':call,'position_base':pos,'layer':l,'distinct_token_prefixes':len(set(prefixes)),**census(ps)};calls.append(c)
 if call in (0,32) and len(ps)>1:
  for row in (0,1):samples.append({'regime':names[mode],'R':r,'call':call,'layer':l,'row':row,'offset':len(raw),'bytes':5600,'sha256':sha(ps[row])});raw.extend(ps[row])
summary=[]
for (mode,r,l),ps in sorted(groups.items()):
 cc=[x for x in calls if x['regime']==names[mode] and x['R']==r and x['layer']==l]
 summary.append({'regime':names[mode],'R':r,'layer':l,**census(ps),'calls':len(cc),'within_call_unique_sum':sum(x['distinct_complete_packets'] for x in cc),'within_call_duplicate_rows':sum(x['exact_duplicate_rows'] for x in cc),'within_call_compression_ratio':len(ps)/sum(x['distinct_complete_packets'] for x in cc),'per_call_distinct':[x['distinct_complete_packets'] for x in cc],'per_call_distinct_token_prefixes':[x['distinct_token_prefixes'] for x in cc],'complete_packet_classes_spanning_distinct_token_prefixes':sum(len(v)>1 for (m,rr,ll,_),v in prefix_by_packet.items() if (m,rr,ll)==(mode,r,l))})
report={'source':'digits-ragged/halo source374c77d','observer':'real h5_step teacher-forced tokens; selected FFN launch_prep boundary captures immediately preceding native bitserial originalA8 produce(D) q/scales/DC/status; no synthetic activation regeneration','weight_contract':'original model Q1 signs and K128 FP16 scales unchanged','packet':'5120 signedA8 bytes, 40 float32 scale bitpatterns, 40 int32 DC, 40 uint32 status; DC audited against q; status all Valid','packet_bytes':5600,'minimal_q_scale_status_bytes':5440,'selected_layers':[0,30,61],'summary':summary,'capture_sha256':sha(data),'capture_bytes':len(data),'observed_rows':len(fingerprints),'calls':calls}
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n');(root/'fingerprints.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in fingerprints));(root/'samples.bin').write_bytes(raw);(root/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
for x in summary:print(json.dumps({k:v for k,v in x.items() if k!='per_call_distinct'}))
