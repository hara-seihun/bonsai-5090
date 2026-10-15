from pathlib import Path
import json,struct,hashlib
p=Path(__file__).parent;old=p.parent/'gu-scale-absorb'/'evidence'
def unpack(block):
 a=[]
 for start,count in [(0,16),(16,8)]:
  for n in range(5):
   for m in range(count):
    q=block[start+m]
    for _ in range(n):q=(q*3)&255
    a.append(((q*3)>>8)-1)
 for n in range(4):
  for h in range(2):
   q=block[24+h]
   for _ in range(n):q=(q*3)&255
   a.append(((q*3)>>8)-1)
 return a
qcode=lambda t:10 if t<0 else 2 if t>0 else 0
def table(c,odd):
 g=u=ng=nu=0
 for j in range(5):
  v=qcode(c%3-1);c//=3
  if (j+odd)%2==0:g|=v<<(4*ng);ng+=1
  else:u|=v<<(4*nu);nu+=1
 return g|(u<<16)
graw=(old/'blk.0.ffn_gate.weight.rows32.bin').read_bytes();uraw=(old/'blk.0.ffn_up.weight.rows32.bin').read_bytes()
words=0;serialized=bytearray();means_bits=[]
for kb in range(40):
 g=[unpack(graw[(r*40+kb)*28:(r*40+kb+1)*28]) for r in range(32)]
 u=[unpack(uraw[(r*40+kb)*28:(r*40+kb+1)*28]) for r in range(32)]
 payload=bytearray(1664)
 for lane in range(32):
  gs=[];us=[]
  for w in range(16):
   mm,h,i=w//8,w//4%2,w%4
   r=16*mm+8*(i&1)+(lane>>2)
   for j in range(8):
    k=64*h+32*(i>>1)+8*(lane&3)+j
    gs.append(g[r][k]);us.append(u[r][k])
  digits=[]
  for i in range(130):digits += [gs[i]+1,us[i]+1] if i<128 else [1,1]
  labels=[]
  for i in range(0,260,5):labels.append(sum(digits[i+j]*3**j for j in range(5)))
  assert len(labels)==52 and max(labels)<243
  for k in range(13):payload[k*128+lane*4:k*128+lane*4+4]=bytes(labels[k*4:k*4+4])
 for lane in range(32):
  labels=b''.join(payload[k*128+lane*4:k*128+lane*4+4] for k in range(13));ge=[];ue=[]
  for i in range(26):
   a=table(labels[2*i],0);b=table(labels[2*i+1],1)
   ge.append((a&0xfff)|((b&255)<<12));ue.append((a>>16)|((b>>16)<<8))
  for w in range(16):
   idx,sh=8*w//5,(8*w%5)*4
   gg=(ge[idx]>>sh)|(ge[idx+1]<<(20-sh));uu=(ue[idx]>>sh)|(ue[idx+1]<<(20-sh))
   if sh>=12:gg|=ge[idx+2]<<(40-sh);uu|=ue[idx+2]<<(40-sh)
   mm,h,i=w//8,w//4%2,w%4;r=16*mm+8*(i&1)+(lane>>2)
   eg=eu=0
   for j in range(8):
    k=64*h+32*(i>>1)+8*(lane&3)+j
    eg|=qcode(g[r][k])<<(4*j);eu|=qcode(u[r][k])<<(4*j)
   assert gg&0xffffffff==eg and uu&0xffffffff==eu
   words+=2
 serialized.extend(payload)
(p/'paired-block0-rows32-k5120.bin').write_bytes(serialized)
r={'gpu_work':False,'native_model_execution':False,'source_fixed_rows_sha256':{'gate':hashlib.sha256(graw).hexdigest(),'up':hashlib.sha256(uraw).hexdigest()},'source_model_sha256':json.loads((old/'bindings.json').read_text())['model_sha256'],'K':5120,'logical_rows':32,'lane_operand_words':words,'word_mismatches':0,'joint_payload_bytes':len(serialized),'joint_payload_sha256':hashlib.sha256(serialized).hexdigest(),'rounding_assumption':'none; original separate operand bits, source MMA call order and mean epilogue retained','scope':'CPU serialized-source/layout proof plus matched compiler probe; no native GPU acceptance or full observer/timing observation.'}
(p/'actual-rows-receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
