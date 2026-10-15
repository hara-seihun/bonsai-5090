from pathlib import Path
import json,struct,hashlib
p=Path(__file__).parent;old=p.parent/'gu-scale-absorb'/'evidence'
# The same original PTQ byte decoder used by the joint5-pair closure.
s=(p/'actual_rows.py').read_text();exec(s[s.index('def unpack'):s.index('qcode=lambda')])
qcode=lambda t:10 if t<0 else 2 if t>0 else 0
dict_words=[]
for code in range(6561):
 q=code;word=0
 for j in range(8):word|=qcode(q%3-1)<<(4*j);q//=3
 assert q==0;dict_words.append(word)
bdict=b''.join(struct.pack('<I',w) for w in dict_words)
(p/'native8-dictionary.bin').write_bytes(bdict)
graw=(old/'blk.0.ffn_gate.weight.rows32.bin').read_bytes();uraw=(old/'blk.0.ffn_up.weight.rows32.bin').read_bytes()
serialized=bytearray();words=0
for kb in range(40):
 g=[unpack(graw[(r*40+kb)*28:(r*40+kb+1)*28]) for r in range(32)]
 u=[unpack(uraw[(r*40+kb)*28:(r*40+kb+1)*28]) for r in range(32)]
 payload=bytearray(1664)
 for lane in range(32):
  packed=0
  for w in range(16):
   mm,h,i=w//8,w//4%2,w%4;r=16*mm+8*(i&1)+(lane>>2)
   codes=[0,0]
   for j in range(8):
    k=64*h+32*(i>>1)+8*(lane&3)+j
    codes[0]+=(g[r][k]+1)*3**j;codes[1]+=(u[r][k]+1)*3**j
   assert max(codes)<6561
   packed|=codes[0]<<(26*w);packed|=codes[1]<<(26*w+13)
  assert packed.bit_length()<=416
  for i in range(13):struct.pack_into('<I',payload,i*128+lane*4,(packed>>(32*i))&0xffffffff)
 for lane in range(32):
  packed=sum(struct.unpack_from('<I',payload,i*128+lane*4)[0]<<(32*i) for i in range(13))
  for w in range(16):
   gc=(packed>>(26*w))&8191;uc=(packed>>(26*w+13))&8191
   assert gc<6561 and uc<6561
   mm,h,i=w//8,w//4%2,w%4;r=16*mm+8*(i&1)+(lane>>2)
   eg=eu=0
   for j in range(8):
    k=64*h+32*(i>>1)+8*(lane&3)+j
    eg|=qcode(g[r][k])<<(4*j);eu|=qcode(u[r][k])<<(4*j)
   assert dict_words[gc]==eg and dict_words[uc]==eu;words+=2
 serialized.extend(payload)
(p/'dict8-block0-rows32-k5120.bin').write_bytes(serialized)
r={'gpu_work':False,'native_model_execution':False,'model_sha256':json.loads((old/'bindings.json').read_text())['model_sha256'],'dictionary_entries':6561,'dictionary_bytes':len(bdict),'dictionary_sha256':hashlib.sha256(bdict).hexdigest(),'K':5120,'rows':32,'original_native_words':words,'mismatches':0,'packed_bytes':len(serialized),'packed_sha256':hashlib.sha256(serialized).hexdigest(),'rounding_assumption':'none; exact separate operand words and original per-output native MMA/scale/mean/SiLU order retained','GPU_acceptance':False}
(p/'dict8-rows-receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
