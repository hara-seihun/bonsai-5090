from pathlib import Path
import struct,json,collections
p=Path(__file__).parent;raw=(p/'dict8-block0-rows32-k5120.bin').read_bytes();hist=collections.Counter();total_rounds=0;total_loads=0
# Static SM shared-memory-bank address model for 32-bit loads: 32 banks,4B.
# Equal addresses in a bank broadcast; different addresses serialize. This is
# address geometry, not a GPU latency or cycle observation.
for kb in range(40):
 tile=raw[kb*1664:(kb+1)*1664]
 values=[sum(struct.unpack_from('<I',tile,i*128+lane*4)[0]<<(32*i) for i in range(13)) for lane in range(32)]
 for w in range(16):
  for part in range(2):
   codes=[(v>>(26*w+13*part))&8191 for v in values]
   banks=[set() for _ in range(32)]
   for code in codes:
    assert code<6561;banks[code%32].add(code)
   rounds=max(map(len,banks));hist[rounds]+=1;total_rounds+=rounds;total_loads+=1
r={'GPU_execution':False,'actual_fixed_rows':32,'K':5120,'dictionary_warp_loads_per_40stage_consumer':total_loads,'predicted_shared_bank_rounds_histogram':dict(sorted(hist.items())),'average_distinct_address_rounds':total_rounds/total_loads,'scope':'4B/32-bank address model including equal-address broadcast. No measured GPU bank/latency or cache-traffic claim.'}
(p/'dict8-bank-price.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
