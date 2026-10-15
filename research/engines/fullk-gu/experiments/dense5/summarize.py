#!/usr/bin/env python3
import json
from pathlib import Path
out=Path('results/dense5')
rows=[json.loads(x) for x in (out/'native.jsonl').read_text().splitlines()]
cert=[x for x in rows if x['stage'] in ('native_correctness','native_register_certificate')]
summary=[x for x in rows if x['stage']=='summary']
assert len(summary)==8 and len(cert)==9
assert all(x.get('bit_mismatches',0)==0 for x in cert)
result={'lane':'dense5-single','execution':'passed','hardware':rows[0],
        'correctness':cert,'summary':summary,
        'scope':'Real layer10 complete FFN, rows1/8. Not full-model single-stream proof.',
        'adopted':False,'all_warm_and_perturbed_win':all(x['ratio_to_current_baseline']<1 for x in summary if x['arm']=='dense5_native')}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
