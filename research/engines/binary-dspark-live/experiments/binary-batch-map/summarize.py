#!/usr/bin/env python3
import collections
import hashlib
import json
import pathlib
import re
import sys
root = pathlib.Path(__file__).resolve().parent
if len(sys.argv) == 2:
    text = pathlib.Path(sys.argv[1]).read_text()
    functions = []
    for block in text.split('Function : ')[1:]:
        name = block.splitlines()[0].strip()
        if not name.startswith('_ZN9batch_map10projection'):
            continue
        opcodes = collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)', block))
        functions.append({'symbol': name, 'sass_sha256': hashlib.sha256(block.encode()).hexdigest(), 'opcodes': dict(opcodes)})
    assert len(functions) == 16
    (root/'results/sass-receipt.json').write_text(json.dumps({'arch':'sm_120a','functions':functions}, indent=2)+'\n')
records = [json.loads(line) for f in sorted((root/'results').glob('*.jsonl')) for line in f.read_text().splitlines()]
if not records:
    sys.exit(0)
keys = sorted({(r['tensor'],r['n'],r['k'],r['r']) for r in records})
summary=[]
for key in keys:
    subset = [r for r in records if (r['tensor'],r['n'],r['k'],r['r']) == key]
    dp = [r for r in subset if r['route']=='dp4a_selected']
    candidates = [r for r in subset if r['route']=='imad_pair']
    assert len(dp)==1 and len(candidates)==8
    best=min(candidates,key=lambda r:r['cold_us'])
    summary.append({'tensor':key[0],'n':key[1],'k':key[2],'r':key[3], 'dp4a':dp[0], 'imad_best':best, 'speedup':dp[0]['cold_us']/best['cold_us']})
    print(f"{key[0]:24} R{key[3]:2} DP4A {dp[0]['cold_us']:8.3f} pair {best['cold_us']:8.3f} ratio {dp[0]['cold_us']/best['cold_us']:.3f} plan {best['chunk']}/{best['threads']}")
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
