#!/usr/bin/env python3
"""Reduce durable batch logs to their numeric guards and measured rates."""
import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('logs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args();reports=[]
for name in a.logs:
 path=pathlib.Path(name);records=[]
 for line in path.read_text().splitlines():
  try:item=json.loads(line)
  except json.JSONDecodeError:continue
  if isinstance(item,dict):records.append(item)
 guards=[v for v in records if 'guard'in v]
 reports.append({'log':str(path),'snapshot':next((v for v in records if 'snapshot'in v),None),'model':next((v for v in records if 'native_batch'in v),None),'timings':[v for v in records if 'timing'in v],'guard_comparisons':len(guards),'compared_u32_words':sum(v['words']for v in guards),'differing_words':sum(v['different_bits_words']for v in guards),'nonfinite_logits':sum(v.get('nonfinite',0)for v in guards),'graph_feedback_guarded':any(v['guard']=='graph_feedback_logits'for v in guards),'terminal':next((v for v in records if 'passed'in v),None)})
out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'runs':reports,'comparison':'bitwise serial-native row16, same original weights and group32 activation values','limitation':'canonical row-parallel routed experts; no expert bucketing; short context; recorded fork rates are not matched same-run rates'},indent=2)+'\n')
print(out)
