#!/usr/bin/env python3
import argparse, collections, json, pathlib
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output');a=p.parse_args()
x=json.loads(pathlib.Path(a.input).read_text());t=x['timestamps']
gdn=['norm_q','input_q','projections','conv_prepare','prepare_unused','recur','finish_q','innerq_unused','output_mv','residual_norm_q','postq_unused','router','select','gate_up_q','expertq_unused','down_sum']
att=['norm_q','projections','norm_rope_append','append_unused','scores','softmax','gate_q','innerq_unused','output_mv','residual_norm_q','router','select','gate_up_q','down_sum']
buckets=collections.defaultdict(float);rows=[];previous=t[0]
for i,now in enumerate(t):
 if i==0 or not now:continue
 layer=None
 if i==1:name='embedding'
 elif i>=642:name={642:'finalnorm_q',643:'head',644:'argmax_partials',645:'argmax_finish'}[i]
 else:
  layer=(i-2)//16;phase=(i-2)%16;name=(att if (layer+1)%4==0 else gdn)[phase]
 us=(now-previous)/1000;previous=now;buckets[name]+=us;rows.append(dict(layer=layer,phase=name,us=us))
result=dict(grid=x['grid'],position=x['position'],token_us=(t[-1]-t[0])/1000,buckets_us=dict(sorted(buckets.items(),key=lambda kv:-kv[1])),rows=rows,note='Block0 globaltimer timestamps after each publication barrier. Includes useful phase+waiting/barrier; not independent kernel sums.')
text=json.dumps(result,indent=2);print(text)
if a.output:pathlib.Path(a.output).write_text(text+'\n')
