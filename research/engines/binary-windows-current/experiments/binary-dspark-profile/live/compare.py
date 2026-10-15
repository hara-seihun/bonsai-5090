#!/usr/bin/env python3
import json, pathlib
root=pathlib.Path(__file__).parent
old=root.parent/'evidence';new=root/'evidence'
load=lambda p:json.loads(p.read_text())
result={'requests':{},'trace':{}}
for arm in ['control','traced']:
    a=load(old/(arm+'-response.json')); b=load(new/(arm+'-response.json'))
    ta,tb=a['timings'],b['timings']
    wa=load(old/(arm+'-timing.json'))['client_wall_s'];wb=load(new/(arm+'-timing.json'))['client_wall_s']
    result['requests'][arm]={'identical_choices':a['choices']==b['choices'],'identical_usage':a['usage']==b['usage'],
      'histogram_equal':ta['prefix_histogram']==tb['prefix_histogram'],
      'old_histogram':ta['prefix_histogram'],'new_histogram':tb['prefix_histogram'],
      'old_accepted':ta['accepted_drafts'],'new_accepted':tb['accepted_drafts'],
      'old_proposed':ta['proposed_drafts'],'new_proposed':tb['proposed_drafts'],
      'old_decode_tps':ta['decode_tok_per_s'],'new_decode_tps':tb['decode_tok_per_s'],
      'decode_rate_ratio':tb['decode_tok_per_s']/ta['decode_tok_per_s'],
      'old_client_wall_s':wa,'new_client_wall_s':wb,'client_wall_rate_ratio':wa/wb,
      'old_http_tps':a['usage']['completion_tokens']/wa,'new_http_tps':b['usage']['completion_tokens']/wb,
      'old_verify_s':ta['verify_s'],'new_verify_s':tb['verify_s'],
      'old_draft_s':ta['draft_s'],'new_draft_s':tb['draft_s']}
a=load(old/'analysis.json');b=load(new/'analysis.json')
result['trace']['attention']={'old_ms':a['draft_query_kernel_parts_ms']['attention'],
  'new_ms':b['draft_query_kernel_parts_ms']['attention'],
  'speedup':a['draft_query_kernel_parts_ms']['attention']/b['draft_query_kernel_parts_ms']['attention']}
for stage in ['decode_target','steady_draft','initial_draft','final_draft_ingest']:
    result['trace'][stage]={'old_wall_ms':a['stages'][stage]['wall_ms'],'new_wall_ms':b['stages'][stage]['wall_ms'],
      'old_busy_ms':a['stages'][stage]['gpu_busy_union_ms'],'new_busy_ms':b['stages'][stage]['gpu_busy_union_ms']}
result['trace']['prepared_rows_equal']=a['prepared_rows']==b['prepared_rows']
(new/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
