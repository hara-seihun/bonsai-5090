#!/usr/bin/env python3
"""Explicit weight/image budgets and assumed cache-miss fractions, not counters."""
import argparse, collections, json, math, pathlib
p=argparse.ArgumentParser();p.add_argument('inspect_json',type=pathlib.Path);p.add_argument('--gb-per-second',type=float,default=1792)
p.add_argument('--state-miss-fraction',type=float,default=1);p.add_argument('--conv-miss-fraction',type=float,default=1);p.add_argument('--router-miss-fraction',type=float,default=1);p.add_argument('--kv-miss-fraction',type=float,default=1);p.add_argument('--context-length',type=int,default=0)
a=p.parse_args()
fractions={k:getattr(a,k+'_miss_fraction') for k in ('state','conv','router','kv')}
if a.gb_per_second<=0 or a.context_length<0 or any(not 0<=v<=1 for v in fractions.values()):p.error('positive bandwidth, nonnegative context, fractions0..1 required')
source=json.loads(a.inspect_json.read_text());groups=collections.defaultdict(lambda:{'source_bytes':0.,'pane_cache_bytes':0.,'active_values':0.})
for t in source['tensor_table']:
    name=t['name'];values=math.prod(t['shape']);cached=values/128*34 if t['type']==143 else values/128*136 if t['type']==8 else t['bytes'] if t['type']==10 else values*4
    if name=='token_embd.weight':role='embedding_row';fraction=1/t['shape'][1]
    elif name=='output.weight':role='vocabulary_head';fraction=1
    elif '_exps.weight' in name:role='routed_experts';fraction=8/256
    elif name.endswith('.rotation_signs'):role='rotation_signs';fraction=1
    elif 'gate_inp' in name:role='routers';fraction=1
    elif '_shexp.weight' in name:role='shared_experts';fraction=1
    elif len(t['shape'])>=2 and 'ssm_conv1d' not in name:role='mixer_projections';fraction=1
    else:role='small_tensors';fraction=1
    groups[role]['source_bytes']+=t['bytes']*fraction;groups[role]['pane_cache_bytes']+=cached*fraction;groups[role]['active_values']+=values*fraction
weights={k:sum(v[k] for v in groups.values()) for k in ('source_bytes','pane_cache_bytes')}
gdn=30*32*128*128*4*2;conv=30*8192*3*4*2;kv=10*2*256*2*2*a.context_length
rate=a.gb_per_second*1e9
cases={}
policies={'current':('source','source',False),'head_q4_0':('q4_0','source',False),'head_q4_0_router_bf16':('q4_0','bf16',False),'head_ternary_router_bf16':('ternary','bf16',False),'all_ternary':('ternary','ternary',True)}
for name,(head,router,all_ternary) in policies.items():
    image={}
    for encoding in weights:
        parts={role:v[encoding] for role,v in groups.items()}
        if all_ternary:
            for role in ('embedding_row','vocabulary_head','mixer_projections','routed_experts','shared_experts','routers'):
                parts[role]=groups[role]['active_values']*(28 if encoding=='source_bytes' else 34)/128
        else:
            if head=='q4_0':parts['vocabulary_head']=groups['vocabulary_head']['active_values']*18/32
            elif head=='ternary':parts['vocabulary_head']=groups['vocabulary_head']['active_values']*(28 if encoding=='source_bytes' else 34)/128
            if router=='bf16':parts['routers']=groups['routers']['active_values']*2
        unique=sum(parts.values());streamed=unique-parts['routers']*(1-fractions['router']);state=gdn*fractions['state']+conv*fractions['conv']+kv*fractions['kv']
        image[encoding]={'unique_weight_bytes':unique,'assumed_streamed_weight_bytes':streamed,'assumed_streamed_state_conv_kv_bytes':state,'total_assumed_hbm_bytes':streamed+state,'weights_only_tokens_s':rate/unique,'ideal_tokens_s':rate/(streamed+state)}
    cases[name]={'quality':'current model' if name=='current' else 'hypothetical DIFFERENT weight model; quality must be measured','head_policy':head,'router_policy':router,'images':image}
result={'source':source['source'],'assumption':'Unique active weights once, eight of256experts. Cache-miss fractions are EXPLICIT ASSUMPTIONS, NOT measured counters. No launch/ALU/activation traffic. KV lower bound reads each shared-KV value once per layer; repeated GQA reads may hit L2.','bandwidth_decimal_gb_s':a.gb_per_second,'groups':dict(groups),'weights':weights,'gdn_read_write_bytes':gdn,'conv_read_write_bytes':conv,'ideal_kv_read_bytes':kv,'cache_miss_fraction_assumptions':fractions,'weights_only_tokens_s':{k:rate/v for k,v in weights.items()},'ideal_tokens_s':{k:v['ideal_tokens_s'] for k,v in cases['current']['images'].items()},'policy_cases':cases}
print(json.dumps(result,indent=2))
