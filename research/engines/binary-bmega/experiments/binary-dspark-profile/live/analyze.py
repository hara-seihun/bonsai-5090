#!/usr/bin/env python3
import bisect, collections, json, pathlib, sqlite3, statistics

root=pathlib.Path(__file__).parent
out=root/'evidence'
db=sqlite3.connect(f'file:{out}/trace.sqlite?mode=ro',uri=True); db.row_factory=sqlite3.Row
ids=dict(db.execute('select id,value from StringIds'))
def load(table): return [dict(r) for r in db.execute('select * from '+table+' order by start')]
kernels=load('CUPTI_ACTIVITY_KIND_KERNEL')
for k in kernels: k['name']=ids[k['demangledName']]
copies=load('CUPTI_ACTIVITY_KIND_MEMCPY')
api=load('CUPTI_ACTIVITY_KIND_RUNTIME')
for k in api: k['name']=ids[k['nameId']]
ranges=load('NVTX_EVENTS')
target=[r for r in ranges if r['text']=='target-forward']
draft=[r for r in ranges if r['text']=='drafter-forward']
assert len(target)==len(draft)==58

def union(items):
    end=None; total=0
    for s,e in sorted(items):
        if end is None or s>end: total+=e-s; end=e
        elif e>end: total+=e-end; end=e
    return total/1e6

def contained(items,r):
    starts=[x['start'] for x in items]
    return [x for x in items[bisect.bisect_left(starts,r['start']):bisect.bisect_right(starts,r['end'])] if x['end']<=r['end']]

def inventory(items):
    groups=collections.defaultdict(list)
    for k in items: groups[k['name']].append(k)
    return sorted([{'name':n,'count':len(ks),'sum_ms':sum(k['end']-k['start'] for k in ks)/1e6,
      'median_us':statistics.median(k['end']-k['start'] for k in ks)/1e3,
      'geometry':dict(collections.Counter(str([k['gridX'],k['gridY'],k['gridZ'],k['blockX'],k['blockY'],k['blockZ']]) for k in ks))}
      for n,ks in groups.items()],key=lambda r:-r['sum_ms'])

def stage(rs):
    kk=[k for r in rs for k in contained(kernels,r)]
    cc=[k for r in rs for k in contained(copies,r)]
    aa=[k for r in rs for k in contained(api,r)]
    wall=sum(r['end']-r['start'] for r in rs)/1e6
    busy=union((k['start'],k['end']) for k in kk+cc)
    return {'calls':len(rs),'wall_ms':wall,'kernel_union_ms':union((k['start'],k['end']) for k in kk),
      'kernel_sum_ms':sum(k['end']-k['start'] for k in kk)/1e6,'gpu_busy_union_ms':busy,'wall_minus_gpu_busy_ms':wall-busy,
      'copy_sum_ms':sum(k['end']-k['start'] for k in cc)/1e6,'copy_bytes':sum(k['bytes'] for k in cc),
      'copy_counts':dict(collections.Counter(k['copyKind'] for k in cc)),
      'api_sum_ms':sum(k['end']-k['start'] for k in aa)/1e6,
      'kernels':inventory(kk)}

result={'stages':{'prefill_target':stage(target[:3]),'prefill_draft_ingest':stage(draft[:2]),
    'initial_draft':stage(draft[2:3]),'decode_target':stage(target[3:]),
    'steady_draft':stage(draft[3:-1]),'final_draft_ingest':stage(draft[-1:])}}
# Native projection order is fixed by pinned source. Convert kernels delimit projects.
projects=collections.defaultdict(list); draft_parts=collections.defaultdict(float)
for r in draft[2:-1]:
    kk=contained(kernels,r)
    converts=[i for i,k in enumerate(kk) if 'convert_kernel' in k['name']]
    assert len(converts)==42
    labels=['tap_fc']+['ingest_kv']*6+['trunk']*30+['head']+['confidence']*4
    for j,(begin,label) in enumerate(zip(converts,labels)):
        # Each project comprises a convert, GEMM/GEMV, and optional splitK reduction.
        for k in kk[begin:]:
            if k is not kk[begin] and not ('cutlass' in k['name'] or 'gemvx' in k['name'] or 'splitKreduce_kernel' in k['name']): break
            projects[label].append(k)
    for k in kk:
        n=k['name']
        family=('attention' if 'attention_kernel' in n else 'markov' if 'markov_kernel' in n
          else 'candidate_argmax' if 'argmax_kernel' in n else 'qkv_rope_cache' if 'qkv_kernel' in n
          else 'norm' if 'rms_kernel' in n else 'other')
        draft_parts[family]+=(k['end']-k['start'])/1e6
ingested=collections.Counter()
for r in draft[3:-1]:
    first=next(k for k in contained(kernels,r) if 'convert_kernel' in k['name'])
    assert first['gridX'] % 100 == 0
    ingested[first['gridX']//100]+=1
result['prepared_rows']={'target_verify':{'5':55},'draft_query':{'4':55},
  'draft_ingested_steady':dict(ingested),'draft_initial_ingested':16,
  'target_projection_evidence':'projection<int5> and producer gridY5; no R16 prepared padding',
  'draft_ingest_evidence':'tap FC f32-to-bf16 launch has ceil(ni*25600/256)=100*ni CTAs'}
result['draft_query_kernel_parts_ms']=dict(draft_parts)
result['draft_projects']={n:{'sum_ms':sum(k['end']-k['start'] for k in ks)/1e6,'kernels':inventory(ks)} for n,ks in projects.items()}
result['requests']={}
for label in ['control','traced']:
    response=json.loads((out/(label+'-response.json')).read_text())
    result['requests'][label]={'usage':response['usage'],'timings':response['timings'],
      'client_wall_s':json.loads((out/(label+'-timing.json')).read_text())['client_wall_s']}
result['identical_text']=json.loads((out/'control-response.json').read_text())['choices']==json.loads((out/'traced-response.json').read_text())['choices']
result['note']='Prefill target and two ingestion-only calls excluded. Initial draft contains last16 prefill rows, separated. Final ingestion has no new draft and is separately charged. Wall minus GPU busy includes host submission, synchronization, scheduling and trace overhead, not just CPU compute. Kernel sums are not elapsed time.'
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for n,s in result['stages'].items(): print(n,{k:v for k,v in s.items() if k not in ['kernels']})
print('prepared_rows',result['prepared_rows'])
print('draft_parts',dict(draft_parts))
print('projects',{n:v['sum_ms'] for n,v in result['draft_projects'].items()})
