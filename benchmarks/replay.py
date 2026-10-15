#!/usr/bin/env python3
"""Offline reduction of exported observations, not a new inference or grading run."""
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'receipts'
def load(name):return json.loads((ROOT/name).read_text())
def require(condition,detail):
    if not condition:raise ValueError(detail)

def coverage(name,partition_name):
    result=load(name);partition=load('partitions/'+partition_name+'.json')
    shards=result['shards']
    if partition_name=='gsm':
        indices=partition['indices'];chunk=partition['chunk']
        windows=[{'shard':i,'count':len(indices[start:start+chunk]),'indices':indices[start:start+chunk]}
                 for i,start in enumerate(range(0,len(indices),chunk))]
        expected_total=partition['count']
    else:
        windows=partition['windows'];expected_total=partition['total']
    require(len(shards)==len(windows),'Incomplete window coverage')
    require(len({s['shard']for s in shards})==len(shards),'Duplicate shard')
    indices=[];correct=count=0
    for shard in shards:
        window=next(w for w in windows if w['shard']==shard['shard'])
        require(shard['state']=='completed','Uncompleted window')
        require(shard['count']==window['count']==len(window['indices']),'Window count mismatch')
        require(0<=shard['correct']<=shard['count'],'Invalid score')
        indices.extend(window['indices']);count+=shard['count'];correct+=shard['correct']
    require(sorted(indices)==list(range(expected_total)),'Missing/duplicate original source indices')
    require((count,correct)==(result['covered'],result['correct']),'Stored aggregate disagrees')
    return {'correct':correct,'count':count,'percent':100*correct/count}

def main():
    canonical=load('canonical-scores.json')
    gsm=coverage('halo-gsm-coverage.json','gsm')
    rows=load('gsm-native-trigger-rows.json')
    require(len(rows)==gsm['count']==len({r['source_id']for r in rows}),'GSM occurrence coverage differs')
    counts=collections.Counter();score=broad=0
    for row in rows:
        parsed=row['native_extraction_success'];rule=row['rule_score']
        require(rule in [0,1]and row['broad_recall_score']in [0,1],'Nonbinary GSM score')
        if parsed:
            value=rule
            require(row['published_trigger_candidate_score']==rule,'Parsed wrong answer received judge credit')
        else:
            require(row['state']=='scored_from_existing_judge_receipt'and bool(row['reused_judge_keys']),
                    'Parse failure lacks retained adjudication')
            value=row['published_trigger_candidate_score']
            require(value in [0,1],'Missing parse-failure adjudication')
        counts[row['trigger']]+=1;score+=value;broad+=row['broad_recall_score']
    require(sum(r['rule_score']for r in rows)==gsm['correct'],'Rule coverage/trigger rows disagree')
    audit=load('gsm-native-trigger-result.json')
    require(dict(counts)==audit['classification'],'Trigger classification differs')
    require(score==audit['parse_failure_only_correct'],'Native trigger total differs')
    results={'gsm8k':{'correct':score,'count':len(rows),'percent':100*score/len(rows)},
             'math_500':coverage('halo-math_500-coverage.json','math_500'),
             'ifeval':coverage('halo-ifeval-coverage.json','ifeval')}
    for row in canonical['results']:
        actual=results[row['benchmark']]
        require(actual['correct']==row['correct']and actual['count']==row['count'],'Canonical score disagrees')
    tau=load('tau2-partial.json');partition=tau['partitions']
    ids=[i for values in partition.values()for i in values]
    require(sorted(ids)==list(range(tau['expected_occurrences'])),'Tau2 partial coverage partitions differ')
    require({k:len(v)for k,v in partition.items()}==tau['counts'],'Tau2 partial counts differ')
    he=load('humaneval-plus-partial.json');done=[s for s in he['shards']if s['status']=='completed']
    require(sum(s['scored_count']for s in done)==he['covered']and sum(s['correct']for s in done)==he['correct'],
            'HumanEval partial disagrees')
    print(json.dumps({'state':'retained_receipts_reduced','results':results,
        'gsm_broad_diagnostic_only':{'correct':broad,'count':len(rows),'percent':100*broad/len(rows)},
        'partials':{'tau2':{'scored':tau['counts']['scored'],'expected':tau['expected_occurrences'],
                           'correct':tau['scored_correct'],'full_score':tau['full_quality']},
                    'humaneval_plus':{k:he[k]for k in ['state','correct','covered','expected','full_accuracy']}},
        'scope':'GSM per-occurrence retained native extraction/judge decisions; MATH/IFEval retained shard scores plus exact disjoint source indices. No fresh model, extractor, judge or program execution.'},indent=2))
if __name__=='__main__':main()
