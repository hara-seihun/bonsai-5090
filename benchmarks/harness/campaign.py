#!/usr/bin/env python3
"""Pinned, full-split EvalScope campaign planner/runner. Planning never invokes an engine."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

DEFAULT_PROJECT=Path(__file__).resolve().parent.parent
EVALSCOPE_REV='78b061cb3d1e9cda9d86a001d65f58e2b19e95ab'
INITIAL=['gsm8k','math_500','humaneval_plus','ifeval']

class Tau2ScoringFailure(RuntimeError):
    """Official task scoring failed; this is not an incorrect agent answer."""

def install_tau2_scoring_guard():
    from evalscope.benchmarks.tau_bench.tau2_bench.tau2_bench_adapter import Tau2BenchAdapter
    official_score=Tau2BenchAdapter.match_score
    def score(self,original_prediction,filtered_prediction,reference,task_state):
        result=official_score(self,original_prediction,filtered_prediction,reference,task_state)
        if 'error' in result.metadata:raise Tau2ScoringFailure(result.metadata['error'])
        return result
    Tau2BenchAdapter.match_score=score

def install_tau2_protocol():
    from evalscope.benchmarks.tau_bench.tau2_bench import generation
    official=generation.run_task
    def run_task(*args,**kwargs):
        if 'max_steps'in kwargs and kwargs['max_steps']!=100:raise ValueError('Tau2 protocol requires100 steps')
        if 'seed'in kwargs and kwargs['seed']!=20261014:raise ValueError('Tau2 reproduction seed changed')
        kwargs.update(max_steps=100,seed=20261014)
        return official(*args,**kwargs)
    generation.run_task=run_task

class InstructionScoringFailure(RuntimeError):
    """Official instruction checks did not produce the required prompt scores."""

def install_ifbench_scoring_guard():
    from evalscope.benchmarks.ifbench.ifbench_adapter import IFBenchAdapter
    official=IFBenchAdapter.match_score
    def score(self,*args,**kwargs):
        result=official(self,*args,**kwargs)
        if not {'prompt_level_strict','prompt_level_loose'}<=result.value.keys():
            raise InstructionScoringFailure('IFBench official scorer returned incomplete prompt metrics')
        return result
    IFBenchAdapter.match_score=score

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def suite(project): return json.loads((project/'protocol.json').read_text())
def benchmark(project,name): return next(x for x in suite(project)['whitepaper20']['benchmarks'] if x['evalscope_id']==name)

def config_for(project,name,profile,api_url,concurrency=4):
    b=benchmark(project,name);root=project/'quality-campaign'
    data={'few_shot_num':0,'shuffle':True}
    overrides={
      'live_code_bench':{'subset_list':['release_v6']},
      'bigcodebench':{'extra_params':{'split':'complete','version':'v0.1.4','calibrate':True}},
      'bfcl_v3':{'extra_params':{'is_fc_model':False}},
      'tau2_bench':{'extra_params':{'generation_config':{'temperature':0.0}}},
      'gpqa_diamond':{'aggregation':'mean'}, 'aime25':{'aggregation':'mean'},'aime26':{'aggregation':'mean'},
    }
    data.update(overrides.get(name,{}))
    pins_path=project/'datasets-original15.lock.json'
    if pins_path.exists():
        pins=json.loads(pins_path.read_text()); pin=pins.get(name,{})
        if pin.get('snapshot_path'): data['dataset_id']=str((project/pin['snapshot_path']).resolve())
        if pin.get('extra_params'): data.setdefault('extra_params',{}).update(pin['extra_params'])
    cfg={'model':profile,'api_url':api_url,'api_key':'EMPTY','eval_type':'openai_api','datasets':[name],
      'dataset_hub':'huggingface','dataset_dir':str(root/'cache/datasets'),'dataset_args':{name:data},
      'repeats':b['completions_per_item'],'eval_batch_size':concurrency,'seed':20261014,
      'generation_config':{'temperature':0.7,'top_p':.95,'top_k':20,'presence_penalty':0.0,
        'repetition_penalty':1.0,'max_tokens':b['max_output_tokens'],'stream':False,'timeout':7200,
        'retries':0,'extra_body':{'min_p':0.0,'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'xhigh'}},
        'reasoning_effort':'xhigh'},
      'judge':{'strategy':'rule'},'work_dir':str(root/'runs'/profile/name), 'no_timestamp':False}
    return cfg

def readiness(project,name):
    b=benchmark(project,name);requires=[]
    if (b['judge_fallback_required'] or b.get('judge_required',False)):requires.append('Gemini 3 Flash fallback endpoint/credentials and authorization')
    if b['simulator_required']:requires.append('Explicit simulator checkpoint/domain set, budgeted endpoint and authorization; actual dual-control tool round-trip qualification')
    if b['vision_required']:requires.append('Validated image transport and fixed vision projector/preprocessing')
    if name in ['humaneval_plus','mbpp_plus','live_code_bench','bigcodebench','omni_doc_bench_v1_6']:
        requires.append('Pinned working code/scoring sandbox')
    if name=='bfcl_v3':requires.append('BFCL dependency and multi-turn execution validation')
    return requires

def write_configs(project,profile,api_url,concurrency):
    root=project/'quality-campaign';out=root/'configs'/profile;out.mkdir(parents=True,exist_ok=True)
    index=[]
    from evalscope import TaskConfig
    for b in suite(project)['whitepaper20']['benchmarks']:
        name=b['evalscope_id'];cfg=config_for(project,name,profile,api_url,concurrency)
        # Validate actual pinned EvalScope schema without invoking inference.
        validated=TaskConfig(**cfg)
        target=out/(name+'.json');target.write_text(json.dumps(cfg,indent=2)+'\n')
        index.append({'benchmark':name,'config':str(target),'published_score':b['published_bonsai_percent'],
          'full_split':True,'limit':None,'prerequisites':readiness(project,name),'schema_validated':True})
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    return index

def run_one(args):
    project=Path(args.project);name=args.benchmark;root=project/'quality-campaign'
    if not args.inference_authorized: raise SystemExit('Pass --inference-authorized only for an endpoint you own and have budgeted')
    b=benchmark(project,name)
    if (b['vision_required'] and not args.vision_ready) or (b['simulator_required'] and not (args.simulator_proxy_url and args.simulator_model and args.simulator_domains)):
        raise SystemExit('This adapter does not yet validate this vision/tool transport; configuration is planned, not runnable')
    if b.get('judge_required') and args.rule_only: raise SystemExit('This task is judge-only; rule-only cannot produce a valid score')
    if (b['judge_fallback_required'] or b.get('judge_required',False)) and not args.rule_only and not args.judge_proxy_url:
        raise SystemExit('Published fallback judge is not configured. --rule-only permits a clearly labeled preliminary result.')
    pins_path=project/'datasets-original15.lock.json';pins=json.loads(pins_path.read_text()) if pins_path.exists() else {}
    if not pins.get(name,{}).get('snapshot_path'):raise SystemExit('Dataset not pinned/materialized; run prepare_assets.py first')
    cfg=config_for(project,name,args.profile,args.api_url,args.concurrency)
    if args.dataset_snapshot:
        cfg['dataset_args'][name]['dataset_id']=args.dataset_snapshot
        cfg['dataset_args'][name]['shuffle']=False
        if name in ['humaneval_plus','mbpp_plus','ifbench']:
            from shard_loading import install
            install(args.dataset_snapshot)
    if args.judge_proxy_url and (b['judge_fallback_required'] or b.get('judge_required',False)):
        cfg['judge']={'strategy':'llm' if b.get('judge_required') else 'auto','models':[{'model_id':'gemini-3-flash-preview','api_url':args.judge_proxy_url,'api_key':'EMPTY','generation_config':{'temperature':0,'max_tokens':4096}}]}
    if name=='gsm8k' and args.judge_proxy_url and not args.rule_only:
        if args.judge_trigger is None:raise SystemExit('Declare GSM judge trigger: published_parse_failure_only or broad_rule_miss_diagnostic')
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from judge_policy import install as install_judge_policy, settings as judge_settings
        cfg['judge']['strategy']='llm_recall'
        cfg['judge']['models'][0]['generation_config']=judge_settings(0,'gsm8k')
        install_judge_policy(root/'gsm-trigger-judge'/args.profile,args.judge_trigger)
    if args.simulator_proxy_url and b['simulator_required']:
        cfg['dataset_args'][name]['extra_params'].update(user_model=args.simulator_model,api_base=args.simulator_proxy_url,api_key='EMPTY',generation_config={'temperature':0,'max_tokens':4096,'retries':0,'timeout':180})
        cfg['dataset_args'][name]['subset_list']=args.simulator_domains
    if args.screen_count:
        cfg['limit']=args.screen_count
        cfg['work_dir']+='--screen-'+str(args.screen_count)
    if args.cache: raise SystemExit('Partial EvalScope cache would break occurrence seeding; restart whole task and reuse proxy receipts')
    if name in ['humaneval_plus','mbpp_plus','live_code_bench','bigcodebench','omni_doc_bench_v1_6']:
        if not args.sandbox_ready:raise SystemExit('Code task requires --sandbox-ready after sandbox startup/validation')
        cfg['use_sandbox']=True
        lock_path=root/({'bigcodebench':'assets/bigcode-sandbox.lock.json','omni_doc_bench_v1_6':'assets/omnidoc-sandbox.lock.json'}.get(name,'assets/sandbox.lock.json'))
        if not lock_path.exists(): raise SystemExit('Missing pinned sandbox.lock.json')
        image=json.loads(lock_path.read_text())['image_id']
        if args.grader_url:
            from docker_backend import install_remote_backend
            install_remote_backend(image,args.grader_url)
        else:
            from docker_backend import install_backend
            install_backend(image)
    if name=='bigcodebench':
        from bigcode_runner import install
        install()
    if name=='live_code_bench' and pins[name].get('lazy_index_path'):
        if not args.asset_base_url: raise SystemExit('LCB lazy assets require --asset-base-url pointing at loopback-forwarded blob server')
        from lcb_lazy import install
        install(pins[name]['lazy_index_path'],args.asset_base_url,pins[name]['index_sha256'])
    if name=='bfcl_v3':
        from bfcl_sandbox import install
        bfcl_worker=install(root/'assets/bfcl-sandbox.lock.json')
    if name=='ifbench':install_ifbench_scoring_guard()
    if name=='tau2_bench':
        os.environ['TAU2_DATA_DIR']=str(Path(cfg['dataset_args'][name]['dataset_id']).resolve())
        install_tau2_scoring_guard();install_tau2_protocol()
    state={'profile':args.profile,'benchmark':name,'status':'running','started_at':time.time(),
      'protocol_status':'preliminary_rule_only' if args.rule_only and (b['judge_fallback_required'] or b.get('judge_required',False)) else 'closest_original_settings_publisher_revisions_unknown',
      'configuration':cfg,'judge_trigger_policy':args.judge_trigger if name=='gsm8k'else None,'dataset_pin':pins[name],'evalscope_revision':EVALSCOPE_REV,
      'full_split':not bool(args.screen_count),'sample_limit':args.screen_count,'limit_scope':'per_dataset_subset' if args.screen_count else None,'screen_count_per_subset':args.screen_count,'publisher_harness_revision_unknown':True}
    if name=='gsm8k'and args.judge_proxy_url and not args.rule_only:
        state['protocol_status']='native_parse_failure_trigger_publisher_extractor_unknown'if args.judge_trigger=='published_parse_failure_only'else'broad_rule_miss_diagnostic_not_published_trigger'
    if name=='ifeval':
        from langdetect import DetectorFactory
        DetectorFactory.seed=0
        state['instruction_check_seeds']={'langdetect':0}
    if name=='tau2_bench':state['tau2_protocol']={'domains':args.simulator_domains,'simulator_model':args.simulator_model,'max_steps':100,'seed':20261014,'repeats':b['completions_per_item'],'publisher_simulator_unknown':True,'AA_reference_repeats':3}
    if args.dataset_snapshot:
        window_pin=Path(args.dataset_snapshot)/'pin.json'
        state['window']={'pin_path':str(window_pin),'pin_sha256':sha(window_pin),'dataset_snapshot':args.dataset_snapshot}
    run_dir=root/'status';run_dir.mkdir(parents=True,exist_ok=True);status=run_dir/(args.profile+'--'+name+'.json')
    status.write_text(json.dumps(state,indent=2)+'\n')
    from evalscope import TaskConfig,run_task
    try:
        result=run_task(task_cfg=TaskConfig(**cfg));state['status']='completed'
        def collect_reports(value):
            if hasattr(value,'model_dump'): value=value.model_dump(mode='json')
            if isinstance(value,dict):
                if 'execution_summary' in value and 'metrics' in value: return [value]
                return [r for child in value.values() for r in collect_reports(child)]
            if isinstance(value,(list,tuple)): return [r for child in value for r in collect_reports(child)]
            return []
        reports=collect_reports(result);state['reports']=reports
        state['actual_requested_samples']=sum((r.get('execution_summary') or {}).get('requested',0) for r in reports)
        state['actual_succeeded_samples']=sum((r.get('execution_summary') or {}).get('succeeded',0) for r in reports)
        state['actual_errored_samples']=sum((r.get('execution_summary') or {}).get('errored',0) for r in reports)
        if not reports or state['actual_errored_samples'] or any((r.get('execution_summary') or {}).get('incomplete') for r in reports):
            state['status']='completed_with_errors_or_incomplete'
        if name in ['ifeval','ifbench']:
            coverage=[]
            primary_name={'ifeval':'prompt_level_strict','ifbench':'prompt_level_loose'}[name]
            for report in reports:
                requested=report['execution_summary']['requested']
                primary=[m for m in report['metrics']if m['identity']['name']==primary_name]
                coverage.append({'requested':requested,'scored':primary[0]['num']if len(primary)==1 else None})
            state['primary_scoring_coverage']=coverage
            if any(x['scored']!=x['requested']for x in coverage):state['status']='scoring_incomplete'
        if not args.screen_count and b.get('item_count'):
            state['expected_full_samples']=b['item_count']*b['completions_per_item']
            if state['actual_requested_samples']!=state['expected_full_samples']:state['status']='incomplete_sample_coverage'
    except BaseException as exc:
        state['status']='failed';state['error']=str(exc)
        if name in ['humaneval_plus','mbpp_plus','live_code_bench','bigcodebench','omni_doc_bench_v1_6'] and not args.grader_url:
            from docker_backend import ExecutionInfrastructure
            if isinstance(exc,ExecutionInfrastructure):
                state['execution_infrastructure']={'kind':exc.kind,'detail':str(exc),'ownership':exc.metadata}
        raise
    finally:
        if name=='bfcl_v3': bfcl_worker.close()
        state['finished_at']=time.time();status.write_text(json.dumps(state,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);ap.add_argument('--project',default=str(DEFAULT_PROJECT))
    ap.add_argument('--profile',required=True);ap.add_argument('--api-url',default='http://127.0.0.1:18331/v1')
    ap.add_argument('--concurrency',type=int,default=4);ap.add_argument('--benchmark',choices=[b['evalscope_id'] for b in suite(DEFAULT_PROJECT)['whitepaper20']['benchmarks']])
    ap.add_argument('--inference-authorized',action='store_true');ap.add_argument('--rule-only',action='store_true')
    ap.add_argument('--grader-url',help='Loopback-forwarded CPU VM Docker grader for BigCodeBench')
    ap.add_argument('--asset-base-url',help='Loopback-forwarded immutable task asset server on the benchmark host')
    ap.add_argument('--simulator-proxy-url',help='Budgeted declared simulator gateway; requires template proxy --allow-tools')
    ap.add_argument('--simulator-model',help='Explicit checkpoint identifier served by the simulator gateway')
    ap.add_argument('--simulator-domains',choices=['airline','retail','telecom'],nargs='+',help='Explicit Tau2 domain set; publisher selection unknown')
    ap.add_argument('--judge-trigger',choices=['published_parse_failure_only','broad_rule_miss_diagnostic'])
    ap.add_argument('--judge-proxy-url',help='Budgeted exact Gemini judge gateway, e.g. http://127.0.0.1:18341/v1')
    ap.add_argument('--vision-ready',action='store_true',help='Require a validated projector/image transport backend and pinned datasets')
    ap.add_argument('--sandbox-ready',action='store_true');ap.add_argument('--cache')
    ap.add_argument('--dataset-snapshot');ap.add_argument('--screen-count',type=int,help='Preliminary limit PER DATASET SUBSET before repeats (e.g. MATH5008→40 items); never a full published score')
    args=ap.parse_args();os.environ.setdefault('HF_HOME',str(Path(args.project)/'quality-campaign/cache/huggingface'))
    os.environ.setdefault('MODELSCOPE_CACHE',str(Path(args.project)/'quality-campaign/cache/modelscope'))
    os.environ.setdefault('NLTK_DATA',str(Path(args.project)/'quality-campaign/assets/nltk_data'))
    if args.action=='plan':print(json.dumps(write_configs(Path(args.project),args.profile,args.api_url,args.concurrency),indent=2))
    else:
        if not args.benchmark:ap.error('--benchmark required for run')
        run_one(args)
if __name__=='__main__':main()
