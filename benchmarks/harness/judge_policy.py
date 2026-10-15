"""Portable GSM native parse-failure trigger; caller supplies a budgeted judge API.
Historical receipt replay is replay.py, independent of this future-run adapter.
"""
from pathlib import Path
import json

TRIGGERS={'published_parse_failure_only','broad_rule_miss_diagnostic'}

def settings(stage,benchmark):
    if stage!=0 or benchmark!='gsm8k':
        raise ValueError('Only a single explicit GSM judge stage is exported')
    return {'temperature':0,'stream':False,'retries':0,'timeout':180,
            'max_tokens':4096}

def install(journal,trigger_policy):
    from evalscope.api.mixin.llm_judge_mixin import LLMJudgeMixin
    from evalscope.metrics.math.parser import _parse
    if trigger_policy not in TRIGGERS:
        raise ValueError('Explicit GSM scoring trigger required')
    original=LLMJudgeMixin.score_with_judge_contracts
    root=Path(journal);root.mkdir(parents=True,exist_ok=True)
    (root/'policy.json').write_text(json.dumps({'trigger':trigger_policy,
        'historical_publisher_extractor':'unknown','parse_recovery':'none; failed judge remains unscored'})+'\n')
    def score(self,*args,**kwargs):
        if self._benchmark_meta.name!='gsm8k':
            return original(self,*args,**kwargs)
        if trigger_policy=='published_parse_failure_only':
            objects,display=_parse(kwargs['original_prediction'],'output')
            if display!=kwargs['filtered_prediction']:
                raise RuntimeError('Native extractor output changed')
            if objects:
                result=self.match_score(**kwargs)
                result.metadata={**(result.metadata or {}),
                    'judge_trigger_policy':trigger_policy,
                    'native_extraction_success':True,
                    'judge_skipped':'parsed_value_even_when_wrong'}
                return result
        return original(self,*args,**kwargs)
    LLMJudgeMixin.score_with_judge_contracts=score
