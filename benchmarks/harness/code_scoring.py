"""Owner-local repairs: full MBPP+ tests and explicit execution-transport errors."""
import json,subprocess
from evalscope.api.metric import Score

class CodeScoringFailure(RuntimeError):
    """No answer score exists because execution infrastructure did not complete."""

def install():
    from evalscope.api.mixin.code_execution_sandbox_mixin import CodeExecutionSandboxMixin
    from evalscope.benchmarks.mbppplus.mbppplus_adapter import MBPPplusAdapter
    original=CodeExecutionSandboxMixin.execute_code_in_sandbox
    if not getattr(original,'original15_transport_guard',False):
        def execute(self,*args,**kwargs):
            try:result=original(self,*args,**kwargs)
            except (OSError,subprocess.SubprocessError,json.JSONDecodeError) as error:
                raise CodeScoringFailure('Code execution transport failed: '+str(error)) from error
            if result.get('status') not in ['success','error','timeout']:
                raise CodeScoringFailure('Unscorable code execution result: '+repr(result))
            return result
        execute.original15_transport_guard=True
        CodeExecutionSandboxMixin.execute_code_in_sandbox=execute
    def match_score(self,original_prediction,filtered_prediction,reference,task_state):
        if not self.use_sandbox:raise CodeScoringFailure('MBPP+ requires sandbox execution')
        problem=task_state.metadata
        if not isinstance(problem['test'],str) or not problem['test'].strip():
            raise CodeScoringFailure('MBPP+ augmented test payload absent')
        program='\n'.join(problem['test_imports'])+'\n'+filtered_prediction+'\n'+'\n'.join(problem['test_list'])+'\n'+problem['test']
        result=self.execute_code_in_sandbox(code=program,timeout=self.review_timeout,language='python')
        return Score(extracted_prediction=filtered_prediction,prediction=original_prediction,
            value={'acc':result['status']=='success'},main_score_name='acc',
            metadata={'task_id':problem['task_id'],'timeout':self.review_timeout,'execution_result':result,
                'test_scope':'original base tests plus pinned augmented test payload'})
    MBPPplusAdapter.match_score=match_score
