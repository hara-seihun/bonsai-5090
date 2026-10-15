from pathlib import Path
import ast, hashlib, json

owner = Path(__file__).resolve().parent
old = owner / 'harness'
out = owner / 'graph-v3'
out.mkdir(exist_ok=True)
plan = json.loads((old / 'h1-test-plan-v1.json').read_text())
helper = (owner / 'h1_graph_contract.py').read_text()

def embedded(source):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and getattr(n.value.func, 'attr', '') == 'write_bytes')

def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)

reference = (old / 'run-h1-v1-strict-serving-reference.py').read_text()
worker = ast.literal_eval(embedded(reference).value.args[0]).decode()
worker = worker.replace('fullk-h1-tail-v1', 'fullk-h1-tail-v3')
worker = replace_once(worker, 'import math\n', 'import math\n' + helper + '\n')
warmup = """warmup_teacher=None if mode=='serving-reference' else json.loads((reference/'warmup-tokens.json').read_text())
warmup_hits_before=dll.qc_graph_stats(engine,0);warmup_caps_before=dll.qc_graph_stats(engine,1)
nexttok,pos,warmup_inputs,warmup_tokens=warmup_decode(step,rows,nexttok,pos,warmup_teacher)
(dest/'warmup-inputs.json').write_text(json.dumps(warmup_inputs));(dest/'warmup-tokens.json').write_text(json.dumps(warmup_tokens))
warmup_result={'warmup_steps':3,'graph_hits_warmup':dll.qc_graph_stats(engine,0)-warmup_hits_before,'graph_captures_warmup':dll.qc_graph_stats(engine,1)-warmup_caps_before,'warmup_inputs_sha256':hashlib.sha256((dest/'warmup-inputs.json').read_bytes()).hexdigest(),'warmup_tokens_sha256':hashlib.sha256((dest/'warmup-tokens.json').read_bytes()).hexdigest()}
"""
worker = replace_once(worker, 'decode_hits_before=dll.qc_graph_stats(engine,0);decode_caps_before=dll.qc_graph_stats(engine,1)', warmup + 'decode_hits_before=dll.qc_graph_stats(engine,0);decode_caps_before=dll.qc_graph_stats(engine,1)')
worker = replace_once(worker, " result['finite_logits']=all(map(math.isfinite,logits));", " result.update(warmup_result);result.update(graph_observation(dll.qc_graph_stats(engine,0),dll.qc_graph_stats(engine,1),decode_hits_before,decode_caps_before,steps))\n result['finite_logits']=all(map(math.isfinite,logits));")
worker = worker.replace("result['graph_hits_decode']>0 and result['graph_hits_decode']+result['graph_captures_decode']==steps", "result['graph_steps_accepted']")
worker = worker.replace("['tokens_sha256','inputs_sha256','logits_sha256']", "['tokens_sha256','inputs_sha256','logits_sha256','warmup_inputs_sha256','warmup_tokens_sha256']")
worker = replace_once(worker, "result['profile']='strict-deterministic';result['intermediate_observer']", "result.update(warmup_result);result.update(graph_observation(dll.qc_graph_stats(engine,0),dll.qc_graph_stats(engine,1),decode_hits_before,decode_caps_before,steps));result['strict_warmup_equal']=(dest/'warmup-inputs.json').read_bytes()==(reference/'warmup-inputs.json').read_bytes() and (dest/'warmup-tokens.json').read_bytes()==(reference/'warmup-tokens.json').read_bytes();result['profile']='strict-deterministic';result['intermediate_observer']")
worker = replace_once(worker, "and result['strict_logits_equal'] and result['strict_inputs_equal']", "and result['strict_warmup_equal'] and result['strict_logits_equal'] and result['strict_inputs_equal']")
(out / 'h1_graph_contract.py').write_text(helper)
(out / 'strict-worker.py').write_text(worker)

for name in plan['runner_sha256']:
    source = (old / name).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == plan['runner_sha256'][name]
    node = embedded(source)
    if '-strict-' in name:
        assert ast.literal_eval(node.value.args[0]).decode() == ast.literal_eval(embedded(reference).value.args[0]).decode()
        substitution = "(d/'strict-worker.py').write_bytes(" + repr(worker.encode()) + ')\n'
    else:
        substitution = ''
    lines = source.splitlines(keepends=True)
    source = ''.join(lines[:node.lineno-1]) + substitution + ''.join(lines[node.end_lineno:])
    source = source.replace('fullk-h1-tail-v1', 'fullk-h1-tail-v3').replace('gpu-window-v1.json', 'gpu-window-v3.json')
    if '-timing-' in name:
        source = replace_once(source, "kernel=ctypes.WinDLL('kernel32',use_last_error=True)", "for required in ['serving-reference','serving-repeat','h1-plain','h1-mixed','tail-plain','tail-mixed']:\n saved=json.loads((d/'strict'/required/'receipt.json').read_text())\n if not saved['accepted']:raise RuntimeError('strict prerequisite failed: '+required)\nkernel=ctypes.WinDLL('kernel32',use_last_error=True)")
    (out / name.replace('h1-v1-', 'h1-v3-')).write_text(source)

begin = (old / 'begin-h1-window-v1.py').read_text().replace('fullk-h1-tail-v1', 'fullk-h1-tail-v3').replace('gpu-window-v1.json', 'gpu-window-v3.json')
(out / 'begin-h1-window-v3.py').write_text(begin)
plan['state'] = 'cpu_prepared_gpu_unrun'
plan['generation'] = 'fullk-h1-tail-v3'
plan['strict_warmup_steps'] = 3
plan['strict_observed_steps'] = 2
plan['strict_observed_graph_contract'] = 'every observed step is cached graph replay: hits==2, captures==0; warmup counters retained separately'
plan['second_predecessor'] = {'source_commit':'cfd08f7df54f86cf09513e51a1a48044ce18f7f5','generation':'fullk-h1-tail-v2','state':'isolated_python_helper_import_failure','native_exit':1,'native_dll_loaded':False,'candidate_or_timing_launched':False}
plan['predecessor'] = {'source_commit': '5079ef7d6d065a939b8640c1b93a286e57fd1b3b', 'state': 'reference_graph_prerequisite_failure', 'native_exit': 7, 'graph_hits': 0, 'graph_captures': 0, 'candidate_or_timing_launched': False}
plan['runner_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('run-h1-v3-*.py'))}
plan['helper_sha256'] = hashlib.sha256(helper.encode()).hexdigest()
plan['worker_sha256'] = hashlib.sha256(worker.encode()).hexdigest()
plan['native_job_bound_s'] = 50
plan['subprocess_timeout_s'] = 50
plan['window_bound_s'] = 240
(out / 'h1-test-plan-v3.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps({'generation': plan['generation'], 'runners': len(plan['runner_sha256']), 'helper': plan['helper_sha256'], 'worker': plan['worker_sha256'], 'gpu': 'unrun'}))
