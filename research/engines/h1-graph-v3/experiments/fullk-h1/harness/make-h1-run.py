from pathlib import Path
import json,ast
r=Path('~/work/five-wide-native');plan=json.loads((r/'engine-smoke-v4-plan.json').read_text());build=json.loads((r/'h1-build-receipt-v1.json').read_text());dllhash=build['binary_sha256'];win='C:/work/models/experiments/bonsai-quality-20261013/fullk-h1-tail-v1'
baseenv=plan['jobs'][0]['environment'].copy()
for k in ['H5_GU_COOP','H5_GU_COOP_CHECK','H5_TAIL_PREP','H5_ORDERED_PREP','H5_GEMM_FLAG','H5_PROF']:baseenv.pop(k,None)
baseenv.update(H5_DETERMINISTIC_GEMM='1',H5_DOWN_CLEAR='0',H5_PLAN_WRITE=win+'/probe-plan.txt')
worker=(r/'fp4-tail-boundary-worker-v2.py').read_text().replace('fullk-fp4-tail-v2','fullk-h1-tail-v1').replace("mode=='serving'", "mode.startswith('serving')").replace("['tail-mixed','tail-plain']", "['h1-mixed','h1-plain','tail-mixed','tail-plain']")
worker=worker.replace("os.environ['H5_GU_COOP_CHECK']='1'", "if mode in ['h1-mixed','h1-plain','tail-mixed','tail-plain']:os.environ['H5_GU_COOP_CHECK']='1'\nelse:os.environ.pop('H5_GU_COOP_CHECK',None)")
worker=worker.replace("reference=base/'fullk-gu-engine-v2/logits-stdlib/serving'", "reference=base/'fullk-h1-tail-v1/strict/serving-reference'")
worker=worker.replace("V=dll.qc_vocab_size(tok);rows=[0,17,33,63];steps=1", "V=dll.qc_vocab_size(tok);rows=list(range(64));steps=2")
# Normal qc_step plus qc_logits keeps the observed graph path; inline qc_step_logits would disable it.
worker=worker.replace("dll.qc_pretune(engine)", "dll.qc_pretune(engine)\ndll.qc_graph_stats.argtypes=[ptr,C.c_int];dll.qc_graph_stats.restype=C.c_long")
worker=worker.replace("dll.h5_fullk_gu_check_stats.argtypes=", "(dest/'tokens.json').write_text(json.dumps(tokens));(dest/'inputs.json').write_text(json.dumps(inputs));(dest/'logits.f32').write_bytes(logits.tobytes())\nif mode.startswith('serving'):\n result={'state':'reference_created','mode':mode,'profile':'strict-deterministic','logit_rows':rows,'steps':steps,'vocab':V,'graph_hits':dll.qc_graph_stats(engine,0),'graph_captures':dll.qc_graph_stats(engine,1),'tokens_sha256':hashlib.sha256((dest/'tokens.json').read_bytes()).hexdigest(),'inputs_sha256':hashlib.sha256((dest/'inputs.json').read_bytes()).hexdigest(),'logits_sha256':hashlib.sha256((dest/'logits.f32').read_bytes()).hexdigest(),'dll_sha256':hashlib.sha256((cohort/'build/quality_custom.dll').read_bytes()).hexdigest()}\n result['accepted']=result['graph_hits']>0 and result['graph_captures']>0\n if mode!='serving-reference':\n  previous=json.loads((reference/'receipt.json').read_text());result['accepted']=result['accepted'] and all(result[k]==previous[k] for k in ['tokens_sha256','inputs_sha256','logits_sha256'])\n (dest/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);raise SystemExit(0 if result['accepted'] else 7)\ndll.h5_fullk_gu_check_stats.argtypes=")
# No scalar Python loop over32 million words: byte equality is the strict endpoint.
worker=worker.replace("result['accepted']=all", "result['profile']='strict-deterministic';result['intermediate_observer']='source-order-first-Hadamard' if mode.startswith('h1') else 'SiLU';result['graph_hits']=dll.qc_graph_stats(engine,0);result['graph_captures']=dll.qc_graph_stats(engine,1);result['logit_rows']=rows;result['steps']=steps;result['logits_sha256']=hashlib.sha256(logits.tobytes()).hexdigest();result['strict_logits_equal']=logits.tobytes()==(reference/'logits.f32').read_bytes();result['strict_inputs_equal']=(dest/'inputs.json').read_bytes()==(reference/'inputs.json').read_bytes();result['strict_tokens_equal']=(dest/'tokens.json').read_bytes()==(reference/'tokens.json').read_bytes();result['accepted']=result['compared_words']>0 and result['graph_hits']>0 and result['graph_captures']>0 and result['strict_logits_equal'] and result['strict_inputs_equal'] and result['strict_tokens_equal'] and all")
worker=worker.replace(" result['accepted']=result['graph_hits']>0", " result['finite_logits']=all(map(math.isfinite,logits));result['accepted']=result['finite_logits'] and result['graph_hits']>0")
worker=worker.replace("result['accepted']=result['compared_words']>0", "result['finite_logits']=all(map(math.isfinite,logits));result['accepted']=result['finite_logits'] and result['compared_words']>0")
# Original mode check saves reference input only for serving-reference, and all comparisons share this freshly generated fixture.
worker=worker.replace("if mode.startswith('serving'):(dest/'initial-input.json')", "if mode=='serving-reference':(dest/'initial-input.json')").replace("teacher=None if mode.startswith('serving') else", "teacher=None if mode=='serving-reference' else")
worker=worker.replace("for i in range(steps):", "decode_hits_before=dll.qc_graph_stats(engine,0);decode_caps_before=dll.qc_graph_stats(engine,1)\nfor i in range(steps):")
worker=worker.replace("'graph_hits':dll.qc_graph_stats(engine,0)","'graph_hits_decode':dll.qc_graph_stats(engine,0)-decode_hits_before,'graph_captures_decode':dll.qc_graph_stats(engine,1)-decode_caps_before,'graph_hits':dll.qc_graph_stats(engine,0)")
worker=worker.replace("result['graph_hits']=dll.qc_graph_stats(engine,0);", "result['graph_hits_decode']=dll.qc_graph_stats(engine,0)-decode_hits_before;result['graph_captures_decode']=dll.qc_graph_stats(engine,1)-decode_caps_before;result['graph_hits']=dll.qc_graph_stats(engine,0);")
worker=worker.replace("result['graph_hits']>0 and result['graph_captures']>0", "result['graph_hits_decode']>0 and result['graph_hits_decode']+result['graph_captures_decode']==steps")
worker=worker.replace("V=dll.qc_vocab_size(tok);", "if mode!='serving-reference':\n assert json.loads((reference/'receipt.json').read_text())['accepted'],'Reference generation failed'\n if not mode.startswith('serving'):assert json.loads((reference.parent/'serving-repeat/receipt.json').read_text())['accepted'],'Strict reference-repeat did not pass'\nV=dll.qc_vocab_size(tok);")
(r/'h1-strict-worker-v1.py').write_text(worker)
# Reuse the established mutex/subprocess receipt writer; strip embedded v3 worker and assert the new CPU build.
s=(r/'run-fp4-tail-v3-boundary-tail-mixed.py').read_text();tree=ast.parse(s);a=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and getattr(n.value.func,'attr',None)=='write_bytes');lines=s.splitlines(keepends=True);s=''.join(lines[:a.lineno-1])+"(d/'strict-worker.py').write_bytes("+repr(worker.encode())+")\n"+''.join(lines[a.end_lineno:]);s=s.replace('fullk-fp4-tail-v3','fullk-h1-tail-v1').replace('3572a8e6aecd89bf96b26b755440ac1e6b5ded7a49b602fe710ff8279f2e3512',dllhash).replace('boundary-worker.py','strict-worker.py')
# Rebind embedded base environment at the AST assignment boundary rather than unrelated string substitutions.
tree=ast.parse(s);envliteral=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'attr',None)=='update' and n.args and isinstance(n.args[0],ast.Dict));lo=sum(len(x) for x in s.splitlines(keepends=True)[:envliteral.args[0].lineno-1])+envliteral.args[0].col_offset;hi=sum(len(x) for x in s.splitlines(keepends=True)[:envliteral.args[0].end_lineno-1])+envliteral.args[0].end_col_offset;s=s[:lo]+repr(baseenv)+s[hi:]
for mode in ['serving-reference','serving-repeat','h1-plain','h1-mixed','tail-plain','tail-mixed']:
 child=s.replace("for mode in ['tail-mixed']:","for mode in "+repr([mode])+":",1).replace("dest=d/('boundary-'+mode)","dest=d/'strict'/mode")
 child=child.replace("env['H5_GU_COOP']=mode;env['PATH']", "\n  if mode.startswith('serving'):env.pop('H5_GU_COOP',None);env.pop('H5_GU_COOP_CHECK',None)\n  else:env['H5_GU_COOP']=mode\n  env['H5_PLAN_WRITE']=str(d/('strict-plan-'+mode+'.txt'));env['PATH']")
 child=child.replace("'SWEEP_','GGML_'", "'SWEEP_','GGML_','QC_'")
 # Preserve all completed or failed receipts rather than implicitly replay a failed generation.
 child=child.replace("assert prev['accepted'];print", "assert prev['accepted'],'Failed generation retained; name a successor';print")
 (r/f'run-h1-v1-strict-{mode}.py').write_text(child)
# Separate ordinary TPS controls: full model producer+consumer+conversion cost, no instrumented checks.
for mode in ['h1-plain','h1-mixed']:
 child=(r/f'run-fp4-tail-v3-timing-tail-{mode.split("-")[1]}.py').read_text().replace('fullk-fp4-tail-v3','fullk-h1-tail-v1').replace('3572a8e6aecd89bf96b26b755440ac1e6b5ded7a49b602fe710ff8279f2e3512',dllhash).replace("for mode in ['tail-"+mode.split('-')[1]+"']:","for mode in "+repr([mode])+":")
 child=child.replace("env['EXPLORER_PASSES']='3';", "env['EXPLORER_PASSES']='3';env['H5_DETERMINISTIC_GEMM']='0';env['H5_DOWN_CLEAR']='0';")
 child=child.replace("if (dest/'receipt.json').exists():print((dest/'receipt.json').read_text(),flush=True);continue", "if (dest/'receipt.json').exists():\n   saved=json.loads((dest/'receipt.json').read_text());assert saved['state']=='measured','Failed generation retained; name a successor';print(json.dumps(saved),flush=True);continue")
 child=child.replace("'SWEEP_','GGML_'", "'SWEEP_','GGML_','QC_'")
 child=child.replace("dest=d/('timing-'+mode)","dest=d/('timing-ordinary-'+mode)")
 (r/f'run-h1-v1-timing-ordinary-{mode}.py').write_text(child)
 strict=child.replace("'H5_DETERMINISTIC_GEMM']='0'","'H5_DETERMINISTIC_GEMM']='1'").replace('timing-ordinary-','timing-deterministic-')
 (r/f'run-h1-v1-timing-deterministic-{mode}.py').write_text(strict)
for p in r.glob('run-h1-v1-*.py'):ast.parse(p.read_text())
ast.parse(worker)
(r/'h1-test-plan-v1.json').write_text(json.dumps({'state':'prepared_not_run','candidate_dll_sha256':dllhash,'candidate_source':build['source'],'strict_profile':baseenv,'strict_workers':['serving-reference','serving-repeat','tail-plain','tail-mixed','h1-plain','h1-mixed'],'strict_rows':64,'strict_steps':2,'strict_graph_hits_required':True,'ordinary_tps_controls':['h1-plain','h1-mixed'],'ordinary_serving_baseline_tps':6777.352952,'strict_serving_baseline_tps':6583.070035,'gpu_authority':'explicit PrismML handoff required; mutex alone is insufficient','subprocess_timeout_s':55},indent=2))
paths=list(r.glob('run-h1-v1-strict-*.py'))+list(r.glob('run-h1-v1-timing-*-*.py'))
for p in paths:
 s=p.read_text();needle="print('MUTEX ACQUIRED',flush=True)\ntry:"
 replacement="print('MUTEX ACQUIRED',flush=True)\ntry:\n window=json.loads((d/'gpu-window-v1.json').read_text())\n if window['candidate_dll_sha256']!="+repr(dllhash)+":raise RuntimeError('window candidate mismatch')\n print('GPU_WINDOW',json.dumps(window),flush=True)\n def native_budget():\n  remaining=window['deadline_epoch_s']-time.time()-2\n  if remaining<=2:raise RuntimeError('window budget exhausted; no native work started')\n  return min(50,remaining)"
 assert s.count(needle)==1
 s=s.replace(needle,replacement).replace('timeout=55','timeout=native_budget()')
 lines=s.splitlines(keepends=True);out=[]
 for line in lines:
  if line.startswith('   a=subprocess.run('):
   out.append('   budget=native_budget()\n   try:\n'+line.replace('   a=','    a=',1).replace('timeout=native_budget()','timeout=budget'))
   out.append("   except subprocess.TimeoutExpired:\n    (dest/'failure-receipt.json').write_text(json.dumps({'state':'timeout','mode':mode,'accepted':False,'window':window,'native_timeout_s':budget,'native_process':'killed_and_waited_by_subprocess_run'},indent=2));raise\n")
  elif line.startswith("  if (dest/'receipt.json').exists():"):
   out.append("  if (dest/'failure-receipt.json').exists():raise RuntimeError('failed original retained; name a successor')\n"+line)
  else:out.append(line)
 s=''.join(out);ast.parse(s);p.write_text(s)
import hashlib
for mode in ['h1-plain','h1-mixed']:(r/f'run-h1-v1-timing-{mode}.py').unlink(missing_ok=True)
planfile=r/'h1-test-plan-v1.json';finalplan=json.loads(planfile.read_text());finalplan.update(window_bound_s=240,native_job_bound_s=50,ordinary_tps_workers=['h1-plain','h1-mixed'],deterministic_tps_workers=['h1-plain','h1-mixed'],runner_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths});planfile.write_text(json.dumps(finalplan,indent=2))
print('Prepared strict reference/repeat, parent-tail and H1 controls plus whole-model TPS; no GPU invocation')
