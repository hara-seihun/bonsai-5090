from pathlib import Path
import ast,json
p=Path(__file__).parent
worker=ast.parse((p/'strict-worker.py').read_text())
definitions=[n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name in ['warmup_decode','graph_observation','endpoint_ok']]
assignment=next(n for n in worker.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_per_step' for t in n.targets))
ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=definitions+[assignment],type_ignores=[])),'pure-contract','exec'),ns)
calls=[]
def step(slots,tokens,positions):calls.append((slots,tokens,positions));return [x+1 for x in tokens]
state=ns['warmup_decode'](step,[0,1],[11,21],[0,0],[[31,41],[51,61],[71,81]])
assert state==([71,81],[3,3],[[11,21],[31,41],[51,61]],[[12,22],[32,42],[52,62]]) and len(calls)==3
assert ns['graph_observation'](2,1,0,1,2)['graph_steps_accepted']
for observed in [(1,2,0,1,2),(2,2,0,1,2),(2,1,0,1,0)]:assert not ns['graph_observation'](*observed)['graph_steps_accepted']
for steps in [2,3]:
 good=[]
 for count in ns['expected_per_step']:good.extend([steps*count,0])
 good.extend([0,0]);assert ns['endpoint_ok'](good,steps)
 for i in range(12):
  bad=good.copy();bad[i]+=1;assert not ns['endpoint_ok'](bad,steps)
for name in ['strict-worker.py','driver.py']:compile((p/name).read_text(),name,'exec')
assert 'h1_graph_contract' not in (p/'strict-worker.py').read_text()
assert not (p/'gpu-handback-v1.json').exists()
print(json.dumps({'GPU_execution':False,'native_model_execution':False,'warmup_contract':True,'old_capture_prefix_rejected':True,'every_endpoint_field_mutation_rejected':True,'isolated_imports':'stdlib-only worker; pure function extraction executed','handback_fabricated':False}))
