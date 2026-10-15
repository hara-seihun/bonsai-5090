import hashlib,json
from pathlib import Path
root=Path('/workspace/binary/row-fibers');root.mkdir(exist_ok=True);p=root/'prompts';p.mkdir(exist_ok=True)
gsm=[json.loads(l) for l in Path('/workspace/binary/results/tasks-reference-b32/gsm8k/responses.jsonl').read_text().splitlines()][:12]
he=[json.loads(l) for l in Path('/workspace/binary/results/tasks-reference-b32/humaneval_plus/responses.jsonl').read_text().splitlines()][:12]
chat=json.loads(Path('/workspace/binary/halo/bench/two_path_prompts.json').read_text())[:8]
items=[]
ordered=[]
for half in range(2):
 ordered += [('gsm',r) for r in gsm[half*6:half*6+6]]+[('he',r) for r in he[half*6:half*6+6]]+[('chat',r) for r in chat[half*4:half*4+4]]
for i,(kind,r) in enumerate(ordered):
 msgs=r.get('messages',[{'role':'user','content':r.get('prompt','')}]);t=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in msgs)+'<|im_start|>assistant\n<think>\n\n</think>\n\n'
 # Retain recorded target continuation after the prompt if a short prompt needs 64 tokens.
 if kind!='chat':t+=r['content']
 (p/f'{i}.txt').write_text(t);items.append({'row':i,'kind':kind,'id':r.get('id',str(i//3)),'sha256':hashlib.sha256(t.encode()).hexdigest(),'teacher_forced_recorded_continuation':kind!='chat'})
(root/'prompts.json').write_text(json.dumps(items,indent=2)+'\n')
