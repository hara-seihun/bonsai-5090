from pathlib import Path
import ctypes as C,json,struct,time,hashlib,sys,os
from array import array
import math
from h1_graph_contract import warmup_decode, graph_observation
root=Path('C:/work/models')
base=root/'experiments/bonsai-quality-20261013';mode=sys.argv[1];dest=Path(sys.argv[2]);dest.mkdir(exist_ok=True,parents=True)
# One native engine per process. Runtime environment comes from the bound parent plan.
for p in [root/'tools/cuda/bin',root/'tools/cuda/bin/x64',root/'build/llama/bin']:
 os.add_dll_directory(str(p))
cohort=base/'custom-wide256' if mode.startswith('serving') else base/'fullk-h1-tail-v2'
os.add_dll_directory(str(cohort/'build'))
if mode in ['h1-mixed','h1-plain','tail-mixed','tail-plain']:os.environ['H5_GU_COOP']=mode
else:os.environ.pop('H5_GU_COOP',None)
dll=C.CDLL(str(cohort/'build/quality_custom.dll'));ll=C.CDLL(str(root/'build/llama/bin/llama.dll'))
ptr=C.c_void_p;ip=C.POINTER(C.c_int);fp=C.POINTER(C.c_float)
dll.qc_create.argtypes=[C.c_char_p,C.c_int,C.c_int,C.c_int];dll.qc_create.restype=ptr
dll.qc_tokenizer.argtypes=[C.c_char_p];dll.qc_tokenizer.restype=ptr
dll.qc_tokenize.argtypes=[ptr,C.c_char_p,C.c_int,ip,C.c_int];dll.qc_tokenize.restype=C.c_int
dll.qc_step.argtypes=[ptr,C.c_int,ip,ip,ip,ip];dll.qc_step.restype=C.c_int
dll.qc_logits.argtypes=[ptr,C.c_int,fp];dll.qc_logits.restype=C.c_int
dll.qc_vocab_size.argtypes=[ptr];dll.qc_vocab_size.restype=C.c_int
dll.qc_pretune.argtypes=[ptr];dll.qc_pretune.restype=None
class Chat(C.Structure):_fields_=[('role',C.c_char_p),('content',C.c_char_p)]
ll.llama_model_chat_template.argtypes=[ptr,C.c_char_p];ll.llama_model_chat_template.restype=C.c_char_p
ll.llama_chat_apply_template.argtypes=[C.c_char_p,C.POINTER(Chat),C.c_size_t,C.c_bool,C.c_char_p,C.c_int32];ll.llama_chat_apply_template.restype=C.c_int32
model=str(root/'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf').encode();tok=dll.qc_tokenizer(model)
if not tok:raise RuntimeError('tokenizer creation')
template=ll.llama_model_chat_template(tok,None);raw=(base/'inputs/video-gsm-prompts256.bin').read_bytes();at=0;prompts=[]
for row in range(64):
 n=struct.unpack_from('<i',raw,at)[0];at+=4;txt=raw[at:at+n];at+=n;b=C.create_string_buffer(2*n+4096);chat=Chat(b'user',txt);nc=ll.llama_chat_apply_template(template,C.pointer(chat),1,True,b,len(b))
 if nc<0 or nc>len(b):raise RuntimeError('chat buffer')
 text=b.raw[:nc]+b'<think>\n\n</think>\n\n';buf=(C.c_int*(len(text)+16))();nt=dll.qc_tokenize(tok,text,len(text),buf,len(buf))
 if nt<0:raise RuntimeError('token capacity')
 prompts.append(list(buf)[:nt])
if mode in ['h1-mixed','h1-plain','tail-mixed','tail-plain']:os.environ['H5_GU_COOP_CHECK']='1'
else:os.environ.pop('H5_GU_COOP_CHECK',None)
engine=dll.qc_create(model,64,512,256)
if not engine:raise RuntimeError('engine creation')
dll.qc_pretune(engine)
dll.qc_graph_stats.argtypes=[ptr,C.c_int];dll.qc_graph_stats.restype=C.c_long
def step(sl,tt,pp):
 n=len(sl);o=(C.c_int*n)();a=(C.c_int*n)(*sl);b=(C.c_int*n)(*tt);c=(C.c_int*n)(*pp)
 if dll.qc_step(engine,n,a,b,c,o):raise RuntimeError('native step')
 return list(o)
done=[0]*64;nexttok=[0]*64
while any(done[r]<len(prompts[r]) for r in range(64)):
 ss=[];tt=[];pp=[];last={}
 for r in range(64):
  while done[r]<len(prompts[r]) and len(ss)<256:
   last[r]=len(ss);ss.append(r);tt.append(prompts[r][done[r]]);pp.append(done[r]);done[r]+=1
 oo=step(ss,tt,pp)
 for r,i in last.items():nexttok[r]=oo[i]
reference=base/'fullk-h1-tail-v2/strict/serving-reference'
if mode=='serving-reference':(dest/'initial-input.json').write_text(json.dumps(nexttok))
else:nexttok=json.loads((reference/'initial-input.json').read_text())
if mode!='serving-reference':
 assert json.loads((reference/'receipt.json').read_text())['accepted'],'Reference generation failed'
 if not mode.startswith('serving'):assert json.loads((reference.parent/'serving-repeat/receipt.json').read_text())['accepted'],'Strict reference-repeat did not pass'
V=dll.qc_vocab_size(tok);rows=list(range(64));steps=2;logits=array('f');buffer=(C.c_float*V)();tokens=[];inputs=[];pos=[len(x) for x in prompts]
teacher=None if mode=='serving-reference' else json.loads((reference/'tokens.json').read_text())
warmup_teacher=None if mode=='serving-reference' else json.loads((reference/'warmup-tokens.json').read_text())
warmup_hits_before=dll.qc_graph_stats(engine,0);warmup_caps_before=dll.qc_graph_stats(engine,1)
nexttok,pos,warmup_inputs,warmup_tokens=warmup_decode(step,rows,nexttok,pos,warmup_teacher)
(dest/'warmup-inputs.json').write_text(json.dumps(warmup_inputs));(dest/'warmup-tokens.json').write_text(json.dumps(warmup_tokens))
warmup_result={'warmup_steps':3,'graph_hits_warmup':dll.qc_graph_stats(engine,0)-warmup_hits_before,'graph_captures_warmup':dll.qc_graph_stats(engine,1)-warmup_caps_before,'warmup_inputs_sha256':hashlib.sha256((dest/'warmup-inputs.json').read_bytes()).hexdigest(),'warmup_tokens_sha256':hashlib.sha256((dest/'warmup-tokens.json').read_bytes()).hexdigest()}
decode_hits_before=dll.qc_graph_stats(engine,0);decode_caps_before=dll.qc_graph_stats(engine,1)
for i in range(steps):
 inputs.append(nexttok.copy());out=step(list(range(64)),nexttok,pos);tokens.append(out)
 for j,row in enumerate(rows):
  if dll.qc_logits(engine,row,buffer):raise RuntimeError('native logits')
  logits.frombytes(C.string_at(buffer,V*4))
 nexttok=out if teacher is None else teacher[i];pos=[x+1 for x in pos]
(dest/'tokens.json').write_text(json.dumps(tokens));(dest/'inputs.json').write_text(json.dumps(inputs));(dest/'logits.f32').write_bytes(logits.tobytes())
if mode.startswith('serving'):
 result={'state':'reference_created','mode':mode,'profile':'strict-deterministic','logit_rows':rows,'steps':steps,'vocab':V,'graph_hits_decode':dll.qc_graph_stats(engine,0)-decode_hits_before,'graph_captures_decode':dll.qc_graph_stats(engine,1)-decode_caps_before,'graph_hits':dll.qc_graph_stats(engine,0),'graph_captures':dll.qc_graph_stats(engine,1),'tokens_sha256':hashlib.sha256((dest/'tokens.json').read_bytes()).hexdigest(),'inputs_sha256':hashlib.sha256((dest/'inputs.json').read_bytes()).hexdigest(),'logits_sha256':hashlib.sha256((dest/'logits.f32').read_bytes()).hexdigest(),'dll_sha256':hashlib.sha256((cohort/'build/quality_custom.dll').read_bytes()).hexdigest()}
 result.update(warmup_result);result.update(graph_observation(dll.qc_graph_stats(engine,0),dll.qc_graph_stats(engine,1),decode_hits_before,decode_caps_before,steps))
 result['finite_logits']=all(map(math.isfinite,logits));result['accepted']=result['finite_logits'] and result['graph_steps_accepted']
 if mode!='serving-reference':
  previous=json.loads((reference/'receipt.json').read_text());result['accepted']=result['accepted'] and all(result[k]==previous[k] for k in ['tokens_sha256','inputs_sha256','logits_sha256','warmup_inputs_sha256','warmup_tokens_sha256'])
 (dest/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);raise SystemExit(0 if result['accepted'] else 7)
dll.h5_fullk_gu_check_stats.argtypes=[ptr,C.POINTER(C.c_uint32)];dll.h5_fullk_gu_check_stats.restype=C.c_int
stats=(C.c_uint32*4)()
if dll.h5_fullk_gu_check_stats(engine,stats):raise RuntimeError('GU check stats unavailable')
result={'state':'observed','mode':mode,'compared_words':stats[0],'different_words':stats[1],'max_abs':struct.unpack('<f',struct.pack('<I',stats[2]))[0],'nonfinite_differences':stats[3],'dll_sha256':hashlib.sha256((cohort/'build/quality_custom.dll').read_bytes()).hexdigest()}
dll.h5_fullk_tail_check_stats.argtypes=[ptr,C.POINTER(C.c_uint32)];dll.h5_fullk_tail_check_stats.restype=C.c_int
ts=(C.c_uint32*6)()
if dll.h5_fullk_tail_check_stats(engine,ts):raise RuntimeError('tail stats unavailable')
result['tail']={'q4_bytes':ts[0],'q4_different':ts[1],'s4_bytes':ts[2],'s4_different':ts[3],'g4_words':ts[4],'g4_different':ts[5]}
result.update(warmup_result);result.update(graph_observation(dll.qc_graph_stats(engine,0),dll.qc_graph_stats(engine,1),decode_hits_before,decode_caps_before,steps));result['strict_warmup_equal']=(dest/'warmup-inputs.json').read_bytes()==(reference/'warmup-inputs.json').read_bytes() and (dest/'warmup-tokens.json').read_bytes()==(reference/'warmup-tokens.json').read_bytes();result['profile']='strict-deterministic';result['intermediate_observer']='source-order-first-Hadamard' if mode.startswith('h1') else 'SiLU';result['graph_hits_decode']=dll.qc_graph_stats(engine,0)-decode_hits_before;result['graph_captures_decode']=dll.qc_graph_stats(engine,1)-decode_caps_before;result['graph_hits']=dll.qc_graph_stats(engine,0);result['graph_captures']=dll.qc_graph_stats(engine,1);result['logit_rows']=rows;result['steps']=steps;result['logits_sha256']=hashlib.sha256(logits.tobytes()).hexdigest();result['strict_logits_equal']=logits.tobytes()==(reference/'logits.f32').read_bytes();result['strict_inputs_equal']=(dest/'inputs.json').read_bytes()==(reference/'inputs.json').read_bytes();result['strict_tokens_equal']=(dest/'tokens.json').read_bytes()==(reference/'tokens.json').read_bytes();result['finite_logits']=all(map(math.isfinite,logits));result['accepted']=result['finite_logits'] and result['compared_words']>0 and result['graph_steps_accepted'] and result['strict_warmup_equal'] and result['strict_logits_equal'] and result['strict_inputs_equal'] and result['strict_tokens_equal'] and all(result[k]==0 for k in ['different_words','nonfinite_differences']) and ts[1]==ts[3]==ts[5]==0 and ts[0]>0 and ts[2]>0 and ts[4]>0
(dest/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if not result['accepted']:raise SystemExit(7)
