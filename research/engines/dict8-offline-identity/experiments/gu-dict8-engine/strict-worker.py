from pathlib import Path
import ctypes as C,json,struct,time,hashlib,sys,os
from array import array
import math
def warmup_decode(step, slots, inputs, positions, teacher):
    warmup_inputs = []
    warmup_tokens = []
    for i in range(3):
        warmup_inputs.append(inputs.copy())
        output = step(slots, inputs, positions)
        warmup_tokens.append(output)
        inputs = output if teacher is None else teacher[i]
        positions = [p + 1 for p in positions]
    return inputs, positions, warmup_inputs, warmup_tokens


def graph_observation(hits, captures, before_hits, before_captures, steps):
    observed = {'graph_hits_decode': hits - before_hits,
                'graph_captures_decode': captures - before_captures}
    observed['graph_steps_accepted'] = (steps > 0 and
        observed['graph_hits_decode'] == steps and
        observed['graph_captures_decode'] == 0)
    return observed

root=Path('C:/work/models')
base=root/'experiments/bonsai-quality-20261013';mode=sys.argv[1];dest=Path(sys.argv[2]);dest.mkdir(exist_ok=True,parents=True)
# One native engine per process. Runtime environment comes from the bound parent plan.
for p in [root/'tools/cuda/bin',root/'tools/cuda/bin/x64',root/'build/llama/bin']:
 os.add_dll_directory(str(p))
cohort=base/'custom-wide256' if mode.startswith('serving') else base/'gu-dict8-engine-build-v2'
os.add_dll_directory(str(cohort/'build'))
if mode=='dict8':os.environ['H5_GU_DICT8']='1'
else:os.environ.pop('H5_GU_DICT8',None)
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
if mode=='dict8':os.environ['H5_GU_DICT8_CHECK']='1'
else:os.environ.pop('H5_GU_DICT8_CHECK',None)
setup_start=time.perf_counter()
engine=dll.qc_create(model,64,512,256)
engine_setup_s=time.perf_counter()-setup_start
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
reference=base/'gu-dict8-engine-build-v2/strict/serving-reference'
initial_actual=nexttok.copy()
initial_equal=mode=='serving-reference' or initial_actual==json.loads((reference/'initial-input.json').read_text())
if mode=='serving-reference':(dest/'initial-input.json').write_text(json.dumps(nexttok))
else:nexttok=json.loads((reference/'initial-input.json').read_text())
if mode!='serving-reference':
 assert json.loads((reference/'receipt.json').read_text())['accepted'],'Reference generation failed'
 if not mode.startswith('serving'):assert json.loads((reference.parent/'serving-repeat/receipt.json').read_text())['accepted'],'Strict reference-repeat did not pass'
V=dll.qc_vocab_size(tok);assert V==248320;rows=list(range(64));steps=2;logits=array('f');buffer=(C.c_float*V)();tokens=[];inputs=[];pos=[len(x) for x in prompts]
teacher=None if mode=='serving-reference' else json.loads((reference/'tokens.json').read_text())
warmup_teacher=None if mode=='serving-reference' else json.loads((reference/'warmup-tokens.json').read_text())
setup=None
if mode=='dict8':
 dll.h5_dict8_check_stats.argtypes=[ptr,C.POINTER(C.c_uint32),C.c_int];dll.h5_dict8_check_stats.restype=C.c_int
 dll.h5_dict8_reset_checks.argtypes=[ptr];dll.h5_dict8_reset_checks.restype=C.c_int
 dll.h5_dict8_setup_stats.argtypes=[ptr,C.POINTER(C.c_uint64),C.c_int];dll.h5_dict8_setup_stats.restype=C.c_int
 data=(C.c_uint64*10)();assert dll.h5_dict8_setup_stats(engine,data,10)==0
 setup=dict(zip(['enabled','prepared_layers','prepared_weight_bytes','dictionary_bytes','retained_native_weight_bytes','retained_fp16_scale_bytes','retained_mean_bytes','total_weight_bytes','encode_ns','upload_ns'],list(data)))
 assert setup['enabled']==1 and setup['prepared_layers']==64 and setup['prepared_weight_bytes']==2317352960 and setup['dictionary_bytes']==26244
 def checks():
  data=(C.c_uint32*12)();assert dll.h5_dict8_check_stats(engine,data,12)==0;return list(data)
 assert dll.h5_dict8_reset_checks(engine)==0
warmup_hits_before=dll.qc_graph_stats(engine,0);warmup_caps_before=dll.qc_graph_stats(engine,1)
nexttok,pos,warmup_inputs,warmup_tokens=warmup_decode(step,rows,nexttok,pos,warmup_teacher)
(dest/'warmup-inputs.json').write_text(json.dumps(warmup_inputs));(dest/'warmup-tokens.json').write_text(json.dumps(warmup_tokens))
warmup_result={'warmup_steps':3,'graph_hits_warmup':dll.qc_graph_stats(engine,0)-warmup_hits_before,'graph_captures_warmup':dll.qc_graph_stats(engine,1)-warmup_caps_before,'warmup_inputs_sha256':hashlib.sha256((dest/'warmup-inputs.json').read_bytes()).hexdigest(),'warmup_tokens_sha256':hashlib.sha256((dest/'warmup-tokens.json').read_bytes()).hexdigest()}
if mode=='dict8':
 warmup_checks=checks()
 assert dll.h5_dict8_reset_checks(engine)==0
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
 result['engine_setup_s']=engine_setup_s;result.update(warmup_result);result.update(graph_observation(dll.qc_graph_stats(engine,0),dll.qc_graph_stats(engine,1),decode_hits_before,decode_caps_before,steps))
 result['finite_logits']=all(map(math.isfinite,logits));result['initial_inputs_equal']=initial_equal;result['accepted']=initial_equal and result['finite_logits'] and result['graph_steps_accepted']
 if mode!='serving-reference':
  previous=json.loads((reference/'receipt.json').read_text());result['accepted']=result['accepted'] and all(result[k]==previous[k] for k in ['tokens_sha256','inputs_sha256','logits_sha256','warmup_inputs_sha256','warmup_tokens_sha256'])
 (dest/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);raise SystemExit(0 if result['accepted'] else 7)
counts=checks()
expected_per_step=[64*64*34816,64*64*17408,64*64*17408//2,64*64*17408//16,64*64]
def endpoint_ok(values,steps):
 return all(values[2*i]==steps*expected_per_step[i] and values[2*i+1]==0 for i in range(5)) and values[10]==values[11]==0
result={'state':'observed','mode':mode,'profile':'strict-deterministic','endpoint_order':['gate/up after activation/row means','SiLU','q4','s4','g4'],'endpoint_comparator':'new source-bound WF0 native-operand diagnostic kernel; frozen unchanged DLL separately supplies logits/control','warmup_checks':warmup_checks,'observed_checks':counts,'setup':setup,'dll_sha256':hashlib.sha256((cohort/'build/quality_custom.dll').read_bytes()).hexdigest(),'initial_inputs_equal':initial_equal,'full_logit_bytes':len(logits)*4,'logit_rows':rows,'steps':steps,'vocab':V,'finite_logits':all(map(math.isfinite,logits))}
result['engine_setup_s']=engine_setup_s;result.update(warmup_result);result.update(graph_observation(dll.qc_graph_stats(engine,0),dll.qc_graph_stats(engine,1),decode_hits_before,decode_caps_before,steps))
for field in ['inputs','tokens','warmup-inputs','warmup-tokens']:
 result[field+'_equal']=(dest/(field+'.json')).read_bytes()==(reference/(field+'.json')).read_bytes()
result['strict_logits_equal']=logits.tobytes()==(reference/'logits.f32').read_bytes()
result['logits_sha256']=hashlib.sha256(logits.tobytes()).hexdigest()
result['accepted']=initial_equal and result['finite_logits'] and result['graph_steps_accepted'] and result['strict_logits_equal'] and all(result[f+'_equal'] for f in ['inputs','tokens','warmup-inputs','warmup-tokens']) and endpoint_ok(warmup_checks,3) and endpoint_ok(counts,2) and result['full_logit_bytes']==127139840
(dest/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
raise SystemExit(0 if result['accepted'] else 7)
