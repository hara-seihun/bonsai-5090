#!/usr/bin/env python3
"""Direct frozen DLL: separate depth0 PP512 and TG128, five repeats, no HTTP."""
import argparse,ctypes as C,hashlib,json,os,pathlib,statistics,sys,time
P=pathlib.Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def rand():
 r=[1]
 for i in range(1,31):r.append(16807*r[-1]%2147483647)
 r.extend(r[:3])
 for i in range(34,344):r.append((r[i-31]+r[i-3])&0xffffffff)
 i=344
 while True:
  r.append((r[i-31]+r[i-3])&0xffffffff);yield r[i]>>1;i+=1

def failed(code,detail):return {'state':'unsupported','code':code,'detail':detail}

def measure(e,kind,ids,clock=time.perf_counter_ns):
 e.reset();start=clock();outs=[];ins=[]
 # A callable/iterator generates random input within the clocked work, as llama-bench does.
 ids=ids() if callable(ids) else iter(ids)
 if kind=='pp512':
  e.pp(ids);ins=list(e.last_pp_ids)
 else:
  token=next(ids)
  for pos in range(128):
   ins.append(token);out=e.step(token,pos);outs.append(out);e.sync()
   token=next(ids) if kind=='tg128_random' else out
 if kind=='pp512':e.sync()
 ns=clock()-start
 assert ns>0
 count=512 if kind=='pp512' else len(outs)
 return dict(kind=kind,n_depth=0,n_prompt=512 if kind=='pp512' else 0,n_gen=0 if kind=='pp512' else 128,elapsed_ns=ns,processed_tokens=count,tokens_per_second=count*1e9/ns,input_token_ids=ins,generated_token_ids=outs,speculative_depth=0,slots=1)

class Native:
 def __init__(self,m):
  self.m=m;self.search=[]
  for n in list(os.environ):
   if n.startswith(('H5_','QC_','P12_FP4_')):del os.environ[n]
  os.environ.update(m['environment'])
  for p in m['runtime_path']:self.search.append(os.add_dll_directory(p))
  def path(name):return next(p for p in m['files_sha256'] if pathlib.PureWindowsPath(p).name==name)
  self.d=C.CDLL(path('quality_custom.dll'))
  def bind(lib,n,r,a):f=getattr(lib,n);f.restype=r;f.argtypes=a;return f
  ip=C.POINTER(C.c_int)
  self.st=bind(self.d,'qc_step',C.c_int,[C.c_void_p,C.c_int,ip,ip,ip,ip])
  self.pf=bind(self.d,'qc_prefill',C.c_int,[C.c_void_p,C.c_int,ip,ip,ip,ip,C.c_int])
  self.rs=bind(self.d,'qc_reset',None,[C.c_void_p,C.c_int])
  self.sy=bind(C.CDLL(path('cudart64_13.dll')),'cudaDeviceSynchronize',C.c_int,[])
  tok=bind(self.d,'qc_tokenizer',C.c_void_p,[C.c_char_p])(os.fsencode(m['model']));assert tok
  self.vocab=bind(self.d,'qc_vocab_size',C.c_int,[C.c_void_p])(tok)
  ll=C.CDLL(path('llama.dll'));v=bind(ll,'llama_model_get_vocab',C.c_void_p,[C.c_void_p])(tok)
  self.bos=bind(ll,'llama_vocab_bos',C.c_int,[C.c_void_p])(v)
  self.add_bos=bool(bind(ll,'llama_vocab_get_add_bos',C.c_bool,[C.c_void_p])(v))
  self.h=bind(self.d,'qc_create',C.c_void_p,[C.c_char_p,C.c_int,C.c_int,C.c_int])(os.fsencode(m['model']),1,m['context'],m['rows']);assert self.h
  bind(self.d,'qc_pretune',None,[C.c_void_p])(self.h)
 def sync(self):assert self.sy()==0,'cuda sync'
 def reset(self):self.rs(self.h,0);self.sync()
 def pp(self,ids):
  self.last_pp_ids=[]
  for off in range(0,512,256):
   n=256;part=[next(ids) for _ in range(n)];self.last_pp_ids.extend(part);s=(C.c_int*n)(*([0]*n));t=(C.c_int*n)(*part);p=(C.c_int*n)(*range(off,off+n));o=(C.c_int*n)()
   assert self.pf(self.h,n,s,t,p,o,int(off+n==512))==0,'prefill'
 def step(self,token,pos):
  s=C.c_int(0);t=C.c_int(token);p=C.c_int(pos);o=C.c_int(-1)
  assert self.st(self.h,1,C.byref(s),C.byref(t),C.byref(p),C.byref(o))==0,'decode'
  assert 0<=o.value<self.vocab
  return o.value

def summary(rows):
 result={}
 for kind in ('pp512','tg128_random'):
  s=[x for x in rows if x['kind']==kind]
  if len(s)!=5:return failed('incomplete_arm',kind)
  ts=[x['tokens_per_second'] for x in s]
  result[kind]=dict(samples_ns=[x['elapsed_ns'] for x in s],samples_ts=ts,avg_ts=statistics.mean(ts),stddev_ts=statistics.stdev(ts))
 return dict(state='measured',metrics=result,official_external_rank=None,published_comparator_position={'state':'uncomputed','reason':'Reducer required; autoregressive companion unmatched'},exact_historical_token_ids={'state':'unsupported','reason':'Historical random IDs were not retained; source-equivalent glibc seed1 sequence used'})

def run(output,arm,repetitions):
 if os.name!='nt':return failed('platform','Windows frozen DLL required; no model loaded')
 if pathlib.Path(output).exists():return failed('existing_run','Inspect retained run; no replay')
 m=json.loads((P/'frozen-manifest.json').read_text())
 for p,h in m['files_sha256'].items():
  if not pathlib.Path(p).is_file():return failed('missing_file',p)
  if sha(p)!=h:return failed('hash_mismatch',p)
 k=C.WinDLL('kernel32',use_last_error=True)
 k.CreateLockW.argtypes=[C.c_void_p,C.c_int,C.c_wchar_p];k.CreateLockW.restype=C.c_void_p
 k.WaitForSingleObject.argtypes=[C.c_void_p,C.c_uint32];k.WaitForSingleObject.restype=C.c_uint32
 handle=k.CreateLockW(None,False,'<GPU_LOCK>')
 if not handle or k.WaitForSingleObject(handle,0) not in (0,128):return failed('gpu_lock_busy','No model loaded')
 # Handle intentionally owned to process exit: DLL has no destroy API.
 e=Native(m);g=rand()
 def seq(n):
  yield e.bos if e.add_bos else next(g)%e.vocab
  for _ in range(n-1):yield next(g)%e.vocab
 rows=[]
 with open(output,'x') as f:
  def emit(o):f.write(json.dumps(o)+'\n');f.flush()
  emit(dict(state='started',source_sha256=sha(__file__),manifest_sha256=sha(P/'frozen-manifest.json'),vocab=e.vocab,bos=e.bos,add_bos=e.add_bos,context=m['context'],rows=m['rows'],environment=m['environment'],eog_policy='fixed work ignores EOG',pid=os.getpid()))
  for kind in ([arm] if arm!='all' else ['pp512','tg128_random']):
   if kind=='pp512':emit(dict(state='warmup',result=measure(e,kind,lambda:seq(512))))
   else:
    anchor=next(seq(1)) if kind=='tg128_random' else e.bos if e.add_bos else 0
    e.reset();e.step(anchor,0);e.sync()
    if kind=='tg128_random':next(g)
    emit(dict(state='warmup',kind=kind,input_token_ids=[anchor]))
   for rep in range(repetitions):
    ids=(lambda:seq(512)) if kind=='pp512' else (lambda:seq(129)) if kind=='tg128_random' else [anchor]
    row=measure(e,kind,ids)
    row['repetition']=rep;rows.append(row);emit(dict(state='sample',**row))
  result=summary(rows) if arm=='all' and repetitions==5 else dict(state='measured',arm=arm,repetitions=repetitions,samples=rows,official_external_rank=None)
  plan=next(p for p in m['files_sha256'] if pathlib.PureWindowsPath(p).name=='frozen-production-plan.txt')
  if sha(plan)!=m['files_sha256'][plan]:return failed('plan_changed',plan)
  emit(result);emit({'state':'completed','cleanup':'process exit owns GPU/lock release; caller observes exit'})
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['native','fixture']);p.add_argument('--output',required=True);p.add_argument('--arm',choices=['pp512','tg128_random','tg128_autoregressive','all'],required=True);p.add_argument('--repetitions',type=int,choices=[1,5],required=True);a=p.parse_args()
 try:
  if a.action=='native':r=run(a.output,a.arm,a.repetitions)
  else:
   from fixture_final import qualify
   r=qualify()
   with open(a.output,'x') as f:json.dump(r,f,indent=2)
 except (OSError,ValueError,KeyError,AttributeError,StopIteration,AssertionError) as ex:r=failed('native_boundary',type(ex).__name__+': '+str(ex))
 print(json.dumps(r));return 0 if r['state'] in ('measured','cpu_qualified') else 2
if __name__=='__main__':sys.exit(main())
