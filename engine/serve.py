"""Raw-completion HTTP server over the native engine: GPU sampler, no template rewriting.

  python engine/serve.py --model MODEL --profile profiles/serving.json --slots 8 --ctx 20480 --port 18322

POST /v1/completions (one request) or /v1/batch/completions ({"requests":[...]}, up to --slots).
Prompts must already be rendered with the model's chat template.
"""
import argparse, ctypes as C, hashlib, json, os, pathlib, sys, threading, time, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from batch_api import complete_many
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parent
BIN=ROOT/'build'/'bin'
LIB=BIN/('quality_custom.dll' if os.name=='nt' else 'libquality_custom.so')
class Sampling(C.Structure):
    _fields_=[('temperature',C.c_double),('top_p',C.c_double),('min_p',C.c_double),('presence_penalty',C.c_double),('repetition_penalty',C.c_double),('top_k',C.c_int)]
class Engine:
    def __init__(self,model,profile,slots,ctx,prefill,kv=None,layout=None):
        env=json.loads(pathlib.Path(profile).read_text())['environment']
        for k,v in env.items():os.environ[k]=str(v)
        os.environ['H5_GRAPH_MAX_ROWS']=str(slots)
        os.environ['H5_PLAN_READ']=str(HERE/'frozen-production-plan.txt')
        os.environ['P12_FP4_SLOTS']=str(slots)
        if kv:os.environ['QC_KV']=kv
        if layout:os.environ['QC_Q6_LAYOUT']=layout
        self.environment={k:v for k,v in os.environ.items()if k.startswith(('H5_','QC_','P12_FP4_'))}
        if os.name=='nt':
            cuda=pathlib.Path(os.environ.get('CUDA_PATH',''))
            self.search_handles=[os.add_dll_directory(str(p))for p in [cuda/'bin'/'x64',cuda/'bin',BIN]if str(p)!='.' and p.is_dir()]
        self.dll=C.CDLL(str(LIB))
        self.model=str(model);self.ctx=ctx;self.slots=slots;self.prefill=prefill;self.profile=pathlib.Path(profile).stem
        self.bind('qc_create',C.c_void_p,[C.c_char_p,C.c_int,C.c_int,C.c_int])
        self.bind('qc_tokenizer',C.c_void_p,[C.c_char_p]);self.bind('qc_vocab_size',C.c_int,[C.c_void_p])
        ip=C.POINTER(C.c_int);fp=C.POINTER(C.c_float)
        self.bind('qc_step',C.c_int,[C.c_void_p,C.c_int,ip,ip,ip,ip]);self.bind('qc_prefill',C.c_int,[C.c_void_p,C.c_int,ip,ip,ip,ip,C.c_int])
        self.bind('qc_step_logits',C.c_int,[C.c_void_p,C.c_int,ip,ip,ip,ip,fp]);self.bind('qc_logits',C.c_int,[C.c_void_p,C.c_int,fp])
        self.bind('qc_reset',None,[C.c_void_p,C.c_int]);self.bind('qc_pretune',None,[C.c_void_p]);self.bind('qc_graph_stats',C.c_long,[C.c_void_p,C.c_int])
        self.bind('qc_tokenize',C.c_int,[C.c_void_p,C.c_char_p,C.c_int,ip,C.c_int]);self.bind('qc_piece',C.c_int,[C.c_void_p,C.c_int,C.c_void_p,C.c_int]);self.bind('qc_is_eog',C.c_int,[C.c_void_p,C.c_int])
        self.bind('qc_sample',C.c_int,[C.c_void_p,C.c_int,C.POINTER(Sampling),C.c_uint64,C.c_uint64,C.POINTER(C.c_ubyte),ip])
        self.tokenizer=self.dll.qc_tokenizer(os.fsencode(model));
        if not self.tokenizer:raise RuntimeError('tokenizer load failed')
        self.vocab=self.dll.qc_vocab_size(self.tokenizer)
        self.handle=self.dll.qc_create(os.fsencode(model),slots,ctx,max(prefill,slots))
        if not self.handle:raise RuntimeError('engine creation failed')
        self.dll.qc_pretune(self.handle)
        self.lock=threading.Lock()
    def bind(self,n,r,a):f=getattr(self.dll,n);f.restype=r;f.argtypes=a
    def tokenize(self,text):
        raw=text.encode('utf-8');n=self.dll.qc_tokenize(self.tokenizer,raw,len(raw),None,0)
        if n>=0:raise RuntimeError('unexpected tokenizer capacity response')
        out=(C.c_int*(-n))();n=self.dll.qc_tokenize(self.tokenizer,raw,len(raw),out,len(out))
        if n<0:raise RuntimeError('tokenize failed')
        return list(out[:n])
    def piece(self,t):
        buf=C.create_string_buffer(256);n=self.dll.qc_piece(self.tokenizer,t,buf,len(buf))
        if n<0:buf=C.create_string_buffer(-n);n=self.dll.qc_piece(self.tokenizer,t,buf,len(buf))
        if n<0:raise RuntimeError('piece failed')
        return bytes(buf[:n])
    def step(self,tokens,positions,slots=None,final=None):
        n=len(tokens);ss=(C.c_int*n)(*(slots if slots is not None else [0]*n));tt=(C.c_int*n)(*tokens);pp=(C.c_int*n)(*positions);oo=(C.c_int*n)()
        rc=self.dll.qc_step(self.handle,n,ss,tt,pp,oo)if final is None else self.dll.qc_prefill(self.handle,n,ss,tt,pp,oo,int(final))
        if rc:raise RuntimeError('step failed '+str(rc))
        return list(oo)
    def complete(self,body):
        if body.get('stream',False):raise ValueError('Use stream=false for retained complete response')
        if body.get('n',1)!=1:raise ValueError('n must be1')
        prompt=body.get('prompt');tokens=body.get('prompt_token_ids')
        if tokens is None:
            if not isinstance(prompt,str):raise ValueError('prompt must be raw pre-rendered string')
            tokens=self.tokenize(prompt)
        elif not isinstance(tokens,list)or not tokens or any(type(x)is not int or x<0 or x>=self.vocab for x in tokens):raise ValueError('invalid prompt_token_ids')
        maximum=body.get('max_tokens',1024)
        if type(maximum)is not int or not 0<=maximum<=81920:raise ValueError('max_tokens requires0..81920')
        if not tokens or len(tokens)+maximum>self.ctx:raise ValueError(f'prompt+max_tokens exceeds ctx {self.ctx}')
        temp=float(body.get('temperature',1));top_p=float(body.get('top_p',.95));top_k=int(body.get('top_k',20));min_p=float(body.get('min_p',0));presence=float(body.get('presence_penalty',0));repetition=float(body.get('repetition_penalty',1));seed=int(body.get('seed',0))
        if not(0<=temp<=2 and 0<top_p<=1 and 0<=top_k<=128 and 0<=min_p<=1 and -2<=presence<=2 and 0<repetition<=2 and 0<=seed<2**32):raise ValueError('invalid sampler parameters')
        stop=body.get('stop',[]);stop=[stop]if isinstance(stop,str)else stop
        if not isinstance(stop,list)or any(not isinstance(x,str)or not x for x in stop):raise ValueError('stop must be nonempty string or list')
        config=Sampling(temp,top_p,min_p,presence,repetition,top_k);seen=(C.c_ubyte*self.vocab)()
        for t in tokens:seen[t]=1
        with self.lock:
            self.dll.qc_reset(self.handle,0);started=time.perf_counter()
            for off in range(0,len(tokens),self.prefill):
                part=tokens[off:off+self.prefill];self.step(part,list(range(off,off+len(part))),final=off+len(part)==len(tokens))
            prefill_seconds=time.perf_counter()-started;start_decode=time.perf_counter();generated=[];raw=bytearray();reason='length';chosen=C.c_int()
            for index in range(maximum):
                rc=self.dll.qc_sample(self.handle,0,C.byref(config),seed,index,seen,C.byref(chosen))
                if rc:raise RuntimeError('sample failed '+str(rc))
                t=chosen.value;generated.append(t);seen[t]=1
                if self.dll.qc_is_eog(self.tokenizer,t):reason='stop';break
                raw.extend(self.piece(t));text=raw.decode('utf-8',errors='replace')
                if any(x in text for x in stop):
                    at=min(text.index(x)for x in stop if x in text);raw=bytearray(text[:at].encode('utf-8'));reason='stop';break
                if index+1<maximum:self.step([t],[len(tokens)+index])
            seconds=time.perf_counter()-start_decode
        return {'id':'cmpl-'+uuid.uuid4().hex,'object':'text_completion','created':int(time.time()),'model':self.profile,'choices':[{'index':0,'text':raw.decode('utf-8',errors='replace'),'finish_reason':reason,'logprobs':None}],'usage':{'prompt_tokens':len(tokens),'completion_tokens':len(generated),'total_tokens':len(tokens)+len(generated)},'quality_campaign':{'profile':self.profile,'sampler':'retained-h5-gpu-gumbel','seed':seed,'prefill_seconds':prefill_seconds,'generation_seconds':seconds,'generated_token_ids':generated}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--profile',default=str(ROOT/'profiles'/'serving.json'));p.add_argument('--ctx',type=int,default=98304);p.add_argument('--slots',type=int,default=1);p.add_argument('--prefill',type=int,default=256);p.add_argument('--port',type=int,default=18322);p.add_argument('--kv',choices=['fp8','fp4-staged','fp4-direct']);p.add_argument('--layout',choices=['planes','striped']);p.add_argument('--jsonl',action='store_true');a=p.parse_args()
    engine=Engine(a.model,a.profile,a.slots,a.ctx,a.prefill,a.kv,a.layout)
    if a.jsonl:
        for line in sys.stdin:
            try:result=engine.complete(json.loads(line))
            except Exception as e:result={'error':{'message':str(e)}}
            print(json.dumps(result),flush=True)
        return
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,obj):
            data=json.dumps(obj,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path in ['/health','/v1/models']:
                self.reply(200,{'status':'ready','data':[{'id':engine.profile,'object':'model'}],'ctx':engine.ctx,'slots':engine.slots,'vocab':engine.vocab,'environment':engine.environment,'sampler':'retained-h5-gpu-gumbel','raw_completions':True});return
            self.reply(404,{'error':'not found'})
        def do_POST(self):
            if self.path not in ['/v1/completions','/v1/batch/completions']:self.reply(404,{'error':'raw completions endpoints only'});return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n>16*1024*1024:raise ValueError('request too large')
                body=json.loads(self.rfile.read(n));self.reply(200,complete_many(engine,body.get('requests'),Sampling) if self.path=='/v1/batch/completions' else engine.complete(body))
            except ValueError as e:self.reply(400,{'error':{'message':str(e)}})
            except Exception as e:self.reply(500,{'error':{'message':str(e)}})
    print(json.dumps({'ready':True,'port':a.port,'profile':engine.profile}),flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
