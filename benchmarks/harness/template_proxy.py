#!/usr/bin/env python3
"""EvalScope chat API -> exact GGUF template -> engine raw-completion API.

Completed requests are immutable, content-addressed receipts. Restart the whole EvalScope
job with a NEW proxy process to replay the same occurrence-indexed requests from cache.
Do not use EvalScope's partial use_cache concurrently with this occurrence-based seeding.
"""
import argparse, collections, hashlib, json, os, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def canonical(value): return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def split_reasoning(raw):
    # Generation starts inside <think> because the exact template opens it.
    if '</think>' not in raw:
        return raw.removeprefix('<think>\n'), '', False
    reasoning, final = raw.split('</think>', 1)
    return reasoning.removeprefix('<think>\n').strip(), final.strip(), True

class Adapter:
    def __init__(self, args):
        from transformers.utils.chat_template_utils import _compile_jinja_template
        self.args=args; self.template_text=Path(args.template).read_text()
        self.template_sha=hashlib.sha256(self.template_text.encode()).hexdigest()
        self.template=_compile_jinja_template(self.template_text)
        self.root=Path(args.receipts);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.keylocks={};self.occurrences=collections.Counter()
        self.inference_slots=threading.BoundedSemaphore(args.backend_concurrency)

    def complete(self, body):
        if body.get('stream',False): raise ValueError('Evaluation proxy requires stream=false')
        if body.get('n',1)!=1: raise ValueError('Use independent EvalScope repeats, not n>1')
        messages=body['messages']
        if body.get('tools') and not getattr(self.args,'allow_tools',False): raise ValueError('Native tool-call response parsing is not validated; use paper BFCL prompt mode')
        for message in messages:
            content=message.get('content')
            if isinstance(content,list) and any(part.get('type')!='text' for part in content) and not getattr(self.args,'allow_vision',False):
                raise ValueError('Vision transport is not yet validated; cannot evaluate vision as text')
        if body.get('tools'):
            from qwen_tools import normalize_history
            messages=normalize_history(messages)
        prompt=self.template.render(messages=messages, tools=body.get('tools'), add_generation_prompt=True,
            enable_thinking=True, reasoning_effort='xhigh', add_vision_id=False)
        images=[];image_refs=[];vision_health=None
        if getattr(self.args,'allow_vision',False): raise ValueError('Vision is outside this text/tool export')
        expected={'temperature':0.7,'top_p':.95,'top_k':20,'min_p':0.0,'presence_penalty':0.0,'repetition_penalty':1.0}
        for k,v in expected.items():
            if k in body and body[k]!=v: raise ValueError(f'Published protocol requires {k}={v}')
        max_tokens=body.get('max_tokens',body.get('max_completion_tokens'))
        if not isinstance(max_tokens,int) or max_tokens<=0: raise ValueError('Positive max_tokens required')
        identity={'prompt':prompt,'sampling':expected,'max_tokens':max_tokens,'template_sha256':self.template_sha}
        if images: identity['images_sha256']=[x['sha256'] for x in image_refs]
        key=digest(identity)
        with self.lock: keylock=self.keylocks.setdefault(key,threading.Lock())
        # Equal-prompt repetitions are serialized. Their stochastic seed is stable across profiles.
        with keylock:
            occurrence=self.occurrences[key]
            seed=int(digest({'request':key,'repeat':occurrence,'campaign_seed':self.args.seed})[:8],16)
            rid=digest({'request':key,'repeat':occurrence})
            receipt=self.root/(rid+'.json')
            if receipt.exists():
                saved=json.loads(receipt.read_text())
                if saved['profile']!=self.args.profile: raise ValueError('Receipt directory belongs to another profile')
                response=saved['response']
            else:
                if not self.args.allow_inference: raise ValueError('Inference disabled: missing cache receipt '+rid)
                request={'model':self.args.backend_model,'prompt':prompt,**expected,'seed':seed,'max_tokens':max_tokens,
                         'stream':False,'special':True,'stop':['<|im_end|>','<|endoftext|>'],'cache_prompt':False}
                if images: request['images_base64']=images
                started=time.time()
                encoded=json.dumps(request,ensure_ascii=False).encode()
                req=urllib.request.Request(self.args.backend.rstrip('/')+'/completions',encoded,{'Content-Type':'application/json'})
                with self.inference_slots:
                    with urllib.request.urlopen(req,timeout=self.args.timeout) as r: result=json.load(r)
                if 'error' in result: raise ValueError('Backend error: '+str(result['error']))
                choice=result['choices'][0];raw=choice['text']
                reasoning,final,closed=split_reasoning(raw)
                response={'id':'chatcmpl-'+rid,'object':'chat.completion','created':int(time.time()),
                    'model':self.args.profile,'choices':[{'index':0,'message':{'role':'assistant','content':final,
                    'reasoning_content':reasoning},'finish_reason':choice.get('finish_reason','stop')}],
                    'usage':result.get('usage',{}),'quality_metadata':{'reasoning_closed':closed,'sample_seed':seed,'receipt_id':rid}}
                parser_pin=None
                if body.get('tools') and closed:
                    from qwen_tools import parse
                    content,calls,parser_pin=parse(final,body['tools'])
                    if calls:
                        response['choices'][0]['message']['content']=content
                        response['choices'][0]['message']['tool_calls']=calls
                        response['choices'][0]['finish_reason']='tool_calls'
                record={'profile':self.args.profile,'request_id':rid,'occurrence':occurrence,'sample_seed':seed,
                    'request_identity':identity,'raw_request':{k:v for k,v in request.items() if k!='images_base64'},'raw_response':result,'response':response,
                    'template_sha256':self.template_sha,'started_at':started,'elapsed_s':time.time()-started,
                    'backend':self.args.backend,'reasoning_closed':closed}
                if parser_pin: record['tool_parser']=parser_pin
                if images: record['vision']={'image_sources':image_refs,'backend_health':vision_health,'image_bytes_omitted_from_receipt':True,'source_custody':'Pinned dataset assets retain original bytes; SHA256/order retained here'}
                tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False)+'\n');os.replace(tmp,receipt)
            self.occurrences[key]+=1
            return response

def handler_for(adapter):
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,value):
            payload=json.dumps(value,ensure_ascii=False).encode();self.send_response(code)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)))
            self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path.rstrip('/')=='/v1/models': return self.send(200,{'object':'list','data':[{'id':adapter.args.profile,'object':'model'}]})
            return self.send(200,{'status':'ready','inference_enabled':adapter.args.allow_inference,
                'profile':adapter.args.profile,'template_sha256':adapter.template_sha})
        def do_POST(self):
            if self.path.rstrip('/')!='/v1/chat/completions': return self.send(404,{'error':{'message':'Only chat completions supported'}})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n>64*1024*1024: raise ValueError('Request exceeds 64 MiB')
                self.send(200,adapter.complete(json.loads(self.rfile.read(n))))
            except (ValueError,KeyError,TypeError) as e: self.send(400,{'error':{'message':str(e)}})
            except Exception as e: self.send(502,{'error':{'message':str(e)}})
        def log_message(self,fmt,*args): print(fmt%args,flush=True)
    return Handler

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',required=True,help='Raw engine API base including /v1')
    ap.add_argument('--profile',required=True);ap.add_argument('--backend-model',default='bonsai')
    ap.add_argument('--template',required=True);ap.add_argument('--receipts',required=True)
    ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=18331)
    ap.add_argument('--seed',type=int,default=20261014);ap.add_argument('--timeout',type=float,default=7200)
    ap.add_argument('--backend-concurrency',type=int,default=4);ap.add_argument('--allow-inference',action='store_true')
    ap.add_argument('--allow-tools',action='store_true',help='Use pinned Qwen/vLLM XML parser for tau2 native tool schemas')
    
    args=ap.parse_args();adapter=Adapter(args)
    print(json.dumps({'profile':args.profile,'template_sha256':adapter.template_sha,'inference_enabled':args.allow_inference}),flush=True)
    ThreadingHTTPServer((args.host,args.port),handler_for(adapter)).serve_forever()
if __name__=='__main__': main()
