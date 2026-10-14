"""Batched quality on real engine rows; preserves per-request RNG and active-row evidence."""
import ctypes as C,time,uuid,collections

def complete_many(engine,bodies,Sampling):
    if not isinstance(bodies,list)or not bodies or len(bodies)>engine.slots:raise ValueError('requests must contain1..allocated slots entries')
    states=[]
    for slot,b in enumerate(bodies):
        if b.get('stream',False)or b.get('n',1)!=1:raise ValueError('batch requests require stream=false,n=1')
        tokens=b.get('prompt_token_ids')
        if tokens is None:
            if not isinstance(b.get('prompt'),str):raise ValueError('each request needs raw prompt or tokens')
            tokens=engine.tokenize(b['prompt'])
        if not isinstance(tokens,list)or not tokens or any(type(t)is not int or t<0 or t>=engine.vocab for t in tokens):raise ValueError('bad tokens')
        maximum=b.get('max_tokens',1024)
        if type(maximum)is not int or not 0<=maximum<=81920 or len(tokens)+maximum>engine.ctx:raise ValueError('request cap/prompt exceeds context')
        temp=float(b.get('temperature',1));p=float(b.get('top_p',.95));k=int(b.get('top_k',20));minp=float(b.get('min_p',0));presence=float(b.get('presence_penalty',0));rep=float(b.get('repetition_penalty',1));seed=int(b.get('seed',0))
        if not(0<=temp<=2 and 0<p<=1 and 0<=k<=128 and 0<=minp<=1 and -2<=presence<=2 and 0<rep<=2 and 0<=seed<2**32):raise ValueError('bad sampling')
        stop=b.get('stop',[]);stop=[stop]if isinstance(stop,str)else stop
        if not isinstance(stop,list)or any(not isinstance(x,str)or not x for x in stop):raise ValueError('bad stop')
        seen=(C.c_ubyte*engine.vocab)()
        for t in tokens:seen[t]=1
        states.append(dict(slot=slot,tokens=tokens,maximum=maximum,seed=seed,config=Sampling(temp,p,minp,presence,rep,k),seen=seen,stop=stop,generated=[],raw=bytearray(),reason='length',done=maximum==0))
    def sample(s,row):
        token=C.c_int();rc=engine.dll.qc_sample(engine.handle,row,C.byref(s['config']),s['seed'],len(s['generated']),s['seen'],C.byref(token))
        if rc:raise RuntimeError('batch sample failed '+str(rc))
        t=token.value;s['generated'].append(t);s['seen'][t]=1
        if engine.dll.qc_is_eog(engine.tokenizer,t):s['reason']='stop';s['done']=True;return
        s['raw'].extend(engine.piece(t));text=s['raw'].decode('utf-8',errors='replace')
        matches=[text.index(x)for x in s['stop']if x in text]
        if matches:s['raw']=bytearray(text[:min(matches)].encode());s['reason']='stop';s['done']=True
        if len(s['generated'])>=s['maximum']:s['done']=True
    with engine.lock:
        started=time.perf_counter()
        for s in states:
            engine.dll.qc_reset(engine.handle,s['slot'])
            for off in range(0,len(s['tokens']),engine.prefill):
                part=s['tokens'][off:off+engine.prefill]
                engine.step(part,list(range(off,off+len(part))),slots=[s['slot']]*len(part),final=off+len(part)==len(s['tokens']))
            # Retained last head overwritten by next prefill; sample this request now.
            if not s['done']:sample(s,0)
        prefill_seconds=time.perf_counter()-started;decode_start=time.perf_counter();hist=collections.Counter()
        while True:
            active=[s for s in states if not s['done']]
            if not active:break
            hist[len(active)]+=1
            engine.step([s['generated'][-1]for s in active],[len(s['tokens'])+len(s['generated'])-1 for s in active],slots=[s['slot']for s in active])
            for row,s in enumerate(active):sample(s,row)
        decode_seconds=time.perf_counter()-decode_start
    results=[]
    for s in states:
        n=len(s['generated']);np=len(s['tokens'])
        results.append({'id':'cmpl-'+uuid.uuid4().hex,'object':'text_completion','created':int(time.time()),'model':engine.profile,'choices':[{'index':0,'text':s['raw'].decode('utf-8',errors='replace'),'finish_reason':s['reason'],'logprobs':None}],'usage':{'prompt_tokens':np,'completion_tokens':n,'total_tokens':np+n},'quality_campaign':{'profile':engine.profile,'slot':s['slot'],'seed':s['seed'],'sampler':'retained-h5-gpu-gumbel','generated_token_ids':s['generated']}})
    return {'object':'batch_text_completion','results':results,'quality_campaign':{'initial_batch':len(states),'active_rows_by_decode_steps':dict(sorted(hist.items())),'prefill_and_first_sample_seconds':prefill_seconds,'batched_decode_seconds':decode_seconds,'prefill_policy':'sequential slots; selected final head','batch_policy':'finished slots removed; per-request seeded draws; actual active-row histogram retained'}}
