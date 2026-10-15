#!/usr/bin/env python3
"""Own a foreground HTTP server; guard scratch-slot reuse and ordinary fallbacks."""
import argparse
import base64
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--server', default='./h5-server')
ap.add_argument('--model', required=True)
ap.add_argument('--drafter', required=True)
ap.add_argument('--mmproj', help='Enable actual image fallback guards')
ap.add_argument('--port', type=int, default=8022)
ap.add_argument('--output', required=True)
a = ap.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
url = f'http://127.0.0.1:{a.port}'
records = []
def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url+path, data=data, headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(req, timeout=30)
def get(path):
    with request(path) as r: return json.load(r)
def body(prompt, count=128):
    return {'messages':[{'role':'user','content':prompt}], 'max_tokens':count, 'temperature':0}
def completion(b):
    with request('/v1/chat/completions', b) as r: ans=json.load(r)
    assert ans['usage']['completion_tokens']>0, ans
    return ans
def wait_idle():
    until = time.monotonic()+10
    while time.monotonic()<until:
        m=get('/metrics')
        if m['active']==0: return m
        time.sleep(.02)
    raise AssertionError('request/scratch lifecycle did not return to idle')
def save(label, ans):
    records.append({'case':label,'response':ans})
    (out/'responses.json').write_text(json.dumps(records,indent=2))
def tree(ans): return ans['timings']['tree_rounds']
try:
    get('/health')
except (OSError, urllib.error.URLError): pass
else: raise RuntimeError('Port already occupied; refuse to test another server')
env = os.environ.copy(); env['H5_TREE2']='1'
command=[a.server,a.model,a.drafter,'--host','127.0.0.1','--port',str(a.port),'--slots','2','--ctx','4096','--prefill','128']
if a.mmproj: command += ['--mmproj',a.mmproj,'--vision-gpu','0','--image-max-tokens','256']
with (out/'server.log').open('w') as log:
    proc=subprocess.Popen(command,env=env,stdout=log,stderr=log)
    try:
        deadline=time.monotonic()+90
        while True:
            if proc.poll() is not None: raise RuntimeError(f'server exited {proc.returncode}')
            try: get('/health'); break
            except (OSError,urllib.error.URLError):
                if time.monotonic()>deadline: raise TimeoutError('readiness timeout')
                time.sleep(.1)
        prompt='Explain why the sky is blue, why sunsets are red, and how clouds scatter sunlight.'
        before=get('/metrics')
        # Repeated admissions reset the original request slot after both primary and rescue rounds.
        for i in range(2):
            ans=completion(body(prompt,192)); save(f'single-{i}',ans)
            assert tree(ans)>0, ans
            wait_idle()
        # Force overlapping text requests: the scratch slot must not become a third request slot.
        start=threading.Barrier(3)
        def concurrent_job(i):
            start.wait()
            return completion(body(f'Count the integers from 1 to 1000, one per line. Then explain a sorting algorithm. Example {i}.',256))
        overlap=False; mid=get('/metrics')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(concurrent_job,i) for i in range(2)]
            start.wait()
            while not all(j.done() for j in jobs):
                m=get('/metrics'); overlap |= m['active']==2
                assert m['active']<=2, 'scratch admitted as a request'
                time.sleep(.01)
            for i,j in enumerate(jobs): save(f'concurrent-{i}',j.result())
        end=wait_idle()
        assert overlap, 'test did not exercise two admitted requests'
        assert end['tree_fallback_rounds']>mid['tree_fallback_rounds'], 'multi-stream did not fall back'
        # Terminate a real streamed response early, then require idle and scratch reuse.
        cancel=body('Count from 1 to 2000, one number per line, with no commentary.',2048)
        cancel['stream']=True
        with request('/v1/chat/completions',cancel) as r:
            while True:
                line=r.readline()
                if not line: raise AssertionError('empty cancellation stream')
                if line.startswith(b'data: ') and b'content' in line: break
        wait_idle()
        ans=completion(body(prompt,128)); save('after-cancel',ans)
        assert tree(ans)>0
        wait_idle()
        if a.mmproj:
            png=(Path(__file__).resolve().parent.parent/'bench/vision/red.png').read_bytes()
            content=[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}},
                     {'type':'text','text':'What color is the large shape? Answer one color word.'}]
            for speculate in (False,True):
                b=body(content,32); b['vision_speculation']=speculate
                ans=completion(b); save(f'image-spec-{speculate}',ans)
                assert 'red' in ans['choices'][0]['message']['content'].lower(), ans
                assert tree(ans)==0, 'image used scratch tree route'
                assert ans['timings']['tree_fallback_rounds']>0, ans
                wait_idle()
            ans=completion(body(prompt,128)); save('after-images',ans)
            assert tree(ans)>0
            wait_idle()
        after=get('/metrics')
        assert after['tree_rescues']>before['tree_rescues'], 'test did not exercise rescue/clone-back'
        assert after['clone_s']>before['clone_s'], 'clone cost missing'
        result={'pass':True,'command':command,'environment':{k:v for k,v in env.items() if k.startswith('H5_')},
                'concurrent_active_two_observed':overlap,'image_fallback_tested':bool(a.mmproj),
                'metrics_before':before,'metrics_after':after,
                'scope':'Lifecycle/counters/color guards, not scalar-greedy identity or task-quality score.'}
        (out/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
