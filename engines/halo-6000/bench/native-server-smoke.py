#!/usr/bin/env python3
"""Actual native generation smoke, not a synthetic HTTP callback test."""
import argparse, json, pathlib, subprocess, time, urllib.request
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--engine',default='/workspace/halo-6000/build/halo-6000');p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
port=18361;base=f'http://127.0.0.1:{port}'
def request(route,data=None):
    body=None if data is None else json.dumps(data).encode()
    with urllib.request.urlopen(urllib.request.Request(base+route,data=body,headers={'Content-Type':'application/json'}),timeout=90) as r:return json.load(r)
with (out/'server.stderr').open('w') as log:
    child=subprocess.Popen([a.engine,'serve','--model',a.model,'--host','127.0.0.1','--port',str(port),'--slots','1','--context-per-slot','1024','--thinking','off','--raw-content'],stdout=log,stderr=log)
    try:
        deadline=time.monotonic()+180
        while True:
            if child.poll() is not None:raise RuntimeError(f'native server exited {child.returncode}')
            try:health=request('/health');break
            except Exception:
                if time.monotonic()>deadline:raise
                time.sleep(.5)
        body={'model':'evaluated-model','messages':[{'role':'user','content':'Reply with the single word hello.'}],'temperature':0,'seed':20261004,'max_tokens':8,'stream':False}
        first=request('/v1/chat/completions',body);second=request('/v1/chat/completions',body)
        assert first['choices'][0]['message']['content']==second['choices'][0]['message']['content'],(first,second)
        assert first['usage']==second['usage']
        assert 1<=first['usage']['completion_tokens']<=8
        result={'actual_native_inference':True,'repeat_after_full_reset_equal':True,'health':health,'first':first,'second':second,'gpu_batched_streams':False}
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    finally:
        child.terminate()
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:child.kill();child.wait()
