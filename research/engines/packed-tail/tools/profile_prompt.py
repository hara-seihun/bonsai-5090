#!/usr/bin/env python3
"""Own a foreground server/profiler and retain one real prompt, output and trace."""
import argparse, json, os, pathlib, signal, subprocess, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument('--mode', required=True, choices=['plain','graph','node','ncu','nvbit'])
p.add_argument('--output', required=True)
p.add_argument('--server', required=True)
p.add_argument('--model', required=True)
p.add_argument('--drafter', required=True)
p.add_argument('--tokens', type=int, required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--nvbit-tool')
p.add_argument('--pdl', required=True, choices=['0','1'])
a = p.parse_args()
out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=False)
base = f'http://127.0.0.1:{a.port}'
prompt = 'Explain in detail what happens from typing an HTTPS URL into a browser until the page appears. Cover DNS, transport, TLS, HTTP, caching, and rendering, in order.'
body = {'model':'ternary-bonsai-2-27b','messages':[{'role':'user','content':prompt}], 'max_tokens':a.tokens, 'temperature':0, 'stream':False, 'chat_template_kwargs':{'enable_thinking':False}}
cmd = [a.server,a.model,a.drafter,'--host','127.0.0.1','--slots','1','--ctx','4096','--port',str(a.port)]
env = dict(os.environ, H5_COMPACT='dense5', H5_DRAFT_MARGIN='0', H5_PROFILE_CONTROL='1', H5_TRACE='1', H5_PDL=a.pdl, H5_PDL_OUTPUT=str(out))
env.pop('H5_PROF',None); env.pop('H5_NOGRAPH',None); env.pop('H5_TREE2',None)
if a.mode in ['graph','node']:
    cmd = ['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--cuda-graph-trace='+a.mode,'--force-overwrite=true','--output='+str(out/'trace')] + cmd
elif a.mode == 'ncu':
    cmd = ['/usr/local/cuda/bin/ncu','--profile-from-start','off','--launch-count','1','--section','LaunchStats','--export',str(out/'compute'),'--force-overwrite'] + cmd
elif a.mode == 'nvbit':
    if not a.nvbit_tool: raise SystemExit('--nvbit-tool required')
    env.update(LD_PRELOAD=a.nvbit_tool, ACTIVE_FROM_START='0', NOBANNER='1', PATH='/usr/local/cuda/bin:'+env.get('PATH',''))
(out/'request.json').write_text(json.dumps(body,indent=2)+'\n')
(out/'command.json').write_text(json.dumps({'argv':cmd,'environment':{k:v for k,v in env.items() if k.startswith('H5_') or k in ['LD_PRELOAD','ACTIVE_FROM_START','NOBANNER']}},indent=2)+'\n')
def req(path, data=None):
    raw = None if data is None else json.dumps(data).encode()
    r = urllib.request.Request(base+path,data=raw,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=180) as response: return json.load(response)
def save(name,value): (out/name).write_text(json.dumps(value,indent=2)+'\n')
proc = None
try:
    with open(out/'server.log','w') as log:
        proc = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline = time.monotonic()+180
        while True:
            if proc.poll() is not None: raise RuntimeError('server exited before ready: '+str(proc.returncode))
            try: save('health.json',req('/health')); break
            except (OSError,ValueError):
                if time.monotonic()>=deadline: raise TimeoutError('server startup')
                time.sleep(.05)
        warm = dict(body,max_tokens=64)
        for i in range(2): save(f'warm-{i}.json',req('/v1/chat/completions',warm))
        save('metrics-before.json',req('/metrics'))
        if a.mode != 'plain': save('capture-start.json',req('/profile/start',{}))
        t = time.monotonic(); result = req('/v1/chat/completions',body); wall = time.monotonic()-t
        save('response.json',result); save('timing.json',{'request_wall_s':wall})
        save('metrics-after.json',req('/metrics'))
        if a.mode != 'plain': save('capture-stop.json',req('/profile/stop',{}))
        save('completed.json',{'completed':True,'mode':a.mode})
finally:
    if proc is not None:
        os.killpg(proc.pid,signal.SIGTERM) if proc.poll() is None else None
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        save('process.json',{'exit_code':proc.returncode})
