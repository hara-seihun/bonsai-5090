#!/usr/bin/env python3
"""Cached binary, one GPU lease, capped diagnostic, durable artifacts."""
import json, os, pathlib, signal, subprocess, sys, time, urllib.request

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
config = json.loads(pathlib.Path(sys.argv[2]).read_text())
cmd = list(config['command'])
cmd[cmd.index('--port') + 1] = '18089'
base = 'http://127.0.0.1:18089'
env = dict(os.environ, **config['env'], H5_PROFILE_CONTROL='1', H5_TRACE='1',
           H5_ROUND_TRACE=str(out/'rounds.txt'))
profile = ['nsys', 'profile', '--trace=cuda,nvtx,cublas', '--sample=none',
           '--cpuctxsw=none', '--capture-range=cudaProfilerApi',
           '--capture-range-end=stop', '--cuda-graph-trace=node',
           '--force-overwrite=true', '--output='+str(out/'trace')]+cmd
body = {'model':'bonsai-27b','messages':[{'role':'user','content':
    'Explain in detail what happens from typing an HTTPS URL into a browser until the page appears. Cover DNS, transport, TLS, HTTP, caching, and rendering, in order.'}],
    'max_tokens':128,'temperature':0,'stream':False,
    'chat_template_kwargs':{'enable_thinking':False}}
def save(name, value): (out/name).write_text(json.dumps(value, indent=2)+'\n')
def req(path, data=None, seconds=5):
    raw = None if data is None else json.dumps(data).encode()
    r = urllib.request.Request(base+path,data=raw,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=seconds) as response: return json.load(response)
def request(name, data):
    save(name+'-metrics-before.json', req('/metrics'))
    t=time.monotonic(); result=req('/v1/chat/completions',data,seconds=15)
    save(name+'-timing.json',{'client_wall_s':time.monotonic()-t})
    save(name+'-response.json',result)
    save(name+'-metrics-after.json', req('/metrics'))
save('command.json',{'argv':profile,'environment':{k:v for k,v in env.items() if k.startswith('H5_')},'source_pin':'90d1577 (e4de708 + dspark kernel93835c0 only; cached b17 object)','cached_binary':cmd[0]})
save('request.json',body)
proc=None; ready=None
try:
    try: req('/health'); raise RuntimeError('foreign endpoint already listening')
    except OSError: pass
    with open(out/'server.log','w') as log:
        proc=subprocess.Popen(profile,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+180
        while True:
            if proc.poll() is not None: raise RuntimeError('server/profiler exited before ready: '+str(proc.returncode))
            try: save('health.json',req('/health')); break
            except OSError:
                if time.monotonic()>=deadline: raise TimeoutError('startup deadline')
                time.sleep(.05)
        ready=time.monotonic()
        # All server descendants share this process group; alarm owns the 55s ready lease.
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('55s diagnostic lease')))
        signal.alarm(45)
        request('warm',dict(body,max_tokens=64))
        request('control',body)
        save('capture-start.json',req('/profile/start',{}))
        request('traced',body)
        save('capture-stop.json',req('/profile/stop',{}))
        signal.alarm(0)
        save('diagnostic.json',{'state':'requests-completed','seconds_after_ready':time.monotonic()-ready})
except Exception as error:
    save('failure.json',{'type':type(error).__name__,'detail':str(error)})
    raise
finally:
    signal.alarm(0)
    if proc is not None:
        if proc.poll() is None: os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=1)
        save('process.json',{'exit_code':proc.returncode,'seconds_after_ready':None if ready is None else time.monotonic()-ready})
