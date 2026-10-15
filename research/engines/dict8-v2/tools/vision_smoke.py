#!/usr/bin/env python3
"""Bounded real HTTP vision test. Owns and cleans up the foreground server child."""
import argparse, base64, concurrent.futures, json, pathlib, subprocess, sys, time, urllib.request, urllib.error
p = argparse.ArgumentParser()
p.add_argument('--server', default='./h5-server')
p.add_argument('--model', required=True)
p.add_argument('--draft', required=True)
p.add_argument('--mmproj', required=True)
p.add_argument('--output', required=True)
p.add_argument('--vision-gpu', default='0')
p.add_argument('--speculate', action='store_true')
p.add_argument('--bench-spec', action='store_true')
p.add_argument('--strict-match', action='store_true')
a = p.parse_args()
out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
port = 18080
url = f'http://127.0.0.1:{port}'
log = (out/'server.log').open('w')
cmd = [a.server, a.model, a.draft, '--mmproj', a.mmproj, '--slots', '2', '--ctx', '4096', '--prefill', '128', '--port', str(port), '--host', '127.0.0.1', '--vision-gpu', a.vision_gpu, '--image-max-tokens', '256']
proc = subprocess.Popen(cmd, stdout=log, stderr=log)
def request(path, body=None):
    if body is not None and a.speculate: body = dict(body, vision_speculation=True)
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(url+path, data=data, headers={'Content-Type':'application/json'}), timeout=20)
def image(color):
    data = (pathlib.Path(__file__).resolve().parent.parent/'bench'/'vision'/f'{color}.png').read_bytes()
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode()}}
try:
    deadline = time.monotonic()+90
    while time.monotonic()<deadline:
        if proc.poll() is not None: raise RuntimeError(f'server exited {proc.returncode}; see {out}/server.log')
        try:
            with request('/health') as r: health = json.load(r)
            break
        except (OSError, urllib.error.URLError): time.sleep(.1)
    else: raise TimeoutError('server readiness exceeded 90s')
    assert health['vision'] and health['vision_language_backend'] == 'halo-native'
    results=[]
    for fixture in ('red','blue','green-large'):
        color = fixture.split('-')[0]
        body={'messages':[{'role':'user','content':[image(fixture), {'type':'text','text':'What color is the large shape? Answer with one color word.'}]}], 'max_tokens':16, 'temperature':0}
        with request('/v1/chat/completions', body) as r: result=json.load(r)
        result['expected']=color; results.append(result)
        text=result['choices'][0]['message']['content'].lower()
        assert color in text, (color, text)
        assert result['timings']['image_tokens'] > 0
        if fixture == 'green-large': assert result['timings']['image_tokens'] > 128
    parallel_bodies = [
        {'messages':[{'role':'user','content':[image('red'), {'type':'text','text':'What color is the large shape? Answer with one color word.'}]}], 'max_tokens':16},
        {'messages':[{'role':'user','content':'Count the integers from 1 to 1000, one per line.'}], 'max_tokens':256},
    ]
    def completion(body):
        with request('/v1/chat/completions', body) as r: return json.load(r)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        concurrent_results = list(pool.map(completion, parallel_bodies))
    assert 'red' in concurrent_results[0]['choices'][0]['message']['content'].lower()
    assert concurrent_results[1]['usage']['completion_tokens'] > 0
    results.extend(concurrent_results)
    body={'messages':[{'role':'user','content':[image('red'), {'type':'text','text':'First image. '}, image('blue'), {'type':'text','text':'Second image. Name both colors in order.'}]}], 'max_tokens':24, 'stream':True, 'stream_options':{'include_usage':True}}
    with request('/v1/chat/completions',body) as r: stream=r.read().decode()
    (out/'stream.txt').write_text(stream)
    pieces=[]
    for line in stream.splitlines():
        if line.startswith('data: ') and line != 'data: [DONE]':
            for choice in json.loads(line[6:])['choices']: pieces.append(choice['delta'].get('content',''))
    answer=''.join(pieces).lower()
    assert 'red' in answer and 'blue' in answer and '[DONE]' in stream, answer
    with request('/v1/chat/completions',{'messages':[{'role':'user','content':'Reply exactly OK.'}], 'max_tokens':8}) as r: results.append(json.load(r))
    try:
        request('/v1/chat/completions',{'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,!!!'}}]}]})
        raise AssertionError('malformed image accepted')
    except urllib.error.HTTPError as e:
        assert e.code == 400
    (out/'responses.json').write_text(json.dumps(results,indent=2))
    if a.bench_spec:
        bench=[sys.executable,str(pathlib.Path(__file__).with_name('vision_spec_bench.py')),'--url',url,'--output',str(out/'speculation.json')]
        if a.strict_match: bench.append('--strict-match')
        subprocess.run(bench,check=True)
    print(json.dumps({'status':'pass','vision_speculation':a.speculate,'colors':['red','blue','green'],'multi_image_stream':answer,'results':str(out)}))
finally:
    proc.terminate()
    try: proc.wait(timeout=3)
    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    log.close()
