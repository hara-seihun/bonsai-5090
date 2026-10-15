#!/usr/bin/env python3
"""Own one foreground server and measure sequential requests, retaining outputs.
No concurrent requests; shutdown is guaranteed even when a request fails.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from loadtest import MIXED, one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='./h5-server')
    ap.add_argument('--model', default='/root/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf')
    ap.add_argument('--drafter', default='/root/models/model.safetensors')
    ap.add_argument('--output', required=True)
    ap.add_argument('--port', type=int, default=8010)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--draft-proposals', type=int, choices=[7,16,32,64], default=7)
    ap.add_argument('--prompts', help='JSON array of prompts; default eight mixed prompts')
    a = ap.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(a.prompts).read_text()) if a.prompts else MIXED
    (out/'prompts.json').write_text(json.dumps(prompts, indent=2))
    env = {k:v for k,v in os.environ.items() if k.startswith('H5_')}
    command = [a.server,a.model,a.drafter,'--slots','1','--ctx',str(a.ctx),'--port',str(a.port),'--draft-proposals',str(a.draft_proposals)]
    url = f'http://127.0.0.1:{a.port}'
    def get(path):
        with urllib.request.urlopen(url+path,timeout=2) as r:return json.load(r)
    try:
        get('/health')
    except (urllib.error.URLError, TimeoutError):
        pass
    else:
        raise RuntimeError('Port already occupied; refusing to benchmark another server')
    rows=[]
    with open(out/'server.log','w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+90
            while True:
                if child.poll() is not None:raise RuntimeError(f'Server exited {child.returncode}')
                try:
                    health=get('/health')
                    if health.get('draft_proposals') != a.draft_proposals or health.get('verifier_rows') != a.draft_proposals+1:
                        raise RuntimeError('Server draft length does not match benchmark arm')
                    break
                except (urllib.error.URLError,TimeoutError):
                    if time.monotonic()>deadline:raise RuntimeError('Server readiness timeout')
                    time.sleep(.1)
            warm=one(url,prompts[0],256,True,False)
            (out/'warmup.json').write_text(json.dumps(warm))
            before=get('/metrics')
            with open(out/'requests.jsonl','w') as records:
                for rep in range(a.repeats):
                    for i,prompt in enumerate(prompts):
                        row=one(url,prompt,a.max_tokens,True,False)
                        if row['ntok']<=0:raise RuntimeError('Missing completion usage')
                        row.update(repeat=rep,prompt_index=i,ttft_s=row['tfirst']-row['t0'],decode_s=row['t1']-row['tfirst'])
                        row['decode_tok_s']=(row['ntok']-1)/max(row['decode_s'],1e-9)
                        row['text_sha256']=hashlib.sha256(row['text'].encode()).hexdigest()
                        rows.append(row);records.write(json.dumps(row)+'\n');records.flush()
                        print(json.dumps({k:row[k] for k in ('repeat','prompt_index','ntok','ttft_s','decode_tok_s')}),flush=True)
            after=get('/metrics')
            total=sum(r['ntok'] for r in rows); duration=sum(r['t1']-r['t0'] for r in rows)
            result={'command':command,'environment':env,'requests':len(rows),'tokens':total,'request_wall_s':duration,'end_to_end_tok_s':total/duration,'weighted_decode_tok_s':sum(r['ntok']-1 for r in rows)/sum(r['decode_s'] for r in rows),'median_decode_tok_s':statistics.median(r['decode_tok_s'] for r in rows),'median_ttft_s':statistics.median(r['ttft_s'] for r in rows),'metrics_before':before,'metrics_after':after}
            (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        finally:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':main()
