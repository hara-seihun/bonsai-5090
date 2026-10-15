#!/usr/bin/env python3
"""Real image ordinary/strict-speculation pairs against one already running native server."""
import argparse, base64, json, pathlib, time, urllib.request
p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:18080')
p.add_argument('--output', required=True)
p.add_argument('--max-tokens', type=int, default=128)
p.add_argument('--strict-match', action='store_true', help='fail on different greedy text/token counts')
a = p.parse_args()
fixtures = pathlib.Path(__file__).resolve().parent.parent/'bench'/'vision'
prompts = [
    ('red', 'Describe the picture in detail. Explain its colors, shape, arrangement, and background in three paragraphs.'),
    ('green-large', 'Explain how to draw this picture using Python and Pillow. Give runnable code and explain each step.'),
    ('blue', 'Neutral context. '*512 + '\nNow describe the image, including its color and shape, in three paragraphs.'),
]
results=[]
for name, text in prompts:
    image={'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode((fixtures/(name+'.png')).read_bytes()).decode()}}
    pair={}
    for spec in (False, True):
        body={'messages':[{'role':'user','content':[image, {'type':'text','text':text}]}], 'max_tokens':a.max_tokens, 'vision_speculation':spec, 'temperature':0}
        start=time.monotonic()
        req=urllib.request.Request(a.url+'/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as response: data=json.load(response)
        data['client_wall_s']=time.monotonic()-start
        pair['speculative' if spec else 'ordinary']=data
    ordinary, speculative=pair['ordinary'],pair['speculative']
    match=(ordinary['choices'][0]['message']['content']==speculative['choices'][0]['message']['content'] and ordinary['usage']==speculative['usage'])
    assert ordinary['timings']['image_tokens']>0 and not ordinary['timings']['vision_speculation']
    assert ordinary['timings']['proposed_drafts']==0
    assert speculative['timings']['vision_speculation'] and speculative['timings']['proposed_drafts']>0
    t=speculative['timings']
    record={'fixture':name,'greedy_match':match,'acceptance':t['accepted_drafts']/t['proposed_drafts'], 'ordinary':ordinary,'speculative':speculative}
    results.append(record)
    pathlib.Path(a.output).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'fixture':name,'greedy_match':match,'acceptance':record['acceptance'],
                      'ordinary_decode_tok_s':ordinary['timings']['decode_tok_per_s'],'speculative_decode_tok_s':t['decode_tok_per_s'],
                      'image_encode_s':t['image_encode_s'],'verify_s':t['verify_s'],'draft_s':t['draft_s']}),flush=True)
if a.strict_match and not all(r['greedy_match'] for r in results): raise SystemExit('image greedy parity failed')
