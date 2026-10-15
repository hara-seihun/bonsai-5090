#!/usr/bin/env python3
"""Deterministic OCR/spatial/chart tasks, ordinary versus speculative image decode.
Synthetic functional checks, not a representative vision benchmark. Owns server lifetime.
"""
import argparse, base64, json, pathlib, re, subprocess, time, urllib.request
from PIL import Image, ImageDraw, ImageFont


def fixtures(root):
    root.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
    im = Image.new('RGB', (768, 512), 'white'); d = ImageDraw.Draw(im)
    d.text((35, 20), 'Warehouse inventory', font=font, fill='black')
    rows = [('Item', 'Count', 'Code'), ('Bolts', '37', 'K8'), ('Washers', '24', 'M3'), ('Nuts', '16', 'R6')]
    for i, row in enumerate(rows):
        y = 90 + i * 90
        for x, value in zip((45, 330, 560), row): d.text((x, y), value, font=font, fill='black')
        d.line((30, y + 65, 735, y + 65), fill='black', width=2)
    im.save(root / 'inventory.png')
    im = Image.new('RGB', (768, 512), 'white'); d = ImageDraw.Draw(im)
    d.text((35, 15), 'Units sold', font=font, fill='black')
    for x, label, value, color in [(110, 'April', 18, '#2868b2'), (320, 'May', 31, '#369445'), (530, 'June', 24, '#be4038')]:
        y = 425 - value * 10
        d.rectangle((x, y, x + 110, 425), fill=color)
        d.text((x + 20, y - 38), str(value), font=small, fill='black')
        d.text((x + 5, 445), label, font=small, fill='black')
    im.save(root / 'chart.png')
    im = Image.new('RGB', (768, 512), 'white'); d = ImageDraw.Draw(im)
    d.rectangle((70, 80, 240, 250), fill='red')
    d.ellipse((480, 80, 650, 250), fill='blue')
    d.polygon([(365, 310), (280, 465), (450, 465)], fill='green')
    im.save(root / 'layout.png')
    return [
        ('inventory', 'How many bolts are listed?', '37'),
        ('inventory', 'What is the code for washers?', 'm3'),
        ('inventory', 'How many nuts are listed?', '16'),
        ('inventory', 'What is the total count of all three items?', '77'),
        ('chart', 'Which month has the most units sold?', 'may'),
        ('chart', 'How many units were sold in June?', '24'),
        ('chart', 'How many more units were sold in May than April?', '13'),
        ('layout', 'What color is the circle?', 'blue'),
        ('layout', 'What shape is below the other two shapes?', 'triangle'),
        ('layout', 'Is the red square left or right of the blue circle?', 'left'),
    ]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--server', default='./h5-server')
    p.add_argument('--model', required=True); p.add_argument('--draft', required=True); p.add_argument('--mmproj', required=True)
    p.add_argument('--output', required=True); p.add_argument('--port', type=int, default=18083)
    a = p.parse_args(); out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    tasks = fixtures(out / 'fixtures'); (out/'tasks.json').write_text(json.dumps(tasks, indent=2))
    url = f'http://127.0.0.1:{a.port}'; records = []
    cmd = [a.server, a.model, a.draft, '--mmproj', a.mmproj, '--slots', '1', '--ctx', '4096', '--prefill', '128', '--port', str(a.port), '--host', '127.0.0.1', '--vision-gpu', '1', '--image-max-tokens', '1024']
    with (out/'server.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                if proc.poll() is not None: raise RuntimeError(f'server exited {proc.returncode}')
                try:
                    with urllib.request.urlopen(url+'/health', timeout=1): break
                except OSError:
                    if time.monotonic() > deadline: raise TimeoutError('server startup')
                    time.sleep(.2)
            for index, (name, prompt, expected) in enumerate(tasks):
                encoded = base64.b64encode((out/'fixtures'/f'{name}.png').read_bytes()).decode()
                # Alternate arm order to avoid assigning all first-use costs to ordinary.
                for spec in ((False, True) if index % 2 == 0 else (True, False)):
                    body = {'messages': [{'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+encoded}},
                        {'type': 'text', 'text': prompt+' Answer only with the requested number or word.'}]}],
                        'temperature': 0, 'max_tokens': 32, 'vision_speculation': spec}
                    request = urllib.request.Request(url+'/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    started = time.monotonic()
                    with urllib.request.urlopen(request, timeout=120) as response: data = json.load(response)
                    text = data['choices'][0]['message']['content']
                    normalized = re.sub(r'[^a-z0-9]', '', text.lower())
                    record = dict(index=index, fixture=name, question=prompt, expected=expected, speculative=spec,
                                  correct=normalized == expected, wall_s=time.monotonic()-started, response=data)
                    records.append(record)
                    with (out/'requests.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                    print(json.dumps({k: record[k] for k in ('index','speculative','correct','wall_s')}), flush=True)
            summary = {arm: {'correct': sum(r['correct'] for r in records if r['speculative']==spec), 'total': len(tasks)} for arm,spec in [('ordinary',False),('speculative',True)]}
            (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n'); print(json.dumps(summary))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()

if __name__ == '__main__': main()
