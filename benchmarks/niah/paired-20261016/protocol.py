import json, hashlib, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CAP = 8192
SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0, 'frequency_penalty': 0.0, 'repeat_penalty': 1.0}

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def error(kind, **details):
    return {'kind': kind, **details}

def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(ROOT / 'assets/tokenizer/tokenizer.json'))

def renderer():
    from transformers.utils.chat_template_utils import _compile_jinja_template
    template = _compile_jinja_template((ROOT / 'assets/chat_template.jinja').read_text())
    def render(user):
        return template.render(messages=[{'role':'user', 'content':user}], tools=None,
                               add_generation_prompt=True, enable_thinking=True,
                               reasoning_effort='xhigh', add_vision_id=False)
    return render

def construct(tok, render, length, mode, depth, seed):
    rng = random.Random(f'custom-niah-v1:{length}:{mode}:{depth}:{seed}')
    count = {'single':1, 'multi':4}[mode]
    keys = ['vault_' + ''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=10)) for _ in range(count)]
    values = [''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=14)) for _ in range(count)]
    gold = dict(zip(keys, values))
    needles = [f'\nRegistry entry: the access code for {k} is {v}.\n' for k, v in gold.items()]
    positions = [depth] if count == 1 else [(depth + i) / 4 for i in range(4)]
    introduction = ('Read this archive and retrieve the access codes requested after it. '
                    'Unrelated archive notes are distractors.\n<archive>\n')
    suffix = ('\n</archive>\nReturn only one JSON object mapping each requested vault to its exact access code. '
              'Requested vaults: ' + ', '.join(keys) + '.\n')
    subjects = ('The library', 'A harbor clerk', 'The survey team', 'A garden committee', 'The local workshop', 'An evening visitor')
    verbs = ('recorded a routine inspection', 'discussed ordinary supplies', 'reviewed the weather report', 'checked the delivery schedule', 'organized the reading room', 'described a quiet afternoon')
    endings = ('and filed the notes.', 'before closing for the day.', 'while the lamps were lit.', 'without changing the schedule.', 'near the west entrance.', 'after a brief conversation.')
    filler = ''.join(f'{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(endings)}\n' for _ in range(length // 9 + 1000))
    ft = tok.encode(filler, add_special_tokens=False).ids
    assert len(ft) > length
    overhead = len(tok.encode(render(introduction + ''.join(needles) + suffix), add_special_tokens=False).ids)
    n = length - overhead - 32
    def build(n, pad):
        chunks, last = [], 0
        for pos, needle in zip(positions, needles):
            at = round(pos * n)
            chunks += [tok.decode(ft[last:at], skip_special_tokens=False), needle]
            last = at
        chunks += [tok.decode(ft[last:n], skip_special_tokens=False), ' a' * pad]
        hay = ''.join(chunks)
        return render(introduction + hay + suffix), hay
    pad = 0
    for _ in range(30):
        prompt, hay = build(n, pad)
        encoded = tok.encode(prompt, add_special_tokens=False)
        delta = length - len(encoded.ids)
        if delta == 0:
            break
        if pad + delta >= 0:
            pad += delta
        else:
            n += delta - 32
            pad = 32
    else:
        raise AssertionError('Exact length construction did not converge')
    assert len(encoded.ids) == length
    observed = []
    archive_start = prompt.index('<archive>') + len('<archive>\n')
    archive_end = prompt.index('</archive>')
    offsets = encoded.offsets
    hay_tokens = [i for i, (start, end) in enumerate(offsets) if end > archive_start and start < archive_end]
    for pos, needle, val in zip(positions, needles, values):
        assert prompt.count(val) == 1
        start = prompt.index(needle)
        # Offsets are measured on the final, fully rendered prompt, never on characters alone.
        token_start = next(i for i, (_, end) in enumerate(offsets) if end > start)
        token_end = next((i for i, (s, _) in enumerate(offsets) if s >= start + len(needle)), len(offsets))
        observed.append({'requested_haystack_depth':pos, 'token_start':token_start, 'token_end':token_end,
                         'prompt_fraction':token_start / length,
                         'haystack_fraction':(token_start - hay_tokens[0]) / len(hay_tokens)})
    identity = {'length':length, 'mode':mode, 'depth_phase':depth, 'content_seed':seed}
    cid = f'{length}-{mode}-d{int(depth*100):03d}-s{seed}'
    raw_request = {'model':'bonsai', 'prompt':prompt, **SAMPLING,
                   'seed':int(digest(identity)[:8], 16), 'max_tokens':CAP, 'stream':False,
                   'special':True, 'stop':['<|im_end|>', '<|endoftext|>'], 'cache_prompt':False}
    return {'id':cid, **identity, 'prompt_tokens':length, 'context':length + CAP,
            'reserved_output_tokens':CAP, 'needle_count':count, 'gold':gold,
            'needle_positions':observed, 'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'prompt_token_ids':encoded.ids, 'request':raw_request, 'request_sha256':digest(raw_request)}

def score(raw, gold):
    if '</think>' not in raw:
        return {'kind':'scored', 'exact_match':False, 'key_recall':0.0, 'reasoning_closed':False, 'final':''}
    final = raw.split('</think>', 1)[1].strip()
    try:
        # Duplicate keys are invalid even if the final repeated value is correct.
        def pairs(items):
            result = {}
            for k, v in items:
                if k in result:
                    raise ValueError('duplicate JSON key')
                result[k] = v
            return result
        answer = json.loads(final, object_pairs_hook=pairs)
    except (ValueError, TypeError):
        answer = None
    recall = sum(answer.get(k) == v for k, v in gold.items()) / len(gold) if isinstance(answer, dict) else 0.0
    return {'kind':'scored', 'exact_match':answer == gold, 'key_recall':recall,
            'reasoning_closed':True, 'final':final}

def validate_response(result, case):
    if result['kind'] != 'response':
        return result
    value = result['value']
    try:
        choice = value['choices'][0]
        raw = choice['text']
        finish = choice['finish_reason']
        usage = value['usage']
        if not isinstance(raw, str) or finish not in ('length', 'stop'):
            return error('response_contract', message='Unknown output or finish reason')
        if usage['prompt_tokens'] != case['prompt_tokens']:
            return error('prompt_token_count_mismatch', expected=case['prompt_tokens'], observed=usage['prompt_tokens'])
        if type(usage['completion_tokens']) is not int or not 0 <= usage['completion_tokens'] <= CAP:
            return error('output_cap_contract', usage=usage)
        return {**score(raw, case['gold']), 'capped':finish == 'length', 'finish_reason':finish,
                'usage':usage, 'request_sha256':case['request_sha256'], 'prompt_sha256':case['prompt_sha256']}
    except (KeyError, IndexError, TypeError) as e:
        return error('response_contract', message=str(e))
