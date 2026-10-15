#!/usr/bin/env python3
"""Bounded strict INT8/F32/F16-KV control; never changes engine or precision.

Native HTTP has no generated-ID field. Report byte equality and canonical
retokenization separately from ordinary-target native IDs. The CPU tokenizer
helper is generated into the receipt directory and linked to the same llama.
"""
import argparse
import codecs
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PROPOSALS = (7, 16, 32, 64)
EOS = (248044, 248046)
MAX_TOKENS = 128
PROMPTS = (
    '<|im_start|>user\nWrite a Python function returning the nth prime, with input validation. Give code only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
    '<|im_start|>user\nA shop has 18 boxes of 24 pencils. It sells 157 pencils. How many remain? Explain briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
    '<|im_start|>user\nExplain why seasons occur on Earth in four concise sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
)
ENV = {
    'H5_WEIGHTS': 'int8', 'H5_ACT': 'int8', 'H5_STATE': 'f32', 'H5_KV': 'f16',
    'H5_DRAFT_MARGIN': '0', 'H5_DF_WINDOW': '1024', 'H5_ATTN_TPS': '16',
    'H5_GDN_DELTA': '0', 'H5_MEGA': '0', 'H5_PDL': '0', 'H5_NOGRAPH': '1',
    'H5_THREADS': '4', 'H5_LOAD_THREADS': '4',
}
TOKENIZER_SOURCE = r'''#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  if (argc != 6) return 2;
  llama_log_set([](ggml_log_level,const char*,void*){},nullptr);
  llama_backend_init();
  auto p=llama_model_default_params(); p.vocab_only=true; p.n_gpu_layers=0;
  auto m=llama_model_load_from_file(argv[1],p); if(!m) return 3;
  auto v=llama_model_get_vocab(m);
  FILE*f=fopen(argv[4],"rb"), *o=fopen(argv[5],"wb"); if(!f||!o) return 4;
  const bool detok=std::string(argv[2])=="detokenize", special=atoi(argv[3])!=0;
  int32_t n;
  while(fread(&n,4,1,f)==1) {
    if(n<0||n>1048576) return 5;
    if(detok) {
      std::vector<int32_t> ids(n); if(fread(ids.data(),4,n,f)!=(size_t)n) return 6;
      std::string text;
      for(int t:ids) { if(t==248044||t==248046) continue;
        char b[256]; int z=llama_token_to_piece(v,t,b,sizeof(b),0,special);
        if(z>0) text.append(b,z);
      }
      int32_t z=text.size(); fwrite(&z,4,1,o); fwrite(text.data(),1,z,o);
    } else {
      std::string s(n,0); if(fread(s.data(),1,n,f)!=(size_t)n) return 6;
      std::vector<llama_token> ids(n+16);
      int z=llama_tokenize(v,s.data(),n,ids.data(),ids.size(),special,true);
      if(z<0) {ids.resize(-z); z=llama_tokenize(v,s.data(),n,ids.data(),ids.size(),special,true);}
      if(z<0) return 7;
      fwrite(&z,4,1,o); fwrite(ids.data(),4,z,o);
    }
  }
  if(ferror(f)||ferror(o)) return 8;
  fclose(f); if(fclose(o)) return 9;
  llama_model_free(m); llama_backend_free(); return 0;
}
'''


class ControlError(Exception):
    def __init__(self, kind, message):
        self.kind = kind
        super().__init__(message)


class Budget:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def remaining(self, cap=None):
        left = self.end - time.monotonic()
        if left <= 0:
            raise ControlError('deadline', 'Exact-control deadline exhausted')
        return left if cap is None else min(left, cap)


def digest(path, budget=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            if budget is not None:
                budget.remaining()
            h.update(block)
    return h.hexdigest()


def asset(path, budget=None):
    path = Path(path).resolve(strict=True)
    s = path.stat()
    return {'path': str(path), 'bytes': s.st_size, 'sha256': digest(path, budget)}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def clean_environment(base):
    return {**{k: v for k, v in base.items() if not k.startswith('H5_')}, **ENV}


def native_failure(log, returncode, stage):
    text = Path(log).read_text(errors='replace').lower()
    oom = any(x in text for x in ('out of memory', 'cudaerrormemoryallocation', 'cannot allocate memory'))
    return ControlError('native_oom' if oom else 'native_exit',
                        f'{stage}: exit={returncode}; OOM evidenced={oom}; retained log={log}')


def run_command(command, env, log, budget):
    with open(log, 'wb') as f:
        try:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=budget.remaining())
        except subprocess.TimeoutExpired as ex:
            raise ControlError('deadline', f'Command timed out; retained log={log}') from ex
    if result.returncode:
        raise native_failure(log, result.returncode, command[0])


@contextlib.contextmanager
def server_process(command, env, log):
    with open(log, 'wb') as f:
        child = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            yield child
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


def http(url, budget, body=None, cap=45, raw_path=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=budget.remaining(cap)) as response:
        raw = response.read()
        if raw_path is not None:
            Path(raw_path).write_bytes(raw)
        return json.loads(raw)


def validate_health(health, proposals):
    if (health.get('status') != 'ok' or health.get('draft_proposals') != proposals or
        health.get('draft_query_rows') != proposals + 1 or health.get('verifier_rows') != proposals + 1 or
        health.get('bonus_policy') != 'greedy'):
        raise ControlError('health_contract', f'Unexpected configured length/policy: {health}')


def response_row(response, proposals, prompt_index):
    try:
        t = response['timings']
        text = response['choices'][0]['text']
        count = response['usage']['completion_tokens']
        reason = response['choices'][0]['finish_reason']
        if not isinstance(text, str) or not isinstance(count, int) or count < 0:
            raise ValueError('invalid text/usage')
        if t['draft_proposals'] != proposals or t['verifier_rows_per_full_round'] != proposals + 1:
            raise ValueError('response length mismatch')
        hist = t['prefix_histogram']
        if len(hist) != proposals + 1 or any(not isinstance(n, int) or n < 0 for n in hist):
            raise ValueError('prefix histogram shape/value')
        if sum(hist) != t['verify_rounds']:
            raise ValueError('histogram rounds disagree')
        if t['verifier_rows'] != t['verify_rounds'] * (proposals + 1):
            raise ValueError('paid verifier rows disagree')
        if sum(i*n for i,n in enumerate(hist)) != t['accepted_drafts']:
            raise ValueError('accepted prefix histogram disagrees')
        arrays = [t[name] for name in ('position_proposed', 'position_reached', 'position_accepted')]
        if any(len(row) != proposals or any(not isinstance(n, int) or n < 0 for n in row) for row in arrays):
            raise ValueError('per-position array shape/value')
        if any(not accepted <= reached <= proposed for proposed,reached,accepted in zip(*arrays)):
            raise ValueError('per-position conditional counts disagree')
        if sum(arrays[2]) != t['accepted_drafts']:
            raise ValueError('per-position accepted counts disagree')
        emitted = t['position_emitted']
        if (len(emitted) != proposals + 1 or any(not isinstance(n, int) or n < 0 for n in emitted) or
            count not in (sum(emitted) + 1, 0) or (count == 0 and sum(emitted))):
            raise ValueError('actual emitted accounting disagrees')
        return {'prompt_index': prompt_index, 'proposals': proposals, 'text': text,
                'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'completion_tokens': count, 'finish_reason': reason, 'timings': t,
                'native_http_generated_ids_exposed': False}
    except (KeyError, IndexError, TypeError, ValueError) as ex:
        raise ControlError('response_contract', str(ex)) from ex


def comparison(reference, candidate):
    a, b = reference['text'].encode(), candidate['text'].encode()
    index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    ids_a, ids_b = reference['canonical_text_token_ids'], candidate['canonical_text_token_ids']
    token_index = next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), min(len(ids_a), len(ids_b)))
    equal = a == b
    return {'prompt_index': candidate['prompt_index'], 'proposals': candidate['proposals'],
            'reference': reference.get('reference', reference['proposals']),
            'text_exact_equal': equal, 'reference_text_sha256': reference['text_sha256'],
            'candidate_text_sha256': candidate['text_sha256'],
            'first_different_byte': None if equal else index,
            'byte_context_reference': a[max(0, index-24):index+48].decode(errors='replace'),
            'byte_context_candidate': b[max(0, index-24):index+48].decode(errors='replace'),
            'completion_tokens_equal': reference['completion_tokens'] == candidate['completion_tokens'],
            'canonical_text_token_ids_equal': ids_a == ids_b,
            'first_different_canonical_token': None if ids_a == ids_b else token_index,
            'canonical_token_context_reference': ids_a[max(0, token_index-4):token_index+8],
            'canonical_token_context_candidate': ids_b[max(0, token_index-4):token_index+8],
            'canonical_token_ids_are_retokenization_not_http_native_ids': True}


def write_strings(path, strings):
    with open(path, 'wb') as f:
        for s in strings:
            data = s.encode()
            f.write(struct.pack('<i', len(data)) + data)


def read_vectors(path, width):
    data = Path(path).read_bytes()
    offset, result = 0, []
    while offset < len(data):
        if offset + 4 > len(data):
            raise ControlError('token_protocol', 'Truncated length')
        n, = struct.unpack_from('<i', data, offset)
        offset += 4
        if n < 0 or offset + n * width > len(data):
            raise ControlError('token_protocol', 'Truncated/invalid payload')
        payload = data[offset:offset+n*width]
        result.append(list(struct.unpack(f'<{n}i', payload)) if width == 4 else payload)
        offset += n * width
    return result


def tokenizer(helper, model, mode, special, input_path, output_path, env, budget):
    command = [str(helper), model, mode, str(int(special)), str(input_path), str(output_path)]
    run_command(command, env, str(output_path) + '.log', budget)
    return read_vectors(output_path, 1 if mode == 'detokenize' else 4)


def self_test():
    env = clean_environment({'H5_DRAFT_MARGIN': '999', 'H5_WEIGHTS': 'both', 'H5_PLAN_READ': 'bad', 'PATH': '/bin'})
    assert env['H5_DRAFT_MARGIN'] == '0' and env['H5_WEIGHTS'] == 'int8' and 'H5_PLAN_READ' not in env
    for p in PROPOSALS:
        health = {'status': 'ok', 'draft_proposals': p, 'draft_query_rows': p+1,
                  'verifier_rows': p+1, 'bonus_policy': 'greedy'}
        validate_health(health, p)
        try:
            validate_health(health, 15)
        except ControlError as ex:
            assert ex.kind == 'health_contract'
        else:
            raise AssertionError('wrong length silently accepted')
        hist = [1] + [0] * p
        emitted = [1] + [0] * p
        response = {'timings': {'draft_proposals': p, 'verifier_rows_per_full_round': p+1,
                               'prefix_histogram': hist, 'verify_rounds': 1, 'verifier_rows': p+1,
                               'position_emitted': emitted, 'accepted_drafts': 0,
                               'position_proposed': [1]*p, 'position_reached': [1]+[0]*(p-1),
                               'position_accepted': [0]*p},
                    'choices': [{'text': 'abc', 'finish_reason': 'length'}], 'usage': {'completion_tokens': 2}}
        a = response_row(response, p, 0)
        a['canonical_text_token_ids'] = [4, 5]
        assert comparison(a, a)['text_exact_equal']
        b = {**a, 'text': 'abd', 'canonical_text_token_ids': [4, 6]}
        c = comparison(a, b)
        assert not c['text_exact_equal'] and c['first_different_byte'] == 2 and c['first_different_canonical_token'] == 1
        response['timings']['verifier_rows'] += 1
        try:
            response_row(response, p, 0)
        except ControlError as ex:
            assert ex.kind == 'response_contract'
        else:
            raise AssertionError('invalid verifier rows silently accepted')
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)/'bad.bin'
        path.write_bytes(struct.pack('<i', 8) + b'x')
        try:
            read_vectors(path, 4)
        except ControlError as ex:
            assert ex.kind == 'token_protocol'
        else:
            raise AssertionError('truncated output silently accepted')
        log = Path(td)/'native.log'; log.write_text('CUDA: out of memory')
        assert native_failure(log, 1, 'load').kind == 'native_oom'
        log.write_text('process killed without memory diagnosis')
        assert native_failure(log, -9, 'load').kind == 'native_exit'
    print('exact-control CPU guards: strict env, lengths/accounting, mismatch retention, token framing, explicit OOM pass')


def execute(a, out, manifest, budget):
    env = clean_environment(os.environ)
    prompts_file = out/'prompts.bin'
    write_strings(prompts_file, PROMPTS)
    save(out/'prompts.json', list(PROMPTS))
    source = out/'tokenizer-control.cpp'; source.write_text(TOKENIZER_SOURCE)
    helper = out/'tokenizer-control'
    llama = Path(a.llama_root).resolve(strict=True)
    command = ['g++', '-O2', '-std=c++17', '-I'+str(llama/'include'), '-I'+str(llama/'ggml/include'),
               str(source), '-L'+str(llama/'build/bin'), '-lllama', '-Wl,-rpath,'+str(llama/'build/bin'), '-o', str(helper)]
    manifest['tokenizer_compile_command'] = command
    run_command(command, env, out/'tokenizer-build.log', budget)
    manifest['tokenizer_helper'] = asset(helper)
    manifest['llama_library'] = asset(llama/'build/bin/libllama.so')
    tokens = tokenizer(helper, a.model, 'tokenize', False, prompts_file, out/'prompt-tokens-server.bin', env, budget)
    if len(tokens) != 3 or any(len(t) + 128 + 66 >= a.ctx for t in tokens):
        raise ControlError('prompt_capacity', 'Three prompts must fit worst-case64 context reserve')
    save(out/'prompt-token-ids.json', tokens)
    ordinary = None
    if a.reference == 'ordinary':
        ordinary_tokens = tokenizer(helper, a.model, 'tokenize', True, prompts_file, out/'prompt-tokens-ordinary.bin', env, budget)
        if ordinary_tokens != tokens:
            save(out/'prompt-tokenization-mismatch.json', {'server': tokens, 'ordinary': ordinary_tokens})
            raise ControlError('prompt_tokenization_mismatch', 'ordinary h5 adds specials; server does not. Native input IDs differ; no fallback')
        command = [a.ordinary_h5, 'gsm', a.model, str(prompts_file), str(out/'ordinary-text.bin'), '1', '128', '128']
        ordinary_env = {**env, 'H5_TOKDUMP': str(out/'ordinary-native-token-ids.txt')}
        manifest['ordinary_command'] = command
        manifest['ordinary_environment'] = {k:v for k,v in ordinary_env.items() if k.startswith('H5_')}
        save(out/'manifest.json', manifest)
        run_command(command, ordinary_env, out/'ordinary.log', budget)
        lines = (out/'ordinary-native-token-ids.txt').read_text().splitlines()
        if len(lines) != 3:
            raise ControlError('ordinary_contract', 'Expected three native token-ID rows')
        ids = [[int(x) for x in line.split()] for line in lines]
        if any(not row or len(row) > 128 or any(t in EOS for t in row[:-1]) for row in ids):
            raise ControlError('ordinary_contract', 'Invalid native token count/EOS placement')
        detok_input = out/'ordinary-detok-input.bin'
        with open(detok_input, 'wb') as f:
            for row in ids:
                f.write(struct.pack('<i', len(row)) + struct.pack(f'<{len(row)}i', *row))
        raw_text = tokenizer(helper, a.model, 'detokenize', False, detok_input, out/'ordinary-canonical-text.bin', env, budget)
        if len(raw_text) != 3:
            raise ControlError('ordinary_contract', 'Expected three decoded outputs')
        ordinary = []
        for i, (row, raw) in enumerate(zip(ids, raw_text)):
            text = codecs.getincrementaldecoder('utf-8')(errors='replace').decode(raw, final=False)
            ordinary.append({'prompt_index': i, 'reference': 'ordinary', 'proposals': None,
                             'native_token_ids': row, 'text': text,
                             'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                             'completion_tokens': len(row) - int(row[-1] in EOS),
                             'finish_reason': 'stop' if row[-1] in EOS else 'length'})
    base = f'http://127.0.0.1:{a.port}'
    try:
        http(base+'/health', budget, cap=1)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        pass
    else:
        raise ControlError('port_occupied', f'Refusing to benchmark existing server at {base}')
    panels = {}
    for p in PROPOSALS:
        arm = out/f'p{p}'; arm.mkdir()
        command = [a.server, a.model, a.drafter, '--slots', '1', '--ctx', str(a.ctx), '--prefill', '128',
                   '--host', '127.0.0.1', '--port', str(a.port), '--draft-proposals', str(p)]
        manifest['server_commands'].append(command)
        save(out/'manifest.json', manifest)
        log = arm/'server.log'
        with server_process(command, env, log) as child:
            ready_end = time.monotonic() + min(120, budget.remaining())
            while True:
                if child.poll() is not None:
                    raise native_failure(log, child.returncode, f'P{p} startup')
                try:
                    health = http(base+'/health', budget, cap=1)
                    save(arm/'health.json', health)
                    validate_health(health, p)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if time.monotonic() >= ready_end:
                        raise ControlError('startup_timeout', f'P{p} readiness timeout; {log}')
                    time.sleep(min(.05, budget.remaining()))
            rows = []
            for i, prompt in enumerate(PROMPTS):
                body = {'model': 'ternary-bonsai-2-27b', 'prompt': prompt, 'max_tokens': 128,
                        'temperature': 0, 'stream': False}
                save(arm/f'request-{i}.json', body)
                try:
                    response = http(base+'/v1/completions', budget, body, raw_path=arm/f'response-{i}.wire.json')
                except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
                    failure = native_failure(log, child.poll(), f'P{p} request{i}')
                    if child.poll() is not None or failure.kind == 'native_oom':
                        raise failure from ex
                    raise ControlError('request_failure', f'P{p} request{i}: {ex}; retained log={log}') from ex
                save(arm/f'response-{i}.json', response)
                row = response_row(response, p, i)
                row['response_wire_sha256'] = digest(arm/f'response-{i}.wire.json')
                rows.append(row)
                save(arm/'rows.json', rows)
            panels[p] = rows
    ordered = ([] if ordinary is None else ordinary) + [r for p in PROPOSALS for r in panels[p]]
    text_input = out/'all-completion-text.bin'; write_strings(text_input, [r['text'] for r in ordered])
    text_tokens = tokenizer(helper, a.model, 'tokenize', False, text_input, out/'all-canonical-text-token-ids.bin', env, budget)
    if len(text_tokens) != len(ordered):
        raise ControlError('token_protocol', 'Retokenizer result count disagrees')
    for row, ids in zip(ordered, text_tokens):
        row['canonical_text_token_ids'] = ids
        row['canonical_text_token_ids_sha256'] = hashlib.sha256(struct.pack(f'<{len(ids)}i', *ids)).hexdigest()
    references = panels[7] if ordinary is None else ordinary
    comparisons = [comparison(references[i], panels[p][i]) for p in PROPOSALS for i in range(3)]
    save(out/'all-rows.json', ordered)
    save(out/'comparisons.json', comparisons)
    save(out/'strict7-versus-long.json', [comparison(panels[7][i], panels[p][i]) for p in PROPOSALS[1:] for i in range(3)])
    mismatch = any(not c['text_exact_equal'] or not c['completion_tokens_equal'] for c in comparisons)
    manifest.update(status='mismatch' if mismatch else 'equal', comparisons=12,
                    text_mismatch_count=sum(not c['text_exact_equal'] for c in comparisons),
                    interpretation='Strict text/usage mismatches are retained target execution diagnostics, not draft-quality scores. HTTP native IDs are not exposed; retokenization is labeled separately.')
    save(out/'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'text_mismatch_count': manifest['text_mismatch_count'],
                      'receipt': str(out/'manifest.json'), 'comparisons': str(out/'comparisons.json')}), flush=True)
    return 2 if mismatch else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--server')
    ap.add_argument('--ordinary-h5')
    ap.add_argument('--reference', choices=['ordinary', 'strict7'])
    ap.add_argument('--model')
    ap.add_argument('--drafter')
    ap.add_argument('--llama-root')
    ap.add_argument('--native-object-dir', help='record the reused native CUDA objects; does not compile them')
    ap.add_argument('--source-commit', help='immutable native source commit, caller-declared')
    ap.add_argument('--output')
    ap.add_argument('--port', type=int, default=8029)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--deadline-seconds', type=int, default=300)
    a = ap.parse_args()
    if a.self_test:
        self_test(); return 0
    if not all((a.server, a.reference, a.model, a.drafter, a.llama_root, a.source_commit, a.output)):
        ap.error('server/reference/model/drafter/llama-root/source-commit/output are required')
    if a.reference == 'ordinary' and not a.ordinary_h5:
        ap.error('--reference ordinary requires --ordinary-h5; no strict7 fallback')
    if len(a.source_commit) != 40 or any(c not in '0123456789abcdef' for c in a.source_commit):
        ap.error('source-commit must be a full immutable lowercase SHA')
    if a.ctx < 512 or a.deadline_seconds <= 0 or not 1 <= a.port <= 65535:
        ap.error('invalid context/deadline/port')
    out = Path(a.output).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except OSError as ex:
        print(json.dumps({'kind': 'output_directory', 'message': str(ex), 'automatic_fallback': False}), file=sys.stderr)
        return 1
    budget = Budget(a.deadline_seconds)
    manifest = {'status': 'preparing', 'argv': sys.argv, 'source_commit_caller_declared': a.source_commit,
                'proposals': PROPOSALS, 'max_tokens': MAX_TOKENS, 'context': a.ctx, 'prefill': 128,
                'reference': a.reference, 'environment': ENV, 'server_commands': [],
                'stripped_inherited_h5_names': sorted(k for k in os.environ if k.startswith('H5_')),
                'native_http_generated_ids_exposed': False, 'control_script': asset(__file__),
                'native_object_receipts': {'status': 'not_requested'},
                'device_environment': {k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES') if k in os.environ}}
    def interrupted(signum, frame):
        raise ControlError('interrupted', f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for key in ('server', 'model', 'drafter'):
            manifest[key] = asset(getattr(a, key), budget)
            setattr(a, key, manifest[key]['path'])
        if a.ordinary_h5:
            manifest['ordinary_h5'] = asset(a.ordinary_h5, budget)
            a.ordinary_h5 = manifest['ordinary_h5']['path']
        if a.native_object_dir:
            objects = Path(a.native_object_dir).resolve(strict=True)
            manifest['native_object_receipts'] = {'status': 'hashed', 'objects':
                {name: asset(objects/name, budget) for name in ('engine.o', 'kernels.o', 'draft.o', 'fp4.o', 'gguf.o')}}
        save(out/'manifest.json', manifest)
        return execute(a, out, manifest, budget)
    except ControlError as ex:
        manifest.update(status='error', error={'kind': ex.kind, 'message': str(ex)}, automatic_fallback=False)
        save(out/'manifest.json', manifest)
        print(json.dumps(manifest['error']), file=sys.stderr)
        return 1
    except (OSError, ValueError, subprocess.SubprocessError) as ex:
        manifest.update(status='error', error={'kind': 'boundary_failure', 'message': str(ex)}, automatic_fallback=False)
        save(out/'manifest.json', manifest)
        print(json.dumps(manifest['error']), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
