"""Offline, standard-library-only replay of the retained custom NIAH evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tarfile
from protocol import validate_response, digest

ROOT = Path(__file__).resolve().parent


def replay():
    provenance = json.loads((ROOT / 'source-provenance.json').read_text())
    archives = {}
    for name, record in provenance['archives'].items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            return {'kind': 'archive_hash_mismatch', 'file': name}
        with tarfile.open(ROOT / name, 'r:gz') as archive:
            members = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
        if {n: hashlib.sha256(b).hexdigest() for n, b in members.items()} != record['members_sha256']:
            return {'kind': 'member_hash_mismatch', 'file': name}
        archives[name] = {n: json.loads(b) for n, b in members.items()}
    prompts = archives['prompts.tar.gz']
    responses = archives['responses.tar.gz']
    rows = json.loads((ROOT / 'results.json').read_text())
    if len(rows) != 120 or len({(r['arm'], r['id']) for r in rows}) != 120:
        return {'kind': 'case_coverage_error'}
    counts = {engine: {n: 0 for n in (32768, 65536, 131072)} for engine in ('halo', 'reference')}
    misses = []
    for row in rows:
        case = prompts['prepared/' + row['id'] + '.json']
        prefix = row['arm'] + '/' + row['id'] + '/'
        raw = responses[prefix + 'raw-response.json']
        wire = responses[prefix + 'wire-request.json']
        effect = responses[prefix + 'effect.json']
        expected = copy.deepcopy(case['request'])
        engine = row['arm'].split('-c')[0]
        if engine == 'halo':
            expected['prompt_token_ids'] = case['prompt_token_ids']
        if digest(case['request']) != case['request_sha256'] or wire != expected:
            return {'kind': 'request_mismatch', 'case': row['id'], 'engine': engine}
        if hashlib.sha256(case['request']['prompt'].encode()).hexdigest() != case['prompt_sha256']:
            return {'kind': 'prompt_hash_mismatch', 'case': row['id']}
        if effect['kind'] != 'response_received' or effect['http_status'] != 200:
            return {'kind': 'incomplete_response', 'case': row['id']}
        if provenance['archives']['responses.tar.gz']['members_sha256'][prefix+'raw-response.json'] != effect['raw_sha256']:
            return {'kind': 'response_hash_mismatch', 'case': row['id']}
        actual = validate_response({'kind': 'response', 'value': raw}, case)
        if actual != {k: v for k, v in row.items() if k not in ('arm', 'id')} or actual != responses[prefix + 'result.json']:
            return {'kind': 'score_mismatch', 'case': row['id'], 'engine': engine}
        if engine == 'reference':
            twire = responses[prefix+'tokenizer/wire-request.json']
            traw = responses[prefix+'tokenizer/raw-response.json']
            if twire != {'content': case['request']['prompt'], 'add_special': False, 'parse_special': True, 'with_pieces': False} or traw['tokens'] != case['prompt_token_ids']:
                return {'kind': 'tokenizer_mismatch', 'case': row['id']}
        counts[engine][case['length']] += int(row['exact_match'])
        if not row['exact_match']:
            misses.append({'engine': engine, 'id': row['id'], 'capped': row['capped'], 'reasoning_closed': row['reasoning_closed'], 'final': row['final']})
    return {'kind': 'verified', 'pairs': 60, 'responses': len(rows), 'correct': counts, 'misses': misses,
            'correct_capped': sum(r['exact_match'] and r['capped'] for r in rows)}


if __name__ == '__main__':
    try:
        result = replay()
    except (OSError, KeyError, ValueError, TypeError, tarfile.TarError) as error:
        result = {'kind': 'invalid_evidence', 'error': str(error)}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['kind'] == 'verified' else 1)
