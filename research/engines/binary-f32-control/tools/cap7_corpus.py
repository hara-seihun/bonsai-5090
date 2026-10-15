#!/usr/bin/env python3
"""Freeze a native7 cap comparison corpus outside the recorded prior160 questions."""
import hashlib
import json
from pathlib import Path
import random
import sys
import unicodedata

PRIOR = Path('~/work/halo-tree/adaptive-commit/fresh32/corpus')
SEED = 20261008


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(q):
    return ' '.join(unicodedata.normalize('NFKC', q).split())


def freeze(output):
    try:
        if output.exists():
            return {'state': 'error', 'error': 'output_exists'}
        dataset_path = PRIOR / 'dataset-test.jsonl'
        if sha(dataset_path) != '752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac':
            return {'state': 'error', 'error': 'dataset_mismatch'}
        if (sha(PRIOR / 'exclusion-manifest.json') != '3f4b3efc15a6c22cd80a0af031ca8041976669d24a9652424da4e7bd3842dc90'
                or sha(PRIOR / 'selection32.json') != '925bb88b38c0dff92d86499c14926e835bdfe6b8a3e8c01c88aa5366bcd25caf'):
            return {'state': 'error', 'error': 'predecessor_hash_mismatch'}
        exclusions = json.loads((PRIOR / 'exclusion-manifest.json').read_text())
        previous = json.loads((PRIOR / 'selection32.json').read_text())
        rows = [json.loads(line) for line in dataset_path.read_text().splitlines()]
        ids = set(exclusions['effective_excluded_source_indices'])
        if len(rows) != 1319 or len(ids) != 128 or len(previous) != 32:
            return {'state': 'error', 'error': 'predecessor_count_mismatch'}
        if len(ids | {r['source_index'] for r in previous}) != 160:
            return {'state': 'error', 'error': 'predecessor_overlap'}
        ids.update(r['source_index'] for r in previous)
        questions = {rows[i]['question'] for i in ids}
        normalized = {norm(q) for q in questions}
        excluded = [i for i, r in enumerate(rows) if i in ids or r['question'] in questions or norm(r['question']) in normalized]
        eligible = [i for i in range(1319) if i not in set(excluded)]
        selected_ids = random.Random(SEED).sample(eligible, 32)
        selection = [{**rows[i], 'source_index': i} for i in selected_ids]
        suffix = (PRIOR / 'suffix.txt').read_text()
        prompts = [r['question'] + suffix for r in selection]
        output.mkdir(parents=True)
        artifacts = {}
        files = {'dataset-test.jsonl': dataset_path.read_bytes(), 'suffix.txt': suffix.encode(),
                 'exclusions160.json': (json.dumps({'source_ids': sorted(ids), 'effective_ids': excluded,
                                                   'questions': sorted(questions),
                                                   'predecessors': {name: sha(PRIOR / name) for name in ['exclusion-manifest.json', 'selection32.json', 'freeze-manifest.json']}}
                                                  , indent=2) + '\n').encode(),
                 'selection32.json': (json.dumps(selection, ensure_ascii=False, indent=2) + '\n').encode(),
                 'prompts32.json': (json.dumps(prompts, ensure_ascii=False, indent=2) + '\n').encode()}
        for name, raw in files.items():
            p = output / name
            with p.open('xb') as stream:
                stream.write(raw)
            p.chmod(0o444)
            artifacts[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        manifest = {'schema': 'halo-cap7-fresh32-v1', 'seed': SEED, 'scoring_performed': False,
                    'exclusion_scope': 'recorded prior128 union frozen transfer32; not a universal campaign scan',
                    'known_prior_questions': 160, 'selected_source_ids': selected_ids, 'artifacts': artifacts}
        p = output / 'manifest.json'
        with p.open('x') as stream:
            json.dump(manifest, stream, indent=2); stream.write('\n')
        p.chmod(0o444)
        return {'state': 'frozen', 'manifest': str(p), 'sha256': sha(p), 'source_ids': selected_ids}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return {'state': 'error', 'error': 'corpus_input_or_io', 'detail': str(error)}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: cap7_corpus.py NEW_DIRECTORY', file=sys.stderr); sys.exit(2)
    result = freeze(Path(sys.argv[1]).resolve())
    print(json.dumps(result)); sys.exit(0 if result['state'] == 'frozen' else 1)
