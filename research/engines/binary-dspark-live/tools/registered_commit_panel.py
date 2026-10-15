#!/usr/bin/env python3
"""An explicit frozen cap7 corpus route over the shared native panel runner."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import sys
import unicodedata
import adaptive_commit_panel as base


def registered(path, expected_sha):
    try:
        path = Path(path).resolve()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            return base.Error('manifest_mismatch', str(path))
        m = json.loads(raw, object_pairs_hook=base.unique_object)
        if m['schema'] != 'halo-cap7-fresh32-v1' or m['seed'] != 20261008 or m['scoring_performed'] is not False:
            return base.Error('unknown_corpus', 'Require frozen cap7 protocol, seed20261008 and no scoring')
        expected_files = {'dataset-test.jsonl', 'exclusions160.json', 'suffix.txt', 'selection32.json', 'prompts32.json'}
        if set(m['artifacts']) != expected_files:
            return base.Error('artifact_set_mismatch', str(path))
        for name, info in m['artifacts'].items():
            p = path.parent / name
            if p.stat().st_size != info['bytes'] or base.sha(p) != info['sha256']:
                return base.Error('corpus_artifact_mismatch', name)
        if m['artifacts']['dataset-test.jsonl']['sha256'] != '752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac':
            return base.Error('dataset_mismatch', 'Require pinned GSM8K source rows')
        if (path.parent / 'suffix.txt').read_text() != base.SUFFIX:
            return base.Error('suffix_mismatch', 'Shared scorer suffix differs')
        dataset = [json.loads(line) for line in (path.parent / 'dataset-test.jsonl').read_text().splitlines()]
        exclusion = json.loads((path.parent / 'exclusions160.json').read_text())
        selection = json.loads((path.parent / 'selection32.json').read_text())
        prompts = json.loads((path.parent / 'prompts32.json').read_text())
        prior = set(exclusion['source_ids'])
        if len(dataset) != 1319 or len(prior) != 160 or not all(type(i) is int and 0 <= i < 1319 for i in prior):
            return base.Error('exclusion_mismatch', 'Require recorded160 source identities')
        def norm(q):
            return ' '.join(unicodedata.normalize('NFKC', q).split())
        questions = {dataset[i]['question'] for i in prior}
        if questions != set(exclusion['questions']):
            return base.Error('exclusion_question_mismatch', 'Original rows differ')
        normalized = {norm(q) for q in questions}
        excluded = {i for i, row in enumerate(dataset) if i in prior or row['question'] in questions or norm(row['question']) in normalized}
        if excluded != set(exclusion['effective_ids']):
            return base.Error('exclusion_union_mismatch', 'ID/exact/normalized union differs')
        ids = random.Random(20261008).sample([i for i in range(1319) if i not in excluded], 32)
        if m['selected_source_ids'] != ids or selection != [{**dataset[i], 'source_index': i} for i in ids]:
            return base.Error('selection_mismatch', 'Deterministic selection replay differs')
        if prompts != [r['question'] + base.SUFFIX for r in selection] or len({norm(r['question']) for r in selection}) != 32:
            return base.Error('prompt_mismatch', 'Prompt content or uniqueness differs')
        identity = {'registered_corpus_sha256': expected_sha, 'registered_adapter_sha256': base.sha(Path(__file__)),
                    'base_runner_sha256': base.sha(Path(base.__file__)), 'scorer_sha256': base.sha(base.ROOT / 'tools/gsm8k.py'),
                    'dataset_sha256': m['artifacts']['dataset-test.jsonl']['sha256'],
                    'exclusion_sha256': m['artifacts']['exclusions160.json']['sha256'],
                    'selection_sha256': m['artifacts']['selection32.json']['sha256'],
                    'prompts_sha256': m['artifacts']['prompts32.json']['sha256']}
        return selection, prompts, identity, raw
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return base.Error('registered_input_or_io', str(error))


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('--corpus-manifest', required=True)
    p.add_argument('--corpus-sha256', required=True)
    p.add_argument('--validate-only', action='store_true')
    own, rest = p.parse_known_args()
    loaded = registered(own.corpus_manifest, own.corpus_sha256)
    if isinstance(loaded, base.Error):
        print(json.dumps({'state': 'error', 'error': loaded.data()})); return 1
    if own.validate_only:
        if rest:
            print(json.dumps({'state': 'error', 'error': {'code': 'unused_arguments', 'detail': 'validate-only takes no panel args'}})); return 1
        print(json.dumps({'state': 'valid', 'identity': loaded[2], 'questions': 32, 'excluded': 160})); return 0
    folder = Path(own.corpus_manifest).resolve().parent
    def corpus(args):
        fresh = registered(own.corpus_manifest, own.corpus_sha256)
        if isinstance(fresh, base.Error):
            return fresh
        if Path(args.selection).resolve() != folder / 'selection32.json' or Path(args.prompts).resolve() != folder / 'prompts32.json':
            return base.Error('corpus_cli_mismatch', 'Use manifest sibling files')
        if not (0 <= args.start < 32 and 1 <= args.count <= 32 - args.start):
            return base.Error('subset_mismatch', 'Subset must lie in frozen32')
        gold, text = fresh[0][args.start:args.start + args.count], fresh[1][args.start:args.start + args.count]
        return gold, text, {**fresh[2], 'start': args.start, 'count': args.count, 'source_ids': [r['source_index'] for r in gold]}
    base.corpus = corpus
    args = base.parser().parse_args(rest)
    signal.signal(signal.SIGTERM, base.interrupted); signal.signal(signal.SIGINT, base.interrupted)
    try:
        result = base.run(args)
        if not isinstance(result, base.Error):
            final = registered(own.corpus_manifest, own.corpus_sha256)
            if isinstance(final, base.Error):
                result.update(state='error', error=final.data(), correct=None, accuracy=None)
            else:
                (Path(args.output).resolve() / 'registered-corpus-manifest.json').write_bytes(final[3])
            base.dump(Path(args.output).resolve() / 'summary.json', result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = base.Error('registered_panel_io', str(error))
    if isinstance(result, base.Error):
        print(json.dumps({'state': 'error', 'error': result.data()})); return 1
    print(json.dumps(result, allow_nan=False)); return 0 if result['state'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
