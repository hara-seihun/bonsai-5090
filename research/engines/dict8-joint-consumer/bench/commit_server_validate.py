#!/usr/bin/env python3
"""Validate real Scheduler mock-engine responses with the actual panel consumer."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import adaptive_commit_panel as panel

p = argparse.ArgumentParser()
p.add_argument('--captures', type=Path, required=True)
a = p.parse_args()
files = sorted(a.captures.glob('*.json'))
assert len(files) == 94, len(files)
valid = failed = 0
for file in files:
    response = json.loads(file.read_text())
    timing = response['timings']
    expected = timing['commitment']['config']
    result = panel.validate_response(response, expected, timing['draft_proposals'], 40)
    if response['choices'][0]['finish_reason'] == 'error':
        assert isinstance(result, panel.Error), (file, result)
        failed += 1
    else:
        assert not isinstance(result, panel.Error), (file, result)
        valid += 1
print(f'actual Scheduler-to-panel contract passes: {valid} valid completions, {failed} retained policy errors')
