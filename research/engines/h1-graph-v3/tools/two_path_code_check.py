#!/usr/bin/env python3
"""Small functional checks on retained mixed-prompt code, not an overall task score."""
import argparse
import ast
import itertools
import json
from pathlib import Path
import re
import struct
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('outputs', nargs='+', type=Path, help='pathmatrix length-prefixed .bin outputs')
a = ap.parse_args()

def texts(p):
    b = p.read_bytes(); out = []
    while b:
        if len(b) < 4: raise ValueError('truncated length')
        n = struct.unpack_from('<I', b)[0]
        if len(b) < n + 4: raise ValueError('truncated text')
        out.append(b[4:4+n].decode()); b = b[4+n:]
    return out

reports = []
for path in a.outputs:
    out = texts(path)
    # Request2's complete Python block precedes truncated requested tests.
    blocks = re.findall(r'```python\n(.*?)```', out[2], re.S)
    tree = ast.parse(blocks[0])
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'merge_sorted_lists')
    # No unrelated generated top-level code is evaluated.
    env = {'__builtins__': {'list': list, 'int': int, 'len': len}}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
    merge = env['merge_sorted_lists']
    lists = [list(x) for n in range(4) for x in itertools.combinations_with_replacement([-1, 0, 1], n)]
    passed = 0
    for x, y in itertools.product(lists, repeat=2):
        aa, bb = x[:], y[:]
        assert merge(aa, bb) == sorted(x+y), (path, x, y)
        assert aa == x and bb == y, (path, 'modified inputs')
        passed += 1
    # All outputs place the function's closing brace at column0. Ignore generated test suffix.
    match = re.search(r'function\s+(\w+)\s*\(', out[3])
    source = out[3][match.start():]
    source = source[:source.index('\n}')+2]
    samples = [('', {}), ('!!! ... ???', {}), ('Hello, hello! HELLO world.', {'hello':3,'world':1}),
               ('the cat sat on the mat', {'the':2,'cat':1,'sat':1,'on':1,'mat':1}),
               ('one\ntwo\tONE', {'one':2,'two':1}), ('a,b;a!', {'a':2,'b':1})]
    script = r'''
const vm = require('vm');
const data = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const sandbox = {};
vm.createContext(sandbox);
vm.runInContext(data.source, sandbox, {timeout:100});
let ordinary=0;
for (const [s, expected] of data.samples) {
  const got = sandbox[data.fn](s);
  if (JSON.stringify(got)!==JSON.stringify(expected)) throw new Error('ordinary mismatch '+s);
  ordinary++;
}
const edge=sandbox[data.fn]('constructor constructor __proto__');
console.log(JSON.stringify({ordinary_passed:ordinary, inherited_key_case_passed:edge.constructor===2 && edge.__proto__===1}));
'''
    result = subprocess.run(['node','-e',script], input=json.dumps({'source':source,'fn':match.group(1),'samples':samples}),
                            text=True, capture_output=True, timeout=5, check=True)
    reports.append({'output':str(path),'python_merge_passed':passed,'javascript_word_frequency':json.loads(result.stdout),
                    'scope':'Completed first functions only; requested generated tests and other tasks remain partly truncated.'})
print(json.dumps(reports, indent=2))
