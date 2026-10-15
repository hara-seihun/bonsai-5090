#!/usr/bin/env python3
"""Print SASS with decoded scoreboard fields: sass_window.py FILE FUNC_SUBSTR START END (hex)."""
import re, subprocess, sys
path, fn, a0, a1 = sys.argv[1], sys.argv[2], int(sys.argv[3], 16), int(sys.argv[4], 16)
L = subprocess.run(['cuobjdump', '-sass', path], check=True, capture_output=True, text=True).stdout.split('\n')
on = False; i = 0
while i < len(L):
    m = re.search(r'Function : (\S+)', L[i])
    if m: on = fn in m.group(1)
    m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*/\* (0x[0-9a-f]+) \*/', L[i])
    if on and m and i + 1 < len(L):
        m2 = re.search(r'/\* (0x[0-9a-f]+) \*/', L[i + 1])
        c = int(m2.group(1), 16) >> 41; a = int(m.group(1), 16)
        if a0 <= a <= a1:
            wb, rb = (c >> 5) & 7, (c >> 8) & 7
            print(f"{a:05x} st={c & 0xf:2d} wb={wb if wb != 7 else '-'} rb={rb if rb != 7 else '-'} wait={(c >> 11) & 0x3f:06b} {m.group(2)}")
        i += 2; continue
    i += 1
