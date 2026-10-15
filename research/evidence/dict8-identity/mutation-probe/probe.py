"""Dict8 model-identity mutation probe v1 (Windows, CPU-only, synthetic file).

Question: can a pre-existing writable mapped view (its file HANDLE and section HANDLE
closed) mutate bytes seen through a read-only mapping while a retained CreateFile read
handle denies FILE_SHARE_WRITE/FILE_SHARE_DELETE? Plus the surrounding mutation vectors
that decide whether "hash once, keep deny-write handle for the view's lifetime" is a
sufficient invariant.

Every actor runs in its own child process (agent.py) controlled over JSON lines, so the
results reflect cross-process kernel behaviour, not same-process aliasing. Only the
synthetic file under EXP is touched. No model, GPU, CUDA DLL, mutex or serving state.
"""
import json, os, subprocess, sys, time, hashlib, platform, ctypes

EXP = 'C:/work/models/experiments/bonsai-quality-20261013/dict8-identity-mutation-probe-v1'
os.makedirs(EXP, exist_ok=True)
SIZE = 1 << 20

AGENT = r'''
import sys, json, ctypes, ctypes.wintypes as W
k = ctypes.WinDLL('kernel32', use_last_error=True)
H = ctypes.c_void_p
k.CreateFileW.restype = H; k.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, W.DWORD, W.DWORD, H]
k.CreateFileMappingW.restype = H; k.CreateFileMappingW.argtypes = [H, ctypes.c_void_p, W.DWORD, W.DWORD, W.DWORD, W.LPCWSTR]
k.MapViewOfFile.restype = ctypes.c_void_p; k.MapViewOfFile.argtypes = [H, W.DWORD, W.DWORD, W.DWORD, ctypes.c_size_t]
k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]; k.CloseHandle.argtypes = [H]
k.WriteFile.argtypes = [H, ctypes.c_void_p, W.DWORD, ctypes.POINTER(W.DWORD), ctypes.c_void_p]
k.ReadFile.argtypes = [H, ctypes.c_void_p, W.DWORD, ctypes.POINTER(W.DWORD), ctypes.c_void_p]
k.SetFilePointerEx.argtypes = [H, ctypes.c_longlong, ctypes.c_void_p, W.DWORD]
k.SetEndOfFile.argtypes = [H]
k.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, W.DWORD, ctypes.POINTER(W.DWORD)]
k.GetCurrentProcess.restype = H
k.WriteProcessMemory.argtypes = [H, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
k.MoveFileExW.argtypes = [W.LPCWSTR, W.LPCWSTR, W.DWORD]
k.DeleteFileW.argtypes = [W.LPCWSTR]
k.CreateHardLinkW.argtypes = [W.LPCWSTR, W.LPCWSTR, ctypes.c_void_p]
k.FlushViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
INV = ctypes.c_void_p(-1).value
def err(ok):
    return {'ok': bool(ok), 'err': 0 if ok else ctypes.get_last_error()}
def h(x): return ctypes.c_void_p(x)
for line in sys.stdin:
    c = json.loads(line); op = c['op']; r = {}
    try:
        if op == 'open':
            v = k.CreateFileW(c['path'], c['access'], c['share'], None, c.get('disp', 3), 0x80, None)
            ok = v is not None and v != INV; r = err(ok); r['h'] = v if ok else None
        elif op == 'section':
            v = k.CreateFileMappingW(h(c['h']), None, c['protect'], 0, 0, None); r = err(v); r['s'] = v
        elif op == 'view':
            v = k.MapViewOfFile(h(c['s']), c['access'], 0, 0, 0); r = err(v); r['a'] = v
        elif op == 'close':
            r = err(k.CloseHandle(h(c['h'])))
        elif op == 'unmap':
            r = err(k.UnmapViewOfFile(c['a']))
        elif op == 'poke':  # store through a view pointer (SEH -> OSError)
            b = bytes.fromhex(c['hex']); ctypes.memmove(c['a'] + c['off'], b, len(b)); r = {'ok': True, 'err': 0}
            if c.get('flush'): r['flush'] = err(k.FlushViewOfFile(c['a'], 0))
        elif op == 'peek':
            r = {'ok': True, 'hex': ctypes.string_at(c['a'] + c['off'], c['n']).hex()}
        elif op == 'writefile':
            b = bytes.fromhex(c['hex']); n = W.DWORD()
            k.SetFilePointerEx(h(c['h']), c['off'], None, 0)
            r = err(k.WriteFile(h(c['h']), b, len(b), ctypes.byref(n), None)); r['n'] = n.value
        elif op == 'readfile':
            buf = ctypes.create_string_buffer(c['n']); n = W.DWORD()
            k.SetFilePointerEx(h(c['h']), c['off'], None, 0)
            r = err(k.ReadFile(h(c['h']), buf, c['n'], ctypes.byref(n), None)); r['hex'] = buf.raw[:n.value].hex()
        elif op == 'seteof':
            k.SetFilePointerEx(h(c['h']), c['size'], None, 0); r = err(k.SetEndOfFile(h(c['h'])))
        elif op == 'vprotect':
            old = W.DWORD(); r = err(k.VirtualProtect(c['a'], 4096, c['protect'], ctypes.byref(old))); r['old'] = old.value
        elif op == 'wpm_self':
            b = bytes.fromhex(c['hex']); n = ctypes.c_size_t()
            r = err(k.WriteProcessMemory(k.GetCurrentProcess(), c['a'] + c['off'], b, len(b), ctypes.byref(n))); r['n'] = n.value
        elif op == 'move':
            r = err(k.MoveFileExW(c['src'], c['dst'], c['flags']))
        elif op == 'delete':
            r = err(k.DeleteFileW(c['path']))
        elif op == 'hardlink':
            r = err(k.CreateHardLinkW(c['link'], c['target'], None))
        else:
            r = {'ok': False, 'err': -1, 'why': 'unknown op'}
    except OSError as e:
        r = {'ok': False, 'exc': str(e)}
    sys.stdout.write(json.dumps(r) + '\n'); sys.stdout.flush()
'''
agent_path = os.path.join(EXP, 'agent.py')
open(agent_path, 'w').write(AGENT)

GR, GW, DEL = 0x80000000, 0x40000000, 0x10000
FILE_WRITE_DATA, FILE_APPEND_DATA = 0x2, 0x4
SR, SW, SD = 1, 2, 4
PRO, PRW, PWC = 0x02, 0x04, 0x08
MR, MW, MC = 0x4, 0x2, 0x1
CREATE_ALWAYS, OPEN_EXISTING = 2, 3

class Actor:
    def __init__(s, name):
        s.name = name
        s.p = subprocess.Popen([sys.executable, '-X', 'utf8', agent_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    def __call__(self_, **c):
        self_.p.stdin.write(json.dumps(c) + '\n'); self_.p.stdin.flush()
        line = self_.p.stdout.readline()
        if not line:
            return {'ok': False, 'dead': True, 'rc': self_.p.wait()}
        return json.loads(line)
    def stop(s):
        try: s.p.stdin.close(); s.p.wait(10)
        except Exception: s.p.kill()

LOG = []
def rec(scenario, step, r):
    LOG.append({'scenario': scenario, 'step': step, **r}); print(scenario, step, json.dumps(r))

def fresh(name):
    path = os.path.join(EXP, name).replace('/', '\\')
    with open(path, 'wb') as f: f.write(bytes(SIZE))  # zero file
    return path

def disk_hex(path, off=0, n=8):
    # independent observer: ordinary read (share all)
    with open(path, 'rb') as f: f.seek(off); return f.read(n).hex()

def reader_capability(a, path, share):
    """dict8-style read-only mapping with a retained deny-write handle."""
    o = a(op='open', path=path, access=GR, share=share); 
    if not o['ok']: return o, None, None
    s = a(op='section', h=o['h'], protect=PRO); v = a(op='view', s=s['s'], access=MR); a(op='close', h=s['s'])
    return o, v['a'], s

PAT = 'a5a5a5a5a5a5a5a5'

def S0_current_code():
    sc = 'S0_current_code_share_RD_handle_closed'; path = fresh('s0.bin')
    rd, wr = Actor('reader'), Actor('writer')
    o = rd(op='open', path=path, access=GR, share=SR | SD); s = rd(op='section', h=o['h'], protect=PRO)
    v = rd(op='view', s=s['s'], access=MR); rd(op='close', h=o['h']); rd(op='close', h=s['s'])
    rec(sc, 'reader view before', rd(op='peek', a=v['a'], off=0, n=8))
    w = wr(op='open', path=path, access=GR | GW, share=SR | SW | SD); rec(sc, 'writer open RW', w)
    if w['ok']:
        rec(sc, 'writer WriteFile', wr(op='writefile', h=w['h'], off=0, hex=PAT))
        rec(sc, 'reader view after WriteFile', rd(op='peek', a=v['a'], off=0, n=8))
        rec(sc, 'writer shrink SetEndOfFile', wr(op='seteof', h=w['h'], size=4096))
    rd.stop(); wr.stop()

def S1_preexisting_writable_view(reader_share, tag, keep_section):
    sc = f'S1_{tag}_writable_view_preexists{"_section_handle_kept" if keep_section else "_all_handles_closed"}'
    path = fresh(f's1_{tag}_{int(keep_section)}.bin')
    rd, wr = Actor('reader'), Actor('writer')
    w = wr(op='open', path=path, access=GR | GW, share=SR | SW | SD)
    s = wr(op='section', h=w['h'], protect=PRW); v = wr(op='view', s=s['s'], access=MW)
    rec(sc, 'writer RW view', v)
    wr(op='close', h=w['h'])
    if not keep_section: wr(op='close', h=s['s'])
    o, ra, _ = reader_capability(rd, path, reader_share)
    rec(sc, 'reader deny-write open while writable view exists', o)
    rec(sc, 'writer store through preexisting view', wr(op='poke', a=v['a'], off=0, hex=PAT))
    if ra:
        rec(sc, 'reader view after store', rd(op='peek', a=ra, off=0, n=8))
    rec(sc, 'disk after store', {'ok': True, 'hex': disk_hex(path)})
    # now remove the writable view; can the deny-write open succeed afterwards?
    rec(sc, 'writer unmap', wr(op='unmap', a=v['a']))
    if not ra:
        o2, ra2, _ = reader_capability(rd, path, reader_share)
        rec(sc, 'reader deny-write open after writer unmapped', o2)
        if keep_section and ra2:
            v2 = wr(op='view', s=s['s'], access=MW)
            rec(sc, 'writer maps NEW writable view from retained section after reader open', v2)
            if v2.get('a'):
                rec(sc, 'writer store through new view', wr(op='poke', a=v2['a'], off=8, hex=PAT))
                rec(sc, 'reader view after new-view store', rd(op='peek', a=ra2, off=8, n=8))
            # copy-on-write view from retained RW section: private, must not reach reader
            v3 = wr(op='view', s=s['s'], access=MC); rec(sc, 'writer FILE_MAP_COPY view from retained section', v3)
            if v3.get('a'):
                rec(sc, 'writer store through COPY view', wr(op='poke', a=v3['a'], off=16, hex=PAT))
                rec(sc, 'reader view after COPY store', rd(op='peek', a=ra2, off=16, n=8))
    rd.stop(); wr.stop()

def S2_retained_handle_blocks_writers():
    sc = 'S2_retained_deny_write_delete_handle_vs_later_writers'; path = fresh('s2.bin')
    link = path[:-4] + '_link.bin'; other = path[:-4] + '_other.bin'
    with open(other, 'wb') as f: f.write(b'\xff' * SIZE)
    rd, wr = Actor('reader'), Actor('writer')
    o, ra, _ = reader_capability(rd, path, SR); rec(sc, 'reader open share=READ', o)
    for nm, acc, sh in [('GENERIC_WRITE', GR | GW, SR | SW | SD), ('FILE_WRITE_DATA only', FILE_WRITE_DATA, SR | SW | SD),
                        ('FILE_APPEND_DATA only', FILE_APPEND_DATA, SR | SW | SD), ('DELETE', DEL, SR | SW | SD),
                        ('GENERIC_READ share=all', GR, SR | SW | SD), ('GENERIC_READ share=READ (second reader)', GR, SR)]:
        rec(sc, 'open ' + nm, wr(op='open', path=path, access=acc, share=sh))
    rec(sc, 'CREATE_ALWAYS overwrite', wr(op='open', path=path, access=GR | GW, share=SR | SW | SD, disp=CREATE_ALWAYS))
    rec(sc, 'hardlink create', wr(op='hardlink', link=link, target=path))
    if os.path.exists(link):
        rec(sc, 'open hardlink GENERIC_WRITE', wr(op='open', path=link, access=GR | GW, share=SR | SW | SD))
    rec(sc, 'MoveFileEx REPLACE_EXISTING onto path', wr(op='move', src=other, dst=path, flags=1))
    rec(sc, 'MoveFileEx rename path away', wr(op='move', src=path, dst=path + '.moved', flags=0))
    rec(sc, 'DeleteFile', wr(op='delete', path=path))
    ro = wr(op='open', path=path, access=GR, share=SR | SW | SD)
    if ro['ok']:
        rec(sc, 'RW section from read-only handle', wr(op='section', h=ro['h'], protect=PRW))
        s = wr(op='section', h=ro['h'], protect=PWC); rec(sc, 'WRITECOPY section from read handle', s)
        if s.get('s'):
            v = wr(op='view', s=s['s'], access=MC); rec(sc, 'COPY view', v)
            if v.get('a'):
                rec(sc, 'store through COPY view', wr(op='poke', a=v['a'], off=0, hex=PAT, flush=True))
    if ra: rec(sc, 'reader view at end', rd(op='peek', a=ra, off=0, n=8))
    rec(sc, 'disk at end', {'ok': True, 'hex': disk_hex(path)})
    rd.stop(); wr.stop()
    for p in (link, other, path + '.moved'):
        try: os.remove(p)
        except OSError: pass

def S3_in_process_vectors():
    sc = 'S3_in_process_vectors_on_own_readonly_view'; path = fresh('s3.bin')
    rd = Actor('reader'); o, ra, _ = reader_capability(rd, path, SR)
    rec(sc, 'plain store into RO view', rd(op='poke', a=ra, off=0, hex=PAT))
    rec(sc, 'VirtualProtect PAGE_READWRITE', rd(op='vprotect', a=ra, protect=PRW))
    rec(sc, 'VirtualProtect PAGE_WRITECOPY', rd(op='vprotect', a=ra, protect=PWC))
    rec(sc, 'WriteProcessMemory(self) into RO view', rd(op='wpm_self', a=ra, off=0, hex=PAT))
    rec(sc, 'reader view after WPM', rd(op='peek', a=ra, off=0, n=8))
    rec(sc, 'disk after WPM', {'ok': True, 'hex': disk_hex(path)})
    rd.stop()

def S4_handle_must_outlive_view():
    sc = 'S4_deny_handle_closed_before_view'; path = fresh('s4.bin')
    rd, wr = Actor('reader'), Actor('writer')
    o, ra, _ = reader_capability(rd, path, SR); rd(op='close', h=o['h'])
    w = wr(op='open', path=path, access=GR | GW, share=SR | SW | SD); rec(sc, 'writer open after reader closed handle (view kept)', w)
    if w['ok']:
        rec(sc, 'writer WriteFile', wr(op='writefile', h=w['h'], off=0, hex=PAT))
        rec(sc, 'reader view after WriteFile', rd(op='peek', a=ra, off=0, n=8))
    rd.stop(); wr.stop()

def S5_preexisting_writer_handle():
    sc = 'S5_open_writer_handle_preexists'; path = fresh('s5.bin')
    rd, wr = Actor('reader'), Actor('writer')
    w = wr(op='open', path=path, access=GR | GW, share=SR | SW | SD)
    o, ra, _ = reader_capability(rd, path, SR); rec(sc, 'reader deny-write open while writer handle open', o)
    rd.stop(); wr.stop()

info = {'python': sys.version, 'platform': platform.platform(), 'admin': bool(ctypes.windll.shell32.IsUserAnAdmin()), 'exp': EXP,
        'fs': None, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
try:
    buf = ctypes.create_unicode_buffer(64); ctypes.windll.kernel32.GetVolumeInformationW(EXP[:3], None, 0, None, None, None, buf, 64); info['fs'] = buf.value
except Exception as e: info['fs'] = str(e)
print('INFO', json.dumps(info))
S0_current_code()
for share, tag in [(SR, 'shareR'), (SR | SD, 'shareRD')]:
    for keep in (False, True):
        S1_preexisting_writable_view(share, tag, keep)
S2_retained_handle_blocks_writers()
S3_in_process_vectors()
S4_handle_must_outlive_view()
S5_preexisting_writer_handle()
out = {'info': info, 'log': LOG}
open(os.path.join(EXP, 'results.json'), 'w').write(json.dumps(out, indent=1))
for f in os.listdir(EXP):
    if f.endswith('.bin'):
        try: os.remove(os.path.join(EXP, f))
        except OSError as e: print('cleanup', f, e)
print('RESULTS_JSON_BEGIN'); print(json.dumps(out)); print('RESULTS_JSON_END')
