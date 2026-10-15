"""Probe v1 part 2: handle-based local-NTFS admission queries (synthetic file only)."""
import os, json, ctypes, ctypes.wintypes as W
EXP = 'C:/work/models/experiments/bonsai-quality-20261013/dict8-identity-mutation-probe-v1'
k = ctypes.WinDLL('kernel32', use_last_error=True); H = ctypes.c_void_p
k.CreateFileW.restype = H; k.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, W.DWORD, W.DWORD, H]
k.GetFileType.argtypes = [H]
k.GetVolumeInformationByHandleW.argtypes = [H, W.LPWSTR, W.DWORD, ctypes.POINTER(W.DWORD), ctypes.POINTER(W.DWORD), ctypes.POINTER(W.DWORD), W.LPWSTR, W.DWORD]
k.GetFileInformationByHandleEx.argtypes = [H, ctypes.c_int, ctypes.c_void_p, W.DWORD]
k.GetFinalPathNameByHandleW.argtypes = [H, W.LPWSTR, W.DWORD, W.DWORD]
k.GetVolumePathNameW.argtypes = [W.LPCWSTR, W.LPWSTR, W.DWORD]
k.GetDriveTypeW.argtypes = [W.LPCWSTR]
k.CreateSymbolicLinkW.argtypes = [W.LPCWSTR, W.LPCWSTR, W.DWORD]; k.CreateSymbolicLinkW.restype = ctypes.c_ubyte
k.CloseHandle.argtypes = [H]
INV = ctypes.c_void_p(-1).value
class STD(ctypes.Structure):
    _fields_ = [('Alloc', ctypes.c_longlong), ('Eof', ctypes.c_longlong), ('Links', W.DWORD), ('DeletePending', ctypes.c_ubyte), ('Directory', ctypes.c_ubyte)]
class TAG(ctypes.Structure):
    _fields_ = [('Attrs', W.DWORD), ('Tag', W.DWORD)]

def query(path):
    r = {'path': path}
    h = k.CreateFileW(path, 0x80000000, 1, None, 3, 0x80, None)
    if h is None or h == INV:
        r['open_err'] = ctypes.get_last_error(); return r
    r['filetype'] = k.GetFileType(h)  # 1 = FILE_TYPE_DISK
    fs = ctypes.create_unicode_buffer(64); flags = W.DWORD()
    r['vol_ok'] = bool(k.GetVolumeInformationByHandleW(h, None, 0, None, None, ctypes.byref(flags), fs, 64)); r['fs'] = fs.value; r['vol_flags'] = hex(flags.value)
    s = STD(); r['std_ok'] = bool(k.GetFileInformationByHandleEx(h, 1, ctypes.byref(s), ctypes.sizeof(s)))
    r.update(eof=s.Eof, links=s.Links, delete_pending=s.DeletePending, directory=s.Directory)
    t = TAG(); r['tag_ok'] = bool(k.GetFileInformationByHandleEx(h, 9, ctypes.byref(t), ctypes.sizeof(t))); r['attrs'] = hex(t.Attrs); r['reparse_tag'] = hex(t.Tag)
    buf = ctypes.create_string_buffer(1024)  # FileRemoteProtocolInfo = 13
    ok = k.GetFileInformationByHandleEx(h, 13, buf, 1024); r['remote_info_ok'] = bool(ok); r['remote_info_err'] = 0 if ok else ctypes.get_last_error()
    fp = ctypes.create_unicode_buffer(1024)
    n = k.GetFinalPathNameByHandleW(h, fp, 1024, 0); r['final_path'] = fp.value if n else None
    vp = ctypes.create_unicode_buffer(1024)
    if n and k.GetVolumePathNameW(fp.value, vp, 1024): r['volume_root'] = vp.value; r['drive_type'] = k.GetDriveTypeW(vp.value)  # 3 = FIXED
    fpv = ctypes.create_unicode_buffer(1024)  # VOLUME_NAME_GUID = 1
    r['final_path_guid'] = fpv.value if k.GetFinalPathNameByHandleW(h, fpv, 1024, 1) else ctypes.get_last_error()
    k.CloseHandle(h); return r

out = []
f = os.path.join(EXP, 'adm.bin').replace('/', '\\'); open(f, 'wb').write(bytes(65536))
out.append(query(f))
lnk = f[:-4] + '_symlink.bin'
ok = k.CreateSymbolicLinkW(lnk, f, 2)  # ALLOW_UNPRIVILEGED_CREATE
out.append({'symlink_create': bool(ok), 'err': 0 if ok else ctypes.get_last_error()})
if ok: out.append(query(lnk))
for p in [r'\\localhost\C$\Windows\win.ini', r'\\127.0.0.1\C$\Windows\win.ini']:
    out.append(query(p))
import string
for d in string.ascii_uppercase:
    root = d + ':\\'
    t = k.GetDriveTypeW(root)
    if t > 1: out.append({'drive': root, 'type': t})
for p in (lnk, f):
    try: os.remove(p)
    except OSError: pass
open(os.path.join(EXP, 'admission-results.json'), 'w').write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
