"""CPU-only exact-model API test with isolated, explicitly repinned tiny fixtures.
No production capability bypass exists. No native model or CUDA DLL is opened.
"""
from pathlib import Path
import hashlib,json,os,struct,subprocess,tempfile
p=Path(__file__).resolve().parent;root=p.parents[1];src=root/'src'
def sha(data):return hashlib.sha256(data).hexdigest()
engine=(src/'engine.cu').read_text()
bootstrap=next(line.strip() for line in engine.splitlines() if 'if(dict8_enabled){dict8_model=cache_take' in line)
selector=next(line.strip() for line in engine.splitlines() if 'const Gguf& model()const' in line)
assert 'g.base,g.size' not in engine and 'pack_pair' not in engine
assert engine.index(bootstrap)<engine.index('CK(cudaStreamCreateWithFlags(&st')
assert 'g.t(' not in engine and 'g.kv' not in engine
model=struct.pack('<IIQQ',0x46554747,3,1,0)+struct.pack('<Q',6)+b'weight'+struct.pack('<I Q I Q',1,1,0,0)
model=model.ljust(64,b'\0')+bytes([42,0,0,0]);model=model.ljust(256,b'\0')
images=[bytes([17+i])*1664 for i in range(2)];table=b''.join(hashlib.sha256(x).digest() for x in images)
with tempfile.TemporaryDirectory(prefix='dict8-identity-test-') as td:
 t=Path(td);fs=t/'fixture';fs.mkdir()
 for f in src.iterdir():
  if f.is_file() and (f.suffix in ['.h','.hpp'] or f.name=='gguf.cpp'):(fs/f.name).write_bytes(f.read_bytes())
 binding='''#pragma once
namespace h5::dict8::cache {
inline constexpr char model_sha256[]="MODEL";
inline constexpr unsigned long long model_size=256ULL;
inline constexpr char semantics_sha256[]="SEMANTICS";
inline constexpr char payload_table_sha256[]="TABLE";
inline constexpr unsigned char payload_table[]={PAYLOAD};
}
'''.replace('MODEL',sha(model)).replace('SEMANTICS','ab'*32).replace('TABLE',sha(table)).replace('PAYLOAD',','.join(map(str,table)))
 (fs/'dict8_binding.hpp').write_text(binding)
 common=(fs/'dict8_common.hpp').read_text();assert common.count('constexpr uint32_t layers=64,K=5120,N=34816;')==1
 (fs/'dict8_common.hpp').write_text(common.replace('constexpr uint32_t layers=64,K=5120,N=34816;','constexpr uint32_t layers=2,K=128,N=64;'))
 test=(p/'identity_test.cpp').read_text().replace('@SELECTOR@',selector).replace('@BOOTSTRAP@',bootstrap)
 (t/'identity_test.cpp').write_text(test);(t/'model.gguf').write_bytes(model)
 h=bytearray(192);h[:8]=b'D8CACHE1';struct.pack_into('<6I',h,8,1,2,128,64,832,1664);h[32:64]=hashlib.sha256(model).digest();h[64:96]=bytes.fromhex('ab'*32);h[128:]=table
 (t/'weights.d8').write_bytes(h+b''.join(images))
 if os.name=='nt':
  exe=t/'identity_test.exe';command=['cl','/nologo','/O2','/EHsc','/std:c++17','/DNOMINMAX','/D_CRT_SECURE_NO_WARNINGS','/I'+str(fs),str(t/'identity_test.cpp'),str(fs/'gguf.cpp'),'/Fe:'+str(exe),'/link','bcrypt.lib']
 else:
  exe=t/'identity_test';command=['g++','-std=c++17','-O2','-I'+str(fs),str(t/'identity_test.cpp'),str(fs/'gguf.cpp'),'-lcrypto','-Wl,--wrap=EVP_DigestUpdate','-o',str(exe)]
 compiled=subprocess.run(command,capture_output=True,text=True,timeout=25,cwd=t);assert compiled.returncode==0,compiled.stdout+compiled.stderr
 run=subprocess.run([str(exe),str(t),'fixture'],capture_output=True,text=True,timeout=10,cwd=t);assert run.returncode==0,(run.returncode,run.stdout,run.stderr)
 receipt=json.loads(run.stdout)
 if os.name!='nt':
  fault_command=['g++','-std=c++17','-O2','-I'+str(fs),str(p/'identity_fault_test.cpp'),str(fs/'gguf.cpp'),'-lcrypto']+['-Wl,--wrap='+x for x in ['mmap','munmap','mprotect','read','EVP_DigestUpdate']]+['-o',str(t/'identity_fault_test')]
  compiled=subprocess.run(fault_command,capture_output=True,text=True,timeout=25,cwd=t);assert compiled.returncode==0,compiled.stdout+compiled.stderr
  run_fault=subprocess.run([str(t/'identity_fault_test'),str(t/'model.gguf')],capture_output=True,text=True,timeout=10,cwd=t);assert run_fault.returncode==0,(run_fault.returncode,run_fault.stdout,run_fault.stderr)
  receipt.update(json.loads(run_fault.stdout))
 receipt.update(proof='Actual generated Engine bootstrap and model selector, production ExactModel/Reader algorithm with isolated tiny fixture binding and reduced cache geometry; no actual native Engine execution',platform=os.name,engine_sha256=sha(engine.encode()),bootstrap_sha256=sha(bootstrap.encode()),model_source_sha256=sha((src/'dict8_model.hpp').read_bytes()),reader_source_sha256=sha((src/'dict8_cache.hpp').read_bytes()),common_source_sha256=sha((src/'dict8_common.hpp').read_bytes()),fixture_model_sha256=sha(model),fixture_table_sha256=sha(table),fixture_test_sha256=sha(test.encode()),compile_command=command)
 (p/'identity-test-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
