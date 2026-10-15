"""Compile the actual exported identity-stats function against a CPU-only sentinel owner."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile
p=Path(__file__).resolve().parent;root=p.parents[1];source=(root/'src/engine.cu').read_text()
a=source.index('extern "C" DICT8_EXPORT int h5_dict8_identity_stats(');b=source.index('{',a);depth=1;i=b+1
while depth:
 if source[i]=='{':depth+=1
 elif source[i]=='}':depth-=1
 i+=1
fn=source[a:i]
exports=(root/'src/dict8.cuh').read_text();export_define=exports[exports.index('#ifdef _WIN32'):exports.index('#endif')+6]
program=export_define+'\n'+'''#if defined(NDEBUG)
#error Dict8 CPU assertions must remain enabled
#endif
#include "dict8_model.hpp"
#include <cassert>
namespace c=h5::dict8::cache;
namespace h5 {
struct SentinelModel {c::ModelCosts value{101,102,103,104,105,106,107,108,109,110,111,112,113,114};const c::ModelCosts& costs()const{return value;}};
struct Engine {std::unique_ptr<SentinelModel> dict8_model;};
}
'''+fn+'''
int main(){uint64_t out[14];for(auto& x:out)x=999;h5::Engine e;
 assert(h5_dict8_identity_stats(nullptr,out,14)==-1);
 assert(h5_dict8_identity_stats(&e,nullptr,14)==-1);
 assert(h5_dict8_identity_stats(&e,out,13)==-1);
 assert(h5_dict8_identity_stats(&e,out,15)==-1);
 assert(h5_dict8_identity_stats(&e,out,14)==-2);for(auto x:out)assert(x==999);
 e.dict8_model=std::make_unique<h5::SentinelModel>();assert(h5_dict8_identity_stats(&e,out,14)==0);
 for(unsigned i=0;i<14;i++)assert(out[i]==101+i);
}
'''
with tempfile.TemporaryDirectory(prefix='dict8-identity-abi-') as td:
 t=Path(td);(t/'abi.cpp').write_text(program)
 if os.name=='nt':
  exe=t/'abi.exe';command=['cl','/nologo','/O2','/EHsc','/std:c++17','/DNOMINMAX','/D_CRT_SECURE_NO_WARNINGS','/I'+str(root/'src'),str(t/'abi.cpp'),'/Fe:'+str(exe),'/link','bcrypt.lib']
 else:
  exe=t/'abi';command=['g++','-std=c++17','-O2','-I'+str(root/'src'),str(t/'abi.cpp'),'-lcrypto','-o',str(exe)]
 r=subprocess.run(command,cwd=t,capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stdout+r.stderr
 r=subprocess.run([str(exe)],cwd=t,capture_output=True,text=True,timeout=5);assert r.returncode==0,r.stdout+r.stderr
 receipt={'actual_export_sha256':hashlib.sha256(fn.encode()).hexdigest(),'ordered_sentinel_fields':14,'null_pointer_bad_count_unset_owner_errors':True,'error_does_not_write_output':True,'gpu_work':False,'native_dll_loaded':False,'proof':'Actual exported identity stats body compiled against CPU sentinel owner; no actual Engine construction','compile_command':command}
 (p/'identity-abi-test-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
