from pathlib import Path
import tempfile,subprocess,json,hashlib
p=Path(__file__).resolve().parent;root=p.parents[1];source=(root/'src/engine.cu').read_text();a=source.index('void* h5_create(const char* path, int slots, int ctx, int rows)');b=source.index('{',a);depth=1;i=b+1
while depth:
 if source[i]=='{':depth+=1
 elif source[i]=='}':depth-=1
 i+=1
fn=source[a:i]
program='''#if defined(NDEBUG)
#error Dict8 CPU assertions must remain enabled
#endif
#include "dict8_cache.hpp"
#include <cassert>
#include <stdexcept>
#include <cstdio>
namespace h5 {int dict8_cache_error=0;}
int mode=0;
struct Engine {Engine(const char*,int,int,int){if(mode==1)throw std::bad_alloc();if(mode==2)throw std::runtime_error("injected");}};
'''+fn+'''
int main(){mode=1;h5::dict8_cache_error=7;assert(!h5_create("",1,1,1));assert(h5::dict8_cache_error==int(h5::dict8::cache::Error::Allocation));mode=2;assert(!h5_create("",1,1,1));assert(h5::dict8_cache_error==int(h5::dict8::cache::Error::Initialization));mode=0;auto* p=(Engine*)h5_create("",1,1,1);assert(p&&h5::dict8_cache_error==0);delete p;}
'''
with tempfile.TemporaryDirectory(prefix='dict8-boundary-') as td:
 d=Path(td);(d/'test.cpp').write_text(program);r=subprocess.run(['g++','-std=c++17','-O2','-I'+str(root/'src'),str(d/'test.cpp'),'-lcrypto','-o',str(d/'test')],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr
 r=subprocess.run([str(d/'test')],capture_output=True,text=True,timeout=5);assert r.returncode==0,r.stderr
 result={'actual_engine_create_source_sha256':hashlib.sha256(fn.encode()).hexdigest(),'allocation_before_constructor_body':'Allocation','generic_exception':'Initialization','successful_creation_clears_stale_error':True,'gpu_work':False,'proof':'Extracted actual C boundary, CPU injected constructor only; no native Engine execution'}
 (p/'boundary-test-receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
