from pathlib import Path
import hashlib,json,re,subprocess,tempfile
p=Path(__file__).parent;src=p.parents[1]/'src';old=(p/'startup-v1-engine.cu').read_text();new=(src/'engine.cu').read_text()
def fragment(s,needle):
 i=s.index(needle);a=s.index('{',i);depth=1;b=a+1
 while depth:
  depth+=(s[b]=='{')-(s[b]=='}');b+=1
 return s[i:b]
def sequence(s):
 ctor=fragment(s,'  Engine(const char* path,');create='CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));'
 block=re.search(r'const auto book=dict8::dictionary\(\);dict8_book=upload<uint32_t>\(book.data\(\),book.size\(\)\);\s*dict8_setup\[0\]=1;dict8_setup\[3\]=book.size\(\)\*4;',ctor).group()
 return '\n'.join(v for _,v in sorted([(ctor.index(create),create),(ctor.index(block),block)]))
a=sequence(old);b=sequence(new)
assert a.index('upload<uint32_t>')<a.index('cudaStreamCreateWithFlags')
assert b.index('cudaStreamCreateWithFlags')<b.index('upload<uint32_t>')
u=fragment(new,'  template <class T> T* upload(')
assert u==fragment(old,'  template <class T> T* upload('),'consumer modified instead of fixing dependency'
assert 'cudaStream_t st;' in new and 'cudaStream_t st = nullptr' not in new,'default stream would mask defect'
code='''#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include "dict8_codec.hpp"
namespace dict8=h5::dict8;
using cudaStream_t=void*;
static void* ready=reinterpret_cast<void*>(0x1234);
static int uploads=0,creates=0;
constexpr int cudaStreamNonBlocking=1,cudaMemcpyHostToDevice=2;
int cudaMalloc(void**p,size_t n){*p=std::malloc(n);return *p?0:1;}
template<class T>int cudaMalloc(T**p,size_t n){return cudaMalloc(reinterpret_cast<void**>(p),n);}
int cudaMemcpyAsync(void*p,const void*s,size_t n,int,cudaStream_t st){if(st!=ready)return 2;std::memcpy(p,s,n);++uploads;return 0;}
int cudaStreamSynchronize(cudaStream_t st){return st==ready?0:2;}
int cudaStreamCreateWithFlags(cudaStream_t*p,int f){assert(f==cudaStreamNonBlocking);*p=ready;++creates;return 0;}
#define CK(x) do{if((x)!=0)throw std::runtime_error("invalid stream dependency");}while(false)
struct Trace {
 cudaStream_t st=reinterpret_cast<void*>(UINTPTR_MAX);
 uint32_t*dict8_book=nullptr;uint64_t dict8_setup[10]{};size_t weight_bytes=0;
UPLOAD
 void before(){OLD}
 void after(){NEW}
};
int main(){Trace x;bool rejected=false;try{x.before();}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&creates==0&&uploads==0);
 Trace y;y.after();assert(creates==1&&uploads==1&&y.dict8_setup[0]==1&&y.dict8_setup[3]==26244&&y.weight_bytes==26244);
 std::free(y.dict8_book);std::cout<<"old source order rejected; successor creates stream before dictionary upload; one upload; GPU execution=false\\n";
}
'''.replace('UPLOAD',u).replace('OLD',a).replace('NEW',b)
with tempfile.TemporaryDirectory() as tmp:
 f=Path(tmp)/'test.cpp';f.write_text(code);exe=Path(tmp)/'test'
 c=subprocess.run(['g++','-std=c++17','-O2','-I'+str(src),str(f),'-o',str(exe)],capture_output=True,text=True,timeout=20)
 assert c.returncode==0,c.stderr
 run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=5);assert run.returncode==0,run.stderr
 result={'state':'CPU_lifecycle_pass','GPU_execution':False,'native_model_execution':False,'old_engine_sha256':hashlib.sha256(old.encode()).hexdigest(),'successor_engine_sha256':hashlib.sha256(new.encode()).hexdigest(),'old_source_order_rejected':True,'successor_source_order_accepted':True,'consumer_body_unchanged':True,'fixture':'actual extracted source upload body and constructor create/upload ordering; host CUDA stubs inject an invalid-stream sentinel, not a claim to reproduce the hardware AV address','compile_exit':c.returncode,'run_exit':run.returncode,'output':run.stdout}
 (p/'startup-cpu-regression.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
