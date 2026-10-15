from pathlib import Path
import subprocess,json,hashlib,tempfile
p=Path(__file__).resolve().parent;root=p.parents[1];src=root/'src';engine=(src/'engine.cu').read_text()
start=engine.index('      if(dict8_prepare) {');end=engine.index('\n      }',start)+8;body=engine[start:end]
assert 'pack_pair' not in engine and 'dict8_cache->layer(index)' in body and 'upload<uint8_t>(image.data(),image.size())' in body
assert engine.index('Reader::open')<engine.index('CK(cudaStreamCreateWithFlags(&st')
assert 'if(dict8_enabled){dict8_model=' in engine and '}else g.open(path);' in engine
head=r'''
#if defined(NDEBUG)
#error Dict8 CPU assertions must remain enabled
#endif
#include "dict8_cache.hpp"
#include "dict8_offline.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
using namespace h5;
struct T {std::string name;};struct M {int wf=0;uint8_t* dict8_weights=nullptr;};
static uint64_t uploads=0;static std::vector<uint8_t> last;
template<class T> T* upload(const T* p,size_t n){++uploads;last.assign(p,p+n);return reinterpret_cast<T*>(uintptr_t(1));}
static void cache_require(dict8::cache::Error){assert(false);}
template<class T> T cache_take(dict8::cache::Result<T> r){assert(std::holds_alternative<T>(r));return std::move(std::get<T>(r));}
struct Stub {unsigned count=0;dict8::cache::Result<std::vector<uint8_t>> layer(unsigned i){assert(i==count++);return std::vector<uint8_t>(dict8::cache::layer_bytes,uint8_t(17+i));}};
void consume(bool dict8_prepare,Stub* dict8_cache,uint64_t (&dict8_setup)[10]){
 uint64_t dict8_original_upload_ns=0,dict8_original_ready=0;bool dict8_enabled=true,wf0=true,interleave=true;M m;const int D=5120,GU=34816,K=D,N=GU;
 std::vector<uint8_t> W(size_t(GU/32)*(D/128)*1024);std::vector<uint16_t> S(size_t(GU)*(D/128));
 auto prefix="blk."+std::to_string(dict8_setup[1])+".";T gate{prefix+"ffn_gate.weight"},up{prefix+"ffn_up.weight"};std::vector<T*> ts{&gate,&up};
'''
tail=r'''
}
int main(int argc,char** argv){
 namespace c=dict8::cache;auto h=c::header();assert(std::get<bool>(c::validate_header(h)));
 for(auto test:std::vector<std::pair<int,c::Error>>{{0,c::Error::Header},{8,c::Error::Version},{12,c::Error::Shape},{32,c::Error::ModelIdentity},{64,c::Error::Semantics},{96,c::Error::Header}}){auto v=h;v[test.first]^=1;assert(std::get<c::Error>(c::validate_header(v))==test.second);}
 assert(c::hex(std::get<c::Digest>(c::hash("abc",3)))=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

 Stub stub;uint64_t stats[10]{};consume(false,&stub,stats);assert(uploads==0&&stub.count==0);
 consume(true,&stub,stats);assert(uploads==1&&stub.count==1&&stats[1]==1&&stats[8]==0&&last.size()==c::layer_bytes);for(auto x:last)assert(x==17);
 consume(true,&stub,stats);assert(uploads==2&&stub.count==2&&stats[1]==2&&stats[8]==0);for(auto x:last)assert(x==18);
 assert(argc==4);auto read=[](const char* p){std::ifstream f(p,std::ios::binary);assert(f);return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),{});};
 auto gate=read(argv[1]),up=read(argv[2]),expected=read(argv[3]);assert(gate.size()==35840&&up.size()==35840&&expected.size()==66560);
 uint64_t checked=0;
 for(unsigned kb=0;kb<40;kb++){int8_t trit[32][128];uint8_t native[2048];
  for(unsigned arm=0;arm<2;arm++){auto& input=arm?up:gate;for(unsigned r=0;r<32;r++)dict8::offline::decode_weight_trits(input.data()+(r*40+kb)*28,trit[r]);dict8::offline::wf0(trit,native+1024*arm);}
  auto packed=std::get<std::array<uint8_t,1664>>(dict8::pack_pair(native,native+1024));assert(!std::memcmp(packed.data(),expected.data()+kb*1664,1664));checked+=packed.size();
 }
 std::cout<<"{\"header_mutations\":6,\"sha256_known_vector\":true,\"actual_engine_branch_stub_uploads\":"<<uploads<<",\"startup_encode_ns\":"<<stats[8]<<",\"unchanged_unset_branch\":true,\"independent_fixture_bytes\":"<<checked<<",\"gpu_work\":false,\"native_cached_startup_ns\":null}\n";
}
'''
with tempfile.TemporaryDirectory(prefix='dict8-cache-test-') as td:
 t=Path(td);(t/'test.cpp').write_text(head+body+tail)
 command=['g++','-std=c++17','-O2','-I'+str(src),str(t/'test.cpp'),str(src/'gguf.cpp'),'-lcrypto','-o',str(t/'test')]
 r=subprocess.run(command,capture_output=True,text=True,timeout=25);assert r.returncode==0,r.stderr
 fixtures=Path('~/work/five-wide-native');r=subprocess.run([str(t/'test'),str(fixtures/'gu-scale-absorb-cpu-v1/blk.0.ffn_gate.weight.rows32.bin'),str(fixtures/'gu-scale-absorb-cpu-v1/blk.0.ffn_up.weight.rows32.bin'),str(fixtures/'gu-joint9-checkpoint-v1/dict8-block0-rows32-k5120.bin')],capture_output=True,text=True,timeout=10)
 assert r.returncode==0,r.stderr
 result=json.loads(r.stdout);result.update(engine_sha256=hashlib.sha256(engine.encode()).hexdigest(),source_branch_sha256=hashlib.sha256(body.encode()).hexdigest(),compile_command=command)
 (p/'cpu-test-receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
