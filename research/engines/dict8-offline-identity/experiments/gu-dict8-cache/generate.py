from pathlib import Path
import hashlib,json,base64
p=Path(__file__).resolve().parent;root=p.parents[1];src=root/'src';frozen=(root/'experiments/gu-dict8-engine/frozen-engine.cu').read_text()
a=frozen.index('static inline void decode_ptq1');b=frozen.index('\n// frag2 layout',a)
decode=frozen[a:b].replace('decode_ptq1','decode_weight_trits').replace(', float* scale','')
decode=decode[:decode.index('  uint16_t hs;')]+'}\n'
a=frozen.index('          for (int w = 0; w < 8; w++) {',frozen.index('dense5::pack_lane(chunk'))
b=frozen.index('\n          }\n        }\n        for (int gid',a)+12
loop=frozen[a:b]
text='#pragma once\n#include "dict8_codec.hpp"\nnamespace h5::dict8::offline {\n'+decode+'\ninline void wf0(const int8_t (&trit)[32][128], uint8_t* chunk) {\n for(int l=0;l<32;l++){const int gid=l>>2,tq=l&3;\n'+loop+'\n}\n}\n'
(src/'dict8_offline.hpp').write_text(text)
gg=(src/'gguf.cpp').read_text();a=gg.index('static size_t tbytes');b=gg.index('\nvoid Gguf::open',a);tb=gg[a:b]
a=gg.index('  const uint8_t* p =',b);b=gg.index('\n}\nconst GTensor&',a);parse=gg[a:b]
model_header=(p/'model.hpp.in').read_text().replace('@TBYTES@',tb).replace('@PARSE@',parse)
(src/'dict8_model.hpp').write_text(model_header)
files=[f.name for f in src.iterdir() if f.is_file() and (f.suffix in ['.h','.hpp','.cuh','.inc'] or f.name in ['dict8.cu','dict8_checks.cu','engine.cu','gguf.cpp']) and f.name!='dict8_binding.hpp']
bind={n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files};bind['frozen-engine.cu']=hashlib.sha256(frozen.encode()).hexdigest();bind['format']='D8CACHE1 retained-local-NTFS-or-private-Linux-snapshot one-identity-scan little-endian kb/group interleaved gate/up pair; 13bits/key lane-striped; scales/means remain original'
bind['generate.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
bind['model.hpp.in']=hashlib.sha256((p/'model.hpp.in').read_bytes()).hexdigest()
bind['cache_tool.cpp']=hashlib.sha256((p/'cache_tool.cpp').read_bytes()).hexdigest()
bind['integrate.py']=hashlib.sha256((root/'experiments/gu-dict8-engine/integrate.py').read_bytes()).hexdigest()
bind['CMakeLists.txt']=hashlib.sha256((root/'experiments/gu-dict8-engine/CMakeLists.txt').read_bytes()).hexdigest()
s=hashlib.sha256(json.dumps(bind,sort_keys=True).encode()).hexdigest()
payload=json.loads((p/'payload-identity.json').read_text());table=base64.b64decode(payload['table_base64']);assert len(table)==2048 and hashlib.sha256(table).hexdigest()==payload['table_sha256']
(src/'dict8_binding.hpp').write_text('#pragma once\nnamespace h5::dict8::cache {\ninline constexpr char model_sha256[]="53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3";\ninline constexpr unsigned long long model_size=5946648928ULL;\ninline constexpr char semantics_sha256[]="'+s+'";\ninline constexpr char payload_table_sha256[]="'+payload['table_sha256']+'";\ninline constexpr unsigned char payload_table[]={'+','.join(str(b) for b in table)+'};\n}\n')
(p/'semantic-binding.json').write_text(json.dumps({'sha256':s,'binding':bind},indent=2))
print(s)
