#!/usr/bin/env python3
"""Extract exact template and scalar metadata from a local GGUF or downloaded header."""
import argparse, hashlib, json, struct
from pathlib import Path

def extract(path):
    data=Path(path).read_bytes() if Path(path).stat().st_size < 64*1024*1024 else None
    f=Path(path).open('rb')
    def read(n):
        b=f.read(n)
        if len(b)!=n: raise ValueError('Incomplete GGUF metadata; download a larger header')
        return b
    def unpack(fmt): return struct.unpack('<'+fmt,read(struct.calcsize('<'+fmt)))[0]
    def string(): return read(unpack('Q')).decode('utf-8')
    formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    def value(kind, keep=True):
        if kind==8: return string()
        if kind==9:
            inner=unpack('I');n=unpack('Q')
            if inner in formats:
                read(n*struct.calcsize('<'+formats[inner]));return {'count':n,'element_type':inner}
            for _ in range(n): value(inner,False)
            return {'count':n,'element_type':inner}
        return unpack(formats[kind])
    if read(4)!=b'GGUF': raise ValueError('Expected GGUF')
    version=unpack('I');tensors=unpack('Q');count=unpack('Q');meta={}
    for _ in range(count):
        key=string();kind=unpack('I');meta[key]=value(kind)
    offset=f.tell();f.close()
    return {'version':version,'tensors':tensors,'metadata_end':offset,'metadata':meta}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('gguf');ap.add_argument('output');a=ap.parse_args()
    result=extract(a.gguf);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    template=result['metadata']['tokenizer.chat_template']
    (out/'chat_template.jinja').write_text(template)
    result['template_sha256']=hashlib.sha256(template.encode()).hexdigest()
    result['source_file']=str(Path(a.gguf).resolve())
    (out/'gguf-metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['metadata_end','template_sha256']}))
if __name__=='__main__': main()
