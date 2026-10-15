#!/usr/bin/env python3
"""Extract the immutable measured Q4_0 arm; never quantize here."""
import hashlib,json,pathlib,sys
import gguf
source,out=map(pathlib.Path,sys.argv[1:3]);out.mkdir(parents=True,exist_ok=True)
r=gguf.GGUFReader(source);t=next(t for t in r.tensors if t.name=='output.weight')
assert int(t.tensor_type)==2 and list(t.shape)==[2048,248320]
raw=t.data.reshape(-1).view('uint8');assert raw.nbytes==286064640
raw.tofile(out/'head-q4_0.raw')
info=dict(source=str(source.resolve()),source_identity=dict(size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns),tensor='output.weight',type=2,shape=[2048,248320],offset=int(t.data_offset),bytes=int(t.n_bytes),head_sha256=hashlib.sha256(raw).hexdigest(),weight_arm='hybrid-head-q4_0',repacking_only=True,not_lossless_to_v1_q8=True)
(out/'q4-fixture.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))
