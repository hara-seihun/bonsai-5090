from pathlib import Path
import os,subprocess,hashlib,json,struct,base64,zipfile,io,time
from fractions import Fraction
root=Path('C:/work/models')
base=root/'experiments/bonsai-quality-20261013'
dest=base/'gu-scale-absorb-cpu-v1'
if dest.exists(): raise RuntimeError('existing CPU generation: inspect before any successor')
dest.mkdir()
image=base/'custom-wide256/build/quality_custom.dll'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(image)=='a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c'
tool=root/'tools/cuda/bin/cuobjdump.exe'
nv=root/'tools/cuda_nvdisasm-windows-x86_64-13.0.85-archive/bin/nvdisasm.exe'
env=os.environ.copy();env['NVDISASM_PATH']=str(nv.parent);env['PATH']=str(nv.parent)+';'+env['PATH']
fn='_ZN2h510k_gemm_fp4ILi8ELi2ELi32ELi3ELi0ELi0ELb0ELb1ELb0EEEvPKhS2_PKfiiS2_S2_iPfiS4_'
cmd=[str(tool),'--dump-sass','--function',fn,str(image)]
r=subprocess.run(cmd,capture_output=True,env=env,timeout=45)
(dest/'serving-carrier-true.sass').write_bytes(r.stdout)
(dest/'serving-carrier-true.stderr').write_bytes(r.stderr)
assert r.returncode==0 and ('Function : '+fn) in r.stdout.decode()
r2=subprocess.run([str(tool),'--dump-resource-usage',str(image)],capture_output=True,env=env,timeout=45)
assert r2.returncode==0
text=r2.stdout.decode();start=text.index('Function '+fn+':');selected=text[start:text.find('\n Function ',start+1)]
(dest/'serving-carrier-true.resources').write_text(selected)
for name in ['engine.cu','fp4.cu']:(dest/name).write_bytes((base/'custom-wide256/src'/name).read_bytes())
# This parser only reads the existing GGUF; it imports/loads no native library.
model=root/'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
f=model.open('rb');size=model.stat().st_size
read=lambda n:f.read(n)
def u(fmt):
 b=read(struct.calcsize('<'+fmt));assert len(b)==struct.calcsize('<'+fmt)
 return struct.unpack('<'+fmt,b)[0]
def string(keep):
 n=u('Q');assert n<=size
 if keep:return read(n).decode('utf8')
 f.seek(n,1)
width={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skip_value(t):
 if t in width:f.seek(width[t],1)
 elif t==8:string(False)
 elif t==9:
  et,n=u('I'),u('Q');assert n<=size
  if et in width:f.seek(width[et]*n,1)
  else:
   assert et==8
   for _ in range(n):string(False)
 else:raise RuntimeError('unsupported GGUF metadata type '+str(t))
assert read(4)==b'GGUF';version=u('I');assert version in [2,3]
nt,nk=u('Q'),u('Q');align=32
for _ in range(nk):
 key=string(True);t=u('I')
 if key=='general.alignment':assert t==4;align=u('I')
 else:skip_value(t)
tensors=[]
for _ in range(nt):
 name=string(True);nd=u('I');dims=[u('Q') for _ in range(nd)];typ,off=u('I'),u('Q')
 tensors.append({'name':name,'dims':dims,'type':typ,'offset':off})
data=(f.tell()+align-1)//align*align
selected_t=[t for t in tensors if t['name'] in ['blk.0.ffn_gate.weight','blk.0.ffn_up.weight']]
assert len(selected_t)==2, [t['name'] for t in tensors if 'ffn_gate' in t['name']][:4]
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
q2=[0,1,2,3,4,6,8,12]
def scale512(c):
 e,m=c>>3,c&7
 return m if e==0 else (8+m)*(1<<(e-1))
coefs={Fraction(q*scale512(c),1024) for q in q2 for c in range(127)}
def odd_core(n):
 if not n:return 0
 while n%2==0:n//=2
 return n
rows=[]
for t in selected_t:
 assert t['type']==143 and t['dims']==[5120,17408]
 count=32;off=data+t['offset'];f.seek(off);raw=read(count*40*28);assert len(raw)==count*40*28
 (dest/(t['name']+'.rows32.bin')).write_bytes(raw)
 for row in range(count):
  hs=[struct.unpack_from('<H',raw,(row*40+k)*28+26)[0] for k in range(40)]
  scales=[struct.unpack('<e',struct.pack('<H',h))[0] for h in hs]
  total=0.
  for s in scales:total=f32(total+s)
  mean=f32(total/40);frac=Fraction(mean)
  rows.append({'tensor':t['name'],'row':row,'mean':mean,'mean_bits':bits(mean),'mean_exact':[frac.numerator,frac.denominator],'half_scales':hs,'one_plane_exact':frac in coefs,'odd_significand':odd_core(frac.numerator),'odd_bound45_failure':odd_core(frac.numerator)>45})
f.close()
# Bind the exact real input, not merely its recorded campaign filename.
h=hashlib.sha256();started=time.time()
with model.open('rb') as mf:
 for b in iter(lambda:mf.read(8<<20),b''):h.update(b)
modelsha=h.hexdigest();assert modelsha=='53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'
record={'gpu_work':False,'dll_loaded':False,'symbol':fn,'image_sha256':sha(image),'cuobjdump_sha256':sha(tool),'nvdisasm_sha256':sha(nv),'command':cmd,'exit_code':r.returncode,'selected_resources':selected,'model':str(model),'model_sha256':modelsha,'model_bytes':size,'model_hash_seconds':time.time()-started,'tensors':selected_t,'tensor_data_base':data,'rows':rows,'one_plane_positive_values':len(coefs),'odd_bound45_failures':sum(x['odd_bound45_failure'] for x in rows),'engine_sha256':sha(dest/'engine.cu'),'fp4_sha256':sha(dest/'fp4.cu')}
(dest/'bindings.json').write_text(json.dumps(record,indent=2))
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(dest.iterdir()):
  if p.is_file():z.write(p,p.name)
b=buf.getvalue();(dest/'cpu-custody.zip').write_bytes(b)
print(json.dumps({'gpu_work':False,'archive_sha256':hashlib.sha256(b).hexdigest(),'archive_bytes':len(b),'archive_base64':base64.b64encode(b).decode()}))
