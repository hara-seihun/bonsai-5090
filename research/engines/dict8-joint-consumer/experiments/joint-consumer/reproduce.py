from pathlib import Path
import concurrent.futures,hashlib,json,os,shutil,subprocess,time
p=Path(__file__).resolve().parent;root=p.parents[1];os.chdir(root)
receipts=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(cmd,log,seconds=55):
 t=time.monotonic();r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=seconds)
 log.write_text(r.stdout)
 item={'command':cmd,'exit':r.returncode,'wall_seconds':time.monotonic()-t,'output':str(log.relative_to(p)),'output_sha256':sha(log)};receipts.append(item)
 if r.returncode:raise RuntimeError(item)
 return r.stdout
run(['python3',str(p/'generate.py')],p/'generation.log')
(p/'build').mkdir(exist_ok=True)
flags=['-std=c++17','-O3','--expt-relaxed-constexpr','-arch=sm_120a','-lineinfo','-Xptxas=-v','-DH5_PROFILE_GEMM','-DH5_PDL_EARLY_GEMM_RELEASE','-DH5_PDL_GDN_STATE_OVERLAP','-DH5_NO_L2PF','-DH5_PROFILE_SCALE_CARRIER','-DH5_WEIGHT_TMA_CACHE_POLICY=2','-Isrc','-I'+str(p)]
names=['native0','baseline','shared8','shared8_compact','shared8_dual','split53_shared','split53_shuffle','literal3']
def build(n):
 run(['nvcc',*flags,'-cubin',str(p/(n+'.cu')),'-o',str(p/'build'/(n+'.cubin'))],p/'build'/(n+'.log'))
 run(['cuobjdump','-sass',str(p/'build'/(n+'.cubin'))],p/'build'/(n+'.sass'))
 run(['cuobjdump','-res-usage',str(p/'build'/(n+'.cubin'))],p/'build'/(n+'.resources'))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(build,names))
fixtures=[str(p/'fixtures/gate.bin'),str(p/'fixtures/up.bin')]
for n,out in [('check','cpu.json'),('check_literal','literal-cpu.json')]:
 run(['g++','-std=c++17','-O3','-Isrc','-I'+str(p),str(p/(n+'.cpp')),'-o',str(p/'build'/n)],p/'build'/(n+'.log'))
 run([str(p/'build'/n),*fixtures],p/out)
run(['g++','-std=c++17','-O3',str(p/'circuit/witness.cpp'),'-o',str(p/'build/circuit_witness')],p/'build/circuit-witness-compile.log')
run([str(p/'build/circuit_witness')],p/'circuit/witness.log')
for n in ['JointCarrier.lean','circuit/Carrier.lean']:run(['lean',str(p/n)],p/(n+'.log'))
# Post-decode arithmetic/control suffix is exactly the same bytes, not an equivalent algebra rewrite.
start='    trace.end(micro::Decode, kb);';end='template<int WF,bool CAPTURE_PAIR>'
base=(p/'baseline.cu').read_text();suffix=base[base.index(start):base.index(end)]
for n in names:
 s=(p/(n+'.cu')).read_text();assert s[s.index(start):s.index(end)]==suffix,n
# The six Boolean immediates in the host proof program and the compiled literal core agree.
host=(p/'circuit/carrier.hpp').read_text();device=(p/'literal.cuh').read_text()
for imm in ['0xd5','0x82','0x97','0xc0','0x3c','0x24']:assert imm in host and imm in device
source={str(x.relative_to(root)):sha(x) for x in (root/'src').rglob('*') if x.is_file()}
source.update({str(x.relative_to(root)):sha(x) for x in p.rglob('*') if x.is_file() and 'build' not in x.relative_to(p).parts and '__pycache__' not in x.parts and x.suffix in ['.cu','.cuh','.hpp','.cpp','.lean','.py','.md','.bin','.csv']})
contract={'gpu_execution':False,'native_model_execution':False,'same_arithmetic_suffix':True,'suffix_sha256':hashlib.sha256(suffix.encode()).hexdigest(),'base_consumer_sha256':sha(root/'src/dict8.cu'),'source_files':source,'limits':['CPU finite decoder and placement plus source suffix equality; no native floating execution or compiler correctness proof','local source headers hashed conservatively; CUDA SDK is external and identified by toolchain receipt','same-build K-loop counts do not equal historical Windows timeline extraction counts']}
(p/'source-contract.json').write_text(json.dumps(contract,indent=2)+'\n')
run(['python3',str(p/'invoice.py')],p/'invoice.log')
versions={tool:subprocess.check_output([tool,'--version'],text=True) for tool in ['nvcc','g++','lean']}
result={'state':'CPU_and_compiler_complete','gpu_execution':False,'native_model_execution':False,'compiler_commands':receipts,'versions':versions,'source_contract_sha256':sha(p/'source-contract.json'),'invoice_sha256':sha(p/'invoice.json')}
(p/'build-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'state':result['state'],'commands':len(receipts),'gpu_execution':False}))
