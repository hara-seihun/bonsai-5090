#!/usr/bin/env python3
"""Replay exact custody-pinned saved producer tapes; never regenerate the producer."""
import argparse, hashlib, io, json, pathlib, subprocess, tarfile, tempfile, time
ROOT = pathlib.Path(__file__).resolve().parents[5]
SOURCE = pathlib.Path(__file__).resolve().parent
RESULT = ROOT / 'results/packing-swarm/round23/cpu'
ARCHIVE = ROOT / 'archive/packing-swarm/round23/cpu'
EVIDENCE = pathlib.Path('~/work/packing-mma-round22-evidence')
REMOTE = pathlib.Path('~/data/research-remotes/program-12-packed-compute.git')
PREREQUISITE = '9c4ed70089114706200215bf60910dbed2c35c95'
SEAL = RESULT / 'inherited20-receipt.json'
ARMS = ('candidate','mergedFW8','current19')
SCRATCH = pathlib.Path('~/work/packing-round23-cpu-sidecar/scratch')

def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')
def run(name,command,payload=None,seconds=45):
    start=time.monotonic()
    try:
        r=subprocess.run(command,input=payload,capture_output=True,timeout=seconds)
        log,code=r.stdout+r.stderr,r.returncode
        state='succeeded' if code==0 and b'runtime error:' not in log else 'failed'
    except (OSError,subprocess.TimeoutExpired) as error:
        log,code,state=str(error).encode(),None,'boundary_error'
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    (ARCHIVE/(name+'.log')).write_bytes(log)
    record=dict(command=command,elapsed_seconds=time.monotonic()-start,timeout_seconds=seconds,exit_code=code,state=state)
    save(ARCHIVE/(name+'.json'),record)
    assert state=='succeeded',(name,record,log[-2000:])
    return log

def git_bytes(path):
    r=subprocess.run(['git','--git-dir='+str(REMOTE),'show',PREREQUISITE+':'+str(path)],capture_output=True,timeout=10)
    assert r.returncode==0,r.stderr
    return r.stdout

def build():
    assert not ARCHIVE.exists(),'fresh evidence required'
    previous=EVIDENCE/'snapshot/results/packing-swarm/round22/cpu/receipt.json'
    receipt=json.loads(previous.read_text())
    sealbytes=git_bytes('results/packing-swarm/round20/cpu/receipt.json')
    assert hashlib.sha256(sealbytes).hexdigest()==receipt['build']['round20_receipt_sha256']
    RESULT.mkdir(parents=True,exist_ok=True);SEAL.write_bytes(sealbytes)
    files=[SOURCE/'check.cpp',SOURCE/'scoped.hpp',SOURCE/'replay.py',
           ROOT/'method/packing-swarm/tools/round17/native/packed.hpp',
           ROOT/'method/packing-swarm/tools/round17/native/model.hpp',
           ROOT/'method/packing-swarm/tools/round18/grammar.hpp']
    identities={str(p.relative_to(ROOT)):sha(p) for p in files}
    for p in files:
        target=ARCHIVE/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    immutable=ARCHIVE/'source/method/packing-swarm/tools/round23/cpu/check.cpp'
    compiler=run('compiler',['g++','--version'],seconds=2).decode().splitlines()[0]
    flags=['-std=c++20','-O3','-march=native','-ffp-contract=off','-fopenmp','-Wall','-Wextra']
    for kind,san in [('ordinary',[]),('ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all'])]:
        run(kind+'-build',['g++',*flags,*san,str(immutable),'-o',str(ARCHIVE/kind)],seconds=20)
    save(RESULT/'build.json',dict(state='shortened_CPU_binaries_built',prerequisite=PREREQUISITE,
         compiler=compiler,flags=flags,source_sha256=identities,binaries={k:sha(ARCHIVE/k) for k in ('ordinary','ubsan')},
         round20_receipt_sha256=sha(SEAL),round22_receipt_sha256=sha(previous),custody_sha256=sha(EVIDENCE/'artifact-custody.json')))

def evidence(name,seal):
    base=pathlib.Path('archive/packing-swarm/round20/cpu')
    archive_path=base/(name+'.raw.tar.zst');manifest_path=base/(name+'.raw.manifest.json')
    packed=git_bytes(archive_path);manifestbytes=git_bytes(manifest_path)
    custody=json.loads((EVIDENCE/'artifact-custody.json').read_text())
    pin=next(v for v in custody['inherited_CPU_tapes'].values() if v['private_tree_path']==str(archive_path))
    assert pin['private_commit']==PREREQUISITE
    assert hashlib.sha256(packed).hexdigest()==pin['SHA256']==seal['artifact_sha256'][str(archive_path)]
    assert hashlib.sha256(manifestbytes).hexdigest()==seal['artifact_sha256'][str(manifest_path)]
    manifest=json.loads(manifestbytes)
    SCRATCH.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='round23-one-case-',dir=SCRATCH) as tmp:
        path=pathlib.Path(tmp)/'saved.raw.tar.zst';path.write_bytes(packed)
        r=subprocess.run(['zstd','-dc',str(path)],capture_output=True,timeout=10)
        assert r.returncode==0,r.stderr
    data={}
    with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:') as tape:
        for member in tape:
            assert member.isfile() and member.name in manifest['members'] and member.name not in data
            expected=manifest['members'][member.name];content=tape.extractfile(member).read()
            assert len(content)==member.size==expected['bytes']
            assert hashlib.sha256(content).hexdigest()==expected['sha256']
            data[member.name.removeprefix(name+'.')]=content
    assert {name+'.'+k for k in data}==set(manifest['members'])
    return data,manifest,archive_path,hashlib.sha256(packed).hexdigest(),hashlib.sha256(manifestbytes).hexdigest()

def case(family,B,kind):
    built=json.loads((RESULT/'build.json').read_text());assert sha(ARCHIVE/kind)==built['binaries'][kind]
    seal=json.loads(SEAL.read_text());assert sha(SEAL)==built['round20_receipt_sha256']
    name=f'{family}-B{B}-final';origin=seal['ordinary_cases'][name]
    data,manifest,path,archivehash,manifesthash=evidence(name,seal)
    for arm in ARMS:
        assert hashlib.sha256(data[arm+'.u32']).hexdigest()==origin['whole_output_sha256'][arm]
        assert hashlib.sha256(data[arm+'.trace.u64']).hexdigest()==origin['whole_trace_sha256'][arm]
    assert len(data['owners.u32'])==B*160*4 and not any(data['owners.u32'])
    assert hashlib.sha256(data['source.f32']).hexdigest()==origin['whole_source_sha256']
    assert hashlib.sha256(data['norm.f32']).hexdigest()==origin['whole_norm_sha256']
    payload=b''.join(data[p+'.'+s] for p in ('primary','innovation') for s in ('q','s','g.f32'))
    payload+=b''.join(data[a+'.u32'] for a in ARMS)+b''.join(data[a+'.trace.u64'] for a in ARMS)
    observed=json.loads(run(name+'-'+kind,[str(ARCHIVE/kind),family,str(B)],payload))
    assert observed['state']=='shortened_q_saved_tape_CPU_pass'
    for ours,inherited in [('source_weight_fnv1a64','merged_weight_fnv1a64'),('native_sign_fnv1a64','merged_sign_fnv1a64'),('native_scale_fnv1a64','merged_scale_fnv1a64')]:
        assert observed[ours].zfill(16)==origin[inherited]
    for ours,inherited in [('primary_q_fnv1a64','primary_q'),('innovation_q_fnv1a64','innovation_q'),('primary_s_fnv1a64','primary_s'),('innovation_s_fnv1a64','innovation_s')]:
        assert observed[ours].zfill(16)==origin['tape_fnv1a64'][inherited]
    predecessor=json.loads((EVIDENCE/'snapshot/results/packing-swarm/round22/cpu'/(name+'-'+kind+'.json')).read_text())
    for arm in ('candidate','equally_paired_FW8'):
        assert observed[arm]==predecessor['observed'][arm]
        inherited='inherited22_candidate' if arm=='candidate' else 'inherited22_FW8'
        assert observed[inherited]==predecessor['observed'][arm]
        assert observed[arm]['exact_K128_dot_scale_fold_checks']==2*B*origin['F']*40
        assert observed[arm]['complete_pass_output_words']==2*B*origin['F']
    save(RESULT/(name+'-'+kind+'.json'),dict(observed=observed,trajectory=origin['trajectory'],
         inherited_archive=str(path),private_remote=str(REMOTE),private_commit=PREREQUISITE,
         inherited_archive_sha256=archivehash,inherited_manifest_sha256=manifesthash,inherited_members=manifest['members'],
         whole_source_sha256=origin['whole_source_sha256'],whole_norm_sha256=origin['whole_norm_sha256'],
         whole_output_sha256=origin['whole_output_sha256'],whole_trace_sha256=origin['whole_trace_sha256']))
    print(json.dumps(dict(state=observed['state'],family=family,B=B,kind=kind,
        seconds=json.loads((ARCHIVE/(name+'-'+kind+'.json')).read_text())['elapsed_seconds'])))

def seal(native):
    built=json.loads((RESULT/'build.json').read_text());cases={}
    for family in ('input0','input3','ffn0'):
        for B in (1,8,32):
            name=f'{family}-B{B}-final';cases[name]={k:json.loads((RESULT/(name+'-'+k+'.json')).read_text()) for k in ('ordinary','ubsan')}
            assert cases[name]['ordinary']==cases[name]['ubsan']
    for p,h in built['source_sha256'].items(): assert sha(ROOT/p)==h,p
    assert native.is_file(),'explicit native source required'
    totals={a:{metric:sum(c['ordinary']['observed'][a][metric] for c in cases.values()) for metric in next(iter(cases.values()))['ordinary']['observed'][a]} for a in ('candidate','equally_paired_FW8')}
    record=dict(state='all_full_shortened_q_saved_tape_witnesses_pass',prerequisite=PREREQUISITE,build=built,cases=cases,totals=totals,
        native_paired_source=str(native),native_paired_source_sha256=sha(native),
        scope='GDN16480, attention14336, FFN34816; B1/8/32; complete saved primary/final words, ordered40 trace digests, one exact fold owner each K128, separate FW4/WG2 and independently enumerated FW8/WG1 output owners; eight-stage shared byte dictionary; q-local E2M1 fragments and scales consumed pass,n,mm before next q; raw/native K64 halves agree individually and combined fold words agree. Original source/norm/all raw archive members are hash-checked and preserved; no producer regeneration.',
        exact_boundary='CPU structural dictionary and raw/native dyadic-dot/fold witness, not executed CUDA MMA. All40 saved digests bind inherited observations rather than recover old intermediate words. Q-local fragment storage is CPU witness storage, not measured CUDA register allocation.',
        CUDA_primitives_executed=False,numerical_producer_regenerated=False,
        unset=['native_MMA_refinement','native_words_fault_time_occupancy','actual_layer3_predecessor','actual_FFN_post_attention_source','whole_controller_B32','engine_model_task_HTTP'],packing_bounds='unchanged')
    record['artifact_sha256']={str(p.relative_to(ROOT)):sha(p) for p in ARCHIVE.rglob('*') if p.is_file()}
    save(RESULT/'receipt.json',record);print(json.dumps(dict(state=record['state'],totals=totals)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true');parser.add_argument('--case',choices=['input0','input3','ffn0']);parser.add_argument('--batch',type=int,choices=[1,8,32]);parser.add_argument('--kind',choices=['ordinary','ubsan']);parser.add_argument('--seal',action='store_true');parser.add_argument('--native',type=pathlib.Path)
    args=parser.parse_args();assert sum((args.build,args.case is not None,args.seal))==1
    if args.build: build()
    elif args.case:
        assert args.batch is not None and args.kind is not None;case(args.case,args.batch,args.kind)
    else:
        assert args.native is not None;seal(args.native)
