#!/usr/bin/env python3
"""Bounded saved-tape witnesses; inherited CPU producer never re-executed."""
import argparse, hashlib, io, json, pathlib, subprocess, tarfile, time
ROOT = pathlib.Path(__file__).resolve().parents[5]
SOURCE = pathlib.Path(__file__).resolve().parent
RESULT = ROOT / 'results/packing-swarm/round22/cpu'
ARCHIVE = ROOT / 'archive/packing-swarm/round22/cpu/run-final'
INHERITED = pathlib.Path('~/work/clones/packing-mma-round20')
SEAL = ROOT / 'results/packing-swarm/round20/cpu/receipt.json'
PREREQUISITE = '9c4ed70089114706200215bf60910dbed2c35c95'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def run(name, command, payload=None, seconds=45):
    start = time.monotonic()
    try:
        result = subprocess.run(command, input=payload, capture_output=True, timeout=seconds)
        log = result.stdout + result.stderr
        code = result.returncode
        state = 'succeeded' if code == 0 and b'runtime error:' not in log else 'failed'
    except (OSError, subprocess.TimeoutExpired) as error:
        log, code, state = str(error).encode(), None, 'boundary_error'
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / (name + '.log')).write_bytes(log)
    record = dict(command=command, elapsed_seconds=time.monotonic()-start,
                  timeout_seconds=seconds, exit_code=code, state=state)
    save(ARCHIVE / (name + '.json'), record)
    assert state == 'succeeded', (name, record, log[-2000:])
    return log

def build():
    assert not ARCHIVE.exists(), 'fresh evidence required'
    source_files = [SOURCE / 'check.cpp', SOURCE / 'replay.py',
                    ROOT / 'method/packing-swarm/tools/round17/native/packed.hpp',
                    ROOT / 'method/packing-swarm/tools/round17/native/model.hpp',
                    ROOT / 'method/packing-swarm/tools/round18/grammar.hpp']
    identities = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    for p in source_files:
        q = ARCHIVE / 'source' / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(p.read_bytes())
    immutable = ARCHIVE / 'source/method/packing-swarm/tools/round22/cpu/check.cpp'
    compiler = run('compiler', ['g++', '--version'], seconds=2).decode().splitlines()[0]
    flags = ['-std=c++20', '-O3', '-march=native', '-ffp-contract=off', '-fopenmp', '-Wall', '-Wextra']
    for kind, sanitizer in [('ordinary', []), ('ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all'])]:
        run(kind+'-build', ['g++', *flags, *sanitizer, str(immutable), '-o', str(ARCHIVE / kind)], seconds=20)
    save(RESULT / 'build.json', dict(state='CPU_binaries_built', prerequisite=PREREQUISITE,
         compiler=compiler, flags=flags, source_sha256=identities,
         binaries={kind: sha(ARCHIVE/kind) for kind in ('ordinary', 'ubsan')},
         round20_receipt_sha256=sha(SEAL)))

def evidence(name, seal):
    relative = pathlib.Path('archive/packing-swarm/round20/cpu')
    manifest_path = INHERITED / relative / (name+'.raw.manifest.json')
    packed = INHERITED / relative / (name+'.raw.tar.zst')
    for p in (manifest_path, packed):
        assert sha(p) == seal['artifact_sha256'][str(relative / p.name)], p
    manifest = json.loads(manifest_path.read_text())
    raw = subprocess.run(['zstd', '-dc', str(packed)], capture_output=True, timeout=10)
    assert raw.returncode == 0, raw.stderr
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw.stdout), mode='r:') as archive:
        for member in archive:
            assert member.isfile() and member.name in manifest['members'] and member.name not in data
            expected = manifest['members'][member.name]
            content = archive.extractfile(member).read()
            assert len(content) == member.size == expected['bytes']
            assert hashlib.sha256(content).hexdigest() == expected['sha256']
            data[member.name.removeprefix(name+'.')] = content
    assert {name+'.'+key for key in data} == set(manifest['members'])
    return data, manifest, manifest_path, packed

def case(family, B, kind):
    build_record = json.loads((RESULT/'build.json').read_text())
    assert sha(ARCHIVE/kind) == build_record['binaries'][kind]
    seal = json.loads(SEAL.read_text())
    name = f'{family}-B{B}-final'
    origin = seal['ordinary_cases'][name]
    data, manifest, manifest_path, packed = evidence(name, seal)
    for arm in ('candidate', 'mergedFW8', 'current19'):
        assert hashlib.sha256(data[arm+'.u32']).hexdigest() == origin['whole_output_sha256'][arm]
        assert hashlib.sha256(data[arm+'.trace.u64']).hexdigest() == origin['whole_trace_sha256'][arm]
    assert len(data['owners.u32']) == B*160*4 and not any(data['owners.u32'])
    payload = b''.join(data[passname+'.'+suffix] for passname in ('primary', 'innovation') for suffix in ('q','s','g.f32'))
    payload += b''.join(data[arm+'.u32'] for arm in ('candidate','mergedFW8','current19'))
    payload += b''.join(data[arm+'.trace.u64'] for arm in ('candidate','mergedFW8','current19'))
    observed = json.loads(run(name+'-'+kind, [str(ARCHIVE/kind), family, str(B)], payload))
    assert observed['state'] == 'paired_saved_tape_CPU_pass'
    for ours, inherited in [('source_weight_fnv1a64','merged_weight_fnv1a64'), ('native_sign_fnv1a64','merged_sign_fnv1a64'), ('native_scale_fnv1a64','merged_scale_fnv1a64')]:
        assert observed[ours].zfill(16) == origin[inherited]
    for ours, inherited in [('primary_q_fnv1a64','primary_q'),('innovation_q_fnv1a64','innovation_q'),('primary_s_fnv1a64','primary_s'),('innovation_s_fnv1a64','innovation_s')]:
        assert observed[ours].zfill(16) == origin['tape_fnv1a64'][inherited]
    for arm in ('candidate', 'equally_paired_FW8'):
        result = observed[arm]
        assert result['exact_K128_dot_scale_fold_checks'] == 2*B*origin['F']*40
        assert result['complete_pass_output_words'] == 2*B*origin['F']
    save(RESULT / (name+'-'+kind+'.json'), dict(observed=observed,
         trajectory=origin['trajectory'], inherited_archive=str(packed),
         inherited_archive_sha256=sha(packed), inherited_manifest_sha256=sha(manifest_path),
         inherited_members=manifest['members'], whole_source_sha256=origin['whole_source_sha256'],
         whole_norm_sha256=origin['whole_norm_sha256'], whole_output_sha256=origin['whole_output_sha256'],
         whole_trace_sha256=origin['whole_trace_sha256']))
    print(json.dumps(dict(state=observed['state'], family=family, B=B, kind=kind,
        seconds=json.loads((ARCHIVE/(name+'-'+kind+'.json')).read_text())['elapsed_seconds'])))

def seal():
    build_record = json.loads((RESULT/'build.json').read_text())
    cases = {}
    for family in ('input0','input3','ffn0'):
        for B in (1,8,32):
            name=f'{family}-B{B}-final'
            cases[name]={kind:json.loads((RESULT/(name+'-'+kind+'.json')).read_text()) for kind in ('ordinary','ubsan')}
            assert cases[name]['ordinary'] == cases[name]['ubsan']
    for p,h in build_record['source_sha256'].items(): assert sha(ROOT/p)==h,p
    native=ROOT/'method/packing-swarm/tools/round22/native/paired.cuh'
    assert native.is_file(), 'native source identity required before seal'
    result = dict(state='all_full_paired_saved_tape_witnesses_pass', prerequisite=PREREQUISITE,
        build=build_record, cases=cases, native_paired_source_sha256=sha(native),
        native_paired_source=str(native.relative_to(ROOT)),
        totals={arm:{metric:sum(case['ordinary']['observed'][arm][metric] for case in cases.values())
                    for metric in next(iter(cases.values()))['ordinary']['observed'][arm]}
                for arm in ('candidate','equally_paired_FW8')},
        scope='Complete inherited sealed20 input0 GDN16480/input3 attention14336/ffn0 FFN34816 tapes and three independent saved full primary/final snapshots plus per-output all40 ordered traces, B1/8/32. Native WG2 and separately enumerated FW8 output dictionaries are bijective; two independent pass accumulators consume the eight-stage shared byte dictionary, raw-positive byte dot and native-negative nibble dot agree exactly at every K128 block; CPU fold words and both ordered pass outputs agree with inherited words. Padded32-token q/scales are staged once per CTA/pass/K128; GDN tail3 groups covered. Preserved model weights are reloaded, native preparation agrees with inherited weight hashes; producer/norm/quantizer not regenerated.',
        exact_boundary='New exact in-loop raw/native K128 dot/scale/fold comparisons plus complete saved output-word equality. Saved ordered-trace digests bind old observations but do not recover each old intermediate word; not CUDA MMA refinement. Source hash binds inspected native code, not executed CUDA code.',
        CUDA_primitives_executed=False, numerical_producer_regenerated=False,
        unset=['native_MMA_refinement','native_words_fault_time_occupancy','actual_layer3_predecessor','actual_FFN_post_attention_source','whole_controller_B32','engine_model_task_HTTP'],
        packing_bounds='unchanged')
    result['retained_pilots']='archive/packing-swarm/round22/cpu contains successful pre-global-scale/explicit-owner instrumented build and two pilot cases; excluded from final totals. run-final includes the matched global-scale and owner dictionary.'
    result['artifact_sha256']={str(p.relative_to(ROOT)):sha(p) for p in ARCHIVE.parent.rglob('*') if p.is_file()}
    save(RESULT/'receipt.json',result)
    print(json.dumps(dict(state=result['state'],totals=result['totals'])))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--case',choices=['input0','input3','ffn0'])
    parser.add_argument('--batch',type=int,choices=[1,8,32])
    parser.add_argument('--kind',choices=['ordinary','ubsan'])
    parser.add_argument('--seal',action='store_true')
    args=parser.parse_args()
    assert sum((args.build,args.case is not None,args.seal))==1
    if args.build: build()
    elif args.case:
        assert args.batch is not None and args.kind is not None
        case(args.case,args.batch,args.kind)
    else: seal()
