#!/usr/bin/env python3
"""Run the native decode benchmark with a measured profile and write a receipt.

  python scripts/bench.py --model models/Ternary-Bonsai-2-27B-PTQ1_0.gguf               # serving profile
  python scripts/bench.py --model ... --profile deterministic                             # H5_DETERMINISTIC_GEMM=1
  python scripts/bench.py --model ... --profile deterministic --check-determinism          # two runs, compare tokens
  python scripts/bench.py --model ... --dry-run                                           # print command/env only

Timing (inside quality_length_bench): every pass prefills all 64 prompts, runs 8 warm-up decode
steps, then times exactly STREAMS*STEPS generated tokens (64*128 = 8192). Model load, weight
packing, plan tuning, tokenization and prefill are outside the timer. The reported figure is the
median of the per-pass generation_tps values.
"""
import argparse, hashlib, json, os, platform, statistics, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'
PROMPTS = ROOT / 'data' / 'video-gsm-prompts256.bin'
PROMPTS_SHA256 = 'd105a31cac29fd42e741b88d0e35bdceeebe1d77a1142a073a8a8c159259670e'
PLAN = ROOT / 'engine' / 'frozen-production-plan.txt'
PLAN_SHA256 = '713c060b627ef9955d68edfef17e12d75f07093c68b74c3fd8ecf4d19b3507bd'
EXE = 'quality_length_bench' + ('.exe' if os.name == 'nt' else '')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 23), b''):
            h.update(block)
    return h.hexdigest()


def device():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,power.limit,memory.total,clocks.max.sm',
                                        '--format=csv,noheader'], text=True, timeout=30).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f'unavailable: {e}'


def run_once(args, env, answers):
    exe = Path(args.build) / 'bin' / EXE
    cmd = [str(exe), str(args.model), str(PROMPTS), str(args.streams), str(args.steps)]
    run_env = {k: v for k, v in os.environ.items() if not k.startswith(('H5_', 'P12_', 'QC_', 'EXPLORER_', 'SWEEP_', 'GGML_'))}
    run_env.update(env)
    if answers:
        run_env['P12_ANSWERS'] = str(answers)
    bin_dir = str(exe.parent)
    search = [bin_dir]
    if os.name == 'nt' and os.environ.get('CUDA_PATH'):  # cudart/cublas DLLs
        cuda = Path(os.environ['CUDA_PATH'])
        search += [str(p) for p in (cuda / 'bin' / 'x64', cuda / 'bin') if p.is_dir()]
    run_env['PATH'] = os.pathsep.join(search + [run_env.get('PATH', '')])
    if args.dry_run:
        print(json.dumps({'command': cmd, 'environment': env}, indent=2))
        return None
    started = time.time()
    proc = subprocess.run(cmd, env=run_env, cwd=bin_dir, capture_output=True, text=True, timeout=args.timeout)
    rows = []
    for line in proc.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    samples = [r for r in rows if 'generation_tps' in r]
    if proc.returncode != 0 or len(samples) != args.passes:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f'benchmark failed: exit {proc.returncode}, {len(samples)}/{args.passes} passes')
    return dict(command=cmd, elapsed_s=time.time() - started, samples=samples,
                median_tps=statistics.median(r['generation_tps'] for r in samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--profile', default='serving', help='profiles/<name>.json (serving, deterministic)')
    ap.add_argument('--build', type=Path, default=ROOT / 'build')
    ap.add_argument('--streams', type=int, default=64)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--out', type=Path, default=ROOT / 'results')
    ap.add_argument('--check-determinism', action='store_true', help='run twice and require identical generated tokens')
    ap.add_argument('--skip-model-hash', action='store_true', help='skip hashing the 5.9 GB model (hash is still required to match if computed)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    profile = json.loads((ROOT / 'profiles' / f'{args.profile}.json').read_text())
    if sha256(PROMPTS) != PROMPTS_SHA256 or sha256(PLAN) != PLAN_SHA256:
        raise SystemExit('fixture or frozen plan differs from the measured bytes')
    model_hash = None
    if not args.dry_run and not args.skip_model_hash:
        model_hash = sha256(args.model)
        if model_hash != MODEL_SHA256:
            raise SystemExit(f'model sha256 {model_hash} != {MODEL_SHA256}')

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    work = Path(tempfile.mkdtemp(prefix=f'bench-{args.profile}-', dir=args.out))
    env = dict(profile['environment'])
    env.update(H5_PLAN_READ=str(PLAN), H5_PLAN_WRITE=str(work / 'plan-written.txt'), EXPLORER_PASSES=str(args.passes))
    if args.streams != 64:
        env['H5_GRAPH_MAX_ROWS'] = str(args.streams)
        env['P12_FP4_SLOTS'] = str(args.streams)

    runs = []
    for i in range(2 if args.check_determinism else 1):
        runs.append(run_once(args, env, work / f'answers-{i}.jsonl' if args.check_determinism else None))
    if args.dry_run:
        return

    receipt = dict(profile=args.profile, streams=args.streams, steps=args.steps, passes=args.passes,
                   environment=env, model_sha256=model_hash, prompts_sha256=PROMPTS_SHA256, plan_sha256=PLAN_SHA256,
                   plan_sha256_after=sha256(PLAN), device=device(), host=platform.platform(),
                   binaries={p.name: sha256(p) for p in sorted((args.build / 'bin').iterdir())
                             if p.suffix in ('.exe', '.dll', '.so') or p.name == EXE},
                   runs=runs)
    if args.check_determinism:
        a, b = [(work / f'answers-{i}.jsonl').read_text() for i in range(2)]
        receipt['determinism'] = dict(identical_tokens=a == b,
                                      answers_sha256=[hashlib.sha256(x.encode()).hexdigest() for x in (a, b)])
    path = args.out / f'{args.profile}-{stamp}.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    for i, r in enumerate(runs):
        tps = ', '.join(f"{s['generation_tps']:.1f}" for s in r['samples'])
        print(f"run {i}: median {r['median_tps']:.1f} tok/s  (passes: {tps})")
    if args.check_determinism:
        print('generated tokens identical across runs:', receipt['determinism']['identical_tokens'])
    print('receipt:', path)
    if args.check_determinism and not receipt['determinism']['identical_tokens']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
