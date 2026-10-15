#!/usr/bin/env python3
"""Explicit binary-trained six-layer DSpark BF16 serving points."""
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--engine-root', required=True)
p.add_argument('--result-root', required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
for name, target in (
    ('packed1', {'H5_WEIGHTS':'packed1','H5_ACT':'f32','H5_CHUNK':'8'}),
    ('hybrid', {'H5_WEIGHTS':'packed1','H5_ACT':'f32-r1-a8','H5_CHUNK':'16'}),
    ('hybrid-fused', {'H5_WEIGHTS':'packed1','H5_ACT':'f32-r1-a8','H5_CHUNK':'16','H5_A8_GU':'fused'}),
    ('a8-fused', {'H5_WEIGHTS':'packed1','H5_ACT':'a8','H5_CHUNK':'16','H5_A8_GU':'fused'}),
    ('joint', {'H5_WEIGHTS':'fp4','H5_ACT':'fp4','H5_COMPACT':'binary1',
               'H5_FP4_PASSES':'2','H5_FP4_PAIR':'joint-r16','H5_CHUNK':'16'}),
):
    for slots in (1, 8, 16, 32):
        point = f'{name}-dspark4-b{slots}'
        # 4608 still accommodates every scoped prompt plus4096 output tokens;
        # the smaller allocation is an explicit operating point, not implicit OOM fallback.
        ctx = 4608 if slots == 32 else 8192
        command = [a.engine_root+'/h5-server','/workspace/binary/models/Bonsai-27B-Q1_0.gguf',
                   '/workspace/binary/models/Bonsai-27B-dspark-bf16.gguf','--host','127.0.0.1',
                   '--port','8088','--slots',str(slots),'--ctx',str(ctx),'--prefill','16',
                   '--draft-proposals','4']
        config = {'command':command,
                  'env':{'H5_DRAFTER':'dspark-bf16','H5_STATE':'f16','H5_KV':'f16',
                         'H5_DRAFT_MARGIN':'0','H5_DETERMINISTIC_GEMM':'1',
                         'H5_ORDERED_PREP':'off',**target},
                  'url':'http://127.0.0.1:8088','out':a.result_root+'/tasks-'+point,
                  'gate_args':['--quick','--limit','32','--concurrency',str(slots)],
                  'startup_seconds':180,'run_seconds':1800,'metrics_path':'/metrics'}
        (a.out/(point+'.json')).write_text(json.dumps(config, indent=2)+'\n')
