#!/usr/bin/env bash
set -euo pipefail
cd /workspace/binary/graph-resource-keyhole
objects=(dspark_native dspark_kernels packed1 bitserial digits native_sf composition fp4_pair graph_memory kernels fp4 sk draft sampling_gpu f02-producer f02-continuation gguf)
args=(engine.o); for name in "${objects[@]}"; do args+=("/workspace/binary/native-sf-engine/halo/build/$name.o"); done
sha256sum /workspace/binary/batch-profile/profile_replay.o "${args[@]}" graph-resource-sf32.events > sf32-profile-inputs.sha256
/usr/local/cuda/bin/nvcc -gencode arch=compute_120a,code=sm_120a -o profile-replay /workspace/binary/batch-profile/profile_replay.o "${args[@]}" -lcublas -L/workspace/binary/llama/build/bin -lllama -Xlinker -rpath,/workspace/binary/llama/build/bin -Xcompiler -pthread
export H5_WEIGHTS=fp4 H5_ACT=fp4 H5_COMPACT=binary1 H5_FP4_PASSES=2 H5_FP4_PAIR=joint-r16 H5_NATIVE_SF=sf1-r32
export H5_STATE=f16 H5_KV=f16 H5_PDL=1 H5_TRACE=1 H5_A8_GU=off
unset H5_A8_PROJECT H5_DRAFTER H5_PROF H5_NOGRAPH H5_TREE2 H5_DRAFT_NEST H5_SYNC_DEBUG H5_WSCALE
sha256sum profile-replay >> sf32-profile-inputs.sha256
for maxrows in 1 64; do
 export H5_GRAPH_MAX_ROWS=$maxrows
 ./profile-replay /workspace/binary/models/Bonsai-27B-Q1_0.gguf graph-resource-sf32.events sf32-resource-$maxrows --slots 32 --ctx 8192 --ordinary --head-rows all --decode-only --graph-upload launch --graph-timing wall-thread --passes 1 > sf32-resource-$maxrows.json 2>sf32-resource-$maxrows.log
 done
python3 - <<'PY'
import csv,json,collections
rows=[]
for maxrows in [1,64]:
 r=list(csv.reader(open(f'sf32-resource-{maxrows}.csv')))
 rows.append([x[:2]+x[3:] for x in r])
 notices=[x.rstrip() for x in open(f'sf32-resource-{maxrows}.log') if x.startswith('GraphResource')]
 print(json.dumps({'graph_max_rows':maxrows,'completed_steps':len(r),'resource_events':len(notices)}))
print(json.dumps({'output_ids_equal':rows[0]==rows[1],'scope':'synthetic32slots ctx8192 token198,rows32..1,32recurringdecode shapes,SF1twoFP4,noHTTP/noDSpark'}))
assert rows[0]==rows[1], 'graph/eager output mismatch'
PY
