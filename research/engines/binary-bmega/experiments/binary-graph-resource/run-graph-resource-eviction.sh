#!/usr/bin/env bash
set -euo pipefail
cd /workspace/binary/graph-resource-keyhole
objects=(dspark_native dspark_kernels packed1 bitserial digits native_sf composition fp4_pair graph_memory kernels fp4 sk draft sampling_gpu f02-producer f02-continuation gguf)
args=(engine.o); for name in "${objects[@]}"; do args+=("/workspace/binary/native-sf-engine/halo/build/$name.o"); done
sha256sum /workspace/binary/batch-profile/profile_replay.o "${args[@]}" graph-resource-eviction.events > eviction-profile-inputs.sha256
/usr/local/cuda/bin/nvcc -gencode arch=compute_120a,code=sm_120a -o profile-replay /workspace/binary/batch-profile/profile_replay.o "${args[@]}" -lcublas -L/workspace/binary/llama/build/bin -lllama -Xlinker -rpath,/workspace/binary/llama/build/bin -Xcompiler -pthread
export H5_WEIGHTS=fp4 H5_ACT=fp4 H5_COMPACT=binary1 H5_FP4_PASSES=2 H5_FP4_PAIR=joint-r16 H5_NATIVE_SF=sf1-r32
export H5_STATE=f16 H5_KV=f16 H5_PDL=1 H5_TRACE=1 H5_A8_GU=off
unset H5_A8_PROJECT H5_DRAFTER H5_PROF H5_NOGRAPH H5_TREE2 H5_DRAFT_NEST H5_SYNC_DEBUG H5_WSCALE
sha256sum profile-replay >> eviction-profile-inputs.sha256
for maxrows in 64; do
 export H5_GRAPH_MAX_ROWS=$maxrows
 ./profile-replay /workspace/binary/models/Bonsai-27B-Q1_0.gguf graph-resource-eviction.events eviction-resource-$maxrows --slots 32 --ctx 8192 --ordinary --head-rows demand --decode-only --graph-upload launch --graph-timing wall-thread --passes 1 > eviction-resource-$maxrows.json 2>eviction-resource-$maxrows.log
 done
python3 - <<'PY'
import csv,json,re
r=list(csv.reader(open('eviction-resource-64.csv')))
log=open('eviction-resource-64.log').read()
counts=[int(x) for x in re.findall(r'GraphResource .*?cache=(\d+)',log)]
evicted=log.count('stage=evicted')
print(json.dumps({'completed_steps':len(r),'admitted':log.count('stage=admitted'),'evicted':evicted,'peak_cache_entries':max(counts),'scope':'synthetic SF1 B32ctx8192 63distinctshapekeys,cacheownershiponly; numericalqualitygateparent'}))
assert evicted>0 and max(counts)<=32, 'LRU eviction contract not exercised'
PY
