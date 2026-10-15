#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=EAGER
root=/workspace/halo-6000
out=${1:?output directory}
model=${2:-/workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf}
mkdir -p "$out"
snapshot() { date -u +%FT%TZ; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv; ps -eo pid,ppid,stat,pcpu,pmem,args; }
snapshot > "$out/context-before.txt"
sha256sum "$root/bench/engine-timing" "$root/build/decode.o" "$root/build/runtime.o" > "$out/binary.sha256"
"$root/bench/engine-timing" "$model" /workspace/models/halo6000-cache > "$out/rows.jsonl"
python3 - "$out" <<'PY'
import json,pathlib,sys,subprocess
out=pathlib.Path(sys.argv[1]);bench=[]
for line in (out/'rows.jsonl').read_text().splitlines():
 row=json.loads(line)
 if row['kind']=='profile':
  p=out/f"profile-{row['row_lanes']}.json";p.write_text(json.dumps(row)+'\n')
  subprocess.run(['python3','/workspace/halo-6000/bench/analyze-profile.py',str(p),'--output',str(out/f"profile-{row['row_lanes']}-analysis.json")],check=True)
 else:bench.append(row)
(out/'timing.json').write_text(json.dumps(bench,indent=2)+'\n')
print(json.dumps(bench))
PY
snapshot > "$out/context-after.txt"
