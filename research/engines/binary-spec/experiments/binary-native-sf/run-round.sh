#!/usr/bin/env bash
set -euo pipefail
root=/workspace/binary/native-sf/halo
mkdir -p "$root/experiments/binary-native-sf/results"
flock -w 120 /tmp/binary-gpu.lock timeout 55 bash -c '
  set -euo pipefail
  cd /workspace/binary/native-sf/halo
  export H5_PDL=1 H5_PREP_CHUNK=0 H5_PREP_WIDE=0
  for kind in gu down head; do
    experiments/binary-native-sf/build/bench /workspace/binary/models/Bonsai-27B-Q1_0.gguf "$kind" 32 "experiments/binary-native-sf/results/$kind-r32.jsonl"
  done
'
tar -czf /workspace/binary/native-sf-results.tgz -C "$root" experiments/binary-native-sf/results
