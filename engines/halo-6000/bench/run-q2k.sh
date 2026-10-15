#!/bin/bash
set -euo pipefail
H=/workspace/halo-6000
F=/workspace/src/prism-llama
nvidia-smi > "$H/bench/q2k-context.txt"
ps -eo pid,stat,args > "$H/bench/q2k-processes.txt"
$H/bench/q2k "$H/bench/q2k-real.bin" timing > "$H/bench/q2k-timing.jsonl"
