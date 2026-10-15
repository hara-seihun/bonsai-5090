#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then
  echo 'Usage: mma_overlap_ncu.sh BINARY OUTPUT_PREFIX <complete mma_overlap arguments>' >&2
  exit 1
fi
binary=$1; output=$2; shift 2
chains=; selected=
args=("$@")
for ((i=0; i<${#args[@]}; i+=2)); do
  if (( i+1 >= ${#args[@]} )); then echo 'ARGUMENT_ERROR expected option/value pairs' >&2; exit 1; fi
  case ${args[i]} in
    --chains) chains=${args[i+1]} ;;
    --case) selected=${args[i+1]} ;;
  esac
done
case $chains in 1|8) ;; *) echo 'ARGUMENT_ERROR --chains 1|8 required' >&2; exit 1 ;; esac
style=0
case $selected in
  tensor) work=0 ;;
  fma) work=1 ;;
  int) work=2 ;;
  fma_int) work=3 ;;
  phases_*|interleave_*|warps_*|ctas_*)
    case ${selected%%_*} in phases) style=1 ;; interleave) style=2 ;; warps) style=3 ;; ctas) style=4 ;; esac
    case ${selected#*_} in fma) work=1 ;; int) work=2 ;; fma_int) work=3 ;; *) echo 'ARGUMENT_ERROR unknown workload' >&2; exit 1 ;; esac ;;
  *) echo 'ARGUMENT_ERROR select a single tensor/fma/int/fma_int/phases/interleave/warps/ctas case; profile streams as isolated kernels, not replayed concurrency' >&2; exit 1 ;;
esac
if [[ -n ${NCU:-} ]]; then
  ncu=$NCU
elif command -v ncu >/dev/null 2>&1; then
  ncu=$(command -v ncu)
else
  echo 'TOOL_ERROR ncu is absent; set NCU to the installed Nsight Compute executable' >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"
# The first matching launch is the warm-up: identical arithmetic, no timed CUDA events.
regex="^_Z14overlap_kernelILi${chains}EL5Style${style}EL4Work${work}E"
"$ncu" --target-processes all --kernel-name-base mangled --kernel-name "regex:$regex" \
  --launch-count 1 --cache-control none --clock-control none \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats --section InstructionStats \
  --section ComputeWorkloadAnalysis --force-overwrite --export "$output" \
  "$binary" "$@" > "$output.log" 2>&1
"$ncu" --import "$output.ncu-rep" --csv --page raw > "$output-raw.csv"
"$ncu" --import "$output.ncu-rep" --csv --page details > "$output-details.csv"
echo "$output.ncu-rep"
