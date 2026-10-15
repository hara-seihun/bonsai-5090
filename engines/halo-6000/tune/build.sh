#!/usr/bin/env bash
# halo-autotune builds on pod B. Usage:
#   tune/build.sh engine            # private engine objects + resident driver (build/tune/engine, build/tune/driver)
#   tune/build.sh variant CFG.hpp OUT.so [schedule|schedule_batch]
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd); cd "$root"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}; FORK=${FORK:-/workspace/src/prism-llama}
inc=(-Iinclude -Isrc)
cuflags=(-O3 -std=c++20 -arch=sm_120a -lineinfo)
case ${1:?mode} in
engine)
 e=build/tune/engine; mkdir -p "$e"
 "$NVCC" "${inc[@]}" "${cuflags[@]}" -Xcompiler -fPIC -c src/runtime.cu -o $e/runtime.o &
 "$NVCC" "${inc[@]}" "${cuflags[@]}" -Xcompiler -fPIC -c src/decode.cu -o $e/decode.o &
 "$NVCC" "${inc[@]}" "${cuflags[@]}" -Xcompiler -fPIC -c src/batch.cu -o $e/batch.o &
 g++ "${inc[@]}" -O3 -std=c++20 -fPIC -fopenmp -c src/gguf.cpp -o $e/gguf.o &
 g++ "${inc[@]}" -O3 -std=c++20 -fPIC -fopenmp -c src/cache.cpp -o $e/cache.o &
 "$NVCC" "${inc[@]}" "${cuflags[@]}" -I"$FORK/vendor" -c tune/driver.cu -o $e/driver.o &
 wait
 "$NVCC" -arch=sm_120a -cudart shared $e/driver.o $e/runtime.o $e/decode.o $e/batch.o $e/gguf.o $e/cache.o -Xcompiler -fopenmp -ldl -o build/tune/driver.tmp
 mv build/tune/driver.tmp build/tune/driver ;;
driver)
 e=build/tune/engine
 "$NVCC" "${inc[@]}" "${cuflags[@]}" -I"$FORK/vendor" -c tune/driver.cu -o $e/driver.o
 "$NVCC" -arch=sm_120a -cudart shared $e/driver.o $e/runtime.o $e/decode.o $e/batch.o $e/gguf.o $e/cache.o -Xcompiler -fopenmp -ldl -o build/tune/driver.tmp
 mv build/tune/driver.tmp build/tune/driver ;;
variant)
 cfg=$(realpath "${2:?config header}"); out=${3:?output .so}; tu=${4:-schedule}; mkdir -p "$(dirname "$out")"
 "$NVCC" "${inc[@]}" "${cuflags[@]}" --ptxas-options=-v -Xcompiler -fPIC,-fvisibility=hidden,-fno-gnu-unique -Xlinker -Bsymbolic -shared -cudart shared \
   -DHALO_TUNE_CONFIG="\"$cfg\"" src/$tu.cu -o "$out.tmp" 2> "${out%.so}.ptxas"
 mv "$out.tmp" "$out" ;;
*) echo "unknown mode $1" >&2; exit 2 ;;
esac
