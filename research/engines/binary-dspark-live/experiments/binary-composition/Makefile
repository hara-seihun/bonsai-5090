all: build/bench build/search

build:
	mkdir -p build

build/bench: composition.cu composition.cuh bench.cu ../binary-bitserial/bitserial.cu ../binary-bitserial/bitserial.cuh ../../src/gguf.cpp ../../src/gguf.h | build
	nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas=-v -I ../../src composition.cu ../binary-bitserial/bitserial.cu bench.cu ../../src/gguf.cpp -o $@

build/search: search.cpp | build
	c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic $< -o $@

search: build/search
	./build/search

.PHONY: all search
