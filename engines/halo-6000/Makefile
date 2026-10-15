.DEFAULT_GOAL := all
CXX ?= g++
NVCC ?= $(shell command -v nvcc 2>/dev/null || printf /usr/local/cuda/bin/nvcc)
CUDA_HOME ?= /usr/local/cuda
FORK ?= /workspace/src/prism-llama
CPPFLAGS := -Iinclude -Isrc
CXXFLAGS := -O3 -std=c++20 -Wall -Wextra
CUDAFLAGS := -O3 -std=c++20 -arch=sm_120a --ptxas-options=-v -lineinfo
TOKEN_FLAGS := -I$(CUDA_HOME)/include -I$(FORK)/include -I$(FORK)/common -I$(FORK)/ggml/include -I$(FORK)/vendor -I$(FORK)/vendor/cpp-httplib
NATIVE_CPP := main debug generate intrinsic tokenizer server gguf cache
NATIVE_OBJECTS := $(addprefix build/,$(addsuffix .o,$(NATIVE_CPP))) build/runtime.o build/decode.o build/httplib.o
SOURCE_KEY := $(shell (find include src -type f -print0 | sort -z | xargs -0 sha256sum; printf '%s\n' '$(CXX) $(NVCC) $(CPPFLAGS) $(TOKEN_FLAGS) $(CXXFLAGS) $(CUDAFLAGS)') | sha256sum | cut -d' ' -f1)
SOURCE_STAMP := build/source-$(SOURCE_KEY)
$(SOURCE_STAMP):
	mkdir -p build
	rm -f build/source-*
	touch $@
$(NATIVE_OBJECTS): $(SOURCE_STAMP)

.PHONY: all clean
all: build/halo-6000 inspect bench/barrier bench/primitives
build/halo-6000: $(NATIVE_OBJECTS)
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fopenmp,-pthread $^ -L$(FORK)/build/bin -Xlinker=-rpath -Xlinker=$(FORK)/build/bin -lllama-common -lllama -lcrypto -o $@.tmp
	mv $@.tmp $@
build/%.o: src/%.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(TOKEN_FLAGS) $(CXXFLAGS) -fopenmp -pthread -MMD -MP -c $< -o $@
build/%.o: src/%.cu
	mkdir -p build
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) -MMD -MP -c $< -o $@
build/httplib.o: $(FORK)/vendor/cpp-httplib/httplib.cpp
	mkdir -p build
	$(CXX) $(TOKEN_FLAGS) -O2 -std=c++20 -pthread -MMD -MP -c $< -o $@
bench/barrier: bench/barrier.cu
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $< -o $@
bench/engine-timing: bench/engine-timing.cu build/runtime.o build/decode.o build/gguf.o build/cache.o
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) -Xcompiler -fopenmp bench/engine-timing.cu build/runtime.o build/decode.o build/gguf.o build/cache.o -o $@.tmp
	mv $@.tmp $@
bench/schedule: bench/schedule.cu build/runtime.o build/decode.o build/gguf.o build/cache.o
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) -Xcompiler -fopenmp bench/schedule.cu build/runtime.o build/decode.o build/gguf.o build/cache.o -o $@.tmp
	mv $@.tmp $@
bench/router: bench/router.cu include/moe.cuh include/device.cuh include/model.hpp include/q2k.cuh
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $< -o $@
bench/primitives: bench/primitives.cu include/device.cuh include/model.hpp include/q2k.cuh
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $< -o $@
inspect: src/gguf.cpp src/cache.cpp src/inspect.cpp src/gguf.hpp src/cache.hpp include/format.hpp include/model.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fopenmp src/gguf.cpp src/cache.cpp src/inspect.cpp -o $@.tmp
	mv $@.tmp $@
bench/loader-test: src/gguf.cpp src/cache.cpp bench/loader-test.cpp src/gguf.hpp src/cache.hpp include/format.hpp include/model.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fopenmp src/gguf.cpp src/cache.cpp bench/loader-test.cpp -o $@.tmp
	mv $@.tmp $@
clean:
	rm -rf build
	rm -f inspect bench/barrier bench/primitives bench/loader-test bench/router bench/schedule bench/engine-timing
-include $(NATIVE_OBJECTS:.o=.d)
