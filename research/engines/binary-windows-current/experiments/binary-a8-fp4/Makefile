ROOT := ../..
NVCC ?= /usr/local/cuda/bin/nvcc
FLAGS := -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -I$(ROOT)/src
BUILD := build
KERNELS_OBJECT ?= ../binary-fp4-pair/build/kernels.o
.PHONY: all clean
all: $(BUILD)/bench
$(BUILD):
	mkdir -p $@
$(BUILD)/digits.o: digits.cu digits.cuh $(ROOT)/src/fp4_ops.cuh $(ROOT)/src/kernels.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/bench.o: bench.cu digits.cuh $(ROOT)/src/gguf.h | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/bitserial.o: ../binary-bitserial/bitserial.cu ../binary-bitserial/bitserial.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/bench: $(BUILD)/bench.o $(BUILD)/digits.o $(BUILD)/bitserial.o $(KERNELS_OBJECT) $(ROOT)/src/gguf.cpp
	$(NVCC) $(FLAGS) $^ -o $@
clean:
	rm -rf $(BUILD)
