ROOT := ../..
NVCC ?= /usr/local/cuda/bin/nvcc
BUILD := build
FLAGS := -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -I$(ROOT)/src --expt-relaxed-constexpr
# An already-built modern kernels.o may be supplied; its source hashes belong in the receipt.
KERNELS_OBJECT ?= $(BUILD)/kernels.o
BITSERIAL_OBJECT ?= $(BUILD)/bitserial.o

.PHONY: all clean
all: $(BUILD)/bench
$(BUILD):
	mkdir -p $@
$(BUILD)/native.o: native.cu native.cuh $(ROOT)/src/fp4_ops.cuh $(ROOT)/src/kernels.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/bench.o: bench.cu native.cuh $(ROOT)/src/fp4.cu $(ROOT)/src/kernels.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/kernels.o: $(ROOT)/src/kernels.cu $(ROOT)/src/prep_chunk.cuh $(ROOT)/src/kernels.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/pair-control.o: pair-control.cu ../binary-fp4-pair/pair.cu ../binary-fp4-pair/pair.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/bitserial.o: ../binary-bitserial/bitserial.cu ../binary-bitserial/bitserial.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/bench: $(BUILD)/bench.o $(BUILD)/native.o $(BUILD)/pair-control.o $(BITSERIAL_OBJECT) $(KERNELS_OBJECT) $(ROOT)/src/gguf.cpp
	$(NVCC) $(FLAGS) $^ -o $@
clean:
	rm -rf $(BUILD)
