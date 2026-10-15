ROOT := ../..
NVCC ?= /usr/local/cuda/bin/nvcc
BUILD := build
FLAGS := -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -I$(ROOT)/src --expt-relaxed-constexpr
# An already-built modern kernels.o may be supplied; its source hashes belong in the receipt.
KERNELS_OBJECT ?= $(BUILD)/kernels.o

.PHONY: all clean
all: $(BUILD)/bench
$(BUILD):
	mkdir -p $@
$(BUILD)/pair.o: pair.cu pair.cuh $(ROOT)/src/fp4_ops.cuh $(ROOT)/src/kernels.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/bench.o: bench.cu pair.cuh $(ROOT)/src/fp4.cu $(ROOT)/src/kernels.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/kernels.o: $(ROOT)/src/kernels.cu $(ROOT)/src/prep_chunk.cuh $(ROOT)/src/kernels.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/bench: $(BUILD)/bench.o $(BUILD)/pair.o $(KERNELS_OBJECT) $(ROOT)/src/gguf.cpp
	$(NVCC) $(FLAGS) $^ -o $@
clean:
	rm -rf $(BUILD)
