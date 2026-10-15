NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_120a
LLAMA ?= /workspace/h5-llama
DEFS ?=
COMPUTE ?= $(subst sm_,compute_,$(ARCH))
FLAGS = -O3 -std=c++17 -gencode arch=$(COMPUTE),code=$(ARCH) -lineinfo -Xcompiler -fPIC,-O3 --expt-relaxed-constexpr $(DEFS)
LIBSRC = src/engine.cu src/kernels.cu src/fp4.cu src/draft.cu src/gguf.cpp experiments/binary-packed/packed1.cu
all: h5 h5-server refdump
build/%.o: src/%.cu src/model.h src/kernels.cuh src/profile_trace.h src/pdl_clock.cuh src/draft_contract.h
	@mkdir -p build; $(NVCC) $(FLAGS) -c -o $@ $<
build/engine.o: src/gguf.h src/compact.hpp experiments/binary-packed/packed1.cuh
build/packed1.o: experiments/binary-packed/packed1.cu experiments/binary-packed/packed1.cuh
	@mkdir -p build; $(NVCC) $(FLAGS) -c -o $@ $<
build/fp4.o: src/compact.cuh
build/kernels.o: src/kernels.cu src/attn.cu src/mega.cu src/model.h src/kernels.cuh src/profile_trace.h src/pdl_clock.cuh src/draft_contract.h
	@mkdir -p build; $(NVCC) $(FLAGS) -c -o $@ $<
build/gguf.o: src/gguf.cpp src/gguf.h
	@mkdir -p build; $(NVCC) $(FLAGS) -c -o $@ $<
build/main.o: src/main.cpp src/two_path_runner.h src/two_path_policy.h
	@mkdir -p build; g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include -c -o $@ $<
h5: build/engine.o build/kernels.o build/fp4.o build/draft.o build/gguf.o build/packed1.o build/main.o
	$(NVCC) -arch=$(ARCH) -o $@ $^ -L$(LLAMA)/build/bin -lllama -Xlinker -rpath,$(LLAMA)/build/bin -Xcompiler -pthread
SERVER_FLAGS = -O2 -std=c++17 -DCPPHTTPLIB_OPENSSL_SUPPORT -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/tools/mtmd -I$(LLAMA)/vendor/cpp-httplib -I$(LLAMA)/vendor
LLAMA_BUILD ?= $(LLAMA)/build
build/server.o: src/server.cpp src/vision.h src/chat.h src/two_path_policy.h src/profile_trace.h src/draft_contract.h
	@mkdir -p build; g++ $(SERVER_FLAGS) $(DEFS) -I$(dir $(NVCC))/../include -c -o $@ $<
build/vision.o: src/vision.cpp src/vision.h src/model.h
	@mkdir -p build; g++ $(SERVER_FLAGS) -c -o $@ $<
build/httplib.o: $(LLAMA)/vendor/cpp-httplib/httplib.cpp
	@mkdir -p build; g++ $(SERVER_FLAGS) -c -o $@ $<
h5-server: build/engine.o build/kernels.o build/fp4.o build/draft.o build/gguf.o build/packed1.o build/server.o build/vision.o build/httplib.o
	$(NVCC) -arch=$(ARCH) -o $@ $^ -L$(LLAMA_BUILD)/bin -lmtmd -lllama -lggml -lggml-base -lssl -lcrypto -Xlinker -rpath,$(LLAMA_BUILD)/bin -Xcompiler -pthread
refdump: tools/refdump.cpp
	g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include -o $@ $< -L$(LLAMA)/build/bin -lllama -Wl,-rpath,$(LLAMA)/build/bin
build/vision_check.o: tools/vision_check.cpp src/vision.h src/model.h
	@mkdir -p build; g++ $(SERVER_FLAGS) -Isrc -c -o $@ $<
h5-vision-check: build/engine.o build/kernels.o build/fp4.o build/draft.o build/gguf.o build/packed1.o build/vision_check.o build/vision.o build/httplib.o
	$(NVCC) -arch=$(ARCH) -o $@ $^ -L$(LLAMA_BUILD)/bin -lmtmd -lllama -lggml -lggml-base -lssl -lcrypto -Xlinker -rpath,$(LLAMA_BUILD)/bin -Xcompiler -pthread
check-chat:
	@mkdir -p build; g++ -O2 -std=c++17 -Isrc -I$(LLAMA)/vendor tools/check_chat.cpp -o build/check-chat && build/check-chat
selector-two-test: bench/selector_two.cu build/draft.o
	$(NVCC) $(FLAGS) -I src -o $@ $^
two-path-policy-test: bench/two_path_policy.cpp src/two_path_policy.h
	g++ -O2 -std=c++17 -I src -o $@ $<
	./$@
.PHONY: draft-contract-test
draft-contract-test: bench/draft_contract.cpp src/draft_contract.h
	@mkdir -p build; g++ -O2 -std=c++17 -Wall -Wextra -Werror -I src $< -o build/draft-contract-test && build/draft-contract-test
clean:
	rm -rf build h5 h5-server h5-vision-check refdump two-path-policy-test selector-two-test
