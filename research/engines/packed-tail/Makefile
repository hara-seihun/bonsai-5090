NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_120a
# Explicit gencode: CUDA 13's -arch=sm_120a also emits plain compute_120 PTX, which rejects sm_120a-only instructions.
ARCH_FLAGS ?= -gencode arch=compute_$(patsubst sm_%,%,$(ARCH)),code=$(ARCH)
LLAMA ?= /workspace/h5-llama
LLAMA_BUILD ?= $(LLAMA)/build
DEFS ?=
NVCC_HOST_CXX ?=
BUILD ?= build
REPLAY ?= profile-replay
SPEC_REPLAY ?= spec-replay
H5_BINARY ?= h5
SERVER_BINARY ?= h5-server
FLAGS = -O3 -std=c++17 -I$(LLAMA)/vendor $(ARCH_FLAGS) -lineinfo -Xcompiler -fPIC,-O3 --expt-relaxed-constexpr $(if $(NVCC_HOST_CXX),-ccbin $(NVCC_HOST_CXX)) $(DEFS)
SERVER_FLAGS = -O2 -std=c++17 -DCPPHTTPLIB_OPENSSL_SUPPORT -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/tools/mtmd -I$(LLAMA)/vendor/cpp-httplib -I$(LLAMA)/vendor
LIBSRC = src/engine.cu src/fullk_gu.cu src/fullk_ring.cu src/graph_memory.cu src/kernels.cu src/fp4.cu src/sk.cu src/draft.cu src/sampling_gpu.cu src/f02-producer.cu src/f02-continuation.cu src/gguf.cpp
ENGINE_OBJECTS = $(addprefix $(BUILD)/,engine.o fullk_gu.o fullk_ring.o graph_memory.o kernels.o fp4.o sk.o draft.o sampling_gpu.o f02-producer.o f02-continuation.o gguf.o)
.PHONY: all clean FORCE sk-bench lowbit-check drafter-exact-test gemm-profile draft-contract-test head-demand-test integrated-single f02-contract-test f07-contract-test f07-native-compile
all: $(H5_BINARY) $(SERVER_BINARY) refdump
integrated-single:
	@defs="$$(python3 tools/selected_build_defines.py results/integrated/selected-preset.json)" || exit; $(MAKE) DEFS="$$defs" "$(SERVER_BINARY)"
FORCE:
# A flag change must invalidate objects even when the same directory is reused.
# Separate BUILD/REPLAY paths allow causal variants to coexist.
$(BUILD)/compile-config: FORCE
	@mkdir -p $(BUILD); printf '%s\n' '$(NVCC)' '$(FLAGS)' '$(SERVER_FLAGS)' '$(LLAMA_BUILD)' > $@.tmp; if ! cmp -s $@.tmp $@; then mv $@.tmp $@; fi; rm -f $@.tmp
$(BUILD)/%.o: src/%.cu src/draft_lz.cuh src/tail_prep.cuh src/smlit.cuh src/model.h src/kernels.cuh src/profile_trace.h src/pdl_clock.cuh src/draft_contract.h
	@mkdir -p $(BUILD); $(NVCC) $(FLAGS) -c -o $@ $<
$(ENGINE_OBJECTS) $(BUILD)/main.o: src/state_format.h src/q8_scale.h
$(ENGINE_OBJECTS) $(BUILD)/profile_replay.o $(BUILD)/spec_replay.o: src/pf_chain.h
$(BUILD)/engine.o: src/gpu_grammar_token_compile.cuh src/gpu_grammar_cache.h src/gpu_grammar_select_engine.cuh src/gpu_grammar_batch_head.cuh src/gpu_grammar_batch_engine.cuh src/gpu_grammar.h src/gpu_grammar_runtime.cuh src/gpu_grammar_select.cuh src/gpu_grammar_engine.cuh src/sparse_head_check.cuh src/sparse_head.cuh src/grammar_host_select.h src/gdn_diagnostic.h src/gdn_single.h src/draft_probe.h src/ternary_operands.h src/gguf.h src/compact.hpp src/graph_probe.h src/graph_memory.h src/scale_carrier.cuh src/head_demand.h src/serving_policy.h src/f02.cuh src/f02_route.h src/f07.cuh src/f07_contract.h
$(BUILD)/engine.o $(BUILD)/draft.o $(BUILD)/sampling_gpu.o: src/sampling_gpu.cuh src/sampling_contract.h src/sampling_noise.h
$(BUILD)/graph_memory.o: src/graph_memory.cu src/graph_memory.h
$(BUILD)/sk.o: src/compact.cuh src/fp4_ops.cuh src/sk.h
$(BUILD)/f02-producer.o: src/fp4_ops.cuh
$(BUILD)/fp4.o: src/fp4_ops.cuh src/compact.cuh src/compact.hpp src/scale_carrier.cuh src/f07.cuh src/f07_contract.h
$(BUILD)/kernels.o: src/kernels.cu src/gdn_single.cuh src/gdn_verify.cuh src/gdn_verify_fused.cuh src/gdn_single.h src/tail_prep.cuh src/smlit.cuh src/prep_chunk.cuh src/attn.cu src/mega.cu src/state_fusion.cuh src/model.h src/kernels.cuh src/profile_trace.h src/pdl_clock.cuh src/draft_contract.h
	@mkdir -p $(BUILD); $(NVCC) $(FLAGS) -c -o $@ $<
$(BUILD)/gguf.o: src/gguf.cpp src/gguf.h
	@mkdir -p $(BUILD); $(NVCC) $(FLAGS) -c -o $@ $<
$(BUILD)/main.o: src/main.cpp src/two_path_runner.h src/two_path_policy.h src/head_demand.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include -c -o $@ $<
$(H5_BINARY): $(ENGINE_OBJECTS) $(BUILD)/main.o
	@mkdir -p $(dir $@); $(NVCC) $(ARCH_FLAGS) -o $@ $^ -L$(LLAMA_BUILD)/bin -lllama -Xlinker -rpath,$(LLAMA_BUILD)/bin -Xcompiler -pthread
$(BUILD)/gpu_grammar_compile.o: src/gpu_grammar_compile.cpp src/gpu_grammar.h
	@mkdir -p $(BUILD); $(CXX) -O2 -std=c++17 -I$(LLAMA)/vendor -c -o $@ $<
$(BUILD)/grammar.o: src/grammar.cpp src/grammar.h
	@mkdir -p $(BUILD); g++ -O3 -std=c++17 -I$(LLAMA)/vendor -c -o $@ $<
.PHONY: grammar-test grammar-jump-test
GPU_SELECT_HEADERS = src/gpu_grammar_select.cuh src/gpu_grammar_runtime.cuh src/gpu_grammar.h src/gpu_grammar_cache.h
gpu-grammar-select-probe: bench/gpu_grammar_select_probe.cu $(GPU_SELECT_HEADERS)
	$(NVCC) $(FLAGS) -Isrc -o $@ $<
gpu-grammar-cache-lifecycle: bench/gpu_grammar_cache_lifecycle.cpp $(ENGINE_OBJECTS) src/gpu_grammar.h
	$(NVCC) $(FLAGS) -Isrc -DGPU_GRAMMAR_CACHE_PROBE_MAIN -o $@ $< $(ENGINE_OBJECTS) -L$(LLAMA_BUILD)/bin -lllama -Xlinker -rpath,$(LLAMA_BUILD)/bin -Xcompiler -pthread

grammar-test: tools/grammar_test.cpp src/grammar.cpp src/grammar.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Isrc -I$(LLAMA)/vendor -o $(BUILD)/grammar-test tools/grammar_test.cpp src/grammar.cpp && $(BUILD)/grammar-test
grammar-jump-test: tools/grammar_jump_test.cpp src/server.cpp src/grammar.cpp src/grammar.h
	@mkdir -p $(BUILD); g++ $(filter-out -O2,$(SERVER_FLAGS)) -O0 -ffunction-sections -fdata-sections -Wl,--gc-sections -I$(dir $(NVCC))/../include tools/grammar_jump_test.cpp src/grammar.cpp -pthread -o $(BUILD)/grammar-jump-test
	@set -e; for route in plain spec; do for scenario in dynamic const cap context; do $(BUILD)/grammar-jump-test $$route $$scenario; done; done
$(BUILD)/server.o: src/server.cpp src/gpu_grammar.h src/grammar.h src/vision.h src/chat.h src/two_path_policy.h src/commit_policy.h src/profile_trace.h src/draft_contract.h src/head_demand.h src/serving_policy.h src/prompt_lookup.h src/sampling.h src/sampling_contract.h src/sampling_noise.h src/relax_policy.h src/lz_penalty.h
	@mkdir -p $(BUILD); g++ $(SERVER_FLAGS) $(DEFS) -I$(dir $(NVCC))/../include -c -o $@ $<
$(BUILD)/vision.o: src/vision.cpp src/vision.h src/model.h
	@mkdir -p $(BUILD); g++ $(SERVER_FLAGS) -c -o $@ $<
$(BUILD)/httplib.o: $(LLAMA)/vendor/cpp-httplib/httplib.cpp
	@mkdir -p $(BUILD); g++ $(SERVER_FLAGS) -c -o $@ $<
$(SERVER_BINARY): $(ENGINE_OBJECTS) $(BUILD)/server.o $(BUILD)/gpu_grammar_compile.o $(BUILD)/grammar.o $(BUILD)/vision.o $(BUILD)/httplib.o
	@mkdir -p $(dir $@); $(NVCC) $(ARCH_FLAGS) -o $@ $^ -L$(LLAMA_BUILD)/bin -lmtmd -lllama -lggml -lggml-base -lssl -lcrypto -Xlinker -rpath,$(LLAMA_BUILD)/bin -Xcompiler -pthread
$(BUILD)/profile_replay.o: tools/profile_replay.cpp src/smlit.cuh src/profile_trace.h src/graph_probe.h src/head_demand.h
	@mkdir -p $(BUILD); $(NVCC) $(FLAGS) -Isrc -c -o $@ $<
$(REPLAY): $(ENGINE_OBJECTS) $(BUILD)/profile_replay.o
	@mkdir -p $(dir $@); $(NVCC) $(ARCH_FLAGS) -o $@ $^ -Xcompiler -pthread
$(BUILD)/spec_replay.o: tools/spec_replay.cpp src/smlit.cuh src/head_demand.h
	@mkdir -p $(BUILD); $(NVCC) $(FLAGS) -Isrc -c -o $@ $<
$(SPEC_REPLAY): $(ENGINE_OBJECTS) $(BUILD)/spec_replay.o
	@mkdir -p $(dir $@); $(NVCC) $(ARCH_FLAGS) -o $@ $^ -Xcompiler -pthread
GDN_DRIFT ?= gdn-state-drift
.PHONY: gdn-diagnostic-contract-test gdn-diagnostic-logits-test gdn-diagnostic-native-compile
$(BUILD)/gdn_diagnostic_build.h: FORCE
	@mkdir -p $(BUILD); python3 -c 'import json, subprocess, sys; source=subprocess.check_output(["git", "describe", "--always", "--dirty"],text=True).strip(); print("#define H5_GDN_DIAGNOSTIC_SOURCE " + json.dumps(source)); print("#define H5_GDN_DIAGNOSTIC_FLAGS " + json.dumps(sys.argv[1]))' '$(FLAGS)' > $@.tmp; if ! cmp -s $@.tmp $@; then mv $@.tmp $@; fi; rm -f $@.tmp
$(BUILD)/gdn_state_drift.o: tools/gdn_state_drift.cpp tools/gdn_diagnostic_input.h tools/gdn_diagnostic_logits.h src/gdn_diagnostic.h src/state_format.h $(BUILD)/gdn_diagnostic_build.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Wextra -Werror -Isrc -Itools -include $(BUILD)/gdn_diagnostic_build.h -c -o $@ $<
$(GDN_DRIFT): $(ENGINE_OBJECTS) $(BUILD)/gdn_state_drift.o
	@mkdir -p $(dir $@); $(NVCC) $(ARCH_FLAGS) -o $@ $^ -Xcompiler -pthread
$(BUILD)/gdn-state-drift-input: tools/gdn_state_drift.cpp tools/gdn_diagnostic_input.h tools/gdn_diagnostic_logits.h src/gdn_diagnostic.h src/state_format.h $(BUILD)/gdn_diagnostic_build.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Wextra -Werror -DH5_GDN_DIAGNOSTIC_CPU_ONLY -Isrc -Itools -include $(BUILD)/gdn_diagnostic_build.h -o $@ $<
gdn-diagnostic-contract-test: bench/gdn_diagnostic_contract.cpp tools/gdn_diagnostic_input.h src/gdn_diagnostic.h src/state_format.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Wextra -Werror -Isrc -Itools $< -o $(BUILD)/gdn-diagnostic-contract-test && $(BUILD)/gdn-diagnostic-contract-test
gdn-diagnostic-logits-test: bench/gdn_diagnostic_logits.cpp tools/gdn_diagnostic_logits.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Wextra -Werror -Itools $< -o $(BUILD)/gdn-diagnostic-logits-test && $(BUILD)/gdn-diagnostic-logits-test
gdn-diagnostic-native-compile: $(BUILD)/engine.o $(BUILD)/gdn_state_drift.o
TAIL_PROBE ?= tail-probe
$(BUILD)/tail_probe.o: tools/tail_probe.cpp
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -c -o $@ $<
$(TAIL_PROBE): $(ENGINE_OBJECTS) $(BUILD)/tail_probe.o
	@mkdir -p $(dir $@); $(NVCC) $(ARCH_FLAGS) -o $@ $^ -Xcompiler -pthread
refdump: tools/refdump.cpp
	g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include -o $@ $< -L$(LLAMA_BUILD)/bin -lllama -Wl,-rpath,$(LLAMA_BUILD)/bin
$(BUILD)/vision_check.o: tools/vision_check.cpp src/vision.h src/model.h
	@mkdir -p $(BUILD); g++ $(SERVER_FLAGS) -Isrc -c -o $@ $<
h5-vision-check: $(ENGINE_OBJECTS) $(BUILD)/vision_check.o $(BUILD)/vision.o $(BUILD)/httplib.o
	$(NVCC) $(ARCH_FLAGS) -o $@ $^ -L$(LLAMA_BUILD)/bin -lmtmd -lllama -lggml -lggml-base -lssl -lcrypto -Xlinker -rpath,$(LLAMA_BUILD)/bin -Xcompiler -pthread
sampling-gpu-test: bench/sampling_gpu_test.cu $(BUILD)/sampling_gpu.o src/sampling.h src/sampling_gpu.cuh src/sampling_contract.h src/sampling_noise.h
	$(NVCC) $(FLAGS) -Isrc -I$(LLAMA)/vendor -o $(BUILD)/sampling-gpu-test $< $(BUILD)/sampling_gpu.o
	$(BUILD)/sampling-gpu-test
sampling-test: bench/sampling_test.cpp src/sampling.h src/sampling_contract.h src/sampling_noise.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Isrc -I$(LLAMA)/vendor $< -o $(BUILD)/sampling-test && $(BUILD)/sampling-test
prompt-lookup-test: bench/prompt_lookup_test.cpp src/prompt_lookup.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Isrc $< -o $(BUILD)/prompt-lookup-test && $(BUILD)/prompt-lookup-test
relax-policy-test: bench/relax_policy_test.cpp src/relax_policy.h src/draft_contract.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Isrc $< -o $(BUILD)/relax-policy-test && $(BUILD)/relax-policy-test

lz-penalty-test: bench/lz_penalty_test.cpp bench/lz_penalty_vectors.h src/lz_penalty.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Isrc -Ibench $< -o $(BUILD)/lz-penalty-test && $(BUILD)/lz-penalty-test

draft-lz-test: bench/draft_lz_test.cu src/draft_lz.cuh src/lz_penalty.h src/kernels.cuh
	@mkdir -p $(BUILD); $(NVCC) -O2 -std=c++17 $(ARCH_FLAGS) -Isrc $< -o $(BUILD)/draft-lz-test && $(BUILD)/draft-lz-test
check-chat:
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Isrc -I$(LLAMA)/vendor tools/check_chat.cpp -o $(BUILD)/check-chat && $(BUILD)/check-chat
drafter-exact-test: bench/drafter_exact.cu $(BUILD)/kernels.o $(BUILD)/draft.o
	$(NVCC) $(FLAGS) -I src -o $@ $^
selector-two-test: bench/selector_two.cu $(BUILD)/draft.o
	$(NVCC) $(FLAGS) -I src -o $@ $^
graph-memory-test: bench/graph_memory.cu $(BUILD)/graph_memory.o
	$(NVCC) $(FLAGS) -Isrc -o $@ $^
pdl-gdn-order: bench/pdl_gdn_order.cu $(BUILD)/kernels.o
	$(NVCC) $(FLAGS) -I src -o $@ $^
state-fusion-test: bench/state_fusion.cu $(BUILD)/kernels.o
	$(NVCC) $(FLAGS) -Isrc -o $@ $^
gdn-q8: bench/gdn_q8.cu $(BUILD)/kernels.o
	$(NVCC) $(FLAGS) -Isrc -o $@ $^
gdn-single-bench: bench/gdn_single.cu bench/gdn_fused_causal.cuh $(BUILD)/kernels.o
	$(NVCC) $(FLAGS) -Isrc -o $@ bench/gdn_single.cu $(BUILD)/kernels.o
two-path-policy-test: bench/two_path_policy.cpp src/two_path_policy.h
	g++ -O2 -std=c++17 -I src -o $@ $<
	./$@
draft-contract-test: bench/draft_contract.cpp src/draft_contract.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Wextra -Werror -I src $< -o $(BUILD)/draft-contract-test && $(BUILD)/draft-contract-test
head-demand-test: bench/head_demand.cpp src/head_demand.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Wextra -Werror -I src $< -o $(BUILD)/head-demand-test && $(BUILD)/head-demand-test
f02-contract-test: bench/f02_contract.cpp src/f02_route.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Wextra -Werror -I src $< -o $(BUILD)/f02-contract-test && $(BUILD)/f02-contract-test
f07-contract-test: bench/f07_contract.cpp src/f07_contract.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Wextra -Werror -I src $< -o $(BUILD)/f07-contract-test && $(BUILD)/f07-contract-test
$(BUILD)/f07_native_compile.o: bench/f07_native_compile.cu src/fp4.cu src/f07.cuh src/f07_contract.h src/kernels.cuh src/model.h src/compact.cuh src/compact.hpp src/scale_carrier.cuh src/pdl_clock.cuh $(BUILD)/compile-config
	@mkdir -p $(BUILD); $(NVCC) $(FLAGS) -Xptxas=-v -c -o $@ $<
f07-native-compile: $(BUILD)/f07_native_compile.o
.PHONY: pf-chain-contract-test gemm-release-native-compile
pf-chain-contract-test: bench/pf_chain_contract.cpp src/pf_chain.h
	@mkdir -p $(BUILD); g++ -O2 -std=c++17 -Wall -Wextra -Werror -I src $< -o $(BUILD)/pf-chain-contract-test && $(BUILD)/pf-chain-contract-test
$(BUILD)/gemm_release_native_compile.o: bench/gemm_release_native_compile.cu src/fp4.cu src/fp4_ops.cuh src/kernels.cuh src/tail_prep.cuh src/smlit.cuh src/model.h src/compact.cuh src/compact.hpp src/scale_carrier.cuh src/f07.cuh src/f07_contract.h src/warp_microscope.cuh src/pdl_clock.cuh $(BUILD)/compile-config
	@mkdir -p $(BUILD); $(NVCC) $(FLAGS) -Xptxas=-v -c -o $@ $<
gemm-release-native-compile: $(BUILD)/gemm_release_native_compile.o
prep-bench: tools/prep_bench.cu src/kernels.cu src/prep_chunk.cuh src/attn.cu src/mega.cu src/state_fusion.cuh src/kernels.cuh src/model.h src/pdl_clock.cuh $(BUILD)/compile-config
	$(NVCC) $(FLAGS) -Xptxas=-v -o $@ $<
prep-rms-bench: tools/prep_rms_bench.cu src/kernels.cu src/prep_chunk.cuh src/attn.cu src/mega.cu src/state_fusion.cuh src/kernels.cuh src/model.h src/pdl_clock.cuh $(BUILD)/compile-config
	$(NVCC) $(FLAGS) -Xptxas=-v -o $@ $<
prep-row-bench: tools/prep_row_bench.cu src/kernels.cu src/prep_chunk.cuh src/attn.cu src/mega.cu src/state_fusion.cuh src/kernels.cuh src/model.h src/pdl_clock.cuh $(BUILD)/compile-config
	$(NVCC) $(FLAGS) -Xptxas=-v -o $@ $<
GEMM_BUILD ?= build-gemm-ring
GEMM_DEFS ?= -DH5_PROFILE_GEMM
$(GEMM_BUILD)/compile-config: FORCE
	@mkdir -p $(GEMM_BUILD); printf '%s\n' '$(NVCC)' '$(FLAGS)' '$(GEMM_DEFS)' > $@.tmp; if ! cmp -s $@.tmp $@; then mv $@.tmp $@; fi; rm -f $@.tmp
$(GEMM_BUILD)/fp4.o: src/fp4.cu src/fp4_ops.cuh src/tail_prep.cuh src/smlit.cuh src/compact.cuh src/compact.hpp src/kernels.cuh src/model.h src/pdl_clock.cuh src/f07.cuh src/f07_contract.h $(GEMM_BUILD)/compile-config
	@mkdir -p $(GEMM_BUILD); $(NVCC) $(FLAGS) $(GEMM_DEFS) -Xptxas=-v -c -o $@ $<
$(GEMM_BUILD)/gemm-residency: bench/gemm_residency.cu $(GEMM_BUILD)/fp4.o
	$(NVCC) $(FLAGS) -I src -o $@ $^
$(GEMM_BUILD)/profile-replay: tools/profile_replay.cpp $(filter-out $(BUILD)/fp4.o,$(ENGINE_OBJECTS)) $(GEMM_BUILD)/fp4.o
	$(NVCC) $(FLAGS) -I src -o $@ $^
gemm-profile: $(GEMM_BUILD)/gemm-residency $(GEMM_BUILD)/profile-replay
$(ENGINE_OBJECTS) $(BUILD)/main.o $(BUILD)/server.o $(BUILD)/vision.o $(BUILD)/httplib.o $(BUILD)/profile_replay.o $(BUILD)/vision_check.o: $(BUILD)/compile-config
clean:
	rm -rf $(BUILD) $(H5_BINARY) $(SERVER_BINARY) h5-vision-check $(REPLAY) $(GDN_DRIFT) graph-memory-test grammar-vocab-bytes gpu-grammar-runtime-probe gpu-grammar-select-probe gpu-grammar-cache-lifecycle predicated-gemm-probe refdump two-path-policy-test selector-two-test prep-bench prep-rms-bench prep-row-bench pdl-gdn-order state-fusion-test gdn-q8 gdn-single-bench
lowbit-check: bench/lowbit_check.cu $(BUILD)/fp4.o
	$(NVCC) $(FLAGS) -Isrc -o $@ $^
batch-gemm: bench/batch_gemm.cu $(BUILD)/fp4.o
	$(NVCC) $(FLAGS) -Isrc -o $@ $^
$(BUILD)/fp4.o $(BUILD)/engine.o: src/deterministic_gemm.cuh
ordered-prep-bench: bench/ordered_prep.cu $(BUILD)/fp4.o $(BUILD)/kernels.o
	$(NVCC) $(FLAGS) -Isrc -o $@ $^
deterministic-gemm: bench/deterministic_gemm.cu $(BUILD)/fp4.o src/deterministic_gemm.cuh
	$(NVCC) $(FLAGS) -Isrc -o $@ bench/deterministic_gemm.cu $(BUILD)/fp4.o
grammar-vocab-bytes: tools/grammar_vocab_bytes.cpp
	g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA_BUILD)/bin -Wl,-rpath,$(LLAMA_BUILD)/bin -lllama -o $@
gpu-grammar-runtime-probe: bench/gpu_grammar_runtime_probe.cu src/gpu_grammar_runtime.cuh src/gpu_grammar_token_compile.cuh src/gpu_grammar_cache.h src/gpu_grammar.h
	$(NVCC) $(FLAGS) -Isrc -o $@ bench/gpu_grammar_runtime_probe.cu
predicated-gemm-probe: bench/predicated_gemm_probe.cu $(BUILD)/fp4.o src/deterministic_gemm.cuh
	$(NVCC) $(FLAGS) -Isrc -o $@ bench/predicated_gemm_probe.cu $(BUILD)/fp4.o
.PHONY: ternary-operands-test
ternary-operands-test: bench/ternary_operands_test.cpp src/ternary_operands.h
	g++ -O2 -std=c++17 -Isrc -o $@ $<
	./$@
ternary-expanded-check: bench/ternary_expanded_check.cu $(BUILD)/fp4.o src/ternary_operands.h
	$(NVCC) $(FLAGS) -Isrc -o $@ bench/ternary_expanded_check.cu $(BUILD)/fp4.o
batch-profile: tools/batch_profile.cu $(ENGINE_OBJECTS)
	$(NVCC) $(FLAGS) -Isrc -o $@ $^ -Xcompiler -pthread
sk-bench: bench/sk_bench.cu $(BUILD)/fp4.o $(BUILD)/sk.o
	$(NVCC) $(FLAGS) -Isrc -o $@ $^
fuse-gu-test: bench/fuse_gu_test.cu $(BUILD)/kernels.o $(BUILD)/fp4.o
	$(NVCC) $(FLAGS) -Isrc -o $@ $^

$(BUILD)/engine.o $(BUILD)/fp4.o $(BUILD)/fullk_gu.o: src/fullk_gu.cuh
$(BUILD)/fullk_gu.o: src/fullk_gu.cu src/fullk_gu.cuh $(BUILD)/compile-config
	@mkdir -p $(BUILD); $(NVCC) $(FLAGS) --fmad=false -c -o $@ $<

$(BUILD)/fullk_ring.o: src/fullk_ring.cu src/fp4.cu src/fullk_gu.cuh src/fp4_ops.cuh src/compact.cuh
