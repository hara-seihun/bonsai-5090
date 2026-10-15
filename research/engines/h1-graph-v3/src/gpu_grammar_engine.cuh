#pragma once
// Included inside Engine: the stream owns all recurring grammar work and buffers.
struct GpuGrammarCacheEntry {
  gpu_grammar::Plan source;
  gpu_grammar::DevicePlan image{};
  gpu_grammar::DevicePlan* device = nullptr;
  uint64_t bytes = 0, token_bytes = 0, wide_states = 0, sparse_edges = 0, used = 0, refs = 0;
  ~GpuGrammarCacheEntry() {
    if (device) cudaFree(device);
    if (image.accepting) cudaFree((void*)image.accepting);
    if (image.counts) cudaFree((void*)image.counts);
    if (image.offsets) cudaFree((void*)image.offsets);
    if (image.sparse_ids) cudaFree((void*)image.sparse_ids);
    if (image.sparse_next) cudaFree((void*)image.sparse_next);
    if (image.wide_next) cudaFree((void*)image.wide_next);
    if (image.wide_mask) cudaFree((void*)image.wide_mask);
    if (image.wide_ids) cudaFree((void*)image.wide_ids);
  }
};
struct GpuGrammarPlan { GpuGrammarCacheEntry* entry; };
struct GpuGrammarCompileTemporary {
  uint32_t* transitions = nullptr;
  uint32_t* dense_next = nullptr;
  ~GpuGrammarCompileTemporary() {
    if (transitions) cudaFree(transitions);
    if (dense_next) cudaFree(dense_next);
  }
};
struct GpuGrammarVocabulary {
  gpu_grammar::DeviceVocabulary image{};
  ~GpuGrammarVocabulary() {
    if (image.offsets) cudaFree((void*)image.offsets);
    if (image.bytes) cudaFree((void*)image.bytes);
    if (image.eos) cudaFree((void*)image.eos);
    if (image.eos_ids) cudaFree((void*)image.eos_ids);
  }
};
std::vector<std::unique_ptr<GpuGrammarPlan>> gpu_plans;
std::vector<std::unique_ptr<GpuGrammarCacheEntry>> gpu_cache;
uint64_t gpu_cache_bytes = 0, gpu_cache_clock = 0, gpu_cache_hits = 0, gpu_cache_misses = 0, gpu_cache_evictions = 0;
double gpu_cache_compile_s = 0;
std::unique_ptr<GpuGrammarVocabulary> gpu_vocab;
gpu_grammar::DeviceInput *d_gpu_input = nullptr, *h_gpu_input = nullptr;
gpu_grammar::DeviceRow* d_gpu_rows = nullptr;
gpu_grammar::RowResult *d_gpu_results = nullptr, *h_gpu_results = nullptr;
int *d_gpu_prefix = nullptr, *h_gpu_prefix = nullptr;
uint32_t *d_gpu_mask = nullptr, *d_gpu_count = nullptr;
int* d_gpu_ids = nullptr;
int gpu_target_pending = 0, gpu_prefix_count = 0, gpu_results_rows = 0;
bool gpu_results_ready = false;
cudaEvent_t gpu_body_start = nullptr, gpu_body_end = nullptr, gpu_selection_end = nullptr;
std::pair<int, int> gpu_head_config;
float* gpu_head_partials = nullptr;
size_t gpu_head_partial_capacity = 0;

void ensure_gpu_grammar() {
  if (d_gpu_input) return;
  d_gpu_input = dalloc<gpu_grammar::DeviceInput>(max_rows);
  d_gpu_rows = dalloc<gpu_grammar::DeviceRow>(max_rows);
  d_gpu_results = dalloc<gpu_grammar::RowResult>(max_rows);
  d_gpu_prefix = dalloc<int>(max_rows);
  d_gpu_mask = dalloc<uint32_t>(GRAMMAR_WORDS);
  d_gpu_count = dalloc<uint32_t>(1);
  d_gpu_ids = dalloc<int>(gpu_grammar::COMPACT_CAPACITY);
  CK(cudaMallocHost(&h_gpu_input, size_t(max_rows) * sizeof(*h_gpu_input)));
  CK(cudaMallocHost(&h_gpu_results, size_t(max_rows) * sizeof(*h_gpu_results)));
  CK(cudaMallocHost(&h_gpu_prefix, size_t(max_rows) * sizeof(int)));
  CK(cudaEventCreate(&gpu_body_start));
  CK(cudaEventCreate(&gpu_body_end));
  CK(cudaEventCreate(&gpu_selection_end));
}
int gpu_vocabulary(const gpu_grammar::TokenTable* table) {
  if (!table || table->offsets.size() != size_t(VOCAB) + 1 || table->eos.size() != VOCAB ||
      table->offsets.front() || table->offsets.back() != table->bytes.size() || gpu_target_pending || !gpu_plans.empty()) return -1;
  std::vector<int> eos_ids;
  for (int t = 0; t < VOCAB; ++t) {
    if (table->offsets[t] > table->offsets[t + 1] || table->eos[t] > 1) return -1;
    if (table->eos[t]) eos_ids.push_back(t);
  }
  if (eos_ids.empty()) return -1;
  auto vocabulary = std::make_unique<GpuGrammarVocabulary>();
  vocabulary->image.offsets = upload<uint32_t>(table->offsets.data(), table->offsets.size());
  if (!table->bytes.empty()) vocabulary->image.bytes = upload<uint8_t>(table->bytes.data(), table->bytes.size());
  vocabulary->image.eos = upload<uint8_t>(table->eos.data(), table->eos.size());
  vocabulary->image.size = VOCAB;
  vocabulary->image.eos_ids = upload<int>(eos_ids.data(), eos_ids.size());
  vocabulary->image.eos_count = uint32_t(eos_ids.size());
  for (auto it = graphs.begin(); it != graphs.end();) {
    if (it->first.kind == 11 || it->first.kind == 12) {
      CK(cudaGraphExecDestroy(it->second.exec));
      it = graphs.erase(it);
    } else ++it;
  }
  gpu_cache.clear();
  gpu_cache_bytes = 0;
  gpu_vocab = std::move(vocabulary);
  ensure_gpu_grammar();
  return 0;
}
int gpu_attach(GpuGrammarCacheEntry* entry, void** handle) {
  auto owned = std::make_unique<GpuGrammarPlan>();
  owned->entry = entry;
  auto* result = owned.get();
  gpu_plans.push_back(std::move(owned));
  ++entry->refs;
  entry->used = ++gpu_cache_clock;
  *handle = result;
  return 0;
}
int gpu_make_cache_space(uint64_t bytes) {
  if (bytes > gpu_grammar::TOKEN_CACHE_BUDGET) return int(gpu_grammar::TokenCacheError::Budget);
  while (gpu_cache_bytes > gpu_grammar::TOKEN_CACHE_BUDGET - bytes) {
    auto victim = gpu_grammar::token_cache_victim(gpu_cache);
    if (victim == gpu_cache.end()) return int(gpu_grammar::TokenCacheError::Budget);
    gpu_cache_bytes -= (*victim)->bytes;
    gpu_cache.erase(victim);
    ++gpu_cache_evictions;
  }
  return 0;
}
int gpu_upload(const gpu_grammar::Plan* plan, void** handle) {
  if (!handle) return -1;
  *handle = nullptr;
  if (!gpu_vocab) return -2;
  if (!plan || !plan->states() || plan->initial >= plan->states() ||
      plan->transitions.size() != size_t(plan->states()) * 256) return -1;
  if (plan->states() > 65535) return int(gpu_grammar::TokenCacheError::Overflow);
  for (uint32_t next : plan->transitions) if (next != gpu_grammar::INVALID && next >= plan->states()) return -1;
  for (uint8_t accept : plan->accepting) if (accept > 1) return -1;
  for (auto& entry : gpu_cache) if (entry->source.initial == plan->initial &&
      entry->source.transitions == plan->transitions && entry->source.accepting == plan->accepting) {
    ++gpu_cache_hits;
    return gpu_attach(entry.get(), handle);
  }
  ++gpu_cache_misses;
  const double begin = now_s();
  auto owned = std::make_unique<GpuGrammarCacheEntry>();
  GpuGrammarCompileTemporary temporary;
  const uint32_t states = plan->states();
  owned->image.states = states;
  owned->image.accepting = upload<uint8_t>(plan->accepting.data(), states);
  owned->image.counts = dalloc<uint32_t>(states);
  temporary.transitions = upload<uint32_t>(plan->transitions.data(), plan->transitions.size());
  const gpu_grammar::BytePlan source{temporary.transitions, owned->image.accepting, states};
  CK(gpu_grammar::launch_token_counts(source, gpu_vocab->image, const_cast<uint32_t*>(owned->image.counts), st));
  std::vector<uint32_t> counts(states);
  CK(cudaMemcpyAsync(counts.data(), owned->image.counts, size_t(states) * sizeof(uint32_t), cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  gpu_grammar::TokenLayout layout;
  const auto layout_error = gpu_grammar::token_layout(counts, VOCAB,
      plan->bytes() + sizeof(gpu_grammar::DevicePlan), &layout);
  if (layout_error != gpu_grammar::TokenCacheError::None) return int(layout_error);
  const int budget = gpu_make_cache_space(layout.bytes);
  if (budget) return budget;
  owned->bytes = layout.bytes;
  owned->token_bytes = layout.bytes - plan->bytes();
  owned->wide_states = layout.wide_states;
  owned->sparse_edges = layout.sparse_edges;
  owned->image.offsets = upload<uint32_t>(layout.offsets.data(), states);
  if (layout.sparse_edges) {
    owned->image.sparse_ids = dalloc<int>(layout.sparse_edges);
    owned->image.sparse_next = dalloc<uint32_t>(layout.sparse_edges);
  }
  if (layout.wide_states) {
    owned->image.wide_next = dalloc<uint32_t>(layout.wide_states * VOCAB);
    owned->image.wide_mask = dalloc<uint32_t>(layout.wide_states * GRAMMAR_WORDS);
    owned->image.wide_ids = dalloc<int>(layout.wide_states * gpu_grammar::COMPACT_CAPACITY);
  }
  temporary.dense_next = dalloc<uint32_t>(size_t(std::min(states, gpu_grammar::TOKEN_COMPILE_CHUNK)) * VOCAB);
  for (uint32_t first = 0; first < states; first += gpu_grammar::TOKEN_COMPILE_CHUNK) {
    CK(gpu_grammar::launch_token_chunk(source, gpu_vocab->image, first,
        std::min(gpu_grammar::TOKEN_COMPILE_CHUNK, states - first), owned->image.counts, owned->image.offsets,
        temporary.dense_next, const_cast<int*>(owned->image.sparse_ids),
        const_cast<uint32_t*>(owned->image.sparse_next), const_cast<uint32_t*>(owned->image.wide_next),
        const_cast<uint32_t*>(owned->image.wide_mask), const_cast<int*>(owned->image.wide_ids), st));
  }
  owned->device = upload<gpu_grammar::DevicePlan>(&owned->image, 1);
  owned->source = *plan;
  CK(cudaFree(temporary.transitions)); temporary.transitions = nullptr;
  CK(cudaFree(temporary.dense_next)); temporary.dense_next = nullptr;
  gpu_cache_compile_s += now_s() - begin;
  auto* entry = owned.get();
  gpu_cache.push_back(std::move(owned));
  gpu_cache_bytes += entry->bytes;
  return gpu_attach(entry, handle);
}
int gpu_cache_stats(void* handle, gpu_grammar::CacheStats* output) const {
  if (!output) return -1;
  GpuGrammarCacheEntry* entry = nullptr;
  if (handle) {
    const auto found = std::find_if(gpu_plans.begin(), gpu_plans.end(), [handle](const auto& p) { return p.get() == handle; });
    if (found == gpu_plans.end()) return -3;
    entry = (*found)->entry;
  }
  *output = gpu_grammar::CacheStats{gpu_cache_bytes, gpu_grammar::TOKEN_CACHE_BUDGET,
      gpu_cache_hits, gpu_cache_misses, gpu_cache_evictions, gpu_cache_compile_s,
      entry ? entry->token_bytes : 0, entry ? entry->wide_states : 0, entry ? entry->sparse_edges : 0};
  return 0;
}
int gpu_release(void* handle) {
  const auto found = std::find_if(gpu_plans.begin(), gpu_plans.end(), [handle](const auto& p) { return p.get() == handle; });
  if (found == gpu_plans.end()) return -3;
  for (int r = 0; r < gpu_target_pending; ++r) if (h_gpu_input[r].plan == (*found)->entry->device) return -3;
  (*found)->entry->used = ++gpu_cache_clock;
  --(*found)->entry->refs;
  gpu_plans.erase(found);
  return 0;
}
int gpu_stage(int rows, const gpu_grammar::Row* input, int prefix_count, const int* prefix) {
  if (!gpu_vocab) return -2;
  if (rows < 0 || rows > max_rows || (rows && !input) || prefix_count < 0 || prefix_count > max_rows ||
      (prefix_count && !prefix)) return -1;
  ensure_gpu_grammar();
  for (int r = 0; r < rows; ++r) {
    const auto& row = input[r];
    const gpu_grammar::DevicePlan* device = nullptr;
    if (row.plan) {
      const auto found = std::find_if(gpu_plans.begin(), gpu_plans.end(), [&](const auto& p) { return p.get() == row.plan; });
      if (found == gpu_plans.end()) return -3;
      device = (*found)->entry->device;
      (*found)->entry->used = ++gpu_cache_clock;
    }
    if (uint64_t(row.prefix_offset) + row.prefix_length > uint64_t(prefix_count)) return -1;
    h_gpu_input[r] = {device, row.state, row.prefix_offset, row.prefix_length};
  }
  std::fill_n(h_gpu_prefix, max_rows, 0);
  if (prefix_count) std::memcpy(h_gpu_prefix, prefix, size_t(prefix_count) * sizeof(int));
  gpu_prefix_count = prefix_count;
  gpu_results_ready = false;
  gpu_results_rows = rows;
  return 0;
}
void gpu_initialize(int rows) {
  CK(cudaMemcpyAsync(d_gpu_input, h_gpu_input, size_t(rows) * sizeof(*h_gpu_input), cudaMemcpyHostToDevice, st));
  CK(cudaMemcpyAsync(d_gpu_prefix, h_gpu_prefix, size_t(max_rows) * sizeof(int), cudaMemcpyHostToDevice, st));
  CK(gpu_grammar::launch_init(rows, d_gpu_input, max_rows, d_gpu_prefix,
      gpu_vocab->image, d_gpu_rows, d_gpu_results, st));
}
void gpu_prepare_head(int tune_rows) {
  if (!act_fp4 || head.il || wscale_e4m3 || (head.wf != 0 && head.wf != 1 && head.wf != 3))
    throw std::runtime_error("GPU grammar head requires the FP4 prepared head");
  if (sync_debug || prof || check_sparse_logits)
    throw std::runtime_error("GPU grammar excludes synchronous per-stage diagnostics");
  SkSel selection;
  if (!deterministic_gemm && sk_select(head, 1, selection))
    throw std::runtime_error("GPU grammar predication is unavailable for stream-K heads");
  ensure_sparse(gpu_grammar::COMPACT_CAPACITY);
  gpu_head_config = tune(head, tune_rows, q4, s4, g4, false);
  if (const char* value = getenv("H5_HEAD_KS")) {
    char* end = nullptr; const long split = strtol(value, &end, 10);
    if (end == value || *end || split < 1 || split > 12 || D / 128 / split < 4)
      throw std::runtime_error("H5_HEAD_KS must be 1..12 with at least 4 K-blocks per split");
    gpu_head_config.second = int(split);
  }
  gpu_head_partial_capacity = deterministic_gemm && gpu_head_config.second > 1 ? size_t(gpu_head_config.second) * VOCAB : 0;
  gpu_head_partials = deterministic_partials(gpu_head_partial_capacity);
  sparse_mat = head;
  sparse_mat.N = gpu_grammar::COMPACT_CAPACITY;
  sparse_mat.w4 = reinterpret_cast<uint4*>(sparse_dst.weights);
  sparse_mat.s4 = reinterpret_cast<__half*>(sparse_dst.scales);
  sparse_mat.rs = head.rs ? sparse_dst.row_scales : nullptr;
  sparse_mat.mean_scale = head.mean_scale ? sparse_dst.row_scales : nullptr;
}
void gpu_project_head() {
  SparseHeadSource source{head.w4, head.s4, head.mean_scale ? head.mean_scale : head.rs, VOCAB, D, sparse_format};
  const auto result = launch_sparse_head_gather_device_count(source, sparse_dst, d_gpu_ids,
      d_gpu_count, gpu_grammar::COMPACT_CAPACITY, st);
  if (!result) throw std::runtime_error("GPU grammar prepared head gather failed");
  const int ci = gpu_head_config.first, split = gpu_head_config.second;
  for (int sparse = 1; sparse >= 0; --sparse) {
    const TMat& matrix = sparse ? sparse_mat : head;
    CK(launch_gemm_fp4_predicated(ci, split, q4, s4, g4, 1, D, matrix.w4, matrix.s4,
        matrix.N, sparse ? sparse_logits : logits, matrix.N, false, st,
        matrix.mean_scale ? matrix.mean_scale : matrix.rs, matrix.wf, matrix.mean_scale != nullptr,
        gpu_head_partials, gpu_head_partial_capacity, deterministic_gemm, d_gpu_count,
        sparse ? 2u : gpu_grammar::COMPACT_CAPACITY + 1, sparse ? gpu_grammar::COMPACT_CAPACITY : uint32_t(VOCAB)));
  }
}
void gpu_copy_results(int rows) {
  CK(cudaMemcpyAsync(h_gpu_results, d_gpu_results, size_t(rows) * sizeof(*h_gpu_results), cudaMemcpyDeviceToHost, st));
}
bool gpu_result_errors() const {
  for (int r = 0; r < gpu_results_rows; ++r) if (h_gpu_results[r].error) return true;
  return false;
}
