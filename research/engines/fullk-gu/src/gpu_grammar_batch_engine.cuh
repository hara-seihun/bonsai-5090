int* gpu_wide_sources=nullptr;
int* gpu_wide_ranks=nullptr;
uint32_t* gpu_wide_count=nullptr;
void ensure_gpu_batch(int rows) {
  if(!gpu_wide_sources) {
    gpu_wide_sources=dalloc<int>(max_rows);
    gpu_wide_ranks=dalloc<int>(max_rows);
    gpu_wide_count=dalloc<uint32_t>(1);
  }
  const size_t bytes=deterministic_gemm && gpu_head_config.second>1?
      size_t(gpu_head_config.second)*rows*VOCAB:0;
  gpu_head_partial_capacity=std::max(gpu_head_partial_capacity,bytes);
  gpu_head_partials=deterministic_partials(gpu_head_partial_capacity);
}
void gpu_project_sparse_only() {
  SparseHeadSource source{head.w4,head.s4,head.mean_scale?head.mean_scale:head.rs,VOCAB,D,sparse_format};
  const auto result=launch_sparse_head_gather_device_count(source,sparse_dst,d_gpu_ids,
      d_gpu_count,gpu_grammar::COMPACT_CAPACITY,st);
  if(!result) throw std::runtime_error("GPU grammar prepared head gather failed");
  CK(launch_gemm_fp4_predicated(gpu_head_config.first,gpu_head_config.second,q4,s4,g4,1,D,
      sparse_mat.w4,sparse_mat.s4,sparse_mat.N,sparse_logits,sparse_mat.N,false,st,
      sparse_mat.mean_scale?sparse_mat.mean_scale:sparse_mat.rs,sparse_mat.wf,
      sparse_mat.mean_scale!=nullptr,gpu_head_partials,gpu_head_partial_capacity,
      deterministic_gemm,d_gpu_count,2u,gpu_grammar::COMPACT_CAPACITY));
}
void gpu_project_wide_batch(int rows) {
  CK(launch_gemm_fp4_predicated(gpu_head_config.first,gpu_head_config.second,q4,s4,g4,rows,D,
      head.w4,head.s4,VOCAB,logits,VOCAB,false,st,head.mean_scale?head.mean_scale:head.rs,
      head.wf,head.mean_scale!=nullptr,gpu_head_partials,gpu_head_partial_capacity,
      deterministic_gemm,gpu_wide_count,1u,uint32_t(rows)));
}
