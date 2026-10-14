#pragma once
float* gpu_select_values = nullptr;
int* gpu_select_ids = nullptr;
void ensure_gpu_select() {
  if (gpu_select_values) return;
  gpu_select_values = dalloc<float>(gpu_grammar::SELECT_PARTS * gpu_grammar::SELECT_TOP);
  gpu_select_ids = dalloc<int>(gpu_grammar::SELECT_PARTS * gpu_grammar::SELECT_TOP);
}
gpu_grammar::SelectWorkspace gpu_select_workspace() const { return {gpu_select_values,gpu_select_ids}; }
