#pragma once
#include <cstdlib>
#include <string>
extern "C" int h5_record_begin(const char*);
extern "C" int h5_record_end();
#ifdef H5_PDLCLOCK
extern "C" void h5_pdl_begin_kernels();
extern "C" void h5_pdl_begin_fp4();
extern "C" void h5_pdl_begin_draft();
extern "C" void h5_pdl_end_kernels(const char*);
extern "C" void h5_pdl_end_fp4(const char*);
extern "C" void h5_pdl_end_draft(const char*);
#endif
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

namespace h5trace {
inline bool enabled() {
  static const bool value = [] { const char* s = std::getenv("H5_TRACE"); return s && std::string(s) == "1"; }();
  return value;
}
struct Range {
  bool active;
  explicit Range(const char* name) : active(enabled()) { if (active) nvtxRangePushA(name); }
  ~Range() { if (active) nvtxRangePop(); }
  Range(const Range&) = delete;
  Range& operator=(const Range&) = delete;
};
}
