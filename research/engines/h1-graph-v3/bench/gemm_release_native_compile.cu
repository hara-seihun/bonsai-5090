// Compile only: instantiate both production release arms; no GPU context is opened.
#define H5_WARP_MICROSCOPE_STANDALONE
#include "../src/fp4.cu"

cudaError_t gemm_release_compile_routes(cudaFuncAttributes* attrs) {
  using namespace h5;
  cudaError_t e = cudaFuncGetAttributes(attrs, k_gemm_fp4<4, 2, 8, 3, 3, 0, false, true>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4_tail<4, 2, 3, 0, TAIL_SIDE>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4<4, 2, 8, 3, 3, 4, false, true>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4_tail<4, 2, 3, 4, TAIL_SIDE>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4<8, 1, 8, 1, 3, 4, false, true>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4_tail<8, 1, 1, 4, TAIL_SIDE>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4<8, 1, 8, 2, 3, 4, false, true>);
  if (e != cudaSuccess) return e;
  return cudaFuncGetAttributes(attrs, k_gemm_fp4_tail<8, 1, 2, 4, TAIL_SIDE>);
}
