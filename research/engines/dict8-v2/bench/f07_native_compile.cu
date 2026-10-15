// Compile only: references production kernels, without opening a GPU context.
#define H5_WARP_MICROSCOPE_STANDALONE
#include "../src/fp4.cu"

cudaError_t f07_compile_existing_routes(cudaFuncAttributes* attrs) {
  using namespace h5;
  cudaError_t e = cudaFuncGetAttributes(attrs, k_gemm_fp4<8, 2, 32, 3, 0>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4<8, 2, 32, 3, 2>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4<8, 2, 32, 3, 3>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4<4, 2, 8, 3, 3, 0, false, true>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4<4, 2, 8, 3, 0, 0, false, true>);
  if (e != cudaSuccess) return e;
  e = cudaFuncGetAttributes(attrs, k_gemm_fp4<8, 1, 64, 0, 1>);
  if (e != cudaSuccess) return e;
  return cudaFuncGetAttributes(attrs, k_gemm_fp4<4, 1, 64, 2, 3>);
}
