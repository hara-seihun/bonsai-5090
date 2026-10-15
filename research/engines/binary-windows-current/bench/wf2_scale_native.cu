// Compile-only production GEMM instantiations for paired WF2 scale-folding SASS.
#define H5_WARP_MICROSCOPE_STANDALONE
#include "../src/fp4.cu"
void wf2_scale_native() {
  cudaFuncSetAttribute(h5::k_gemm_fp4<8, 1, 128, 0, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       h5::WsGeom<8, 1, 128, 2>::SMEM);
  cudaFuncSetAttribute(h5::k_gemm_fp4<4, 2, 64, 3, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       h5::WsGeom<4, 2, 64, 2>::SMEM);
}
