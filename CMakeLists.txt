cmake_minimum_required(VERSION 3.28)
project(bonsai5090 LANGUAGES C CXX CUDA)

# Serving engine for Ternary Bonsai 2 27B (PTQ1_0) on one RTX 5090 (sm_120a).
# llama.cpp (PrismML fork) is used only to read the GGUF vocabulary and chat template.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_BUILD_RPATH "$ORIGIN")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
find_package(CUDAToolkit 12.8 REQUIRED)

# --- Tokenizer: PrismML llama.cpp fork, pinned commit, CPU-only build -------------------
set(PRISM_LLAMA_COMMIT 6067dbea986ed1359db98c42ab06eb49426bbf16)
set(PRISM_LLAMA_URL https://github.com/PrismML-Eng/llama.cpp/archive/${PRISM_LLAMA_COMMIT}.tar.gz)
set(PRISM_LLAMA_SHA256 079a80ae6047b75f3df4593cb90c06a17b4dbe6c751589c70dd31f631a19b80f)
set(LLAMA_CPP_SOURCE_DIR "" CACHE PATH "Optional local checkout of PrismML-Eng/llama.cpp at ${PRISM_LLAMA_COMMIT}")

set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

include(FetchContent)
if(LLAMA_CPP_SOURCE_DIR)
  FetchContent_Declare(prism_llama SOURCE_DIR ${LLAMA_CPP_SOURCE_DIR} EXCLUDE_FROM_ALL)
else()
  FetchContent_Declare(prism_llama URL ${PRISM_LLAMA_URL} URL_HASH SHA256=${PRISM_LLAMA_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON EXCLUDE_FROM_ALL)
endif()
FetchContent_MakeAvailable(prism_llama)
set(llama_src ${prism_llama_SOURCE_DIR})

# --- Engine -------------------------------------------------------------------------------
set(ENGINE engine)
add_library(quality_custom SHARED
  ${ENGINE}/src/engine.cu ${ENGINE}/src/graph_memory.cu ${ENGINE}/src/kernels.cu ${ENGINE}/src/fp4.cu
  ${ENGINE}/src/sk.cu ${ENGINE}/src/draft.cu ${ENGINE}/src/sampling_gpu.cu
  ${ENGINE}/src/f02-producer.cu ${ENGINE}/src/f02-continuation.cu ${ENGINE}/src/gguf.cpp
  ${ENGINE}/src/gdn_q6.cu ${ENGINE}/dispatch.cu ${ENGINE}/api.cpp)
target_include_directories(quality_custom PRIVATE ${ENGINE}/src ${llama_src}/vendor)
target_compile_definitions(quality_custom PRIVATE
  H5_PROFILE_GEMM H5_PDL_EARLY_GEMM_RELEASE H5_PDL_GDN_STATE_OVERLAP H5_NO_L2PF
  H5_PROFILE_SCALE_CARRIER H5_WEIGHT_TMA_CACHE_POLICY=2
  $<$<PLATFORM_ID:Windows>:NOMINMAX _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES>)
target_compile_options(quality_custom PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--generate-code=arch=compute_120a,code=sm_120a;-lineinfo>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/O2,/EHsc,/bigobj>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Xcompiler=-O3>"
  "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/O2;/EHsc;/bigobj>"
  "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3>")
target_link_libraries(quality_custom PRIVATE CUDA::cudart CUDA::cublas llama ggml ggml-base)

add_executable(quality_length_bench ${ENGINE}/length_bench.cpp)
target_compile_options(quality_length_bench PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/O2;/EHsc,-O3>)
target_link_libraries(quality_length_bench PRIVATE quality_custom llama ggml ggml-base)
