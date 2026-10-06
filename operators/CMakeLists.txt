cmake_minimum_required(VERSION 3.20)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(edge_infer_operators LANGUAGES CXX)
  include(CTest)
endif()

option(EDGE_INFER_OPERATORS_ENABLE_CUDA "Build direct CUDA operators" ON)
if(NOT TARGET core_headers)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../core" "${CMAKE_CURRENT_BINARY_DIR}/core")
endif()

add_library(operators_core INTERFACE)
add_library(EdgeInfer::operators_core ALIAS operators_core)
target_include_directories(operators_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(operators_core INTERFACE core_headers)
target_compile_features(operators_core INTERFACE cxx_std_17)

if(EDGE_INFER_OPERATORS_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to compile")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(operators_cuda STATIC
    src/cuda/execution.cu
    src/cuda/conditioning.cu
    src/cuda/random.cu
    src/cuda/add_cuda.cu
    src/cuda/multiply_cuda.cu
    src/cuda/rms_norm_cuda.cu
    src/cuda/silu_cuda.cu
    src/cuda/kernels/awq_gemv_gemm.cu
    src/cuda/kernels/flash_attn_prefill.cu
    src/cuda/kernels/flash_attention_variable.cu
    src/cuda/kernels/flash_attention_graph.cu
    src/cuda/kernels/gather_fa_variable.cu
    src/cuda/kernels/gather_fa_graph.cu)
  add_library(EdgeInfer::operators_cuda ALIAS operators_cuda)
  target_include_directories(operators_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(operators_cuda PUBLIC operators_core CUDA::cublas CUDA::cudart)
  set_target_properties(operators_cuda PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(operators_cuda PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;--threads=0;--maxrregcount=64>)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
