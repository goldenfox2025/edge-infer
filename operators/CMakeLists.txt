cmake_minimum_required(VERSION 3.20)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(edge_infer_operators LANGUAGES CXX)
  include(CTest)
endif()

option(EDGE_INFER_OPERATORS_ENABLE_CUDA "Build CUDA operators and compatibility adapters" ON)

if(NOT TARGET core_headers)
  set(EDGE_INFER_CORE_ENABLE_CUDA "${EDGE_INFER_OPERATORS_ENABLE_CUDA}")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../core" "${CMAKE_CURRENT_BINARY_DIR}/core")
endif()

add_library(operators_core INTERFACE)
add_library(EdgeInfer::operators_core ALIAS operators_core)
target_include_directories(operators_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(operators_core INTERFACE core_headers)
target_compile_features(operators_core INTERFACE cxx_std_17)

if(EDGE_INFER_OPERATORS_ENABLE_CUDA)
  if(NOT TARGET core_cuda)
    message(FATAL_ERROR "CUDA operators require EDGE_INFER_CORE_ENABLE_CUDA=ON")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to compile")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass" CACHE PATH "CUTLASS source directory")
  if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS headers are missing. Initialize the pinned cutlass submodule or set CUTLASS_DIR.")
  endif()

  set(UNIFIED_CPU_OPERATOR_SOURCES
      src/cpu/add_cpu.cpp
      src/cpu/kv_cache_write_cpu.cpp
      src/cpu/matmul_cpu.cpp
      src/cpu/multiply_cpu.cpp
      src/cpu/rms_norm_cpu.cpp
      src/cpu/rope_cpu.cpp
      src/cpu/silu_cpu.cpp
  )

  set(UNIFIED_CUDA_OPERATOR_SOURCES
      src/cuda/add_cuda.cu
      src/cuda/attention_output_prefill_cuda.cu
      src/cuda/attention_scores_prefill_cuda.cu
      src/cuda/dynamic_flash_attention_cuda.cu
      src/cuda/flash_attention_prefill_cuda.cu
      src/cuda/gather_cuda.cu
      src/cuda/kv_cache_write_cuda.cu
      src/cuda/legacy_bridge.cu
      src/cuda/multiply_cuda.cu
      src/cuda/rms_norm_cuda.cu
      src/cuda/rope_cuda.cu
      src/cuda/sample_cuda.cu
      src/cuda/sampling_runtime.cu
      src/cuda/silu_cuda.cu
      src/cuda/softmax_cuda.cu
  )

  set(UNIFIED_CUDA_MATMUL_SOURCES
      src/cuda/matmul/awq_matmul_cuda.cu
      src/cuda/matmul/cublas_matmul_cuda.cu
      src/cuda/matmul/cutlass_matmul_cuda.cu
      src/cuda/matmul/matmul_cuda.cu
  )

  set(UNIFIED_CUDA_LEGACY_KERNEL_SOURCES
      src/cuda/legacy/add.cu
      src/cuda/legacy/add_rms.cu
      src/cuda/legacy/attention_cublas.cu
      src/cuda/legacy/attention_wmma.cu
      src/cuda/legacy/awq_gemv_gemm.cu
      src/cuda/legacy/basic.cu
      src/cuda/legacy/cutlass_c_wrapper.cu
      src/cuda/legacy/flash_attention.cu
      src/cuda/legacy/flash_attention_graph.cu
      src/cuda/legacy/flash_attention_variable.cu
      src/cuda/legacy/flash_attn_prefill.cu
      src/cuda/legacy/gather.cu
      src/cuda/legacy/gather_fa.cu
      src/cuda/legacy/gather_fa_graph.cu
      src/cuda/legacy/gather_fa_variable.cu
      src/cuda/legacy/gemm_gqa_prefill.cu
      src/cuda/legacy/gemv_mlp.cu
      src/cuda/legacy/gemv_qkv.cu
      src/cuda/legacy/gemv_qkv_rope.cu
      src/cuda/legacy/gemv_scores.cu
      src/cuda/legacy/matmul.cu
      src/cuda/legacy/matmul_awq.cu
      src/cuda/legacy/multiply.cu
      src/cuda/legacy/rope_device_offset.cu
      src/cuda/legacy/rope_kv_fused.cu
      src/cuda/legacy/sample.cu
      src/cuda/legacy/sample_with_prob.cu
      src/cuda/legacy/silu.cu
      src/cuda/legacy/silu_mul.cu
      src/cuda/legacy/softmax.cu
  )

  add_library(unified_operators STATIC
      ${UNIFIED_CPU_OPERATOR_SOURCES}
      ${UNIFIED_CUDA_OPERATOR_SOURCES}
      ${UNIFIED_CUDA_MATMUL_SOURCES}
      ${UNIFIED_CUDA_LEGACY_KERNEL_SOURCES}
  )

  add_library(EdgeInfer::unified_operators ALIAS unified_operators)
  target_include_directories(unified_operators PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  target_include_directories(unified_operators PRIVATE
      ${CUTLASS_DIR}/include
      ${CUTLASS_DIR}/tools/util/include
  )

  target_link_libraries(unified_operators PUBLIC
      operators_core
      core_cuda
      CUDA::cublas
  )

  set_target_properties(unified_operators PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  target_compile_options(unified_operators PRIVATE
      $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;--resource-usage;--display-error-number;-DDEBUG_AWQ>
      $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;--threads=0;--maxrregcount=64>
  )

  add_subdirectory(benchmarks)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
