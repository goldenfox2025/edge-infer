cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

set(CMAKE_CUDA_FLAGS_DEBUG
    "-lineinfo -O2 --generate-line-info --resource-usage --display-error-number --threads 0 --maxrregcount=64"
)
set(CMAKE_CXX_FLAGS_DEBUG "-g1 -O2 -mavx2 -mfma")

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

target_include_directories(unified_operators PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/backend/cpp/include
    ${CMAKE_SOURCE_DIR}
)
target_include_directories(unified_operators PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/cutlass")
target_include_directories(unified_operators PUBLIC
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

target_link_libraries(unified_operators PUBLIC
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(unified_operators PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(unified_operators PRIVATE
    $<$<CONFIG:Debug>:-g1;-O2>
    $<$<CONFIG:Release>:-O3;-march=native;-funroll-loops;-DNDEBUG>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-O2;--generate-line-info;--resource-usage;--display-error-number;-DDEBUG_AWQ>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;--use_fast_math;-DNDEBUG>
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler;--expt-relaxed-constexpr;--extended-lambda;-gencode;arch=compute_89,code=sm_89;--threads;0;--maxrregcount=64>
)

target_compile_options(unified_operators PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2;-mfma>
)

add_subdirectory(benchmarks)
add_subdirectory(tests)
