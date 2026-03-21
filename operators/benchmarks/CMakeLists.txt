# 设置最低CMake版本
cmake_minimum_required(VERSION 3.15)

# 创建可执行程序
add_executable(avx_matmul_bench
    EXCLUDE_FROM_ALL
    avx_matmul_bench.cpp
    ${CMAKE_SOURCE_DIR}/backend/cpp/src/CudaMemoryPool.cpp
)

# 头文件包含路径
target_include_directories(avx_matmul_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/backend/cpp/include
    ${CMAKE_SOURCE_DIR}/operators/include
)

# 链接 CUDA 运行时（tensor.hpp 依赖 CUDA 头，链接上更稳妥）
target_link_libraries(avx_matmul_bench PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    unified_operators
)

# 编译选项：开启优化与 AVX2/FMA
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(avx_matmul_bench PRIVATE -O3 -march=native -funroll-loops -flto -mavx2 -mfma)
else()
  target_compile_options(avx_matmul_bench PRIVATE -O2 -mavx2 -mfma)
endif() 
