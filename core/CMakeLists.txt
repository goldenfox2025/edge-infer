cmake_minimum_required(VERSION 3.20)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(edge_infer_core LANGUAGES CXX)
endif()

option(EDGE_INFER_CORE_ENABLE_CUDA "Build the CUDA memory runtime" OFF)

add_library(core_headers INTERFACE)
add_library(EdgeInfer::core_headers ALIAS core_headers)
target_include_directories(core_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(core_headers INTERFACE cxx_std_17)

if(EDGE_INFER_CORE_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(core_cuda STATIC src/CudaMemoryPool.cpp)
  add_library(EdgeInfer::core_cuda ALIAS core_cuda)
  target_link_libraries(core_cuda PUBLIC core_headers CUDA::cudart CUDA::cuda_driver)
  set_target_properties(core_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
