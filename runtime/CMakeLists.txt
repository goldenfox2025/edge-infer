cmake_minimum_required(VERSION 3.20)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(edge_infer_runtime LANGUAGES CXX)
  include(CTest)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to compile")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(EDGE_INFER_CORE_ENABLE_CUDA ON)
  set(EDGE_INFER_OPERATORS_ENABLE_CUDA ON)
endif()

find_package(Threads REQUIRED)

if(NOT TARGET unified_operators)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../operators"
                   "${CMAKE_CURRENT_BINARY_DIR}/operators")
endif()

add_library(inference_runtime STATIC
  src/inference.cpp
  src/thread_pool.cpp
  src/qwen.cpp
  src/qwen3.cpp
  src/qwen3_forward.cpp
  src/qwen3_prefill.cpp
  src/speculative_decoder.cpp
)
add_library(EdgeInfer::runtime ALIAS inference_runtime)
target_include_directories(inference_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(inference_runtime PUBLIC cxx_std_17)
target_link_libraries(inference_runtime PUBLIC
  unified_operators
  Threads::Threads
)
set_target_properties(inference_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING)
  add_executable(runtime_callback_test tests/runtime_callback_test.cpp)
  target_link_libraries(runtime_callback_test PRIVATE EdgeInfer::runtime)
  add_test(NAME runtime_callback_test COMMAND runtime_callback_test)
  set_tests_properties(runtime_callback_test PROPERTIES LABELS "runtime;cpu" TIMEOUT 15)
endif()
