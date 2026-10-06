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

if(NOT TARGET operators_cuda)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../operators"
                   "${CMAKE_CURRENT_BINARY_DIR}/operators")
endif()

if(NOT TARGET core_cuda)
  edge_infer_enable_core_cuda()
endif()

add_library(inference_runtime STATIC
  src/inference.cpp
  src/thread_pool.cpp
  src/qwen.cpp
  src/qwen3.cpp
  src/speculative_decoder.cpp
  src/speech/qwen_tts_conditioning.cpp
)
add_library(EdgeInfer::runtime ALIAS inference_runtime)
target_include_directories(inference_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(inference_runtime PUBLIC cxx_std_17)
target_link_libraries(inference_runtime PUBLIC
  operators_cuda
  core_cuda
  Threads::Threads
)
set_target_properties(inference_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../core/tests/support")
  add_executable(cuda_workspace_arena_test tests/cuda_workspace_arena_test.cpp)
  target_link_libraries(cuda_workspace_arena_test PRIVATE EdgeInfer::runtime)
  add_test(NAME cuda_workspace_arena_test COMMAND cuda_workspace_arena_test)
  set_tests_properties(cuda_workspace_arena_test PROPERTIES
    LABELS "runtime;cuda;gpu;storage" SKIP_RETURN_CODE 77 TIMEOUT 30)
  add_executable(tensor_storage_test tests/tensor_storage_test.cpp ../core/tests/support/allocation_probe.cpp)
  target_link_libraries(tensor_storage_test PRIVATE EdgeInfer::runtime)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND UNIX)
    target_compile_definitions(tensor_storage_test PRIVATE EDGE_INFER_TEST_CUDA_WRAPPING=1)
    target_link_options(tensor_storage_test PRIVATE
      LINKER:--wrap=cudaMalloc LINKER:--wrap=cudaMallocAsync
      LINKER:--wrap=cudaFree LINKER:--wrap=cudaFreeAsync)
  endif()
  add_test(NAME tensor_storage_test COMMAND tensor_storage_test)
  set_tests_properties(tensor_storage_test PROPERTIES
    LABELS "runtime;cuda;gpu;storage" SKIP_RETURN_CODE 77 TIMEOUT 30)
  add_executable(qwen3_session_test tests/qwen3_session_test.cpp ../core/tests/support/allocation_probe.cpp tests/test_cuda_rng.cu)
  set_target_properties(qwen3_session_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND UNIX)
    target_compile_definitions(qwen3_session_test PRIVATE EDGE_INFER_TEST_CUDA_WRAPPING=1)
    target_link_options(qwen3_session_test PRIVATE
      LINKER:--wrap=cudaMalloc LINKER:--wrap=cudaMallocAsync
      LINKER:--wrap=cudaFree LINKER:--wrap=cudaFreeAsync)
  endif()
  target_link_libraries(qwen3_session_test PRIVATE EdgeInfer::runtime)
  add_test(NAME qwen3_session_test COMMAND qwen3_session_test)
  set_tests_properties(qwen3_session_test PROPERTIES
    LABELS "runtime;cuda;gpu;session" SKIP_RETURN_CODE 77 TIMEOUT 120)
  add_executable(qwen2_float_test tests/qwen2_float_test.cpp)
  target_link_libraries(qwen2_float_test PRIVATE EdgeInfer::runtime)
  add_test(NAME qwen2_float_test COMMAND qwen2_float_test)
  set_tests_properties(qwen2_float_test PROPERTIES LABELS "runtime;cuda;gpu;reference" SKIP_RETURN_CODE 77 TIMEOUT 120)
  add_executable(inference_generation_test tests/inference_generation_test.cpp)
  target_link_libraries(inference_generation_test PRIVATE EdgeInfer::runtime)
  add_test(NAME inference_generation_test COMMAND inference_generation_test)
  set_tests_properties(inference_generation_test PROPERTIES LABELS "runtime;cuda;gpu;generation" SKIP_RETURN_CODE 77 TIMEOUT 120)

  add_executable(qwen_tts_conditioning_parity tests/qwen_tts_conditioning_parity.cpp)
  target_link_libraries(qwen_tts_conditioning_parity PRIVATE EdgeInfer::runtime)
  set(EDGE_INFER_QWEN_TTS_FIXTURES "" CACHE PATH "Optional Torch conditioning reference fixture directory")
  if(EDGE_INFER_QWEN_TTS_FIXTURES)
    add_test(NAME qwen_tts_conditioning_parity
             COMMAND qwen_tts_conditioning_parity "${EDGE_INFER_QWEN_TTS_FIXTURES}")
    set_tests_properties(qwen_tts_conditioning_parity PROPERTIES
      LABELS "speech;cuda;gpu;reference" SKIP_RETURN_CODE 77 TIMEOUT 60)
  endif()
  add_executable(runtime_callback_test tests/runtime_callback_test.cpp)
  target_link_libraries(runtime_callback_test PRIVATE EdgeInfer::runtime)
  add_test(NAME runtime_callback_test COMMAND runtime_callback_test)
  set_tests_properties(runtime_callback_test PROPERTIES LABELS "runtime;cpu" TIMEOUT 15)
endif()
