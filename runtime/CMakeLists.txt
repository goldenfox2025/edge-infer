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
  src/speech/qwen_tts_conditioning.cpp
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
  add_executable(qwen3_session_test tests/qwen3_session_test.cpp)
  target_link_libraries(qwen3_session_test PRIVATE EdgeInfer::runtime)
  add_test(NAME qwen3_session_test COMMAND qwen3_session_test)
  set_tests_properties(qwen3_session_test PROPERTIES
    LABELS "runtime;cuda;gpu;session" SKIP_RETURN_CODE 77 TIMEOUT 120)

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
