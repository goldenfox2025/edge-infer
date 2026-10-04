cmake_minimum_required(VERSION 3.20)
project(llm_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
option(LLM_INFER_BUILD_ENGINE "Build the CUDA operators and Python inference module" ON)
set(LLM_CORE_ENABLE_CUDA ${LLM_INFER_BUILD_ENGINE})
set(LLM_OPERATORS_ENABLE_CUDA ${LLM_INFER_BUILD_ENGINE})

if(LLM_INFER_BUILD_ENGINE)
  # 89 is the original RTX 4070 Laptop target. Other devices must select
  # their architecture explicitly; this does not establish device support.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to compile")
  endif()

  set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cutlass" CACHE PATH "CUTLASS source directory")
  if(NOT EXISTS "${CUTLASS_DIR}/include/cute/tensor.hpp")
    message(FATAL_ERROR
      "CUTLASS headers are missing. Run git submodule update --init --recursive -- cutlass, "
      "or set CUTLASS_DIR to the recorded CUTLASS checkout. "
      "Use -DLLM_INFER_BUILD_ENGINE=OFF to run the standalone workspace tests.")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  find_package(Threads REQUIRED)

endif()

add_subdirectory(core)
add_subdirectory(operators)

if(LLM_INFER_BUILD_ENGINE)
  pybind11_add_module(model_bridge MODULE
    bindings/python/model_bridge.cpp
    bindings/python/model_initializer.cpp
    runtime/src/inference.cpp
    runtime/src/thread_pool.cpp
    runtime/src/qwen.cpp
    runtime/src/qwen3.cpp
    runtime/src/qwen3_forward.cpp
    runtime/src/qwen3_prefill.cpp
    runtime/src/speculative_decoder.cpp
  )

  target_link_libraries(model_bridge PRIVATE
    unified_operators
    CUDA::cublas
    CUDA::cudart
    CUDA::cuda_driver
    Threads::Threads
  )
  target_include_directories(model_bridge PRIVATE
    runtime/include
    bindings/python
  )
  target_compile_options(model_bridge PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--threads=0;--maxrregcount=64>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;--resource-usage;--display-error-number>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>
  )
  set_target_properties(model_bridge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
