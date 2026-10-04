cmake_minimum_required(VERSION 3.20)
project(edge_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
option(EDGE_INFER_BUILD_RUNTIME "Build the native CUDA inference runtime" ON)
option(EDGE_INFER_BUILD_PYTHON "Build the Python bindings" ${EDGE_INFER_BUILD_RUNTIME})
set(EDGE_INFER_CORE_ENABLE_CUDA ${EDGE_INFER_BUILD_RUNTIME})
set(EDGE_INFER_OPERATORS_ENABLE_CUDA ${EDGE_INFER_BUILD_RUNTIME})

if(EDGE_INFER_BUILD_PYTHON AND NOT EDGE_INFER_BUILD_RUNTIME)
  message(FATAL_ERROR "Python bindings require EDGE_INFER_BUILD_RUNTIME=ON")
endif()

if(EDGE_INFER_BUILD_RUNTIME)
  # 89 is the original RTX 4070 Laptop target. Other devices must select
  # their architecture explicitly; this does not establish device support.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to compile")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(Threads REQUIRED)
endif()

if(EDGE_INFER_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
endif()

add_subdirectory(core)
add_subdirectory(operators)

if(EDGE_INFER_BUILD_RUNTIME)
  add_subdirectory(runtime)
endif()

if(EDGE_INFER_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()
