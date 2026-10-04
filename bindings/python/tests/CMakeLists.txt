cmake_minimum_required(VERSION 3.20)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(edge_infer_python_callback_tests LANGUAGES CXX)
  include(CTest)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
endif()

find_package(Threads REQUIRED)
pybind11_add_module(_edge_callback_boundary_test MODULE callback_boundary_module.cpp)
target_compile_features(_edge_callback_boundary_test PRIVATE cxx_std_17)
target_include_directories(_edge_callback_boundary_test PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(_edge_callback_boundary_test PRIVATE Threads::Threads)

if(BUILD_TESTING)
  add_test(NAME python_callback_boundary_test
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/callback_boundary_test.py"
            "$<TARGET_FILE_DIR:_edge_callback_boundary_test>"
  )
  set_tests_properties(python_callback_boundary_test PROPERTIES
                       LABELS "bindings;python;cpu" TIMEOUT 15)
endif()
